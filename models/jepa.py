import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.layers import (
    PatchEmbed,
    Block,
    RotaryPositionEmbedding2D,
    PositionGetter,
)

from .autoencoder import Encoder, Predictor


# ================================================================
# Target branch (online + EMA copies share this class)
# ================================================================

class TargetEncoder(nn.Module):
    """
    Target view -> patch embed -> [mask tokens] -> 4 self-attn -> norm

    Online copy:  called with a mask   -> query tokens for the predictor
    EMA copy:     called without mask  -> latent targets

    Input:  [B, S, 3, H, W]
    Output: [B, S, P, D]
    """

    def __init__(
        self,
        *,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=512,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        qk_norm=True,
        rope_freq=0,
        init_values=0.01,
        block_fn=Block,
        norm_layer=nn.LayerNorm,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Mask token lives here so the EMA copy mirrors the online one.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positions: RoPE if enabled, otherwise a learned absolute
        # embedding so masked tokens are distinguishable.
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.grid = img_size // patch_size
        if self.rope is None:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope=self.rope,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

    def _abs_pos(self, Hp, Wp, dtype):
        pos = self.pos_embed
        if Hp != self.grid or Wp != self.grid:
            pos = pos.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(Hp, Wp), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, Hp * Wp, -1)
        return pos.to(dtype)

    def forward(self, images, mask=None):
        """
        images: [B, S, 3, H, W]
        mask:   optional [B*S, P], 1 = masked
        """
        B, S, C, H, W = images.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        x = self.patch_embed(images.reshape(B * S, C, H, W))       # [B*S, P, D]
        _, P, D = x.shape

        if mask is not None:
            x = torch.where(mask.bool()[..., None], self.mask_token.to(x.dtype), x)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, Hp, Wp, device=x.device).reshape(B * S, P, -1)
        else:
            x = x + self._abs_pos(Hp, Wp, x.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x, pos=pos)

        x = self.norm(x)
        return x.reshape(B, S, P, D)



# ================================================================
# Vista JEPA
# ================================================================

class VistaJEPA(nn.Module):
    """
    JEPA model for 3D.
    """

    def __init__(
        self,
        *,
        img_size=518,
        patch_size=14,
        in_chans=3,

        # source encoder (scene representation)
        embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,

        # target branch + predictor share this width (no target projector)
        target_embed_dim=512,
        target_depth=4,
        target_num_heads=16,
        predictor_depth=8,
        predictor_num_heads=16,

        # masking
        mask_mode="block",               # "block" (I-JEPA) or "random"
        mask_ratio=0.75,                 # used when mask_mode == "random"
        num_blocks=4,                    # I-JEPA npred
        block_scale=(0.15, 0.20),        # I-JEPA pred_mask_scale
        block_aspect=(0.75, 1.5),        # I-JEPA aspect_ratio
        enc_mask_scale=(0.85, 1.0),      # I-JEPA enc_mask_scale (context block)
        min_keep=4,                      # I-JEPA min_keep
        allow_overlap=False,             # I-JEPA: context may not overlap target blocks
        full_mask_prob=0.25,             # Vista: fraction of fully masked target views

        # EMA
        ema_start=0.996,
        ema_end=1.0,

        # other
        n_storage_tokens=0,
        rope_freq=0,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.target_embed_dim = target_embed_dim

        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.num_blocks = num_blocks
        self.block_scale = block_scale
        self.block_aspect = block_aspect
        self.enc_mask_scale = enc_mask_scale
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.full_mask_prob = full_mask_prob
        self._mask_itr = -1

        self.ema_start = ema_start
        self.ema_end = ema_end

        # 1. Source encoder: its own network, learns the scene representation
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            n_storage_tokens=n_storage_tokens,
            rope_freq=rope_freq,
            use_reentrant=False,
        )

        # 2. Online target branch (sees the context block of the target view)
        self.target_branch = TargetEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=target_embed_dim,
            depth=target_depth,
            num_heads=target_num_heads,
            rope_freq=rope_freq,
            gradient_checkpointing=gradient_checkpointing,
        )

        # 3. EMA target encoder (sees the full target view, produces targets)
        self.target_encoder = copy.deepcopy(self.target_branch)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

        # 4. Predictor: self-attends over target tokens, cross-attends to source tokens
        #    (projects source features embed_dim -> target_embed_dim itself)
        self.predictor = Predictor(
            embed_dim=embed_dim,
            decoder_embed_dim=target_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            self_attn=True,
            rope_freq=rope_freq,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.predictor_norm = nn.LayerNorm(target_embed_dim)

    # ------------------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    # ------------------------------------------------------------
    # EMA: target_encoder <- target_branch
    # (train_jepa.py does this itself with I-JEPA's linear momentum schedule;
    #  kept here for other training scripts)
    # ------------------------------------------------------------
    @torch.no_grad()
    def update_target_encoder(self, step=None, total_steps=None):
        if step is None or total_steps is None:
            m = self.ema_start
        else:
            t = min(step / max(total_steps, 1), 1.0)
            m = self.ema_start + t * (self.ema_end - self.ema_start)        # linear, as in I-JEPA

        torch._foreach_lerp_(
            list(self.target_encoder.parameters()),
            list(self.target_branch.parameters()),
            1.0 - m,
        )
        for b_t, b_o in zip(self.target_encoder.buffers(), self.target_branch.buffers()):
            b_t.copy_(b_o)

    # ------------------------------------------------------------
    # Masking, following I-JEPA's MaskCollator (1 = masked)
    #
    # Returns
    #   context_mask : [n, P]  1 = hidden from the target branch (outside the context block)
    #   target_mask  : [n, P]  1 = inside a target block (scored by the loss)
    # ------------------------------------------------------------
    def _sample_block_size(self, Hp, Wp, scale, aspect_ratio, generator):
        # one random number drives both scale and aspect ratio (as in I-JEPA)
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(Hp * Wp * mask_scale)
        min_ar, max_ar = aspect_ratio
        ar = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))
        h = max(1, min(h, Hp - 1))
        w = max(1, min(w, Wp - 1))
        return h, w

    def _sample_block_mask(self, Hp, Wp, h, w, acceptable_regions=None):
        tries, timeout = 0, 20
        while True:
            top = torch.randint(0, Hp - h, (1,)).item()
            left = torch.randint(0, Wp - w, (1,)).item()
            mask = torch.zeros(Hp, Wp, dtype=torch.bool)
            mask[top:top + h, left:left + w] = True
            # restrict to acceptable regions (i.e. outside the target blocks)
            if acceptable_regions is not None:
                for k in range(max(len(acceptable_regions) - tries, 0)):
                    mask &= acceptable_regions[k]
            if mask.sum() > self.min_keep:
                break
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = 20
        complement = torch.ones(Hp, Wp, dtype=torch.bool)
        complement[top:top + h, left:left + w] = False
        return mask, complement

    def block_masking(self, n, Hp, Wp, device):
        # block sizes shared across the batch for this iteration (I-JEPA: seeded by an iteration counter)
        self._mask_itr += 1
        g = torch.Generator()
        g.manual_seed(self._mask_itr)
        p_h, p_w = self._sample_block_size(Hp, Wp, self.block_scale, self.block_aspect, g)
        e_h, e_w = self._sample_block_size(Hp, Wp, self.enc_mask_scale, (1.0, 1.0), g)

        context_mask = torch.ones(n, Hp, Wp, dtype=torch.bool)    # start fully hidden
        target_mask = torch.zeros(n, Hp, Wp, dtype=torch.bool)

        for i in range(n):
            # Vista: some views are fully masked -> pure cross-view prediction
            if torch.rand(1).item() < self.full_mask_prob:
                target_mask[i] = True
                continue

            # target blocks (I-JEPA npred)
            complements = []
            for _ in range(self.num_blocks):
                m, comp = self._sample_block_mask(Hp, Wp, p_h, p_w)
                target_mask[i] |= m
                complements.append(comp)

            # context block (I-JEPA nenc = 1), kept outside the target blocks
            acceptable = None if self.allow_overlap else complements
            m, _ = self._sample_block_mask(Hp, Wp, e_h, e_w, acceptable_regions=acceptable)
            context_mask[i] = ~m

        return (context_mask.flatten(1).float().to(device),
                target_mask.flatten(1).float().to(device))

    def random_masking(self, n, P, device):
        len_keep = int(P * (1.0 - self.mask_ratio))
        ids_restore = torch.rand(n, P, device=device).argsort(1).argsort(1)
        mask = torch.ones(n, P, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return mask, mask                      # hidden patches are the scored patches

    def sample_mask(self, n, Hp, Wp, device):
        if self.mask_mode == "block":
            return self.block_masking(n, Hp, Wp, device)
        return self.random_masking(n, Hp * Wp, device)

    # ------------------------------------------------------------
    @staticmethod
    def jepa_loss(pred, target, mask):
        """Smooth-L1 (I-JEPA) averaged over target-block patches."""
        loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    # ------------------------------------------------------------
    def forward(self, reference_imgs, target_imgs, mask=None):
        """
        reference_imgs: [B, S_ref, 3, H, W]
        target_imgs:    [B, S_tgt, 3, H, W]
        mask:           optional, either
                          (context_mask, target_mask), each [B*S_tgt, P], 1 = masked
                          or a single [B*S_tgt, P] mask used for both

        Returns: loss, pred [B,S_tgt,P,D], target [B,S_tgt,P,D], target_mask [B,S_tgt,P]
        """
        B, S_t, C, H, W = target_imgs.shape
        assert reference_imgs.shape[0] == B
        Hp, Wp = H // self.patch_size, W // self.patch_size
        P, D = Hp * Wp, self.target_embed_dim

        # 1. Scene representation from source views (predictor projects to D)
        ref_feats, psi = self.encoder(reference_imgs)
        reference_tokens = ref_feats[:, :, psi:, :]                             # [B,S_ref,P,embed_dim]

        # 2. Latent targets: EMA encoder on the full target view
        with torch.no_grad():
            target = self.target_encoder(target_imgs)                           # [B,S_t,P,D]
            target = F.layer_norm(target, (D,))

        # 3. Masks: context (input to the target branch) and target blocks (loss)
        if mask is None:
            context_mask, target_mask = self.sample_mask(B * S_t, Hp, Wp, target_imgs.device)
        elif isinstance(mask, (tuple, list)):
            context_mask, target_mask = mask
        else:
            context_mask = target_mask = mask

        target_tokens = self.target_branch(target_imgs, mask=context_mask)      # [B,S_t,P,D]

        # 4. Cross-view prediction
        pred = self.predictor(target_tokens=target_tokens, reference_tokens=reference_tokens)
        pred = self.predictor_norm(pred.reshape(B, S_t, P, D))

        # 5. Loss on target blocks only
        target_mask = target_mask.reshape(B, S_t, P)
        loss = self.jepa_loss(pred, target, target_mask)

        return loss, pred, target, target_mask


def vista_jepa_base(
    patch_size=14,
    **kwargs,
):

    model = VistaJEPA(
        patch_size=patch_size,
        img_size=518,

        # --------------------------------------------------------
        # Online encoder
        # --------------------------------------------------------

        embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,

        # --------------------------------------------------------
        # Target encoder (predictor runs at this width)
        # --------------------------------------------------------

        target_embed_dim=512,
        target_depth=4,
        target_num_heads=16,

        # --------------------------------------------------------
        # Predictor
        # --------------------------------------------------------

        predictor_depth=8,
        predictor_num_heads=16,

        # --------------------------------------------------------
        # Masking
        # --------------------------------------------------------

        mask_ratio=0.75,

        # --------------------------------------------------------
        # EMA
        # --------------------------------------------------------

        ema_start=0.996,
        ema_end=1.0,

        **kwargs,
    )

    return model



def vista_jepa_large(
    patch_size=14,
    **kwargs,
):

    model = VistaJEPA(
        patch_size=patch_size,
        img_size=518,

        # Online encoder
        embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,

        # Target encoder (predictor runs at this width)
        target_embed_dim=768,
        target_depth=4,
        target_num_heads=16,

        # Predictor
        predictor_depth=8,
        predictor_num_heads=16,

        mask_ratio=0.75,
        ema_start=0.996,
        ema_end=1.0,

        **kwargs,
    )

    return model