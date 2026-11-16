import math
import random
from functools import partial
from typing import Any, Sequence, Tuple, Union, Optional

import torch
import torch.nn.init
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .autoencoder import EncoderV2, Decoder
from models.layers import PatchEmbed, LayerScale, Block, RotaryPositionEmbedding2D, PositionGetter
from .utils import named_apply


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()


# ================================================================
# Target encoder
# ================================================================

class TargetEncoder(nn.Module):
    """
    Masked target view -> target tokens for the decoder.

        target image
            ↓
        patch embedding
            ↓
        mask tokens at masked positions
            ↓
        self-attention × depth   (Q = K = V = target, per view)
            ↓
        target tokens [B, S, P, D]

    Input:
        images: [B, S, 3, H, W]
        mask:   [B*S, P]   1 = masked, 0 = visible

    Output:
        [B, S, P, D]
    """

    def __init__(
        self,
        *,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        norm_layer=nn.LayerNorm,
        block_fn=Block,
        gradient_checkpointing=False,
        use_cls=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing

        # Optional per-view CLS: position 0, patches at +1 (same convention as EncoderV2)
        self.use_cls = use_cls
        self.patch_start_idx = 1 if use_cls else 0
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # ---------------------------------------------------------
        # Patch embedding + mask token
        # ---------------------------------------------------------

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ---------------------------------------------------------
        # Positions: RoPE, or a learned absolute embedding when RoPE
        # is off (otherwise all mask tokens are identical)
        # ---------------------------------------------------------

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.grid = img_size // patch_size
        if self.rope is None:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # ---------------------------------------------------------
        # Self-attention blocks
        # ---------------------------------------------------------

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

    def _abs_pos(self, Hp, Wp, dtype):
        pos = self.pos_embed
        if Hp != self.grid or Wp != self.grid:
            pos = pos.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(Hp, Wp), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, Hp * Wp, -1)
        return pos.to(dtype)

    def forward(self, images, mask):
        """Returns [B, S, patch_start_idx + P, D] (CLS first when use_cls)."""
        B, S, C, H, W = images.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # [B*S, P, D]
        x = self.patch_embed(images.reshape(B * S, C, H, W))
        _, P, D = x.shape

        # Replace masked patches with the mask token
        x = torch.where(mask.reshape(B * S, P).bool()[..., None], self.mask_token.to(x.dtype), x)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, Hp, Wp, device=x.device).reshape(B * S, P, -1)
        else:
            x = x + self._abs_pos(Hp, Wp, x.dtype)

        if self.use_cls:
            x = torch.cat([self.cls_token.to(x.dtype).expand(B * S, -1, -1), x], dim=1)
            if pos is not None:
                pos = torch.cat([torch.zeros(B * S, 1, 2, device=pos.device, dtype=pos.dtype), pos + 1], dim=1)
        P = x.shape[1]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x, pos=pos)

        return x.reshape(B, S, P, D)
# ================================================================
# Vista V4
# ================================================================

class VistaV4(nn.Module):
    """
    Three view sets per scene:
        A: clean views            (reference_imgs[:, :num_clean])
        B: masked views           (reference_imgs[:, num_clean:])
        C: masked target views    (target_imgs, never seen by the encoder)

        A (clean) ───┐
                     ├─► EncoderV2 (frame + global, A and B jointly) ─► A + B tokens ─┐
        B ─► mask ───┘                         │                                      │ K,V
                                               │ B tokens                             ▼
                                               ▼                C ─► mask ─► TargetEncoder (+CLS) ─► Decoder (cross-attn)
                                        encoder_pred (linear)                                 │
                                               ↓                                              ↓
                                         masked MSE on B                              masked MSE on C

        loss = L_B + lambda_c * L_C      (B: main loss, C: auxiliary signal)

    V4 = V2 + a per-view CLS token in the target encoder. The CLS is processed
    with the target patches in the target encoder, passed through the decoder,
    and removed before decoder_pred (only patch features are predicted).

    V4 masks C less than B (c_mask_ratios): with more of the target visible, the
    target camera can be placed relative to A + B, so C is a precise-matching
    task rather than guessing content that the sources do not show.

    Confidence (conf_loss=True, B and C), CroCo-style with two passes:
        pixels:     from the usual masked pass
        confidence: from a second, UNMASKED pass of the same views (same heads),
                    one value per pixel, c = 1 + exp(x) > 1
        err  = (pred - target)^2 averaged over RGB, per pixel
        loss = sum over masked pixels of (c * err - conf_alpha * log c) / #masked pixels
    B: second encoder pass with A + B all clean. C: second target-encoder + decoder pass
    with the target unmasked (same A + B reference tokens). The extra passes roughly
    double the cost of each branch.

    L_B forces the encoder itself to complete B from A (cross-view matching
    inside the encoder). L_C asks the A + B representation to answer queries
    from a view the encoder never saw.
    """

    def __init__(
        self,
        *,
        img_size=224,
        patch_size=16,
        in_chans=3,

        # ====================================================
        # Encoder (A clean + B masked, frame + global attention)
        # ====================================================

        embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        # num_clean=2,

        # ====================================================
        # Target encoder (C, self-attention)
        # ====================================================

        target_depth=4,
        target_num_heads=8,

        # ====================================================
        # Decoder (cross-attention only, C -> A + B)
        # ====================================================

        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,

        # ====================================================
        # Masking (Muskie-style), same strategies for B and C;
        # C uses its own, lower ratios (c_mask_ratios)
        #
        # A (mode, ratio) pair is drawn from these lists:
        #   mask_per_view=False -> one draw per batch: all views
        #                          share the strategy, each gets its own
        #                          random mask instance
        #   mask_per_view=True  -> one draw per view: strategies
        #                          are mixed inside the batch
        # ====================================================

        masking_types=("random", "ellipse", "rectangle"),
        mask_ratios=(0.90, 0.75, 0.75),       # B (masked reference views)
        c_mask_ratios=(0.75, 0.50, 0.50),     # C (held-out targets): less masking
        mask_per_view=True,

        # ====================================================
        # Loss weighting
        # ====================================================

        lambda_c=0.40,         # weight of the auxiliary C loss (0 skips the C branch: B-only, Muskie-style)

        norm_pix_loss=True,
        conf_loss=True,        # CroCo-style confidence (from an unmasked pass) for both B and C
        conf_alpha=0.2,        # weight of the -log c regulariser

        n_storage_tokens=0,
        rope_freq=100,

        gradient_checkpointing=False,

        **kwargs,
    ):
        super().__init__()

    
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.n_storage_tokens = n_storage_tokens
        # self.num_clean = num_clean
        self.lambda_c = lambda_c

        assert len(masking_types) == len(mask_ratios), "need one mask ratio per masking type"
        assert len(masking_types) == len(c_mask_ratios), "need one C mask ratio per masking type"
        self.masking_types = tuple(masking_types)
        self.mask_ratios = tuple(mask_ratios)
        self.c_mask_ratios = tuple(c_mask_ratios)
        self.mask_per_view = mask_per_view
        self.norm_pix_loss = norm_pix_loss
        self.conf_loss = conf_loss
        self.conf_alpha = conf_alpha
        self.gradient_checkpointing = gradient_checkpointing

        self.in_chans = in_chans
        self.patch_dim = patch_size ** 2 * in_chans
        # pixel prediction (+ one confidence logit per pixel)
        head_dim = self.patch_dim + (patch_size ** 2 if conf_loss else 0)

        # ====================================================
        # 1. ENCODER: A (clean) + B (masked) jointly, all tokens out
        # ====================================================

        self.encoder = EncoderV2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            n_storage_tokens=n_storage_tokens,
            rope_freq=rope_freq,
            # num_clean=num_clean,
        )

        # ====================================================
        # 1b. B RECONSTRUCTION HEAD
        #
        # Deliberately linear: the completion of B must happen
        # inside the encoder, not in a head.
        # ====================================================

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder_pred = nn.Linear(embed_dim, head_dim, bias=True)

        # ====================================================
        # 2. TARGET ENCODER for C
        #
        # target image -> patch embed -> mask -> self-attention
        # Runs at the decoder width, so no projection is needed.
        # ====================================================

        self.target_encoder = TargetEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=decoder_embed_dim,
            depth=target_depth,
            num_heads=target_num_heads,
            rope_freq=rope_freq,
            gradient_checkpointing=gradient_checkpointing,
            use_cls=True,                  # V4: target CLS, removed after the decoder
        )

        # ====================================================
        # 3. DECODER (cross-attention only)
        #
        # Q = C tokens, K,V = A + B tokens
        # ====================================================

        self.decoder = Decoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            rope_freq=rope_freq,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, head_dim, bias=True)  # decoder to patch (+ confidence)

        named_apply(init_weights_vit, self)

    # ========================================================
    # VISUALISATION HELPERS
    # ========================================================

    def mask_image_with_white(self, imgs, mask):
        """
        Replace masked patches with solid white.

        imgs: [B, 3, H, W]
        mask: [B, h, w] or [B, h*w]
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h, w = H // p, W // p

        if mask.ndim == 2:
            mask = mask.reshape(B, h, w)

        pixel_mask = (
            mask.bool()
            .repeat_interleave(p, dim=1)
            .repeat_interleave(p, dim=2)
            .unsqueeze(1)
        )

        return torch.where(pixel_mask, torch.ones_like(imgs), imgs)

    def fill_masked_patches(self, imgs, mask, pred):
        """
        Replace masked patches in imgs with predicted RGB patches.

        imgs: [B, 3, H, W]
        mask: [B, h, w] or [B, h*w]   1 = masked
        pred: [B, h*w, p*p*3]
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h, w = H // p, W // p

        if mask.ndim == 2:
            mask = mask.reshape(B, h, w)

        pred = pred.reshape(B, h, w, p, p, C)
        pred = torch.einsum("nhwpqc->nchpwq", pred).reshape(B, C, H, W)

        pixel_mask = (
            mask.bool()
            .repeat_interleave(p, dim=1)
            .repeat_interleave(p, dim=2)
            .unsqueeze(1)
        )

        return torch.where(pixel_mask, pred, imgs)

    # ========================================================
    # PATCHIFY
    # ========================================================

    def patchify(self, imgs):
        """
        imgs:    [N, 3, H, W]
        returns: [N, L, p*p*3]
        """
        p = self.patch_size
        N, C, H, W = imgs.shape
        assert H % p == 0 and W % p == 0
        h, w = H // p, W // p

        x = imgs.reshape(N, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(N, h * w, p * p * C)

    def unpatchify(self, x):
        """
        x:    [N, L, p*p*3]
        imgs: [N, 3, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], 3, h * p, w * p)

    # ========================================================
    # MASKING
    # ========================================================

    def connected_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        mode: str = "rectangle",
        blob_kernel: int = 9,
        ar_range: tuple[float, float] = (0.3, 3.0),
    ):
        """
        GPU masking

        Args:
            x:          (N, L, D), where N = batch_size * num_views
            mask_ratio: fraction of patches to mask
            mode:       rectangle, ellipse, random, blob, or random_walk
            blob_kernel: odd smoothing kernel size for blob masking
            ar_range:   minimum and maximum rectangle/ellipse aspect ratios

        Returns:
            mask:       (N, L), 0 = keep, 1 = mask, in original patch order
        """
        N, L, D = x.shape
        H = W = math.isqrt(L)

        if H * W != L:
            raise ValueError(f"L={L} must be a square number.")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1).")
        if mode not in {"rectangle", "ellipse", "random", "blob", "random_walk"}:
            raise ValueError(f"Unknown masking mode: {mode}")
        if mode == "blob" and blob_kernel % 2 == 0:
            raise ValueError("blob_kernel must be odd.")

        device = x.device
        len_keep = int(L * (1.0 - mask_ratio))
        target = L - len_keep

        if target == 0:
            return torch.zeros(N, L, device=device)

        noise = torch.rand(N, H, W, device=device)

        # Higher scores are masked; lower scores are kept.
        if mode == "random":
            scores = noise.flatten(1)

        elif mode == "blob":
            smooth = F.avg_pool2d(
                noise.unsqueeze(1),
                kernel_size=blob_kernel,
                stride=1,
                padding=blob_kernel // 2,
            ).squeeze(1)
            scores = smooth.flatten(1)

        elif mode == "rectangle":
            # Sample rectangle dimensions in parallel.
            log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
            ar = torch.empty(N, device=device).uniform_(log_min, log_max).exp()

            h = (target / ar).sqrt().round().long().clamp_(1, H)
            w = (target / h.float()).round().long().clamp_(1, W)

            top = (torch.rand(N, device=device) * (H - h + 1)).long()
            left = (torch.rand(N, device=device) * (W - w + 1)).long()

            yy = torch.arange(H, device=device)[None, :, None]
            xx = torch.arange(W, device=device)[None, None, :]

            inside_y = (yy >= top[:, None, None]) & (yy < (top + h)[:, None, None])
            inside_x = (xx >= left[:, None, None]) & (xx < (left + w)[:, None, None])
            region = inside_y & inside_x

            # Prefer patches inside the rectangle, then patches near its boundary.
            dy = torch.maximum(
                top[:, None, None] - yy,
                yy - (top + h - 1)[:, None, None],
            ).clamp_min(0)
            dx = torch.maximum(
                left[:, None, None] - xx,
                xx - (left + w - 1)[:, None, None],
            ).clamp_min(0)

            distance = torch.sqrt(dx.float().square() + dy.float().square())
            scores = torch.where(
                region,
                2.0 + noise * 1e-3,
                -distance + noise * 1e-3,
            ).flatten(1)

        elif mode == "ellipse":
            log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
            ar = torch.empty(N, device=device).uniform_(log_min, log_max).exp()

            # Approximate ellipse area with pi * rx * ry ~= target.
            ry = (target / (math.pi * ar)).sqrt().clamp_(min=0.5, max=H)
            rx = (target * ar / math.pi).sqrt().clamp_(min=0.5, max=W)

            cy = torch.rand(N, device=device) * H
            cx = torch.rand(N, device=device) * W

            yy = torch.arange(H, device=device)[None, :, None]
            xx = torch.arange(W, device=device)[None, None, :]

            rho = torch.sqrt(
                ((yy - cy[:, None, None]) / ry[:, None, None]).square()
                + ((xx - cx[:, None, None]) / rx[:, None, None]).square()
            )
            region = rho <= 1.0

            # The ellipse has priority; if its area is insufficient, fill nearby.
            scores = torch.where(
                region,
                2.0 + noise * 1e-3,
                -rho + noise * 1e-3,
            ).flatten(1)

        else:  # random_walk: GPU frontier growth
            seed = torch.randint(L, (N, 1), device=device)
            region = torch.zeros(N, L, device=device, dtype=torch.bool)
            region.scatter_(1, seed, True)
            region = region.view(N, H, W)

            # Record growth step so the final selected region favors the seed
            # and earlier growth, preserving a connected core.
            age = torch.full((N, H, W), -1, device=device, dtype=torch.int32)
            age.view(N, L).scatter_(1, seed, 0)
            step = 0

            while (region.sum(dim=(1, 2)) < target).any():
                expanded = F.max_pool2d(
                    region.float().unsqueeze(1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).squeeze(1).bool()

                frontier = expanded & ~region
                region |= frontier
                step += 1
                age.masked_fill_(frontier, step)

            # The seed and earlier growth receive higher masking priority.
            scores = torch.where(
                region,
                (step - age).float() + 2.0 + noise * 1e-3,
                noise * 1e-3,
            ).flatten(1)

        # One sort per batch of views. The last `target` patches are masked.
        ids_shuffle = torch.argsort(scores, dim=1)

        # Write masked positions directly in original patch order.
        mask = torch.zeros(N, L, device=device)
        mask.scatter_(1, ids_shuffle[:, len_keep:], 1.0)

        return mask

    def sample_mask(self, N, P, device, mask_ratio=None, mask_type=None, ratios=None):
        """
        Pick masking strategies (Muskie-style) and build the masks.

        N:          number of views to mask
        P:          patches per view
        ratios:     per-strategy ratios (None -> self.mask_ratios, i.e. B; C passes c_mask_ratios)
        mask_ratio: None  -> each strategy uses its own ratio
                    float -> fixed ratio for every view (e.g. for evaluation)
        mask_type:  optional fixed strategy (e.g. "rectangle" for evaluation)

        Every view gets its own random mask instance.
        mask_per_view=True  -> each view draws its own strategy (and with it its ratio)
        mask_per_view=False -> one strategy for the whole batch

        Returns: mask [N, P], 0 = visible, 1 = masked
        """
        x = torch.empty(N, P, 1, device=device)                # only shape and device are used

        # ratio follows the strategy, e.g. B: random 0.90, ellipse / rectangle 0.75
        type_to_ratio = dict(zip(self.masking_types, ratios if ratios is not None else self.mask_ratios))

        def ratio_of(mode):
            return mask_ratio if mask_ratio is not None else type_to_ratio[mode]

        # one strategy for the whole batch (or a fixed one, e.g. "rectangle" for evaluation)
        if mask_type is not None or not self.mask_per_view:
            if mask_type is None:
                mask_type = random.choice(self.masking_types)
            return self.connected_masking(x, ratio_of(mask_type), mode=mask_type)

        # per view: any strategy, each with its own ratio, selected on the GPU
        modes = list(self.masking_types)
        if len(modes) == 1:
            return self.connected_masking(x, ratio_of(modes[0]), mode=modes[0])
        mode_ids = torch.randint(len(modes), (N,), device=x.device)
        outs = [self.connected_masking(x, ratio_of(m), mode=m) for m in modes]
        mask = outs[0]
        for k in range(1, len(modes)):
            sel = mode_ids == k
            mask = torch.where(sel[:, None], outs[k], mask)

        return mask

    # ========================================================
    # LOSS
    # ========================================================

    def split_head(self, out):
        """Head output -> (pixel prediction [.., patch_dim], confidence logits [.., p*p] or None)."""
        if not self.conf_loss:
            return out, None
        return out[..., :self.patch_dim], out[..., self.patch_dim:]

    def forward_loss(self, imgs, pred, mask, conf_logits=None):
        """
        imgs:        [B*S, 3, H, W], RGB in [0, 1]
        pred:        [B*S, L, patch_dim]   (p, p, 3 layout, as patchify)
        mask:        [B*S, L], 0 = keep, 1 = masked
        conf_logits: [B*S, L, p*p] confidence logits from the UNMASKED pass, or None (plain MSE)

        Returns: loss, mse (plain masked MSE, for logging), masked_rgb, reconstructed_rgb
        """
        target = self.patchify(imgs)                       # [B*S, L, patch_dim]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = torch.sqrt(target.var(dim=-1, keepdim=True) + 1e-6)

            target_loss = (target - mean) / std
            pred_loss = pred                                # predicts normalized pixels
            pred_rgb = (pred_loss * std + mean).clamp(0.0, 1.0)
        else:
            target_loss = target                            # RGB in [0, 1]
            pred_rgb = pred.sigmoid()
            pred_loss = pred_rgb

        diff = pred_loss - target_loss                         # [B*S, L, patch_dim]
        mse = diff.pow(2).mean(dim=-1)                         # [B*S, L]
        denom = mask.sum() + 1e-6

        if conf_logits is not None:
            # CroCo-style: per-pixel squared error (mean over RGB), weighted by c = 1 + exp(x)
            err = diff.pow(2).reshape(*diff.shape[:-1], -1, self.in_chans).mean(dim=-1)   # [B*S, L, p*p]
            conf = 1.0 + torch.exp(conf_logits.clamp(max=20.0))                           # > 1, overflow-safe
            per_pix = conf * err - self.conf_alpha * torch.log(conf)
            loss = (per_pix * mask.unsqueeze(-1)).sum() / (per_pix.shape[-1] * denom)
        else:
            loss = (mse * mask).sum() / denom                  # masked patches only

        mse_masked = (mse.detach() * mask).sum() / denom       # comparable across runs

        # RGB visualisations: only outside training, never tracked by autograd
        if self.training:
            return loss, mse_masked, None, None

        with torch.no_grad():
            masked_rgb = self.mask_image_with_white(imgs, mask)
            reconstructed_rgb = self.fill_masked_patches(imgs, mask, pred_rgb.detach())

        return loss, mse_masked, masked_rgb, reconstructed_rgb



    def forward(self, reference_imgs, target_imgs, mask_ratio=None, mask_type=None):
        """
        reference_imgs: [B, S_ref, 3, H, W]
        target_imgs:    [B, S_target, 3, H, W]

        Randomly selects num_clean reference views per batch.
        First num_clean views are clean (A); remaining views are masked (B).
        """
        B, S_ref = reference_imgs.shape[:2]
        _, S_target, C, H, W = target_imgs.shape
        assert target_imgs.shape[0] == B

        # Random Muskie-style reference-view selection.
        num_clean = random.randint(max(1, S_ref // 4), max(1, S_ref // 2))

        assert S_ref > num_clean, (
            f"need at least one masked reference view: S_ref={S_ref}, num_clean={num_clean}"
        )

        P = (H // self.patch_size) * (W // self.patch_size)
        S_masked = S_ref - num_clean

        # 1. Mask B (reference views after the first num_clean)
        ref_mask = self.sample_mask(
            B * S_masked, P, reference_imgs.device,
            mask_ratio=mask_ratio, mask_type=mask_type,
        )

        # 2. Encode A + B jointly
        reference_tokens, patch_start_idx = self.encoder(
            reference_imgs, ref_mask.reshape(B, S_masked, P).bool()
        )

        # 3. B reconstruction loss (main loss, always on)
        b_tokens = reference_tokens[:, num_clean:, patch_start_idx:]
        out_b = self.encoder_pred(self.encoder_norm(b_tokens)).reshape(
            B * S_masked, P, -1
        )
        pred_b, _ = self.split_head(out_b)

        # Confidence from a second pass with A + B all clean (CroCo-style)
        conf_b = None
        if self.conf_loss:
            full_tokens, _ = self.encoder(reference_imgs, None)
            out_b_full = self.encoder_pred(
                self.encoder_norm(full_tokens[:, num_clean:, patch_start_idx:])
            ).reshape(B * S_masked, P, -1)
            _, conf_b = self.split_head(out_b_full)

        with torch.autocast(device_type=pred_b.device.type, enabled=False):
            loss_b, mse_b, b_masked_rgb, b_recon_rgb = self.forward_loss(
                reference_imgs[:, num_clean:].reshape(B * S_masked, C, H, W).float(),
                pred_b.float(),
                ref_mask.float(),
                None if conf_b is None else conf_b.float(),
            )

        # 4. C branch (auxiliary); skipped entirely when lambda_c == 0
        loss_c = torch.zeros((), device=reference_imgs.device)
        mse_c = torch.zeros((), device=reference_imgs.device)
        masked_rgb = reconstructed_rgb = None
        if self.lambda_c > 0:
            loss_c, mse_c, masked_rgb, reconstructed_rgb = self._forward_c(
                target_imgs, reference_tokens, patch_start_idx, mask_ratio, mask_type
            )

        # 5. Total loss
        loss = loss_b + self.lambda_c * loss_c
        loss_dict = {
            "loss_b": loss_b.detach(), "loss_c": loss_c.detach(),
            "mse_b": mse_b, "mse_c": mse_c,          # plain masked MSE, comparable across runs
        }

        # 6. Visualisations
        if masked_rgb is not None:
            masked_rgb = masked_rgb.reshape(B, S_target, 3, H, W)
            reconstructed_rgb = reconstructed_rgb.reshape(B, S_target, 3, H, W)

        if b_masked_rgb is not None:
            b_masked_rgb = b_masked_rgb.reshape(B, S_masked, 3, H, W)
            b_recon_rgb = b_recon_rgb.reshape(B, S_masked, 3, H, W)

        return loss, masked_rgb, reconstructed_rgb, loss_dict, b_masked_rgb, b_recon_rgb

    def _forward_c(self, target_imgs, reference_tokens, patch_start_idx, mask_ratio=None, mask_type=None):
        """Auxiliary C branch: mask the held-out targets, encode them, decode against A + B."""
        B, S_target, C, H, W = target_imgs.shape
        P = (H // self.patch_size) * (W // self.patch_size)

        # Mask C (lower ratios than B)
        mask = self.sample_mask(
            B * S_target, P, target_imgs.device,
            mask_ratio=mask_ratio, mask_type=mask_type,
            ratios=self.c_mask_ratios,
        )

        # Masked pass -> pixels
        pred, _ = self.split_head(self._decode_c(target_imgs, mask, reference_tokens, patch_start_idx))

        # Unmasked pass of the target (same A + B references) -> confidence (CroCo-style)
        conf = None
        if self.conf_loss:
            out_full = self._decode_c(target_imgs, torch.zeros_like(mask), reference_tokens, patch_start_idx)
            _, conf = self.split_head(out_full)

        # C reconstruction loss
        with torch.autocast(device_type=pred.device.type, enabled=False):
            loss_c, mse_c, masked_rgb, reconstructed_rgb = self.forward_loss(
                target_imgs.reshape(B * S_target, C, H, W).float(),
                pred.float(),
                mask.float(),
                None if conf is None else conf.float(),
            )

        return loss_c, mse_c, masked_rgb, reconstructed_rgb

    def _decode_c(self, target_imgs, mask, reference_tokens, patch_start_idx):
        """Target encoder (+CLS) -> decoder against A + B -> head output [B*S_target, P, head_dim]."""
        B, S_target, C, H, W = target_imgs.shape
        P = (H // self.patch_size) * (W // self.patch_size)

        target_tokens = self.target_encoder(target_imgs, mask)

        # target CLS goes through the decoder and is dropped at its output,
        # so only patch features reach decoder_pred
        t_psi = self.target_encoder.patch_start_idx
        decoder_out = self.decoder(
            target_tokens=target_tokens,
            reference_tokens=reference_tokens,
            patch_start_idx=patch_start_idx,
        )
        decoder_out = decoder_out.reshape(B, S_target, t_psi + P, -1)[:, :, t_psi:].reshape(B * S_target, P, -1)
        return self.decoder_pred(self.decoder_norm(decoder_out))

# ================================================================
# VISTA V4 factories
#
# Each factory holds its defaults in a dict; any keyword argument
# overrides the matching default (e.g. decoder_depth=4, lambda_c=1.0)
# instead of raising "got multiple values for keyword argument".
# e.g. vista_v4_base(lambda_c=0.5); lambda_c=0 -> B only (Muskie-style baseline)
# ================================================================

def vista_v4_small(patch_size=16, img_size=224, **kwargs):

    config = dict(
        patch_size=patch_size,
        img_size=img_size,

        # Encoder (frame + global)
        embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,

        # Target encoder
        target_depth=4,
        target_num_heads=8,

        # Decoder
        decoder_embed_dim=256,
        decoder_depth=8,
        decoder_num_heads=8,
    )

    return VistaV4(**{**config, **kwargs})


def vista_v4_base(patch_size=16, img_size=224, **kwargs):

    config = dict(
        patch_size=patch_size,
        img_size=img_size,

        # Encoder (frame + global)
        embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=12,

        # Target encoder
        target_depth=4,
        target_num_heads=16,

        # Decoder
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    )

    return VistaV4(**{**config, **kwargs})


def vista_v4_large(patch_size=16, img_size=224, **kwargs):

    config = dict(
        patch_size=patch_size,
        img_size=img_size,

        # Encoder (frame + global)
        embed_dim=1024,
        encoder_depth=12,
        encoder_num_heads=16,

        # Target encoder
        target_depth=4,
        target_num_heads=16,

        # Decoder
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
    )

    return VistaV4(**{**config, **kwargs})