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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing

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

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x, pos=pos)

        return x.reshape(B, S, P, D)
# ================================================================
# Vista V2
# ================================================================

class VistaV2(nn.Module):
    """
    Three view sets per scene:
        A: clean views            (reference_imgs[:, :num_clean])
        B: masked views           (reference_imgs[:, num_clean:])
        C: masked target views    (target_imgs, never seen by the encoder)

        A (clean) ───┐
                     ├─► EncoderV2 (frame + global, A and B jointly) ─► A + B tokens ─┐
        B ─► mask ───┘                         │                                      │ K,V
                                               │ B tokens                             ▼
                                               ▼                C ─► mask ─► TargetEncoder ─► Decoder (cross-attn)
                                        encoder_pred (linear)                                 │
                                               ↓                                              ↓
                                         masked MSE on B                              masked MSE on C

        loss = L_C + lambda_b * L_B

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
        num_clean=2,

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
        # Masking (Muskie-style), same strategies for B and C
        #
        # A (mode, ratio) pair is drawn from these lists:
        #   mask_per_view=False -> one draw per batch: all views
        #                          share the strategy, each gets its own
        #                          random mask instance
        #   mask_per_view=True  -> one draw per view: strategies
        #                          are mixed inside the batch
        # ====================================================

        masking_types=("random", "ellipse", "rectangle"),
        mask_ratios=(0.90, 0.75, 0.75),
        mask_per_view=True,

        # ====================================================
        # Loss weighting
        # ====================================================

        lambda_b=0.40,         # weight of the B reconstruction loss (0 disables it)

        norm_pix_loss=True,

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
        self.num_clean = num_clean
        self.lambda_b = lambda_b

        assert len(masking_types) == len(mask_ratios), "need one mask ratio per masking type"
        self.masking_types = tuple(masking_types)
        self.mask_ratios = tuple(mask_ratios)
        self.mask_per_view = mask_per_view
        self.norm_pix_loss = norm_pix_loss
        self.gradient_checkpointing = gradient_checkpointing

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
            num_clean=num_clean,
        )

        # ====================================================
        # 1b. B RECONSTRUCTION HEAD
        #
        # Deliberately linear: the completion of B must happen
        # inside the encoder, not in a head.
        # ====================================================

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

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

    def sample_mask(self, N, P, device, mask_ratio=None, mask_type=None):
        """
        Pick masking strategies (Muskie-style) and build the masks.

        N:          number of views to mask
        P:          patches per view
        mask_ratio: None  -> each strategy uses its own ratio
                    float -> fixed ratio for every view (e.g. for evaluation)
        mask_type:  optional fixed strategy (e.g. "rectangle" for evaluation)

        Every view gets its own random mask instance.
        mask_per_view=True  -> each view draws its own strategy (and with it its ratio)
        mask_per_view=False -> one strategy for the whole batch

        Returns: mask [N, P], 0 = visible, 1 = masked
        """
        x = torch.empty(N, P, 1, device=device)                # only shape and device are used

        # ratio follows the strategy: rectangle / ellipse -> 0.75, random -> 0.90
        type_to_ratio = dict(zip(self.masking_types, self.mask_ratios))

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

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [B*S, 3, H, W], RGB in [0, 1]
        pred: [B*S, L, patch_dim]
        mask: [B*S, L], 0 = keep, 1 = masked

        Returns: loss, masked_rgb, reconstructed_rgb
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

        loss = (pred_loss - target_loss).pow(2).mean(dim=-1)   # [B*S, L]
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)       # masked patches only

        # RGB visualisations: only outside training, never tracked by autograd
        if self.training:
            return loss, None, None

        with torch.no_grad():
            masked_rgb = self.mask_image_with_white(imgs, mask)
            reconstructed_rgb = self.fill_masked_patches(imgs, mask, pred_rgb.detach())

        return loss, masked_rgb, reconstructed_rgb

    # ========================================================
    # FULL FORWARD
    # ========================================================

    def forward(self, reference_imgs, target_imgs, mask_ratio=None, mask_type=None):
        """
        reference_imgs: [B, S_ref, 3, H, W]
                        [:, :num_clean]  A, clean
                        [:, num_clean:]  B, masked inside the encoder
        target_imgs:    [B, S_target, 3, H, W]   C
        mask_ratio:     optional fixed ratio for B and C (e.g. evaluation)
        mask_type:      optional fixed strategy for B and C (e.g. evaluation)

        Returns:
            loss:              total loss = loss_c + lambda_b * loss_b
            masked_rgb:        C with masked patches in white      [B, S_target, 3, H, W]  (None in training)
            reconstructed_rgb: C decoder reconstruction            [B, S_target, 3, H, W]  (None in training)
            loss_dict:         {"loss_b": ..., "loss_c": ...}      (detached, for logging)
            b_masked_rgb:      B with masked patches in white      [B, S_masked, 3, H, W]  (None in training / lambda_b=0)
            b_recon_rgb:       B encoder reconstruction            [B, S_masked, 3, H, W]  (None in training / lambda_b=0)
        """
        B, S_ref = reference_imgs.shape[:2]
        _, S_target, C, H, W = target_imgs.shape
        assert target_imgs.shape[0] == B
        assert S_ref > self.num_clean, (
            f"need at least one masked reference view: S_ref={S_ref}, num_clean={self.num_clean}"
        )

        P = (H // self.patch_size) * (W // self.patch_size)
        S_masked = S_ref - self.num_clean

        # 1. Mask B (reference views after the first num_clean)
        ref_mask = self.sample_mask(
            B * S_masked, P, reference_imgs.device,
            mask_ratio=mask_ratio, mask_type=mask_type,
        )                                                                        # [B*S_masked, P]

        # 2. Encode A + B jointly; tokens of all reference views
        reference_tokens, patch_start_idx = self.encoder(
            reference_imgs, ref_mask.reshape(B, S_masked, P).bool()
        )                                                                        # [B, S_ref, psi + P, D]

        # 3. B reconstruction loss from B's encoder tokens
        loss_b = torch.zeros((), device=reference_imgs.device)
        b_masked_rgb = b_recon_rgb = None
        if self.lambda_b > 0:
            b_tokens = reference_tokens[:, self.num_clean:, patch_start_idx:]     # [B, S_masked, P, D]
            pred_b = self.encoder_pred(self.encoder_norm(b_tokens)).reshape(B * S_masked, P, -1)
            with torch.autocast(device_type=pred_b.device.type, enabled=False):
                loss_b, b_masked_rgb, b_recon_rgb = self.forward_loss(
                    reference_imgs[:, self.num_clean:].reshape(B * S_masked, C, H, W).float(),
                    pred_b.float(),
                    ref_mask.float(),
                )

        # 4. Mask C
        mask = self.sample_mask(
            B * S_target, P, target_imgs.device,
            mask_ratio=mask_ratio, mask_type=mask_type,
        )                                                                        # [B*S_target, P]

        # 5. Target encoder: patch embed + mask tokens + self-attention
        target_tokens = self.target_encoder(target_imgs, mask)                  # [B, S_target, P, D_dec]

        # 6. Decoder: C cross-attends to A + B tokens
        decoder_out = self.decoder(
            target_tokens=target_tokens,
            reference_tokens=reference_tokens,
            patch_start_idx=patch_start_idx,
        ).reshape(B * S_target, P, -1)

        pred = self.decoder_pred(self.decoder_norm(decoder_out))                # [B*S_target, P, p*p*3]

        # 7. C reconstruction loss
        with torch.autocast(device_type=pred.device.type, enabled=False):
            loss_c, masked_rgb, reconstructed_rgb = self.forward_loss(
                target_imgs.reshape(B * S_target, C, H, W).float(),
                pred.float(),
                mask.float(),
            )

        # 8. Total loss + per-term losses for logging
        loss = loss_c + self.lambda_b * loss_b
        loss_dict = {"loss_b": loss_b.detach(), "loss_c": loss_c.detach()}

        # 9. Visualisations (None during training)
        if masked_rgb is not None:
            masked_rgb = masked_rgb.reshape(B, S_target, 3, H, W)
            reconstructed_rgb = reconstructed_rgb.reshape(B, S_target, 3, H, W)

        if b_masked_rgb is not None:
            b_masked_rgb = b_masked_rgb.reshape(B, S_masked, 3, H, W)
            b_recon_rgb = b_recon_rgb.reshape(B, S_masked, 3, H, W)

        return loss, masked_rgb, reconstructed_rgb, loss_dict, b_masked_rgb, b_recon_rgb


# ================================================================
# VISTA V2 factories
# ================================================================

def vista_v2_small(patch_size=16, img_size=224, **kwargs):

    model = VistaV2(
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

        **kwargs,
    )

    return model


def vista_v2_base(patch_size=16, img_size=224, **kwargs):

    model = VistaV2(
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

        **kwargs,
    )

    return model


def vista_v2_large(patch_size=16, img_size=224, **kwargs):

    model = VistaV2(
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

        **kwargs,
    )

    return model