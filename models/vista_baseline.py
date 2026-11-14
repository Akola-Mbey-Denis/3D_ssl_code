import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from .autoencoder import EncoderV2, Decoder
from models.layers import PatchEmbed
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
# Vista baseline: encoder + decoder, no target encoder
# ================================================================

class VistaBaseline(nn.Module):
    """
    Vista without the target encoder and without a held-out view set C.

    Two view sets per scene:
        A: clean views   (imgs[:, :num_clean])
        B: masked views  (imgs[:, num_clean:])

        A (clean) ───┐
                     ├─► EncoderV2 (frame + global) ─► A + B tokens ──────────────┐ K,V
        B ─► mask ───┘                                    │                       ▼
                                                          └─► query_proj ─► Decoder (cross-attn)
                                                              (Q = all A + B          │
                                                               patch tokens)          ↓
                                                                              masked MSE on B

        loss = L_dec

    The decoder cross-attends every encoder patch token (A and B) to all
    encoder tokens (A + B, including storage/camera tokens). Only B's masked
    patches carry a loss; A is unmasked, so its outputs have zero weight.

    fold_targets=True: if target_imgs are passed, they are appended to the
    reference views, so the baseline sees the same images per batch as Vista.
    """

    def __init__(
        self,
        *,
        img_size=224,
        patch_size=16,
        in_chans=3,

        # Encoder (A clean + B masked, frame + global attention)
        embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,

        # Decoder (cross-attention only, B -> A + B)
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,

        # Masking (Muskie-style)
        masking_types=("random", "ellipse", "rectangle"),
        mask_ratios=(0.90, 0.75, 0.75),
        mask_per_view=True,

        # Loss
        norm_pix_loss=True,

        # Data
        fold_targets=True,    # append target_imgs to the reference views

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
        self.fold_targets = fold_targets

        assert len(masking_types) == len(mask_ratios), "need one mask ratio per masking type"
        self.masking_types = tuple(masking_types)
        self.mask_ratios = tuple(mask_ratios)
        self.mask_per_view = mask_per_view
        self.norm_pix_loss = norm_pix_loss
        self.gradient_checkpointing = gradient_checkpointing

        # ====================================================
        # 1. ENCODER
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
        )

        # ====================================================
        # 2. QUERY PROJECTION: encoder patch tokens -> decoder width
        #    (replaces the target encoder)
        # ====================================================

        self.query_norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # ====================================================
        # 3. DECODER (cross-attention only)
        #    Q = A + B patch tokens, K,V = A + B tokens
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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        named_apply(init_weights_vit, self)

    # ========================================================
    # VISUALISATION HELPERS
    # ========================================================

    def mask_image_with_white(self, imgs, mask):
        """
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
        ar_range: tuple = (0.3, 3.0),
    ):
        """
        x:       (N, L, D), N = batch_size * num_views
        Returns: mask (N, L), 0 = keep, 1 = mask, in original patch order
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

            scores = torch.where(
                region,
                2.0 + noise * 1e-3,
                -rho + noise * 1e-3,
            ).flatten(1)

        else:  # random_walk
            seed = torch.randint(L, (N, 1), device=device)
            region = torch.zeros(N, L, device=device, dtype=torch.bool)
            region.scatter_(1, seed, True)
            region = region.view(N, H, W)

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

            scores = torch.where(
                region,
                (step - age).float() + 2.0 + noise * 1e-3,
                noise * 1e-3,
            ).flatten(1)

        ids_shuffle = torch.argsort(scores, dim=1)

        mask = torch.zeros(N, L, device=device)
        mask.scatter_(1, ids_shuffle[:, len_keep:], 1.0)

        return mask

    def sample_mask(self, N, P, device, mask_ratio=None, mask_type=None):
        """
        Returns: mask [N, P], 0 = visible, 1 = masked
        """
        x = torch.empty(N, P, 1, device=device)

        type_to_ratio = dict(zip(self.masking_types, self.mask_ratios))

        def ratio_of(mode):
            return mask_ratio if mask_ratio is not None else type_to_ratio[mode]

        if mask_type is not None or not self.mask_per_view:
            if mask_type is None:
                mask_type = random.choice(self.masking_types)
            return self.connected_masking(x, ratio_of(mask_type), mode=mask_type)

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
        imgs: [N, 3, H, W], RGB in [0, 1]
        pred: [N, L, patch_dim]
        mask: [N, L], 0 = keep, 1 = masked

        Returns: loss, masked_rgb, reconstructed_rgb
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = torch.sqrt(target.var(dim=-1, keepdim=True) + 1e-6)

            target_loss = (target - mean) / std
            pred_loss = pred
            pred_rgb = (pred_loss * std + mean).clamp(0.0, 1.0)
        else:
            target_loss = target
            pred_rgb = pred.sigmoid()
            pred_loss = pred_rgb

        loss = (pred_loss - target_loss).pow(2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        if self.training:
            return loss, None, None

        with torch.no_grad():
            masked_rgb = self.mask_image_with_white(imgs, mask)
            reconstructed_rgb = self.fill_masked_patches(imgs, mask, pred_rgb.detach())

        return loss, masked_rgb, reconstructed_rgb

    # ========================================================
    # FULL FORWARD
    # ========================================================

    def forward(self, reference_imgs, target_imgs=None, mask_ratio=None, mask_type=None, num_clean=None):
        """
        reference_imgs: [B, S_ref, 3, H, W]
        target_imgs:    [B, S_target, 3, H, W] or None
                        appended to the reference views if fold_targets=True,
                        ignored otherwise
        num_clean:      optional fixed number of clean views.
                        None in training -> random per batch.
                        None in eval with target_imgs -> num_clean = S_ref, so the
                        masked views are exactly the target views (same views
                        Vista reconstructs, output shape matches target_imgs).

        Returns:
            loss:              decoder loss on B's masked patches
            masked_rgb:        B with masked patches in white   [B, S_masked, 3, H, W]  (None in training)
            reconstructed_rgb: B decoder reconstruction         [B, S_masked, 3, H, W]  (None in training)
            loss_dict:         {"loss_b": detached loss,
                                "num_clean": int}
                               The views are imgs = cat([reference_imgs, target_imgs], 1)
                               (or reference_imgs if fold_targets=False / no targets):
                                   clean  views: imgs[:, :num_clean]
                                   masked views: imgs[:, num_clean:]  <- what masked_rgb /
                                                                         reconstructed_rgb show
        """
        if target_imgs is not None and self.fold_targets:
            imgs = torch.cat([reference_imgs, target_imgs], dim=1)
        else:
            imgs = reference_imgs

        B, S, C, H, W = imgs.shape

        if num_clean is None:
           
           num_clean = random.randint(max(1, S // 4), max(1, S // 2))

        assert 0 < num_clean < S, (
            f"need at least one clean and one masked view: S={S}, num_clean={num_clean}"
        )

        P = (H // self.patch_size) * (W // self.patch_size)
        S_masked = S - num_clean

        # 1. Mask B
        mask = self.sample_mask(
            B * S_masked, P, imgs.device,
            mask_ratio=mask_ratio, mask_type=mask_type,
        )                                                                    # [B*S_masked, P]

        # 2. Encode A + B jointly
        tokens, patch_start_idx = self.encoder(
            imgs, mask.reshape(B, S_masked, P).bool()
        )                                                                    # [B, S, psi + P, D]

        # 3. Decoder: all A + B patch tokens cross-attend to all encoder tokens
        queries = self.query_proj(self.query_norm(tokens[:, :, patch_start_idx:]))   # [B, S, P, D_dec]

        decoder_out = self.decoder(
            target_tokens=queries,
            reference_tokens=tokens,
            patch_start_idx=patch_start_idx,
        )                                                                    # [B, S, P, D_dec]

        # 4. Prediction head on B only (A is unmasked, its loss weight would be 0)
        b_out = decoder_out.reshape(B, S, P, -1)[:, num_clean:].reshape(B * S_masked, P, -1)
        pred = self.decoder_pred(self.decoder_norm(b_out))                   # [B*S_masked, P, p*p*3]

        # 5. Loss on B's masked patches
        b_imgs = imgs[:, num_clean:].reshape(B * S_masked, C, H, W).float()
        with torch.autocast(device_type=pred.device.type, enabled=False):
            loss, masked_rgb, reconstructed_rgb = self.forward_loss(
                b_imgs, pred.float(), mask.float(),
            )

        loss_dict = {"loss_b": loss.detach(), "num_clean": num_clean}

        # 6. Visualisations
        if masked_rgb is not None:
            masked_rgb = masked_rgb.reshape(B, S_masked, 3, H, W)
            reconstructed_rgb = reconstructed_rgb.reshape(B, S_masked, 3, H, W)

        return loss, masked_rgb, reconstructed_rgb, loss_dict


# ================================================================
# Factories (same encoder/decoder sizes as the VistaV2 factories)
# ================================================================

def vista_baseline_small(patch_size=16, img_size=224, **kwargs):
    config = dict(
        patch_size=patch_size,
        img_size=img_size,
        embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=8,
        decoder_num_heads=8,
    )
    return VistaBaseline(**{**config, **kwargs})


def vista_baseline_base(patch_size=16, img_size=224, **kwargs):
    config = dict(
        patch_size=patch_size,
        img_size=img_size,
        embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    )
    return VistaBaseline(**{**config, **kwargs})


def vista_baseline_large(patch_size=16, img_size=224, **kwargs):
    config = dict(
        patch_size=patch_size,
        img_size=img_size,
        embed_dim=1024,
        encoder_depth=12,
        encoder_num_heads=16,
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
    )
    return VistaBaseline(**{**config, **kwargs})