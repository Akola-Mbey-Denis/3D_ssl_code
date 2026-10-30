# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True,
                 masking_types=("random", "ellipse", "rectangle"), mask_per_view=True,
                 mask_ratios=(0.75, 0.90)):
        super().__init__()

        # --------------------------------------------------------------------------
        # Masking (Muskie-style): a strategy is drawn per view (mask_per_view=True)
        # or once per batch (False). The mask ratio is shared by the whole batch,
        # because the encoder drops masked tokens and every view must keep the
        # same number of tokens.
        self.masking_types = tuple(masking_types)
        self.mask_per_view = mask_per_view
        self.mask_ratios = tuple(mask_ratios)          # one is drawn per batch
        # --------------------------------------------------------------------------
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



    def connected_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        mode: str = "rectangle",
        blob_kernel: int = 9,
        ar_range: tuple[float, float] = (0.3, 3.0),
    ):
        """
        GPU masking for MAE.

        Args:
            x:          (N, L, D), where N = batch_size * num_views
            mask_ratio: fraction of patches to mask
            mode:       rectangle, ellipse, random, blob, or random_walk
            blob_kernel: odd smoothing kernel size for blob masking
            ar_range:   minimum and maximum rectangle/ellipse aspect ratios

        Returns:
            x_masked:   (N, len_keep, D)
            mask:       (N, L), 0 = keep, 1 = mask, in original patch order
            ids_restore:(N, L), inverse permutation for MAE decoder
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
            ids_restore = torch.arange(L, device=device).expand(N, -1)
            mask = torch.zeros(N, L, device=device)
            return x, mask, ids_restore

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

            inside_y = (yy >= top[:, None, None]) & (
                yy < (top + h)[:, None, None]
            )
            inside_x = (xx >= left[:, None, None]) & (
                xx < (left + w)[:, None, None]
            )
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
            age = torch.full(
                (N, H, W), -1, device=device, dtype=torch.int32
            )
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
        ids_keep = ids_shuffle[:, :len_keep]

        # Invert the permutation without a second argsort.
        ids_restore = torch.empty_like(ids_shuffle)
        positions = torch.arange(L, device=device).expand(N, -1)
        ids_restore.scatter_(1, ids_shuffle, positions)

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        # Write masked positions directly in original patch order.
        mask = torch.zeros(N, L, device=device)
        mask.scatter_(1, ids_shuffle[:, len_keep:], 1.0)

        return x_masked, mask, ids_restore

 
    def sample_masking(self, x, mask_ratio, mask_type=None):
        """
        Pick masking strategies (Muskie-style) and mask x.

        x:          [N, L, D]
        mask_ratio: None -> one ratio drawn from self.mask_ratios for the whole batch
                    float -> fixed ratio (e.g. for evaluation)
                    The ratio is always shared by all views in the batch
                    (equal kept-token count is required).
        mask_type:  optional fixed strategy (e.g. "rectangle" for evaluation)

        Every view gets its own random mask instance.
        mask_per_view=True  -> each view also draws its own strategy
        mask_per_view=False -> one strategy for the whole batch

        Returns: x_masked [N, len_keep, D], mask [N, L], ids_restore [N, L]
        """
        # ratio follows the strategy: rectangle / ellipse -> 0.75, random -> 0.90
        type_to_ratio = {"rectangle": 0.75, "ellipse": 0.75, "random": 0.90}

        # one strategy drawn per batch sets the batch ratio
        # (all views must keep the same number of tokens)
        fixed_type = mask_type is not None                  # e.g. "rectangle" for evaluation
        if mask_type is None:
            mask_type = random.choice(self.masking_types)
        if mask_ratio is None:
            mask_ratio = type_to_ratio[mask_type]

        if fixed_type or not self.mask_per_view:
            return self.connected_masking(x, mask_ratio, mode=mask_type)

        # per view: any strategy with the same ratio
        # (0.90 batch -> random; 0.75 batch -> rectangle / ellipse mixed per view)
        modes = [t for t in self.masking_types if type_to_ratio[t] == mask_ratio] or [mask_type]

        # draw a strategy per view ON THE GPU and select with torch.where: no Python index lists,
        # so no blocking host->device copy. Each strategy is computed for all views (cheap).
        N = x.shape[0]
        if len(modes) == 1:
            return self.connected_masking(x, mask_ratio, mode=modes[0])
        mode_ids = torch.randint(len(modes), (N,), device=x.device)
        outs = [self.connected_masking(x, mask_ratio, mode=m) for m in modes]
        x_masked, mask, ids_restore = outs[0]
        for k in range(1, len(modes)):
            sel = mode_ids == k
            x_masked = torch.where(sel[:, None, None], outs[k][0], x_masked)
            mask = torch.where(sel[:, None], outs[k][1], mask)
            ids_restore = torch.where(sel[:, None], outs[k][2], ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, images, mask_ratio):

        B, S, C, H, W = images.shape

        # embed patches
        x = self.patch_embed(images.reshape(B * S, C, H, W))

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.sample_masking(x=x, mask_ratio=mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

  

    def forward_features(self, images, norm=True):
        """
        Encoder features from every transformer block (no masking: every patch is kept).

        images: [B, S, C, H, W] (or [B, C, H, W] -> S = 1) at the model's img_size
        norm:   apply the encoder's final LayerNorm to every block output

        Returns a list of tensors [B, S, 1 + L, D]; token 0 is the cls token, so patch
        tokens start at index 1 (patch_start_idx = 1).
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)
        B, S, C, H, W = images.shape

        # embed patches
        x = self.patch_embed(images.reshape(B * S, C, H, W))                 # [BS, L, D]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)                                # [BS, 1 + L, D]

        # apply Transformer blocks, keep every block's output
        outputs = []
        for blk in self.blocks:
            x = blk(x)
            y = self.norm(x) if norm else x
            outputs.append(y.reshape(B, S, x.shape[1], -1))
        return outputs

 
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
 
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
 
        # add pos embed
        x = x + self.decoder_pos_embed
 
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
 
        # predictor projection
        x = self.decoder_pred(x)
 
        # remove cls token
        x = x[:, 1:, :]
 
        return x
 

    
    # ========================================================
    # PATCHIFY
    # ========================================================
    def mask_image_with_white(
        self,
        imgs,
        mask,
    ):
        """
        Replace masked patches with solid white.

        imgs:
            [B, 3, 518, 518]

        mask:
            [B, 37, 37] or [B, 1369]

        returns:
            [B, 3, 518, 518]
        """

        B, C, H, W = imgs.shape
        p = self.patch_size

        h = H // p
        w = W // p

        # [B, 1369] -> [B, 37, 37]
        if mask.ndim == 2:
            mask = mask.reshape(B, h, w)

        # Expand patch mask to pixel mask
        pixel_mask = (
            mask.bool()
            .repeat_interleave(p, dim=1)
            .repeat_interleave(p, dim=2)
        )

        # [B, 1, H, W] -> [B, 3, H, W]
        pixel_mask = pixel_mask.unsqueeze(1)

        # Replace masked pixels with white
        masked_imgs = torch.where(
            pixel_mask,
            torch.ones_like(imgs),
            imgs,
        )

        return masked_imgs

    def fill_masked_patches(
        self,
        imgs,
        mask,
        pred,
    ):
        """
        Replace masked patches in imgs with predicted RGB patches.

        imgs:
            [B, 3, 518, 518]

        mask:
            [B, 37, 37] or [B, 1369]
            1 = masked
            0 = unmasked

        pred:
            [B, 1369, patch_size * patch_size * 3]

        returns:
            [B, 3, 518, 518]
        """

        B, C, H, W = imgs.shape
        p = self.patch_size

        h = H // p
        w = W // p

        # ---------------------------------------------------------
        # Convert mask to [B, h, w]
        # ---------------------------------------------------------

        if mask.ndim == 2:
            mask = mask.reshape(B, h, w)

        # ---------------------------------------------------------
        # Inverse of your patchify()
        #
        # patchify:
        #
        # [N,C,h,p,w,p]
        #       ↓ einsum
        # [N,h,w,p,p,C]
        #       ↓ reshape
        # [N,h*w,p*p*C]
        #
        # So we reverse that here.
        # ---------------------------------------------------------

        pred = pred.reshape(
            B,
            h,
            w,
            p,
            p,
            C,
        )

        # [B,h,w,p,p,C]
        #       ↓
        # [B,C,h,p,w,p]
        pred = torch.einsum(
            "nhwpqc->nchpwq",
            pred,
        )

        # [B,C,H,W]
        pred = pred.reshape(
            B,
            C,
            H,
            W,
        )

        # ---------------------------------------------------------
        # Expand patch mask to pixels
        # ---------------------------------------------------------

        pixel_mask = (
            mask.bool()
            .repeat_interleave(p, dim=1)
            .repeat_interleave(p, dim=2)
        )

        pixel_mask = pixel_mask.unsqueeze(1)

        # ---------------------------------------------------------
        # Keep GT in unmasked regions
        # and prediction in masked regions
        # ---------------------------------------------------------

        output = torch.where(
            pixel_mask,
            pred,
            imgs,
        )

        return output

 

    

    def forward_loss(
        self,
        imgs,
        pred,
        mask,
    ):
        """
        imgs:
            [B*S, 3, H, W], RGB in [0, 1]

        pred:
            [B*S, L, patch_dim]

        mask:
            [B*S, L], 0 = keep, 1 = masked

        Returns:
            loss
            masked_rgb:
                [B*S, 3, H, W]
            reconstructed_rgb:
                [B*S, 3, H, W]
        """

        # ------------------------------------------------------------
        # Ground-truth patches in RGB space
        # ------------------------------------------------------------
        target = self.patchify(imgs)
        # [B*S, L, patch_dim]

        # ------------------------------------------------------------
        # Normalize target if using pixel-normalized loss
        # ------------------------------------------------------------
        if self.norm_pix_loss:

            mean = target.mean(
                dim=-1,
                keepdim=True,
            )

            var = target.var(
                dim=-1,
                keepdim=True,
            )

            std = torch.sqrt(
                var + 1e-6
            )

            target_loss = (
                target - mean
            ) / std

            # Model predicts normalized pixels.
            pred_loss = pred

            # Recover RGB from normalized prediction.
            pred_rgb = (
                pred_loss * std
                + mean
            )

            pred_rgb = pred_rgb.clamp(
                0.0,
                1.0,
            )

        else:

            # Target is already RGB [0, 1].
            target_loss = target

            # Decoder output -> RGB.
            pred_rgb = pred.sigmoid()

        # ------------------------------------------------------------
        # Reconstruction loss
        # ------------------------------------------------------------
        loss = (
            pred_loss - target_loss
        ).pow(2)

        # Mean over RGB values within each patch
        loss = loss.mean(
            dim=-1
        )
        # [B*S, L]

        # Only masked patches contribute
        loss = (
            loss * mask
        ).sum() / (
            mask.sum() + 1e-6
        )

        # ------------------------------------------------------------
        # RGB visualizations: only outside training (unused there), never tracked by autograd
        # ------------------------------------------------------------
        if self.training:
            return loss, None, None

        with torch.no_grad():
            # White out masked GT patches
            masked_rgb = self.mask_image_with_white(imgs, mask)
            # Fill masked patches with reconstructed RGB
            reconstructed_rgb = self.fill_masked_patches(imgs, mask, pred_rgb.detach())

        return (
            loss,
            masked_rgb,
            reconstructed_rgb,
        )


    def forward(self, imgs, mask_ratio=None):
        B, S, C, H, W = imgs.shape

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        with torch.autocast(device_type=pred.device.type, enabled=False):
            imgs = imgs.reshape(B*S, C, H, W)
            loss, masked_rgb, reconstructed_rgb = self.forward_loss(imgs.float(), pred.float(), mask.float())

        if masked_rgb is not None:                      # None during training
            masked_rgb = masked_rgb.reshape(B, S, 3, H, W)
            reconstructed_rgb = reconstructed_rgb.reshape(B, S, 3, H, W)

        return loss, masked_rgb, reconstructed_rgb


def mae_vit_base_patch16_dec512d8b(img_size = 224, **kwargs):
    model = MaskedAutoencoderViT(
        img_size = img_size,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(img_size = 224, **kwargs):
    model = MaskedAutoencoderViT(
        img_size = img_size,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch16_dec512d8b(img_size =224, **kwargs):
    model = MaskedAutoencoderViT(
        img_size = img_size,
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch16 = mae_vit_huge_patch16_dec512d8b  # decoder: 512 dim, 8 blocks