import math
import random
from functools import partial
from typing import Any, Sequence, Tuple, Union, Optional

import torch
import torch.nn.init
import torch.nn.functional as F
from torch import nn

from .layers import LayerScale, Mlp, PatchEmbed, RopePositionEmbedding, SelfAttentionBlock
from .utils import named_apply

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()

def build_model(cfg):
    vit_kwargs = dict(**cfg.model)
    vit_kwargs['device'] = 'cuda'
    model = globals()[cfg.model.name](**vit_kwargs)
    model.init_weights()
    return model

class MuMAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = True,
        
        norm_layer: str = "layernorm",
        n_storage_tokens: int = 0,
        device: Optional[Any] = None,
        gradient_checkpointing: bool = False,
        masking_types=("random", "ellipse", "rectangle"), 
        mask_per_view=True,
        mask_ratios=(0.90, 0.75, 0.75),
        default_masking: bool = True,       # True: MuM default (all views, random, ratio 0.75)
        n_ref_views : int = 0,
        default_mask_ratio: float = 0.75,   # False: mixed random / ellipse / rectangle above
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

         # --------------------------------------------------------------------------
        # Masking (Muskie-style): a strategy is drawn per view (mask_per_view=True)
        # or once per batch (False). The mask ratio is shared by the whole batch,
        # because the encoder drops masked tokens and every view must keep the
        # same number of tokens.
        self.masking_types = tuple(masking_types)
        self.mask_per_view = mask_per_view
        self.mask_ratios = tuple(mask_ratios)          # one is drawn per batch
        self.default_masking = default_masking
        self.n_ref_views = n_ref_views
        self.default_mask_ratio = default_mask_ratio

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.gradient_checkpointing = gradient_checkpointing

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        rope_cls = partial(
            RopePositionEmbedding,
            device=device,
        )
        self.rope_embed = rope_cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        block_cls = partial(SelfAttentionBlock, 
                ffn_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                mask_k_bias=False,
                device=device,
        )
        self.blocks = nn.ModuleList([block_cls(dim=embed_dim, num_heads=num_heads) for i in range(depth)])
        self.norm = norm_layer_cls(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.rope_embed_decoder = rope_cls(
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True, device=device)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim, device=device))
        # self.mask_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim, device=device))
        self.mask_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim, device=device)) if self.n_ref_views > 0 else nn.Identity()

        self.decoder_frame_blocks = nn.ModuleList([
            block_cls(dim=decoder_embed_dim, num_heads=decoder_num_heads)
            for i in range(decoder_depth//2)])

        self.decoder_global_blocks = nn.ModuleList([
            block_cls(dim=decoder_embed_dim, num_heads=decoder_num_heads)
            for i in range(decoder_depth//2)])
        
        self.decoder_norm = norm_layer_cls(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True, device=device) # decoder to patch
        # --------------------------------------------------------------------------

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)

        # nn.init.zeros_(self.mask_token)
        nn.init.normal_(self.mask_token, std=.02)
        if self.n_ref_views > 0:
            nn.init.normal_(self.mask_placeholder, std=0.02)
        named_apply(init_weights_vit, self)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h, w = imgs.shape[2] // p, imgs.shape[3] // p 
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
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
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

        return x_masked, mask, ids_restore, ids_keep

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
            x:           (N, L, D), where N = batch_size * num_views
            mask_ratio:  fraction of patches to mask
            mode:        rectangle, ellipse, random, blob, or random_walk
            blob_kernel: odd smoothing kernel size for blob masking
            ar_range:    minimum and maximum rectangle/ellipse aspect ratios

        Returns:
            x_masked:    (N, len_keep, D)
            mask:        (N, L), 0 = keep, 1 = mask, in original patch order
            ids_restore: (N, L), inverse permutation for MAE decoder
            ids_keep:    (N, len_keep), indices of visible patches
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
            ids_keep = torch.arange(L, device=device).expand(N, -1)
            ids_restore = ids_keep
            mask = torch.zeros(N, L, device=device)
            return x, mask, ids_restore, ids_keep

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

            inside_y = (yy >= top[:, None, None]) & (
                yy < (top + h)[:, None, None]
            )
            inside_x = (xx >= left[:, None, None]) & (
                xx < (left + w)[:, None, None]
            )
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

            scores = torch.where(
                region,
                (step - age).float() + 2.0 + noise * 1e-3,
                noise * 1e-3,
            ).flatten(1)

        # Sort patches by masking score.
        # Lower scores are kept; higher scores are masked.
        ids_shuffle = torch.argsort(scores, dim=1)

        # Indices of visible patches.
        ids_keep = ids_shuffle[:, :len_keep]

        # Invert the permutation for the MAE decoder.
        ids_restore = torch.empty_like(ids_shuffle)
        positions = torch.arange(L, device=device).expand(N, -1)
        ids_restore.scatter_(1, ids_shuffle, positions)

        # Gather visible patches.
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        # Mask in original patch order: 0 = keep, 1 = mask.
        mask = torch.zeros(N, L, device=device)
        mask.scatter_(1, ids_shuffle[:, len_keep:], 1.0)

        return x_masked, mask, ids_restore, ids_keep
 
   
    def sample_masking(self, x, mask_ratio=None, mask_type=None, n_ref=0, num_views=None):
        """
        x: [B*S, L, C]
 
        Returns:
            x_masked, mask, ids_restore, ids_keep
            n_ref == 0:  x_masked [N, K, C], mask [N, L], ids_restore [N, L], ids_keep [N, K]     (tokens dropped)
            n_ref  > 0:  x_masked [N, L, C], mask [N, L], ids_restore [N, L], ids_keep [N, L]     (identity ids)
        """
        N, L, C = x.shape
 
        if n_ref < 0:                                    # random number of clean views, as in the reference model
            assert num_views is not None, "num_views is required when n_ref < 0"
            n_ref = random.randint(num_views // 4, num_views // 2)
 
        # ---------------------------------------------------------------- n_ref == 0: original behaviour
        if n_ref == 0:
            # MuM default: every view masked with plain MAE random masking at a fixed ratio
            if self.default_masking and mask_type is None:
                ratio = self.default_mask_ratio if mask_ratio is None else mask_ratio
                return self.random_masking(x, ratio)
 
            type_to_ratio = dict(zip(self.masking_types, self.mask_ratios))   # e.g. random->0.90, ellipse/rectangle->0.75
 
            # one strategy drawn per batch decides the batch ratio
            if mask_type is None:
                mask_type = random.choice(self.masking_types)
            if mask_ratio is None:
                mask_ratio = type_to_ratio[mask_type]
 
            if not self.mask_per_view:
                return self.connected_masking(x, mask_ratio, mode=mask_type)
 
            # per view: any strategy that shares this batch's ratio
            modes = [t for t, r in type_to_ratio.items() if r == mask_ratio] or [mask_type]
 
            if len(modes) == 1:
                return self.connected_masking(x, mask_ratio, mode=modes[0])
            mode_ids = torch.randint(len(modes), (N,), device=x.device)
            outs = [self.connected_masking(x, mask_ratio, mode=m) for m in modes]
            x_masked, mask, ids_restore, ids_keep = outs[0]
            for k in range(1, len(modes)):
                sel = mode_ids == k
                x_masked = torch.where(sel[:, None, None], outs[k][0], x_masked)
                mask = torch.where(sel[:, None], outs[k][1], mask)
                ids_restore = torch.where(sel[:, None], outs[k][2], ids_restore)
                ids_keep = torch.where(sel[:, None], outs[k][3], ids_keep)
            return x_masked, mask, ids_restore, ids_keep
 
        # ---------------------------------------------------------------- n_ref > 0: clean + masked views together
        assert num_views is not None, "num_views is required when n_ref > 0"
        S = num_views
        assert N % S == 0 and n_ref < S, f"need 0 < n_ref < num_views (n_ref={n_ref}, num_views={S})"
        B = N // S
 
        # one strategy + ratio per batch
        if self.default_masking and mask_type is None:
            mode = "random"
            ratio = self.default_mask_ratio if mask_ratio is None else mask_ratio
        else:
            type_to_ratio = dict(zip(self.masking_types, self.mask_ratios))
            mode = random.choice(self.masking_types) if mask_type is None else mask_type
            ratio = type_to_ratio[mode] if mask_ratio is None else mask_ratio
 
        # mask only the non-clean views (dummy 1-channel x: only the mask is used, no need to gather the tokens)
        x_rest = x.reshape(B, S, L, C)[:, n_ref:, :, :1].reshape(B * (S - n_ref), L, 1)
        _, mask_rest, _, _ = self.connected_masking(x_rest, ratio, mode=mode)
 
        mask = torch.cat([torch.zeros(B, n_ref, L, device=x.device, dtype=mask_rest.dtype),
                          mask_rest.reshape(B, S - n_ref, L)], dim=1).reshape(N, L)
 
        # encoder-dim placeholder: learned if the model has self.mask_placeholder, zeros otherwise
        # (self.mask_token is decoder_embed_dim wide, the encoder tokens are embed_dim wide)
        placeholder = getattr(self, "mask_placeholder", None)
        placeholder = torch.zeros(1, 1, C, device=x.device, dtype=x.dtype) if placeholder is None else placeholder.to(x.dtype)
        x_masked = torch.where(mask.unsqueeze(-1) > 0.5, placeholder, x)
 
        ids_keep = torch.arange(L, device=x.device).expand(N, L)
        ids_restore = ids_keep
        return x_masked, mask, ids_restore, ids_keep



    def forward_encoder(self, x, mask_ratio = 0.0, n_views = 6, return_all_blocks=False):
    
        # embed patches
        if x.ndim == 5: 
            B, S, C_in, H, W = x.shape
            x = x.reshape(B*S, C_in, H, W)
        
        x = self.patch_embed(x)
        rope_sincos = self.rope_embed(H=x.shape[1], W=x.shape[2])
        x = x.flatten(1,2)  # [SB, L, C], with L=H*W
 
        # masking: length -> length * mask_ratio
        if not return_all_blocks: 
            x, mask, ids_restore, ids_keep = self.sample_masking(x = x, mask_ratio = mask_ratio, n_ref = self.n_ref_views, num_views = n_views)
            # breakpoint()
 
            # Let's just drop the masked patches in the rope
            # (only when tokens were dropped; with clean views all tokens are kept: full rope)
            sin, cos = rope_sincos 
            if ids_keep.shape[1] < sin.shape[0]:
                sin_vis, cos_vis = sin[ids_keep], cos[ids_keep]  # [B, N_vis, D_head]
                sin_vis, cos_vis = sin_vis.unsqueeze(1).repeat(1, self.num_heads, 1, 1), cos_vis.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                rope_sincos = (sin_vis, cos_vis)
 
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        BS, P, C  = x.shape
 
        # apply Transformer blocks
        if return_all_blocks:
            out = []
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, rope_sincos, use_reentrant=False)
                else:
                    x = blk(x, rope_sincos)
                out.append(x.reshape(B, S, -1, C))
            return out
        else:
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, rope_sincos, use_reentrant=False)
                else:
                    x = blk(x, rope_sincos)
            x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, B:int, S:int, H=None, W=None):
        # embed tokens
        num_patches_h, num_patches_w = H//self.patch_size, W//self.patch_size
        x = self.decoder_embed(x)

        sin, cos = self.rope_embed_decoder(H=num_patches_h, W=num_patches_w)
        rope_sincos_frame = (sin, cos)
        pos_special = torch.zeros(1 + self.n_storage_tokens, sin.shape[-1], device=sin.device, dtype=sin.dtype)
        sin, cos = torch.cat([pos_special, sin]), torch.cat([pos_special, cos])
        sin, cos = sin.repeat(S, 1), cos.repeat(S, 1)
        rope_sincos_global = (sin, cos)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        _, P, C = x.shape

        # apply alternating attention
        for frame_block, global_block in zip(self.decoder_frame_blocks, self.decoder_global_blocks):
            # Frame-wise attention
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(frame_block, x, rope_sincos_frame, use_reentrant=False)
            else:
                x = frame_block(x, rope_sincos_frame)
            
            # Global attention
            x = x.view(B, S, P, C).view(B, S * P, C)

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(global_block, x, rope_sincos_global, use_reentrant=False)
            else:
                x = global_block(x, rope_sincos_global)
        
        x = x.view(B, S, P, C).view(B*S, P, C)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    
    def mask_image_with_white(
        self,
        imgs,
        mask,
    ):
        """
        Replace masked patches with solid white.

        imgs:
            [B, 3, 224, 224]

        mask:
            [B, 14, 14] or [B, 196]

        returns:
            [B, 3, 224, 224]
        """

        B, C, H, W = imgs.shape
        p = self.patch_size

        h = H // p
        w = W // p

        # [B, 196] -> [B, 14, 14]
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
            [B, 3, 224, 224]

        mask:
            [B, 14, 14] or [B, 196]
            1 = masked
            0 = unmasked

        pred:
            [B, 196, patch_size * patch_size * 3]

        returns:
            [B, 3, 224, 224]
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
        B, S, C_in, H, W = imgs.shape
        imgs = imgs.view(B*S, C_in, H, W)  # [B*S, C, H, W]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, n_views = S)
        pred = self.forward_decoder(latent, ids_restore, B = B, S = S, H=H, W=W)  # [N, L, p*p*3]
        
        with torch.autocast(device_type=pred.device.type, enabled=False):
            loss, masked_rgb, reconstructed_rgb = self.forward_loss(imgs.float(), pred.float(), mask.float())
        

        if masked_rgb is not None:                      # None during training
            masked_rgb = masked_rgb.reshape(B, S, C_in, H, W)
            reconstructed_rgb = reconstructed_rgb.reshape(B, S, C_in, H, W)
        return loss, masked_rgb, reconstructed_rgb
        

    def forward_features(self, x):
        x = self.forward_encoder(x, 0, return_all_blocks=True)
        x = self.norm(x[-1])
        return {
            "x_norm_patchtokens": x[:, 1+self.n_storage_tokens :],  # [B, L, C]
            "x_norm_cls_token": x[:, 0],  # [B, C]
        }
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        B, _, w, h = x.shape

        outputs = self.forward_encoder(x, 0, return_all_blocks=True)
        outputs = [output for i, output in enumerate(outputs) if i in blocks_to_take]

        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1+self.n_storage_tokens:] for out in outputs]
        assert len(outputs) == len(blocks_to_take)
        if reshape:
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


def mum_vit_base(patch_size=16, img_size = 224, **kwargs):
    model = MuMAutoEncoder(
        patch_size=patch_size,
        img_size = img_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    return model


def mum_vit_large(patch_size=16, img_size = 224,  **kwargs):
    model = MuMAutoEncoder(
        patch_size=patch_size,
        embed_dim=1024,
        img_size = img_size,
        depth=24,
        num_heads=16,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=16,
        **kwargs,
    )
    return model

def mum_vit_huge(patch_size=16, img_size =224, **kwargs):
    model = MuMAutoEncoder(
        patch_size=patch_size,
        embed_dim=1280,
        img_size = img_size,
        depth=32,
        num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=24,
        decoder_num_heads=16,
        **kwargs,
    )
    return model

def mum_vitl16_decoderb(pretrained=True, **kwargs) -> MuMAutoEncoder:
    model = mum_vit_large(**kwargs)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/davnords/mum/releases/download/weights/MuM_ViTLarge_BaseDecoder.pth"
        )
        msg = model.load_state_dict(weights, strict=True)
    return model