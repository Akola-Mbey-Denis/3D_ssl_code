import math
import random
import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from .util.rope import RotaryPositionEmbedding2D, PositionGetter
from .util.generate_mask import generate_connected_masks
from .util.misc import broadcast
from einops import rearrange

# ---------------------------
# Utilities: PatchEmbed
# ---------------------------
class PatchEmbed(nn.Module):
    """Image to patch tokens"""
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/ps, W/ps)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, E)
        return x, (H, W)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------
# Alternative Attention
# ---------------------------
class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.qk_norm = qk_norm

    def forward(self, view, view_pos, ret_attn=False):
        B, L, C = view.shape

        qkv = self.qkv(view).reshape(B, L, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = qkv.unbind(2)
        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        q = self.rope(q, view_pos)
        k = self.rope(k, view_pos)
        if ret_attn:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_tmp = attn.clone()
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            view = attn @ v
        else:
            view = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        view = view.transpose(1, 2).reshape(B, L, C)
        view = self.proj(view)
        view = self.proj_drop(view)
        if ret_attn:
            return view, attn_tmp
        else:
            return view

class AABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.frame_attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.frame_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.global_atten = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm4 = norm_layer(dim)
        self.global_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, view, view_pos, ret_attn=False):
        B, V, L, C = view.shape
        view = view.view(B*V, L, C)
        view_pos = view_pos.view(B*V, L, 2)
        view = view + self.drop_path(self.frame_attn(self.norm1(view), view_pos))
        view = view + self.drop_path(self.frame_mlp(self.norm2(view)))

        view = view.view(B, V*L, C)
        view_pos = view_pos.view(B, V*L, 2)
        if ret_attn:
            x_tmp, attn_map = self.global_atten(self.norm3(view), view_pos, ret_attn=True)
            view = view + self.drop_path(x_tmp)
        else:
            view = view + self.drop_path(self.global_atten(self.norm3(view), view_pos))
        view = view + self.drop_path(self.global_mlp(self.norm4(view)))

        view = view.view(B, V, L, C)
        if ret_attn:
            return view, attn_map.mean(1) # average over head_dim: (B, num_heads, hp*wp, hp*wp) -> (B, hp*wp, hp*wp)
        else:
            return view


# ---------------------------
# Full Multi-view Croco
# ---------------------------
class MultiViewCroco(nn.Module):
    def __init__(
        self,
        block_type=AABlock,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        rope_freq=100,
        enable_checkpoint=False,
        num_ref=-1,             # -1: Muskie default (random count); n >= 0: first n views are references
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.depth = depth
        self.num_register_tokens = num_register_tokens
        self.enable_checkpoint = enable_checkpoint
        self.num_ref = num_ref

        # register tokens per view (learnable)
        self.register_tokens = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))

        self.mask_placeholder = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()

        # create transformer blocks
        self.blocks = nn.ModuleList([
            block_type(embed_dim, num_heads, mlp_ratio, qkv_bias=True, rope=self.rope)
            for _ in range(depth)
        ])

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans * 2, bias=True) # decoder to patch

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

    def random_masking(self, x, Hp, Wp, mask_mode, mask_ratio):
        """
        Apply connected masking to non-reference views.

        x: [B, V, L, C]

        The first num_ref views remain unmasked.
        Each remaining view independently selects a masking strategy.
        Masked tokens are replaced, not dropped.
        """
        B, V, L, C = x.shape

        # num_ref >= 0: fixed split -- the first num_ref views are always clean.
        # num_ref < 0: Muskie default -- random number of reference views per batch.
        if self.num_ref >= 0:
            num_ref = self.num_ref
            assert num_ref < V, f"num_ref={num_ref} must be smaller than V={V}"
        else:
            num_ref = random.randint(V // 4, V // 2)

        # Each masked view draws its own strategy; ratio is paired with strategy.
        n_masked = V - num_ref
        mode_ids = torch.randint(
            len(mask_mode), (B, n_masked), device=x.device
        )

        mask = torch.zeros(
            B, n_masked, Hp, Wp,
            device=x.device,
            dtype=torch.float32,
        )

        # Flatten only the non-reference views for batched masking.
        x_masked = x[:, num_ref:].reshape(B * n_masked, L, C)
        mode_ids_flat = mode_ids.reshape(-1)
        mask_flat = mask.reshape(B * n_masked, L)

        for i, (mode, ratio) in enumerate(zip(mask_mode, mask_ratio)):
            # Only process views assigned to this masking strategy.
            indices = torch.where(mode_ids_flat == i)[0]
            if indices.numel() == 0:
                continue

            x_i = x_masked.index_select(0, indices)

            _, m_i, _ = self.connected_masking(
                x=x_i,
                mask_ratio=ratio,
                mode=mode,
            )

            mask_flat.index_copy_(0, indices, m_i.float())

        mask = mask_flat.reshape(B, n_masked, L)

        # Add the unmasked reference views.
        mask = torch.cat(
            (
                torch.zeros(
                    B, num_ref, L,
                    device=x.device,
                    dtype=mask.dtype,
                ),
                mask,
            ),
            dim=1,
        )

        # Replace masked tokens; do not drop them.
        x = torch.where(
            mask.unsqueeze(-1) > 0.5,
            self.mask_placeholder,
            x,
        )

        return x, mask


    # def random_masking(self, x, Hp, Wp, mask_mode, mask_ratio):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     B, V, L, C = x.shape  # batch, length, dim

    #     # num_ref >= 0: fixed split -- the first num_ref views are always the clean
    #     #               reference views (e.g. 4 sources + 2 targets from the dataloader)
    #     # num_ref < 0:  Muskie default -- a random number of reference views per batch
    #     # In both cases the reference views come first; the remaining views are masked.
    #     if self.num_ref >= 0:
    #         num_ref = self.num_ref
    #         assert num_ref < V, f"num_ref={num_ref} must be smaller than V={V}"
    #     else:
    #         num_ref = random.randint(V//4,V//2)

    #     # each masked view draws its own strategy; the ratio is paired with it
    #     # (mask_mode[i] <-> mask_ratio[i], e.g. rectangle/ellipse 0.75, random 0.90).
    #     # Masked tokens are replaced (not dropped), so ratios can differ per view.
    #     # SPEED: one batched generate_connected_masks call per strategy for ALL masked views
    #     # (was one call per (sample, view): ~B*(V-num_ref) calls per step), and the per-view
    #     # strategy is drawn on the GPU and selected with torch.where (same distribution).
    #     n_masked = V - num_ref
    #     mode_ids = torch.randint(len(mask_mode), (B, n_masked), device=x.device)
    #     mask = torch.zeros(B, n_masked, Hp, Wp, device=x.device, dtype=torch.float32)
    #     for i, (mode, ratio) in enumerate(zip(mask_mode, mask_ratio)):
    #         m_i = generate_connected_masks(B, n_masked, Hp, Wp, ratio=ratio,
    #                                        device=x.device, dtype=torch.float32, mode=mode)
    #         mask = torch.where((mode_ids == i)[..., None, None], m_i, mask)
    #     mask = mask.flatten(2, 3)                                         # [B, V-num_ref, L]
    #     mask = torch.cat((torch.zeros(B, num_ref, L, device=x.device, dtype=mask.dtype), mask), dim=1)

    #     x = torch.where(mask.unsqueeze(-1)>0.5, self.mask_placeholder, x)

    #     return x, mask

    def compute_confloss(self, imgs, pred, conf, mask):
        """
        imgs: [B, V, 3, H, W]
        pred: [B, V, L, p*p*3]
        conf: [B, V, L, p*p*3]
        mask: [B, V, L], 0=keep, 1=remove
        """
        target = F.pixel_unshuffle(imgs, downscale_factor=self.patch_size)
        target = target.permute(0,1,3,4,2).flatten(2, 3)

        mse = (pred - target) ** 2
        loss = ((conf+0.1) * mse).mean(dim=-1)  # [N, L], mean loss per patch

        per_instance_loss = (loss * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2))

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        conf_reg = -torch.log(conf + 1e-6).mean(dim=-1)
        conf_reg = (conf_reg * mask).mean()

        return {
            'mse': (mse.detach().mean(dim=-1) * mask).sum() / mask.sum(),
            'conf_reg': conf_reg,
            'loss': loss + 0.1 * conf_reg,
        }, per_instance_loss.detach()

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

    def fill_masked_patches(self, imgs, mask, pred):
        """
        Paste predicted RGB into the masked patches, keep the GT elsewhere.

        imgs: [N, 3, H, W]    mask: [N, L] (1 = masked)    pred: [N, 3, H, W] (already pixel_shuffled)
        returns: [N, 3, H, W]
        """
        N, C, H, W = imgs.shape
        p = self.patch_size
        pixel_mask = (mask.reshape(N, H // p, W // p).bool().repeat_interleave(p, dim=1).repeat_interleave(p, dim=2).unsqueeze(1))

        return torch.where(pixel_mask, pred.clamp(0.0, 1.0), imgs)      # Muskie predicts raw RGB: just clamp

    def forward(self, images: torch.Tensor, random_aspect_ratio=False, random_num_views=False,
                mask_mode=('rectangle', 'ellipse', 'random'), mask_ratio=(0.75, 0.75, 0.90)):
        """
        images: (B, V, C, H, W)
        returns: per-view token lists
        """
        if random_aspect_ratio and images.shape[-1] == 512:
            # randomly select an aspect ratio from 1:1, 4:3, 16:9.
            H, W = random.choice([(512, 288), (512, 384), (512, 512), (288, 512), (384, 512), (512, 512)])

            # synchronize ratio across GPUs to avoid waiting.
            H, W = broadcast((H, W))
            margin_H, margin_W = (512-H) // 2, (512-W) // 2
            images = images[..., margin_H:512-margin_H, margin_W:512-margin_W]
        if random_num_views:
            V = random.choice([1,2,4,8])
            V, = broadcast((V,))
            total_views = images.shape[0] * images.shape[1]
            images = images.view(total_views//V, V, *images.shape[2:])
        B, V, C_in, H, W = images.shape

        x, (Hp, Wp) = self.patch_embed(images.view(B*V,3,H,W))

        x = x.view(B, V, Hp*Wp, -1) # (B, N_patches, E)
        x, mask = self.random_masking(x, Hp, Wp, mask_mode=mask_mode, mask_ratio=mask_ratio)
        x = torch.cat((self.register_tokens.expand(B, V, -1, -1), x), dim=2)

        pos = self.position_getter(B * V, Hp, Wp, device=images.device).view(B, V, Hp*Wp, -1)
        pos = pos + 1
        pos = torch.cat((torch.zeros(B, V, self.num_register_tokens, 2, device=pos.device, dtype=pos.dtype), pos), dim=2)

        # run blocks
        for i, blk in enumerate(self.blocks):
            if self.enable_checkpoint:
                x = checkpoint.checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)

        # final norm on concatenated views
        x = x[:,:,self.num_register_tokens:]
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        pred, conf = torch.chunk(x, 2, dim=-1)
        conf = torch.sigmoid(conf)

        loss_dict, per_instance_loss = self.compute_confloss(images, pred, conf, mask)

        pred = F.pixel_shuffle(pred.view(B,V,Hp,Wp,-1).permute(0,1,4,2,3), upscale_factor=self.patch_size)
        conf = F.pixel_shuffle(conf.view(B,V,Hp,Wp,-1).permute(0,1,4,2,3), upscale_factor=self.patch_size)

        # visualization only outside training (unused there), never tracked by autograd

        if self.training:
            return loss_dict, per_instance_loss, None , conf, mask, None
        else:
            with torch.no_grad():
                masked_images = self.mask_image_with_white(imgs = images.reshape(B*V, C_in, H, W), mask = mask.reshape(B*V, -1))
                masked_images = masked_images.reshape(B, V, C_in, H, W)
                reconstructed_images = self.fill_masked_patches(imgs = images.reshape(B*V, C_in, H, W), mask = mask.reshape(B*V, -1), pred = pred.detach().reshape(B * V, C_in, H, W)).reshape(B, V, C_in, H, W)
                
                return loss_dict, per_instance_loss, reconstructed_images, conf, mask, masked_images


    def get_latent_embeddings(self, images: torch.Tensor, ret_attn: bool = False):
        """
        images: (B, V, C, H, W)
        returns: per-view token lists
        """
        B, V, _, H, W = images.shape
        
        # Get patch embeddings from encoder
        x, (Hp, Wp) = self.patch_embed(images.view(B*V, 3, H, W))
        x = x.view(B, V, Hp*Wp, -1)  # (B, V, L, E)
        
        # Add register tokens
        x = torch.cat((self.register_tokens.expand(B, V, -1, -1), x), dim=2)
        
        # Get positional embeddings
        pos = self.position_getter(B * V, Hp, Wp, device=images.device).view(B, V, Hp*Wp, -1)
        pos = pos + 1
        pos = torch.cat((torch.zeros(B, V, self.num_register_tokens, 2, device=pos.device, dtype=pos.dtype), pos), dim=2)
        
        # Run encoder blocks
        enc_tokens = []
        attn_maps = []
        for blk in self.blocks:
            if self.enable_checkpoint:
                x = checkpoint.checkpoint(blk, x, pos, use_reentrant=False)
            else:
                if ret_attn:
                    x, attn_map = blk(x, pos, ret_attn=True)
                    attn = rearrange(attn_map, 
                        'b (v1 N1) (v2 N2) -> b v1 N1 v2 N2',
                        v1=V, v2=V, 
                        N1=self.num_register_tokens+Hp*Wp,
                        N2=self.num_register_tokens+Hp*Wp
                    )
                    attn = attn[:, :, self.num_register_tokens:, :, self.num_register_tokens:]
                    attn_maps.append(attn)
                else:
                    x = blk(x, pos, ret_attn=False)
            enc_tokens.append(
                # rearrange(x, 'b v l e -> (b v) l e')
                x
            )
        
        if ret_attn:
            return enc_tokens, attn_maps
        else:
            return enc_tokens

def muskie_small(**kwargs):
    return MultiViewCroco(embed_dim=384, depth=6, num_heads=6, **kwargs)

def muskie_base(**kwargs):
    return MultiViewCroco(embed_dim=768, depth=6, num_heads=12, **kwargs)

def muskie_large(**kwargs):
    return MultiViewCroco(embed_dim=1024, depth=12, num_heads=16, **kwargs)

def muskie_huge(**kwargs):
    return MultiViewCroco(embed_dim=1280, depth=16, num_heads=16, **kwargs)