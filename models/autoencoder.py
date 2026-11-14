import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Callable, Union, Optional, List, Any, Tuple, Dict

from models.layers import PatchEmbed
from models.layers import Block, CrossAttentionBlock
from models.layers import RotaryPositionEmbedding2D, PositionGetter

import math

class Encoder(nn.Module):
    """
    Encoder with alternating frame-level and global attention.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        embed_dim: Token embedding dimension.
        depth: Number of attention blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        n_storage_tokens: Number of storage/register tokens.
        block_fn: Transformer block class.
        in_chans: Number of input image channels.
        qkv_bias: Whether to use QKV bias.
        proj_bias: Whether to use attention projection bias.
        patch_embed: Patch embedding type.
        aa_order: Alternating attention order.
        aa_block_size: Number of consecutive blocks per attention type.
        qk_norm: Whether to normalize Q/K.
        rope_freq: RoPE frequency. Set <= 0 to disable.
        init_values: LayerScale initialization.
        cached_layer_indices: Transformer layers whose outputs are returned.
        use_reentrant: Whether to use reentrant checkpointing.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        n_storage_tokens: int = 0,
        block_fn=Block,
        in_chans: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        aa_order: Tuple[str, ...] = ("frame", "global"),
        aa_block_size: int = 1,
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
        cached_layer_indices: Tuple[int, ...] = (3, 5, 7, 11),
        use_reentrant: bool = False,
    ):
        super().__init__()

        self.depth = depth
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.use_reentrant = use_reentrant

        # ---------------------------------------------------------
        # Patch embedding
        # ---------------------------------------------------------

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # ---------------------------------------------------------
        # Special tokens
        # ---------------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.empty(1, 1, embed_dim)
        )

        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim)
            )

        self.patch_start_idx = 1 + n_storage_tokens

        # ---------------------------------------------------------
        # RoPE
        # ---------------------------------------------------------

        self.rope = (
            RotaryPositionEmbedding2D(
                frequency=rope_freq
            )
            if rope_freq > 0
            else None
        )

        self.position_getter = (
            PositionGetter()
            if self.rope is not None
            else None
        )

        # ---------------------------------------------------------
        # Cached layers
        # ---------------------------------------------------------

        self.cached_layer_indices = set(
            cached_layer_indices
        )

        # Always retain final layer.
        self.cached_layer_indices.add(depth - 1)

        # ---------------------------------------------------------
        # Initialize special tokens
        # ---------------------------------------------------------

        nn.init.normal_(
            self.cls_token,
            std=0.02,
        )

        if n_storage_tokens > 0:
            nn.init.normal_(
                self.storage_tokens,
                std=0.02,
            )

        # ---------------------------------------------------------
        # Frame attention blocks
        # ---------------------------------------------------------

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        # ---------------------------------------------------------
        # Global attention blocks
        # ---------------------------------------------------------

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.aa_block_num = depth // aa_block_size

    # =============================================================
    # Forward
    # =============================================================

    def forward(
        self,
        images: torch.Tensor,
        return_intermediate_outputs: bool = False,
    ):
        """
        Args:
            images:
                [B, S, 3, H, W]

        Returns:
            If return_intermediate_outputs=False:
                tokens, patch_start_idx

            If return_intermediate_outputs=True:
                output_list, patch_start_idx
        """

        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(
                f"Expected 3 input channels, got {C_in}"
            )

        # ---------------------------------------------------------
        # Patch embedding
        # ---------------------------------------------------------

        images = images.reshape(
            B * S,
            C_in,
            H,
            W,
        )

        x = self.patch_embed(images)


        _, _, C = x.shape

        # ---------------------------------------------------------
        # Special tokens
        # ---------------------------------------------------------
        cls_tokens = self.cls_token.expand(B*S, -1, -1)
        tokens = torch.cat((cls_tokens, x), dim=1)
        _, P, C = tokens.shape

        # ---------------------------------------------------------
        # RoPE positions
        # ---------------------------------------------------------

        pos = None

        if self.rope is not None:
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size

            pos = self.position_getter(
                B * S,
                grid_h,
                grid_w,
                device=images.device,
            )

            # Shift patch positions so special tokens occupy
            # the prefix.
            pos = pos + 1

            pos_special = torch.zeros(
                B * S,
                self.patch_start_idx,
                2,
                device=images.device,
                dtype=pos.dtype,
            )

            pos = torch.cat(
                [
                    pos_special,
                    pos,
                ],
                dim=1,
            )
        
        # ---------------------------------------------------------
        # Alternating attention
        # ---------------------------------------------------------

        frame_idx = 0
        global_idx = 0

        output_list = []

        for _ in range(self.aa_block_num):

            frame_intermediates = []
            global_intermediates = []

            for attn_type in self.aa_order:

                if attn_type == "frame":

                    (
                        tokens,
                        frame_idx,
                        frame_intermediates,
                    ) = self._process_frame_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        frame_idx,
                        pos=pos,
                    )

        
                elif attn_type == "global":

                    (
                        tokens,
                        global_idx,
                        global_intermediates,
                    ) = self._process_global_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        global_idx,
                        pos=pos,
                    )

                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_type}"
                    )

            # -----------------------------------------------------
            # Save selected intermediate features
            # -----------------------------------------------------

            if return_intermediate_outputs:
                for layer_idx in range(len(frame_intermediates)):
                        output_list.append(torch.cat([frame_intermediates[layer_idx],global_intermediates[layer_idx]],dim=-1))

        if return_intermediate_outputs:
            return (
                output_list,
                self.patch_start_idx,
            )

        return tokens.view(B, S, P, C),  self.patch_start_idx

    # =============================================================
    # Frame attention
    # =============================================================

    def _process_frame_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        frame_idx: int,
        pos: Optional[torch.Tensor] = None,
    ):

        """
        Frame attention operates independently on each frame.

        Input/output:
            [B*S, P, C]
        """

        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(
                B * S,
                P,
                C,
            )

        if pos is not None:
            if pos.shape != (B * S, P, 2):
                pos = pos.reshape(
                    B * S,
                    P,
                    2,
                )

        intermediates = []

        for _ in range(self.aa_block_size):

            block = self.frame_blocks[frame_idx]

            if self.training:
                tokens = checkpoint(
                    block,
                    tokens,
                    pos,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = block(
                    tokens,
                    pos=pos,
                )

            frame_idx += 1

            intermediates.append(
                tokens.reshape(
                    B,
                    S,
                    P,
                    C,
                )
            )

        return (
            tokens,
            frame_idx,
            intermediates,
        )

    # =============================================================
    # Global attention
    # =============================================================

    def _process_global_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        global_idx: int,
        pos: Optional[torch.Tensor] = None,
    ):

        """
        Global attention operates across all frames.

        Input/output:
            [B, S*P, C]
        """

        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(
                B,
                S * P,
                C,
            )

        if pos is not None:
            if pos.shape != (B, S * P, 2):
                pos = pos.reshape(
                    B,
                    S * P,
                    2,
                )

        intermediates = []

        for _ in range(self.aa_block_size):

            block = self.global_blocks[global_idx]

            if self.training:
                tokens = checkpoint(
                    block,
                    tokens,
                    pos,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = block(
                    tokens,
                    pos=pos,
                )

            global_idx += 1

            intermediates.append(
                tokens.reshape(
                    B,
                    S,
                    P,
                    C,
                )
            )

        return (
            tokens,
            global_idx,
            intermediates,
        )


# =================================================================
# Token utilities
# =================================================================

def slice_expand_and_flatten(
    token_tensor: torch.Tensor,
    B: int,
    S: int,
) -> torch.Tensor:
    """
    Expand special tokens across batch and frames.

    Input:
        [1, X, ...] where X is 1 or n_storage_tokens.

    For two-position tokens, the first position is used for the
    first frame and the second position is shared by remaining
    frames.

    Output:
        [B*S, X, ...]
    """

    # First token / position.
    first = token_tensor[:, 0:1].expand(
        B,
        1,
        *token_tensor.shape[2:],
    )

    # Remaining positions.
    if token_tensor.shape[1] > 1:

        others = token_tensor[:, 1:].expand(
            B,
            S - 1,
            *token_tensor.shape[2:],
        )

        combined = torch.cat(
            [
                first,
                others,
            ],
            dim=1,
        )

    else:
        combined = first.expand(
            B,
            S,
            *token_tensor.shape[2:],
        )

    return combined.reshape(
        B * S,
        *combined.shape[2:],
    )

class EncoderV2(nn.Module):
    """
    Encoder for clean + masked views with alternating frame-level and global attention.

    The first `num_clean` views are clean (A); the rest are masked (B). Masked
    patches of B are replaced by a learned mask token after patch embedding.
    A and B are processed jointly, and tokens of ALL views are returned.

    mask=None -> all views clean (downstream / inference).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        n_storage_tokens: int = 0,
        block_fn=Block,
        in_chans: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        aa_order: Tuple[str, ...] = ("frame", "global"),
        aa_block_size: int = 1,
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
        cached_layer_indices: Tuple[int, ...] = (3, 5, 7, 11),
        use_reentrant: bool = False,
        # num_clean: int = 4,
    ):
        super().__init__()

        # if num_clean < 1:
        #     raise ValueError(f"num_clean must be >= 1, got {num_clean}")

        # self.num_clean = num_clean
        self.depth = depth
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.aa_block_num = depth // aa_block_size
        self.use_reentrant = use_reentrant

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Special tokens
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
            nn.init.normal_(self.storage_tokens, std=0.02)

        self.patch_start_idx = 1 + n_storage_tokens

        # Mask token for masked patches of B views
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # RoPE
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Cached layers (always retain final layer)
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices.add(depth - 1)

        # Frame and global attention blocks
        def make_blocks():
            return nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )

        self.frame_blocks = make_blocks()
        self.global_blocks = make_blocks()

    # =============================================================
    # Forward
    # =============================================================

    def forward(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_intermediate_outputs: bool = False,
    ):
        """
        Args:
            images: [B, S, 3, H, W], first `num_clean` views clean, rest masked.
            mask:   [B, S - num_clean, N] bool, True = masked patch.
                    None -> all views clean (downstream / inference).

        Returns:
            If return_intermediate_outputs=False:
                tokens of all views [B, S, P, C], patch_start_idx
            If return_intermediate_outputs=True:
                list of [B, S, P, 2C], patch_start_idx
        """
        # B, S, C_in, H, W = images.shape
        # if C_in != 3:
        #     raise ValueError(f"Expected 3 input channels, got {C_in}")

        # num_clean = S if mask is None else self.num_clean
        # S_m = S - num_clean
        # if mask is not None and not 0 < num_clean < S:
        #     raise ValueError(f"Got {S} views but num_clean={num_clean}; need at least one masked view")

        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if mask is None:
            num_clean = S
            S_m = 0
        else:
            S_m = mask.shape[1]
            num_clean = S - S_m

        # Patch embedding
        x = self.patch_embed(images.reshape(B * S, C_in, H, W))  # [B*S, N, C]
        _, N, C = x.shape

        # Replace masked patches of B views with the mask token
        if S_m > 0:
            if mask.shape != (B, S_m, N):
                raise ValueError(f"Expected mask of shape {(B, S_m, N)}, got {tuple(mask.shape)}")
            x = x.view(B, S, N, C)
            masked_views = torch.where(
                mask.bool().unsqueeze(-1),
                self.mask_token.to(x.dtype).expand(B, S_m, N, C),
                x[:, num_clean:],
            )
            x = torch.cat([x[:, :num_clean], masked_views], dim=1).reshape(B * S, N, C)

        # Special tokens
        special = [self.cls_token.expand(B * S, -1, -1)]
        if self.n_storage_tokens > 0:
            special.append(self.storage_tokens.expand(B * S, -1, -1))
        tokens = torch.cat(special + [x], dim=1)
        _, P, C = tokens.shape

        # RoPE positions (masked tokens keep their patch positions)
        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            ) + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Alternating attention over A and B jointly
        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            frame_intermediates, global_intermediates = [], []
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if return_intermediate_outputs:
                for i in range(len(frame_intermediates)):
                    output_list.append(
                        torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    )

        if return_intermediate_outputs:
            return output_list, self.patch_start_idx

        return tokens.view(B, S, P, C), self.patch_start_idx

    # =============================================================
    # Frame attention: [B*S, P, C]
    # =============================================================

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B * S, P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]
            if self.training:
                tokens = checkpoint(block, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    # =============================================================
    # Global attention: [B, S*P, C]
    # =============================================================

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S * P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]
            if self.training:
                tokens = checkpoint(block, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = block(tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, global_idx, intermediates
        

class Decoder(nn.Module):
    """
    Cross-attention decoder.

    Input:
        target_tokens:    [B, S_target, P, D]                    D = decoder_embed_dim
        reference_tokens: [B, S_ref, patch_start_idx + P, C]     C = embed_dim
                          (CLS / storage tokens first, then patches)

    Each block:
        Q   = target tokens      (target positions)
        K,V = reference tokens   (reference positions; prefix tokens at position 0)

    Returns:
        [B, S_target*P, D]
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = True,
        rope_freq: int = 0,
        init_values: float = 0.01,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        cross_block_fn: Callable[..., nn.Module] = CrossAttentionBlock,
        gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.depth = depth
        self.gradient_checkpointing = gradient_checkpointing

        # ====================================================
        # Reference feature projection
        # ====================================================

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # ====================================================
        # RoPE
        # ====================================================

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # ====================================================
        # Cross-attention
        #
        # Q   = target
        # K,V = reference
        # ====================================================

        self.cross_blocks = nn.ModuleList(
            [
                cross_block_fn(
                    dim=decoder_embed_dim,
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
            ]
        )

        self.init_weights()

    # ========================================================
    # Initialization
    # ========================================================

    def init_weights(self):
        nn.init.trunc_normal_(self.decoder_embed.weight, std=0.02)
        if self.decoder_embed.bias is not None:
            nn.init.zeros_(self.decoder_embed.bias)

    # ========================================================
    # Build 2D positions
    # ========================================================

    def build_positions(self, batch_size: int, height: int, width: int, device):
        """
        Returns:
            [B*num_views, H*W, 2]  or None if RoPE is disabled
        """
        if self.position_getter is None:
            return None
        return self.position_getter(batch_size, height, width, device=device)

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        target_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        patch_start_idx: int = 0,
    ):
        """
        target_tokens:    [B, S_target, P, D]
        reference_tokens: [B, S_ref, patch_start_idx + P, C]
        patch_start_idx:  number of prefix tokens per reference view (0 = patches only)

        Returns:
            [B, S_target*P, D]
        """

        B, S_target, P, D = target_tokens.shape
        _, S_ref, P_ref, _ = reference_tokens.shape

        assert P_ref == P + patch_start_idx, (
            f"reference has {P_ref} tokens per view, expected {P} patches + {patch_start_idx} prefix tokens"
        )

        H_patch = W_patch = int(math.sqrt(P))
        assert H_patch * W_patch == P, f"P={P} is not a square"

        # Project reference features (prefix + patches): [B, S_ref*P_ref, D]
        context_tokens = self.decoder_embed(reference_tokens).reshape(B, S_ref * P_ref, D)

        # ====================================================
        # Positions
        #
        # Patches start at 1; position 0 is reserved for the
        # prefix tokens (same convention as the encoder).
        # RoPE is relative, so the +1 shift does not change
        # patch-to-patch attention.
        # ====================================================

        target_pos = self.build_positions(B * S_target, H_patch, W_patch, target_tokens.device)
        reference_pos = self.build_positions(B * S_ref, H_patch, W_patch, target_tokens.device)

        if target_pos is not None:
            target_pos = (target_pos + 1).reshape(B, S_target * P, -1)            # [B, S_target*P, 2]

            prefix_pos = torch.zeros(
                B * S_ref, patch_start_idx, 2,
                device=reference_pos.device, dtype=reference_pos.dtype,
            )
            reference_pos = torch.cat([prefix_pos, reference_pos + 1], dim=1)     # [B*S_ref, P_ref, 2]
            reference_pos = reference_pos.reshape(B, S_ref * P_ref, -1)           # [B, S_ref*P_ref, 2]

        # ====================================================
        # Cross attention
        #
        # Q   = target       [B, S_target*P,   D]
        # K,V = reference    [B, S_ref*P_ref,  D]
        # ====================================================

        x = target_tokens.reshape(B, S_target * P, D)

        for block in self.cross_blocks:

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    query_tokens=x,
                    context_tokens=context_tokens,
                    query_pos=target_pos,
                    context_pos=reference_pos,
                    use_reentrant=False,
                )
            else:
                x = block(
                    query_tokens=x,
                    context_tokens=context_tokens,
                    query_pos=target_pos,
                    context_pos=reference_pos,
                )

        return x


class Predictor(nn.Module):
    """
    JEPA predictor: cross-attention from target tokens to source tokens.

    Input:
        target_tokens:    [B, S_target, P, D]    D = decoder_embed_dim
        reference_tokens: [B, S_ref, P, C]       C = embed_dim

    Returns:
        [B, S_target, P, D]
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = True,
        rope_freq: int = 0,
        init_values: float = 0.01,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        cross_block_fn: Callable[..., nn.Module] = CrossAttentionBlock,
        gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.decoder = Decoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope_freq=rope_freq,
            init_values=init_values,
            norm_layer=norm_layer,
            cross_block_fn=cross_block_fn,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, target_tokens: torch.Tensor, reference_tokens: torch.Tensor):
        B, S_target, P, D = target_tokens.shape
        x = self.decoder(target_tokens, reference_tokens)
        return x.reshape(B, S_target, P, D)
