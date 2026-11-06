class DecoderV2(nn.Module):
    """
    Decoder alternating self-attention and cross-attention (ablation of the cross-attention-only Decoder).

    Input:
        target_tokens:    [B, S_target, P, D]                    D = decoder_embed_dim
        reference_tokens: [B, S_ref, patch_start_idx + P, C]     C = embed_dim
                          (CLS / storage tokens first, then patches)

    Each of the `depth` stages:
        1. self-attention   Q = K = V = target tokens
                            self_attn_scope="view":    within each target view   [B*S_target, P, D]
                            self_attn_scope="targets": across all target views   [B, S_target*P, D]
        2. cross-attention  Q = target tokens, K,V = reference tokens (prefix tokens at position 0)

    Self-attention only mixes target tokens, so the number of reference views does not matter: S_ref and
    S_target can differ. With depth = Decoder depth / 2 the parameter count is about the same as Decoder.

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
        self_block_fn: Callable[..., nn.Module] = Block,
        cross_block_fn: Callable[..., nn.Module] = CrossAttentionBlock,
        self_attn_scope: str = "view",
        gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()
        if self_attn_scope not in ("view", "targets"):
            raise ValueError(f"self_attn_scope must be 'view' or 'targets', got {self_attn_scope!r}")

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.depth = depth
        self.self_attn_scope = self_attn_scope
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
        # Self-attention (Q = K = V = target)
        # ====================================================

        self.self_blocks = nn.ModuleList(
            [
                self_block_fn(
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

        # ====================================================
        # Cross-attention (Q = target, K,V = reference)
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
    # Blocks
    # ========================================================

    def _self_attention(self, block, x, pos, B, S_target, P, D):
        """x: [B, S_target*P, D]; pos: [B*S_target, P, 2] or None."""
        if self.self_attn_scope == "view":
            x = x.reshape(B * S_target, P, D)
            p = pos
        else:
            p = pos.reshape(B, S_target * P, -1) if pos is not None else None

        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(block, x, p, use_reentrant=False)
        else:
            x = block(x, pos=p)

        return x.reshape(B, S_target * P, D)

    def _cross_attention(self, block, x, context_tokens, query_pos, context_pos):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block,
                query_tokens=x,
                context_tokens=context_tokens,
                query_pos=query_pos,
                context_pos=context_pos,
                use_reentrant=False,
            )
        return block(
            query_tokens=x,
            context_tokens=context_tokens,
            query_pos=query_pos,
            context_pos=context_pos,
        )

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
        # Positions (patches start at 1, prefix tokens at 0)
        # ====================================================

        target_pos_view = self.build_positions(B * S_target, H_patch, W_patch, target_tokens.device)
        reference_pos = self.build_positions(B * S_ref, H_patch, W_patch, target_tokens.device)

        target_pos = None
        if target_pos_view is not None:
            target_pos_view = target_pos_view + 1                                    # [B*S_target, P, 2]
            target_pos = target_pos_view.reshape(B, S_target * P, -1)                # [B, S_target*P, 2]

            prefix_pos = torch.zeros(
                B * S_ref, patch_start_idx, 2,
                device=reference_pos.device, dtype=reference_pos.dtype,
            )
            reference_pos = torch.cat([prefix_pos, reference_pos + 1], dim=1)       # [B*S_ref, P_ref, 2]
            reference_pos = reference_pos.reshape(B, S_ref * P_ref, -1)             # [B, S_ref*P_ref, 2]

        # ====================================================
        # Alternating self-attention / cross-attention
        # ====================================================

        x = target_tokens.reshape(B, S_target * P, D)

        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            x = self._self_attention(self_block, x, target_pos_view, B, S_target, P, D)
            x = self._cross_attention(cross_block, x, context_tokens, target_pos, reference_pos)

        return x