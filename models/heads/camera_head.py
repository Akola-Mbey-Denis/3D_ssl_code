# import math
# from typing import Optional
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.layers import Mlp
# from models.layers import Block
# from .head_act import activate_pose


# class ResMLPBlock(nn.Module):
#     """
#     Residual block of three per-token linear layers (the 1x1-conv residual block of marepo,
#     https://github.com/nianticlabs/marepo, written with nn.Linear).
#     """

#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.head_skip = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
#         self.res_conv1 = nn.Linear(in_channels, out_channels)
#         self.res_conv2 = nn.Linear(out_channels, out_channels)
#         self.res_conv3 = nn.Linear(out_channels, out_channels)

#     def forward(self, res: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.res_conv1(res))
#         x = F.relu(self.res_conv2(x))
#         x = F.relu(self.res_conv3(x))
#         return self.head_skip(res) + x


# class PatchPool(nn.Module):
#     """
#     marepo-style pooling of each view's patch tokens into one token:
#     residual per-token MLP blocks -> global average over the patch grid -> MLP.
#     (B, S, P, C) -> (B, S, C)
#     """

#     def __init__(self, dim: int, num_res_blocks: int = 2):
#         super().__init__()
#         self.res_blocks = nn.Sequential(*[ResMLPBlock(dim, dim) for _ in range(num_res_blocks)])
#         self.more_mlps = nn.Sequential(
#             nn.Linear(dim, dim), nn.ReLU(),
#             nn.Linear(dim, dim), nn.ReLU(),
#         )

#     def forward(self, tokens: torch.Tensor) -> torch.Tensor:
#         x = self.res_blocks(tokens)               # (B, S, P, C), per patch
#         x = x.mean(dim=2)                         # average over the h x w grid (= AdaptiveAvgPool2d(1))
#         return self.more_mlps(x)    


# class CameraHead(nn.Module):
#     """
#     CameraHead predicts camera parameters from token representations using iterative refinement.

#     It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
#     """

#     def __init__(
#         self,
#         dim_in: int = 2048,
#         trunk_depth: int = 2,
#         pose_encoding_type: str = "absT_quaR_FoV",
#         num_heads: int = 16,
#         mlp_ratio: int = 2,
#         init_values: float = 0.01,
#         global_feat: bool = True,
#         model_type: Optional[str] = None,
#         trans_act: str = "linear",
#         quat_act: str = "linear",
#         fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
#     ):
#         super().__init__()

#         if pose_encoding_type == "absT_quaR_FoV":
#             self.target_dim = 9
#         else:
#             raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

#         self.trans_act = trans_act
#         self.quat_act = quat_act
#         self.fl_act = fl_act
#         self.trunk_depth = trunk_depth
#         self.global_feat = global_feat
        
#         trunk_dim = dim_in // 2 if model_type == "vista" else dim_in          # Vista: 1536 -> 768


#         self.patch_pool = nn.Identity() if global_feat else PatchPool(trunk_dim)

#         # Vista: frame + global tokens are concatenated (2 * embed_dim); project them to embed_dim so the
#         # trunk has the same size as the other models' camera heads. Everything after input_proj uses trunk_dim.
#         self.input_proj = nn.Linear(dim_in, trunk_dim) if model_type == "vista" else nn.Identity()

#         # Build the trunk using a sequence of transformer blocks.
#         self.trunk = nn.Sequential(
#             *[
#                 Block(dim=trunk_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
#                 for _ in range(trunk_depth)
#             ]
#         )

#         # Normalizations for camera token (before the projection) and trunk output.
#         self.token_norm = nn.LayerNorm(trunk_dim)
#         self.trunk_norm = nn.LayerNorm(trunk_dim)

#         # Learnable empty camera pose token.
#         self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
#         self.embed_pose = nn.Linear(self.target_dim, trunk_dim)

#         # Module for producing modulation parameters: shift, scale, and a gate.
#         self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(trunk_dim, 3 * trunk_dim, bias=True))

#         # Adaptive layer normalization without affine parameters.
#         self.adaln_norm = nn.LayerNorm(trunk_dim, elementwise_affine=False, eps=1e-6)
#         self.pose_branch = Mlp(in_features=trunk_dim, hidden_features=trunk_dim // 2, out_features=self.target_dim, drop=0)

#     def forward(self, aggregated_tokens_list: list, patch_start_idx: int = 0, num_iterations: int = 4) -> list:
#         """
#         Forward pass to predict camera parameters.

#         Args:
#             aggregated_tokens_list (list): List of token tensors from the network;
#                 the last tensor is used for prediction.
#             patch_start_idx (int): index of the first patch token (used when global_feat=False).
#             num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

#         Returns:
#             list: A list of predicted camera encodings (post-activation) from each iteration.
#         """
#         # Use tokens from the last block for camera prediction.
#         tokens = aggregated_tokens_list[-1]                                          # (B, S, N, dim_in)

#         # Extract the camera tokens (special tokens never enter the projection in the patch path).
#         if self.global_feat:
#             pose_tokens = self.input_proj(tokens[:, :, 0])                           # (B, S, dim) 
#         else:
#             patch_tokens = self.input_proj(tokens[:, :, patch_start_idx:, :])        # Vista: 1536 -> 768; others: identity
#             pose_tokens = self.patch_pool(patch_tokens)                              # (B, S, dim)

#         pose_tokens = self.token_norm(pose_tokens)                                   # norm AFTER pooling (as before)

#         pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
#         return pred_pose_enc_list



#     def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
#         """
#         Iteratively refine camera pose predictions.

#         Args:
#             pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
#             num_iterations (int): Number of refinement iterations.

#         Returns:
#             list: List of activated camera encodings from each iteration.
#         """
#         B, S, C = pose_tokens.shape
#         pred_pose_enc = None
#         pred_pose_enc_list = []

#         for _ in range(num_iterations):
#             # Use a learned empty pose for the first iteration.
#             if pred_pose_enc is None:
#                 module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
#             else:
#                 # Detach the previous prediction to avoid backprop through time.
#                 pred_pose_enc = pred_pose_enc.detach()
#                 module_input = self.embed_pose(pred_pose_enc)

#             # Generate modulation parameters and split them into shift, scale, and gate components.
#             shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

#             # Adaptive layer normalization and modulation.
#             pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
#             pose_tokens_modulated = pose_tokens_modulated + pose_tokens

#             pose_tokens_modulated = self.trunk(pose_tokens_modulated)
#             # Compute the delta update for the pose encoding.
#             pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

#             if pred_pose_enc is None:
#                 pred_pose_enc = pred_pose_enc_delta
#             else:
#                 pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

#             # Apply final activation functions for translation, quaternion, and field-of-view.
#             activated_pose = activate_pose(
#                 pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
#             )
#             pred_pose_enc_list.append(activated_pose)

#         return pred_pose_enc_list


# def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
#     """
#     Modulate the input tensor using scaling and shifting parameters.
#     """
#     # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
#     return x * (1 + scale) + shift

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import Block


class ResMLPBlock(nn.Module):
    """
    Residual block of three per-token linear layers (the 1x1-conv residual block of marepo,
    https://github.com/nianticlabs/marepo, written with nn.Linear).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head_skip = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
        self.res_conv1 = nn.Linear(in_channels, out_channels)
        self.res_conv2 = nn.Linear(out_channels, out_channels)
        self.res_conv3 = nn.Linear(out_channels, out_channels)

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        return self.head_skip(res) + x


class PatchPool(nn.Module):
    """
    marepo-style pooling of each view's patch tokens into one token:
    residual per-token MLP blocks -> global average over the patch grid -> MLP.
    (B, S, P, C) -> (B, S, C)
    """

    def __init__(self, dim: int, num_res_blocks: int = 2):
        super().__init__()
        self.res_blocks = nn.Sequential(*[ResMLPBlock(dim, dim) for _ in range(num_res_blocks)])
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.res_blocks(tokens)               # (B, S, P, C), per patch
        x = x.mean(dim=2)                         # average over the h x w grid (= AdaptiveAvgPool2d(1))
        return self.more_mlps(x)


class CameraHead(nn.Module):
    """
    Single-shot absolute-pose head, following Poincar3's CameraHead: per-frame pose tokens are mixed across
    frames once by `self.trunk`, then a 2-layer MLP regresses the 9D pose encoding (absT_quaR_FoV) directly --
    no iterative delta refinement.

    Per-frame pose tokens (kept from the previous head):
        global_feat=True : the special (prefix) tokens of each frame, tokens[:, :, :patch_start_idx]
                           (camera / cls + register tokens, as Poincar3); the pose is read from token 0.
                           With patch_start_idx == 0 only token 0 is used.
        global_feat=False: the patch tokens of each frame pooled into one token by PatchPool (marepo).
    Vista: frame + global tokens are concatenated (2 * embed_dim) and projected to embed_dim by `input_proj`, so the
    trunk has the same size as the other models' camera heads.

    `forward` returns a one-element list, so CameraLoss's per-stage weighting (a no-op over one stage) is unchanged.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: float = 1e-5,
        global_feat: bool = True,
        model_type: Optional[str] = None,
        **kwargs,                                  # old args (trans_act, quat_act, fl_act, ...) are accepted, unused
    ):
        super().__init__()
        if pose_encoding_type != "absT_quaR_FoV":
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.target_dim = 9
        self.global_feat = global_feat

        trunk_dim = dim_in // 2 if model_type == "vista" else dim_in          # Vista: 1536 -> 768
        self.input_proj = nn.Linear(dim_in, trunk_dim) if model_type == "vista" else nn.Identity()
        self.patch_pool = nn.Identity() if global_feat else PatchPool(trunk_dim)

        self.token_norm = nn.LayerNorm(trunk_dim, eps=1e-5)
        # head-local transformer blocks: attention over all pose tokens of all frames of a sample
        self.trunk = nn.ModuleList(
            [Block(dim=trunk_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
             for _ in range(trunk_depth)]
        )
        self.trunk_norm = nn.LayerNorm(trunk_dim, eps=1e-5)
        self.camera_branch = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2, bias=True),
            nn.GELU(),
            nn.Linear(trunk_dim // 2, self.target_dim, bias=True),
        )

    def forward(self, aggregated_tokens_list: list, patch_start_idx: int = 0, num_iterations: int = 1) -> list:
        """
        Args:
            aggregated_tokens_list: token tensors from the backbone; the last one, (B, S, N, dim_in), is used.
            patch_start_idx: index of the first patch token (prefix tokens come before it).
            num_iterations: ignored (single-shot); kept so existing callers keep working.
        Returns:
            [pose_enc]: one-element list, pose_enc (B, S, 9) activated (translation, quaternion, FoV).
        """
        tokens = aggregated_tokens_list[-1]                                       # (B, S, N, dim_in)
        if tokens.dtype != torch.float32:
            tokens = tokens.float()
        B, S, N, _ = tokens.shape
        if patch_start_idx > N:
            raise ValueError(f"patch_start_idx ({patch_start_idx}) exceeds token length ({N})")

        if self.global_feat:
            n_pose = max(patch_start_idx, 1)                                     # prefix tokens (>= token 0)
            pose_tokens = self.input_proj(tokens[:, :, :n_pose])                 # (B, S, R, dim)
        else:
            patch_tokens = self.input_proj(tokens[:, :, patch_start_idx:])       # Vista: 1536 -> 768
            pose_tokens = self.patch_pool(patch_tokens)[:, :, None]              # (B, S, 1, dim)
            n_pose = 1

        pose_tokens = self.token_norm(pose_tokens)
        x = pose_tokens.reshape(B, S * n_pose, -1)                               # all frames of a sample together
        for block in self.trunk:
            x = block(x)
        x = x.reshape(B, S, n_pose, -1)[:, :, 0]                                 # pose read from token 0 per frame

        pose_enc = _apply_camera_activation(self.camera_branch(self.trunk_norm(x)))
        return [pose_enc]


def _apply_camera_activation(raw_camera: torch.Tensor) -> torch.Tensor:
    """Poincar3: linear translation and quaternion, FoV = ReLU + 0.01 (strictly positive)."""
    translation = raw_camera[..., :3]
    quaternion = raw_camera[..., 3:7]
    fov = F.relu(raw_camera[..., 7:]) + 0.01
    return torch.cat([translation, quaternion, fov], dim=-1)