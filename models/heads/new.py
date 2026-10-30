import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import Mlp
from models.layers import Block
from .head_act import activate_pose


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
        return self.more_mlps(x)                  # (B, S, C)


class CameraHead(nn.Module):
    """
    Predicts camera parameters per view with iterative refinement (VGGT-style).

    Input features (last entry of the token list, or a tensor):
        global_feat=True  -> (B, S, C): one global token per view (e.g. the cls token).
                             A (B, S, P, C) tensor is also accepted; its token 0 is used.
        global_feat=False -> (B, S, P, C): patch tokens only (cls / register tokens removed).
                             If they are still present, pass patch_start_idx to drop them.
                             The P patch tokens of each view are pooled into one token
                             marepo-style: residual MLP blocks, average over the patches, MLP.
    Either way the trunk sees one token per view, (B, S, C), and attends across the S views,
    so poses are predicted jointly.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        global_feat: bool = True,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # field-of-view activation: keeps FoV positive
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.global_feat = global_feat

        # Patch tokens -> one token per view (only needed without a global token).
        self.patch_pool = None if global_feat else PatchPool(dim_in)

        # Trunk: transformer blocks over the S per-view tokens.
        self.trunk = nn.Sequential(
            *[Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
              for _ in range(trunk_depth)]
        )

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty pose for the first refinement iteration.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Adaptive LayerNorm modulation (shift, scale, gate) from the current pose estimate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def get_pose_tokens(self, tokens: torch.Tensor, patch_start_idx: int = None) -> torch.Tensor:
        """Feature tensor -> one token per view, (B, S, C)."""
        if self.global_feat:
            if tokens.ndim == 4:                                   # (B, S, P, C): use the first (cls) token
                tokens = tokens[:, :, 0]
            assert tokens.ndim == 3, f"global_feat=True expects (B, S, C), got {tuple(tokens.shape)}"
            return tokens

        assert tokens.ndim == 4, f"global_feat=False expects (B, S, P, C), got {tuple(tokens.shape)}"
        if patch_start_idx:                                        # drop cls / register tokens if present
            tokens = tokens[:, :, patch_start_idx:]
        return self.patch_pool(tokens)

    def forward(self, aggregated_tokens_list, patch_start_idx: int = None, num_iterations: int = 4) -> list:
        """
        aggregated_tokens_list: list of feature tensors (the last one is used) or a single tensor,
                                shaped as described in the class docstring.
        patch_start_idx:        only for global_feat=False with cls / register tokens still present.
        Returns a list of activated pose encodings (B, S, 9), one per refinement iteration.
        """
        tokens = aggregated_tokens_list[-1] if isinstance(aggregated_tokens_list, (list, tuple)) \
            else aggregated_tokens_list

        pose_tokens = self.get_pose_tokens(tokens, patch_start_idx)    # (B, S, C)
        pose_tokens = self.token_norm(pose_tokens)
        return self.trunk_fn(pose_tokens, num_iterations)

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """Iterative refinement on per-view tokens (B, S, C)."""
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            if pred_pose_enc is None:                              # first iteration: learned empty pose
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:                                                  # no backprop through earlier iterations
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)

            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            pred_pose_enc = pred_pose_enc_delta if pred_pose_enc is None else pred_pose_enc + pred_pose_enc_delta

            pred_pose_enc_list.append(
                activate_pose(pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act)
            )

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale-and-shift modulation (from DiT)."""
    return x * (1 + scale) + shift