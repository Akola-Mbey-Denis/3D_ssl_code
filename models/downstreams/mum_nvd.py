"""
Novel-view depth (NVD) probe on a FROZEN MuM encoder, LVSM-decoder style.
Identical probe to vista_nvd.py; only the encoder (MuMAutoEncoder) and encode_sources differ.

    source images (S ≥ 1) ─► frozen MuM encoder ─► 4 layers concat ─► Linear ─► source tokens (+ 2D pos, + view emb)
    target K + pose rel. to source 0 ─► Plücker rays ─► patchify + Linear ─► target tokens
                                                                                   │
    target tokens, source tokens ─► LayerNorm ─► layers by `probe_pattern` (default S, C, S, C)
                                                                                   │
                  per selected encoder layer: same decoder ─► the 4 target feature maps ─► DPT head ─► depth (exp), conf (1 + exp)

LVSM decoder recipe (target ray tokens concatenated with the scene tokens, plain transformer); the
"scene tokens" are the frozen backbone's source tokens. LVSM's linear patch decoder gave 16x16 patch
seams in depth, so the output is a DPT head on the target tokens of every block.
Each target is decoded on its own (repeated source tokens), so the prediction does not depend on T.
The target image is never given to the model. Only `encode_sources` changes between backbones.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange

from models.mum.model import MuMAutoEncoder
from models.heads.dpt_ndv import DPTHead
from models.layers import Block, CrossAttentionBlock      # adjust the import if the blocks live elsewhere


def even_layers(depth: int, n: int = 4):
    return [int(round(i * (depth - 1) / (n - 1))) for i in range(n)]


# DINOv2 / Depth Anything DPT layers per encoder depth (same blocks as the MuM depth/camera probes)
DINO_DPT_LAYERS = {
    12: [2, 5, 8, 11],      # ViT-B
    24: [4, 11, 17, 23],    # ViT-L
}


def dpt_layers(depth: int) -> list:
    return DINO_DPT_LAYERS.get(depth, [depth * (q + 1) // 4 - 1 for q in range(4)])


def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding [grid*grid, dim] (MAE style), values in [-1, 1]."""
    assert dim % 4 == 0
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing="ij")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    out = []
    for c in (ys.flatten(), xs.flatten()):
        a = c[:, None] * omega[None]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, dim=1)


# =============================================================================
# Camera conditioning
# =============================================================================
def relative_c2w(src_w2c, w2c):
    """[B, S, 4, 4] source w2c, [B, N, 4, 4] view w2c -> [B, N, 4, 4] view camera -> source-0 camera (metric)."""
    return src_w2c[:, 0, None] @ torch.linalg.inv(w2c)


def plucker_rays(K, c2w, H, W):
    """K [N, 3, 3], c2w [N, 4, 4] -> Plücker rays [N, 6, H, W] = (o x d, d), as in LVSM's default_plucker."""
    N = K.shape[0]
    ys, xs = torch.meshgrid(torch.arange(H, device=K.device, dtype=K.dtype) + 0.5,
                            torch.arange(W, device=K.device, dtype=K.dtype) + 0.5, indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).T          # [3, HW]
    d = F.normalize(c2w[:, :3, :3] @ (torch.linalg.inv(K) @ pix), dim=1)          # [N, 3, HW]
    o = c2w[:, :3, 3:4].expand_as(d)
    return torch.cat([torch.cross(o, d, dim=1), d], 1).reshape(N, 6, H, W)


# =============================================================================
# Model
# =============================================================================
class MumNVD(nn.Module):
    """Frozen MuM encoder + the same novel-view depth probe as VistaNVD."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        n_storage_tokens: int = 0,
        n_ref_views: int = 0,                           # as in the MuM depth/camera probe
        patch_start_idx: int = 1,                       # MuM tokens: [cls, patches]
        enc_layers: Sequence[int] = (2, 5, 8, 11),   # encoder layers fed to the probe (4; MuM ViT-B: 12 blocks)
        probe_dim: int = 768,
        probe_pattern: str = "SC",   # S: self-attn over target tokens, C: cross-attn target -> sources,
                                       # J: joint self-attn over [target ; source] (LVSM). "JJJJ" = LVSM decoder
        probe_heads: int = 12,
        dpt_features: int = 128,
        dpt_out_channels: Sequence[int] = (96, 192, 384, 768),   # standard DPT widths for 768-d tokens
        predict_conf: bool = False,     # False: DPT head outputs depth only, depth_conf = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.enc_layers = list(enc_layers)
        assert len(self.enc_layers) == 4, f"the DPT head needs exactly 4 layers, got {self.enc_layers}"
        self.patch_start_idx = patch_start_idx

        # ---- frozen backbone ----
        self.encoder = MuMAutoEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            n_ref_views=n_ref_views,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            n_storage_tokens=n_storage_tokens,
        )
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        enc_token_dim = embed_dim                                        # MuM tokens are embed_dim wide

        # ---- probe (identical for every backbone; only enc_token_dim differs) ----
        p = patch_size
        # source tokens: frozen features -> probe_dim, plus where the patch is and which view it is from
        self.src_proj = nn.ModuleList([nn.Linear(enc_token_dim, probe_dim) for _ in self.enc_layers])  # one per layer
        self.register_buffer("pos_2d", sincos_2d(self.grid, probe_dim)[None, None], persistent=False)  # [1,1,P,d]
        self.view_ref = nn.Parameter(torch.zeros(1, 1, 1, probe_dim))   # source 0 (frame of the target pose)
        self.view_other = nn.Parameter(torch.zeros(1, 1, 1, probe_dim))  # every other source

        # target tokens: LVSM pose tokenizer (patchify the Plücker rays, linear, no bias)
        self.target_pose_tokenizer = nn.Sequential(
            Rearrange("n c (hh ph) (ww pw) -> n (hh ww) (ph pw c)", ph=p, pw=p),
            nn.Linear(6 * p * p, probe_dim, bias=False),
        )

        # decoder layers (no LayerScale, no RoPE: tokens carry rays / 2D positions)
        assert set(probe_pattern) <= {"S", "C", "J"} and set(probe_pattern) & {"C", "J"}, probe_pattern
        self.probe_pattern = probe_pattern
        self.input_norm = nn.LayerNorm(probe_dim, bias=False)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim=probe_dim, num_heads=probe_heads, rope=None, init_values=None) if k == "C"
            else Block(dim=probe_dim, num_heads=probe_heads, rope=None, init_values=None)       # S and J
            for k in probe_pattern
        ])

        # DPT head on the target tokens of every block (multi-scale fusion -> no patch seams)
        self.dpt_head = DPTHead(
            dim_in=probe_dim,
            output_dim=2 if predict_conf else 1,                         # depth (+ confidence)
            patch_size=patch_size,
            activation="exp",
            conf_activation="expp1",
            features=dpt_features,
            out_channels=list(dpt_out_channels),
            intermediate_layer_idx=None,          # outs = the 4 per-layer target feature maps, used as given
            return_conf=predict_conf,
        )
        self.predict_conf = predict_conf

        for prm in (self.view_ref, self.view_other):
            nn.init.trunc_normal_(prm, std=0.02)

    # keep the frozen encoder in eval mode while the probe trains
    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    # -------------------------------------------------------------------------
    # The only backbone-specific part: source images -> [B, S, P, C_enc_total]
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode_sources(self, src_images):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            tokens_list = self.encoder.forward_encoder(x=src_images, return_all_blocks=True)   # depth x (B, S, 1 + P, D)
        feats = [tokens_list[i][:, :, self.patch_start_idx:].float() for i in self.enc_layers]   # each [B, S, P, D]
        return torch.stack(feats, dim=0)                               # [L, B, S, P, D]

    def forward(self, src_images, src_w2c, tgt_K, tgt_w2c):
        """
        src_images: [B, S, 3, H, W] in [0, 1], any S >= 1
        src_w2c:    [B, S, 4, 4]   world-to-camera
        tgt_K:      [B, T, 3, 3]   target intrinsics at (H, W) ([B, 3, 3] also accepted: T = 1)
        tgt_w2c:    [B, T, 4, 4]   target world-to-camera      ([B, 4, 4] also accepted)
        Returns dict: depth [B, T, H, W] (metric), depth_conf [B, T, H, W].
        """
        if tgt_w2c.dim() == 3:
            tgt_K, tgt_w2c = tgt_K[:, None], tgt_w2c[:, None]
        B, S, _, H, W = src_images.shape
        T = tgt_w2c.shape[1]
        p = self.patch_size

        L = len(self.enc_layers)

        # source tokens, one set per selected encoder layer  [L*B*T, S*P, d]  (each target gets its own copy)
        view = torch.cat([self.view_ref, self.view_other.expand(-1, S - 1, -1, -1)], dim=1)   # [1, S, 1, d]
        feats = self.encode_sources(src_images)                                         # [L, B, S, P, D]
        src = torch.stack([proj(f) for proj, f in zip(self.src_proj, feats)])           # [L, B, S, P, d]
        src = src + self.pos_2d + view
        src = src.flatten(2, 3).repeat_interleave(T, dim=1).flatten(0, 1)               # [L*B*T, S*P, d]

        # target tokens  [B*T, P, d]  from Plücker rays in the source-0 frame, the same for every layer
        with torch.autocast(device_type=src_images.device.type, enabled=False):
            # c2w = relative_c2w(src_w2c.float(), tgt_w2c.float()).reshape(B * T, 4, 4)
            c2w = torch.linalg.inv(tgt_w2c.float()).reshape(B * T, 4, 4)
            rays = plucker_rays(tgt_K.float().reshape(B * T, 3, 3), c2w, H, W)          # [B*T, 6, H, W]
        tgt = self.target_pose_tokenizer(rays)
        n_tgt = tgt.shape[1]
        tgt = tgt.repeat(L, 1, 1)                                                       # [L*B*T, P, d]

        # decoder (shared weights), run once per selected layer as a batch: the target-pose tokens read
        # that layer's source tokens
        tgt, src = self.input_norm(tgt), self.input_norm(src)
        for k, blk in zip(self.probe_pattern, self.blocks):
            if k == "S":                                   # target patches attend to each other
                tgt = blk(tgt, pos=None)
            elif k == "C":                                 # target patches read the (fixed) source tokens
                tgt = blk(query_tokens=tgt, context_tokens=src, query_pos=None, context_pos=None)
            else:                                          # "J": joint, sources are updated too (LVSM)
                x = blk(torch.cat([tgt, src], dim=1), pos=None)
                tgt, src = x[:, :n_tgt], x[:, n_tgt:]

        # one target feature map per selected layer (low -> high level), passed to DPT as is
        tgt = tgt.reshape(L, B, T, n_tgt, -1)
        outs = list(tgt.unbind(0))                                                      # L x [B, T, P, d]

        # the DPT head treats the T targets as frames
        head_out = self.dpt_head(
            images=torch.zeros(B, T, 3, H, W, device=src_images.device),               # only shape is used
            aggregated_tokens_list=outs,                                               # used as they are
        )
        if self.predict_conf:
            depth, conf = head_out
            depth, conf = depth.reshape(B, T, H, W), conf.reshape(B, T, H, W)
        else:
            depth = head_out.reshape(B, T, H, W)
            conf = None                        # no confidence: DepthLoss / SILogLoss skip the conf term
        return {"depth": depth, "depth_conf": conf}


# =============================================================================
# Configs (same probe defaults as vista_nvd_base / vista_nvd_large)
# =============================================================================
def mum_nvd_base(**kwargs) -> MumNVD:
    """ViT-B/16 MuM: 768-d, 12 blocks; probe reads blocks 2, 5, 8, 11."""
    cfg = dict(embed_dim=768, encoder_depth=12, encoder_num_heads=12, enc_layers=dpt_layers(12))
    return MumNVD(**{**cfg, **kwargs})


def mum_nvd_large(**kwargs) -> MumNVD:
    """ViT-L/16 MuM: 1024-d, 24 blocks; probe reads blocks 4, 11, 17, 23."""
    cfg = dict(embed_dim=1024, encoder_depth=24, encoder_num_heads=16, enc_layers=dpt_layers(24))
    return MumNVD(**{**cfg, **kwargs})