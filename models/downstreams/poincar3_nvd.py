"""
Novel-view depth (NVD) probe on a FROZEN Poincar3 backbone, LVSM-decoder style.
Identical probe to vista_nvd.py; only the backbone and encode_sources differ:
    encoder    load_backbone("poincar3", run_path) (run_path=None -> released checkpoint), frozen, bf16
    sources    resized 224 -> input_size (256) for Poincar3, as in Poincar3_LDI; the source tokens come out on
               Poincar3's own patch grid (e.g. 16 x 16), which the cross-attention handles as is (sources only
               give keys / values; their 2D positions use that grid)
    taps       block outputs [4, 11, 17, 23] (same taps as Poincar3_LDI)
    prefix     the camera + register tokens (tokens[:, :, :patch_token_start]) of every source view are kept and
               added to the scene representation: same per-layer projection as the patch tokens, a learned
               embedding per prefix slot (instead of the 2D patch position), and the view embedding
    target     unchanged: 224 x 224 rays, 14 x 14 queries, DPT head and loss at the GT resolution

    source images (S ≥ 1) ─► resize 256 ─► frozen Poincar3 ─► 4 taps concat ─► Linear ─► source tokens (+ 2D pos, + view emb)
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

from models.poincar3.baselines import load_backbone
from models.heads.dpt_ndv import DPTHead
from models.layers import Block, CrossAttentionBlock      # adjust the import if the blocks live elsewhere


def even_layers(depth: int, n: int = 4):
    return [int(round(i * (depth - 1) / (n - 1))) for i in range(n)]


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
class Poincar3NVD(nn.Module):
    """Frozen Poincar3 backbone + the same novel-view depth probe as VistaNVD."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        run_path: Optional[str] = None,                 # None: released Poincar3 checkpoint (auto-download)
        input_size: int = 256,                          # Poincar3's resolution
        enc_layers: Sequence[int] = (4, 11, 17, 23),   # encoder layers fed to the probe (4; Poincar3 block-output taps, as Poincar3_LDI)
        probe_dim: int = 768,
        probe_pattern: str = "SC",   # S: self-attn over target tokens, C: cross-attn target -> sources,
                                       # J: joint self-attn over [target ; source] (LVSM). "JJJJ" = LVSM decoder
        probe_heads: int = 12,
        dpt_features: int = 128,
        dpt_out_channels: Sequence[int] = (96, 192, 384, 768),   # standard DPT widths for 768-d tokens
        predict_conf: bool = False,     # False: DPT head outputs depth only, depth_conf = None
    ):
        super().__init__()
        self.img_size = img_size                                         # target / GT resolution (224)
        self.patch_size = patch_size                                     # replaced by Poincar3's in load_pretrained
        self.enc_layers = list(enc_layers)
        assert len(self.enc_layers) == 4, f"the DPT head needs exactly 4 layers, got {self.enc_layers}"
        self.input_size = input_size
        self.probe_dim = probe_dim
        self.probe_pattern = probe_pattern
        self.probe_heads = probe_heads
        self.dpt_features = dpt_features
        self.dpt_out_channels = list(dpt_out_channels)
        self.predict_conf = predict_conf

        self.encoder = None                                              # named `encoder` for freeze_encoder
        self.run_path = None
        self.load_pretrained(run_path)                                   # always build: None = released checkpoint

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def load_pretrained(self, run_path=None):
        """Load the frozen Poincar3 and build the probe on its taps. Must run before the optimizer and DDP are
        created (the probe's parameters are made here). Idempotent: a second call with None or the same
        run_path is a no-op."""
        if self.encoder is not None:
            if run_path not in (None, self.run_path):
                raise ValueError(f"Poincar3NVD already loaded from {self.run_path or 'released checkpoint'}, "
                                 f"not {run_path}")
            return self

        encoder, step = load_backbone("poincar3", run_path)
        self.encoder = encoder.eval().requires_grad_(False)
        self.run_path = run_path

        # Read the tap layout off one dummy forward: number of block outputs, token width, prefix length.
        dev = next(self.encoder.parameters()).device
        taps = self._taps(torch.zeros(1, 1, 3, self.input_size, self.input_size, device=dev))
        num_blocks = len(taps) // 2                                      # block_outputs hold (frame, global) pairs
        enc_token_dim = taps[0].shape[-1]
        patch_start = self.encoder.patch_token_start
        self.n_prefix = patch_start                                      # camera + register tokens per view
        self.patch_size = self.encoder.patch_size                        # Poincar3's patch size (rays, DPT, grids)
        self.grid = self.img_size // self.patch_size                     # target query grid (224 / 16 = 14)
        self.src_grid = self.input_size // self.patch_size               # source grid (256 / 16 = 16)
        source = run_path if run_path is not None else "released checkpoint"
        print(f"Poincar3NVD: {source} (step {step}), {num_blocks} blocks ({len(taps)} block outputs), "
              f"dim {enc_token_dim}, patch {self.patch_size}, prefix {patch_start}, "
              f"source grid {self.src_grid}, taps {self.enc_layers}, input {self.input_size} x {self.input_size}")

        self._build_probe(enc_token_dim, dev)
        return self

    def _build_probe(self, enc_token_dim, dev):
        """The probe: identical for every backbone; only enc_token_dim and the source grid differ."""
        probe_dim, p = self.probe_dim, self.patch_size

        # source tokens: frozen features -> probe_dim, plus where the patch is and which view it is from
        self.src_proj = nn.ModuleList([nn.Linear(enc_token_dim, probe_dim) for _ in self.enc_layers])  # one per layer
        self.register_buffer("pos_2d", sincos_2d(self.src_grid, probe_dim)[None, None],
                             persistent=False)                         # [1, 1, P_src, d] (Poincar3 grid)
        self.prefix_pos = nn.Parameter(torch.zeros(1, 1, self.n_prefix, probe_dim))  # one per camera / register slot
        self.view_ref = nn.Parameter(torch.zeros(1, 1, 1, probe_dim))   # source 0 (frame of the target pose)
        self.view_other = nn.Parameter(torch.zeros(1, 1, 1, probe_dim))  # every other source

        # target tokens: LVSM pose tokenizer (patchify the Plücker rays, linear, no bias)
        self.target_pose_tokenizer = nn.Sequential(
            Rearrange("n c (hh ph) (ww pw) -> n (hh ww) (ph pw c)", ph=p, pw=p),
            nn.Linear(6 * p * p, probe_dim, bias=False),
        )

        # decoder layers (no LayerScale, no RoPE: tokens carry rays / 2D positions)
        assert set(self.probe_pattern) <= {"S", "C", "J"} and set(self.probe_pattern) & {"C", "J"}, self.probe_pattern
        self.input_norm = nn.LayerNorm(probe_dim, bias=False)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim=probe_dim, num_heads=self.probe_heads, rope=None, init_values=None) if k == "C"
            else Block(dim=probe_dim, num_heads=self.probe_heads, rope=None, init_values=None)       # S and J
            for k in self.probe_pattern
        ])

        # DPT head on the 4 per-layer target feature maps
        self.dpt_head = DPTHead(
            dim_in=probe_dim,
            output_dim=2 if self.predict_conf else 1,                    # depth (+ confidence)
            patch_size=p,
            activation="exp",
            conf_activation="expp1",
            features=self.dpt_features,
            out_channels=self.dpt_out_channels,
            intermediate_layer_idx=None,          # outs = the 4 per-layer target feature maps, used as given
            return_conf=self.predict_conf,
        )

        for prm in (self.prefix_pos, self.view_ref, self.view_other):
            nn.init.trunc_normal_(prm, std=0.02)
        self.to(dev)

    # ------------------------------------------------------------------
    # Poincar3 helpers (as in Poincar3_LDI)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _taps(self, images):
        """images (B, S, 3, h, w) in [0, 1] -> list of [B, S, prefix + P, D] block outputs.
        Poincar3 normalizes with ImageNet mean/std internally, so images stay in [0, 1]."""
        B, S, _, H, W = images.shape
        num_patches = (H // self.encoder.patch_size) * (W // self.encoder.patch_size)
        # all-False head_mask: the prototype head runs on zero patches; only block outputs are needed
        head_mask = torch.zeros(B, S, num_patches, dtype=torch.bool, device=images.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            *_, block_outputs = self.encoder(images, mask=None, head_mask=head_mask, capture_block_outputs=True)
        return [t.reshape(B, S, -1, t.shape[-1]).float() for t in block_outputs]

    @staticmethod
    def _resize(x, size):
        """(B, S, C, h, w) -> (B, S, C, *size), bilinear."""
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        B, S = x.shape[:2]
        y = F.interpolate(x.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False)
        return y.reshape(B, S, *y.shape[1:])

    # keep the frozen encoder in eval mode while the probe trains
    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    # -------------------------------------------------------------------------
    # The only backbone-specific part: source images -> [B, S, P, C_enc_total]
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode_sources(self, src_images):
        """[B, S, 3, 224, 224] (resized to 256) -> per selected layer:
            patch tokens                [L, B, S, P_src, D]  on Poincar3's grid
            camera + register tokens    [L, B, S, R, D]      tokens[:, :, :patch_token_start]"""
        images = self._resize(src_images, (self.input_size, self.input_size)).clamp(0, 1)
        taps = self._taps(images)
        psi = self.encoder.patch_token_start
        patches = torch.stack([taps[i][:, :, psi:] for i in self.enc_layers], dim=0)
        prefix = torch.stack([taps[i][:, :, :psi] for i in self.enc_layers], dim=0)
        return patches, prefix

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

        # source tokens (camera + register + patch), one set per selected encoder layer  [L*B*T, S*(R+P), d]
        view = torch.cat([self.view_ref, self.view_other.expand(-1, S - 1, -1, -1)], dim=1)   # [1, S, 1, d]
        patches, prefix = self.encode_sources(src_images)                  # [L, B, S, P, D], [L, B, S, R, D]
        src_p = torch.stack([proj(f) for proj, f in zip(self.src_proj, patches)])      # [L, B, S, P, d]
        src_c = torch.stack([proj(f) for proj, f in zip(self.src_proj, prefix)])       # [L, B, S, R, d]
        src_p = src_p + self.pos_2d + view                                             # patch position + view
        src_c = src_c + self.prefix_pos + view                                         # prefix slot + view
        src = torch.cat([src_c, src_p], dim=3)                                         # [L, B, S, R + P, d]
        src = src.flatten(2, 3).repeat_interleave(T, dim=1).flatten(0, 1)              # [L*B*T, S*(R+P), d]

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
# Configs (same probe defaults as vista_nvd_base)
# =============================================================================
def poincar3_nvd(**kwargs) -> Poincar3NVD:
    """run_path=None: released checkpoint; run_path=dir: a Poincar3 pretraining run."""
    return Poincar3NVD(**kwargs)