"""
Poincar3_LDI: frozen Poincar3 backbone + LayeredDPTHead, for LDI probing with finetune_ldi.py (same outputs as
Vista_LDI, so LayeredDepthLoss / the logging / the visualisation work unchanged).

Loaded in finetune_ldi.py like the other probes:
    --model "poincar3_ldi(num_layers=4)"                 # released checkpoint (auto-download)
    --model "poincar3_ldi(num_layers=4, run_path=...)"   # a Poincar3 pretraining run directory
    model = eval(args.model)                 # backbone + head are built here
    load_encoder(model, args.ckpt_path)      # no-op when ckpt_path is None or matches the loaded run
    freeze_encoder(model)                    # model.encoder = the Poincar3 backbone (already frozen)

    encoder    load_backbone("poincar3", run_path): the pretrained Poincar3, frozen (eval, no grad, bf16).
               run_path=None -> released checkpoint, downloaded by Poincar3()
    taps       Poincar3's block outputs (capture_block_outputs=True). They come as a (frame, inter-frame) pair per
               block; the head reads the inter-frame output of the blocks at 1/4, 1/2, 3/4 and the end of the
               network (dense_head_hook_indices_poincar3, as the released ffrecon depth head)
    head       LayeredDPTHead(dim_in=embed_dim, patch_size=Poincar3's patch size): N depth layers + N + 1 stopping
               logits (LaRI); the only trainable part
    resolution Poincar3 runs at 256 x 256: the 224 x 224 images are resized (bilinear) to input_size, backbone
               and head run there, and the predictions are resized back (bilinear) to the input resolution, so
               the loss, the GT and the other backbones stay at 224 x 224

Predictions (B, S views):
    layered_depth       (B, S, N, H, W)
    layered_depth_stop  (B, S, N + 1, H, W)
    depth               (B, S, H, W)       = layer 0 (visible surface)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.poincar3.baselines import load_backbone

from models.heads.dpt_head import LayeredDPTHead


def dense_head_hook_indices_poincar3(depth: int) -> list[int]:
    """Four evenly spaced taps; 2 * i + 1 = the inter-frame output of block i (block_outputs holds pairs)."""
    return [2 * (round(depth * f) - 1) + 1 for f in (0.25, 0.5, 0.75, 1.0)]


class Poincar3_LDI(nn.Module):
    def __init__(self, num_layers=4, features=256, input_size=256, run_path=None, **head_kwargs):
        """
        run_path=None -> released Poincar3 checkpoint (auto-downloaded by Poincar3()).
        run_path=dir  -> a Poincar3 pretraining run directory (cfg.pkl + weights.pth).
        """
        super().__init__()
        self.num_layers = num_layers
        self.features = features
        self.input_size = input_size                    # Poincar3's resolution (256 x 256)
        self.head_kwargs = head_kwargs
        self.encoder = None                             # named `encoder` for freeze_encoder
        self.head = None
        self.run_path = None
        self.load_pretrained(run_path)                  # always build: None = released checkpoint

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def load_pretrained(self, run_path=None):
        """Load the frozen Poincar3 and build the head on its taps. Must run before the optimizer and DDP are
        created (the head's parameters are made here). Idempotent: a second call with None or the same
        run_path is a no-op."""
        if self.encoder is not None:
            if run_path not in (None, self.run_path):
                raise ValueError(f"Poincar3_LDI already loaded from {self.run_path or 'released checkpoint'}, "
                                 f"not {run_path}")
            return self

        encoder, step = load_backbone("poincar3", run_path)
        self.encoder = encoder.eval().requires_grad_(False)
        self.run_path = run_path

        # Read the tap layout off one dummy forward: number of blocks, token width, prefix length.
        dev = next(self.encoder.parameters()).device
        taps, patch_start = self._taps(torch.zeros(1, 1, 3, self.input_size, self.input_size, device=dev))
        num_blocks = len(taps) // 2
        embed_dim = taps[0].shape[-1]
        self.patch_size = self.encoder.patch_size
        self.hook_indices = [4, 11, 17, 23]
        source = run_path if run_path is not None else "released checkpoint"
        print(f"Poincar3_LDI: {source} (step {step}), {num_blocks} blocks, dim {embed_dim}, "
              f"patch {self.patch_size}, prefix {patch_start}, head taps {self.hook_indices}, "
              f"input {self.input_size} x {self.input_size}")

        self.head = LayeredDPTHead(dim_in=embed_dim, patch_size=self.patch_size,
                                   intermediate_layer_idx=self.hook_indices, num_layers=self.num_layers,
                                   features=self.features, **self.head_kwargs).to(dev)
        return self

    def train(self, mode=True):                          # the backbone always stays in eval mode
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _taps(self, images):
        """images (B, S, 3, H, W) in [0, 1] -> ([B, S, P_total, D] per block output, patch_token_start).

        Poincar3 normalizes with ImageNet mean/std internally, so images must stay in [0, 1].
        """
        B, S, _, H, W = images.shape
        num_patches = (H // self.encoder.patch_size) * (W // self.encoder.patch_size)

        # All-False head_mask: the 65536-dim prototype head runs on zero patches instead of all of them.
        # Only block_outputs are needed for the probe.
        head_mask = torch.zeros(B, S, num_patches, dtype=torch.bool, device=images.device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            *_, block_outputs = self.encoder(images, mask=None, head_mask=head_mask, capture_block_outputs=True)
        taps = [t.reshape(B, S, -1, t.shape[-1]).float() for t in block_outputs]
        return taps, self.encoder.patch_token_start

    @staticmethod
    def _resize(x, size):
        """(B, S, C, h, w) -> (B, S, C, *size), bilinear."""
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        B, S = x.shape[:2]
        y = F.interpolate(x.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False)
        return y.reshape(B, S, *y.shape[1:])

    def forward(self, images):
        out_hw = images.shape[-2:]                                    # 224 x 224 (data / GT resolution)
        images = self._resize(images, (self.input_size, self.input_size)).clamp(0, 1)
        taps, patch_start = self._taps(images)
        layered_depth, stop = self.head(taps, images, patch_start)    # at input_size
        layered_depth = self._resize(layered_depth, out_hw)           # back to the GT resolution
        stop = self._resize(stop, out_hw)
        return {"layered_depth": layered_depth, "layered_depth_stop": stop, "depth": layered_depth[:, :, 0]}


def poincar3_large_ldi(**kwargs):
    return Poincar3_LDI(**kwargs)


def poincar3_ldi(**kwargs):
    return Poincar3_LDI(**kwargs)