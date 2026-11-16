"""
Overfit ONE fixed batch with FIXED masks, adding one component at a time.

Each variant should drive mse_b (and mse_c) towards ~0 and pred_std towards target_std.
The first variant that stalls is the culprit:

  1  B only,  raw RGB,  no confidence          -> sanity of encoder + linear head
  2  + norm-pix                                 -> norm-pix on flat Hypersim patches
  3  + confidence (CroCo two-pass)              -> confidence loss
  4  + C branch (lambda_c = 0.4)                -> held-out branch

Printed every LOG_EVERY steps:
  mse_b / mse_c   plain masked MSE in the loss space (raw: RGB^2, norm-pix: normalised^2)
  pred_std        std of predicted pixel values inside masked patches
  target_std      the same for the target  (~1 for norm-pix). pred_std << target_std = flat blocks
  grad_head       grad norm of encoder_pred (0 -> no gradient reaches the B head)
  grad_enc        grad norm of the encoder

Run from the repo root:  python debug_overfit.py
Outputs debug_overfit/<variant>.png (B: GT | masked | reconstruction) after each variant.
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# EDIT THESE TWO IMPORTS to match your repo layout
# ---------------------------------------------------------------------------
from models.modelv4 import vista_v4_base                       # noqa: E402
# /projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/datasets
from datasets import Hypersim_NVS                    # noqa: E402

CONFIG = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs/hypersim_nvs.yaml"
BATCH, MAX_VIEWS, N_TARGET = 4, 6, 2
STEPS, LOG_EVERY, LR = 400, 50, 2e-4
OUT = "debug_overfit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = [
    ("1_raw_noconf_Bonly",   dict(norm_pix_loss=False, conf_loss=False, lambda_c=0.0)),
    ("2_normpix_noconf_Bonly", dict(norm_pix_loss=True,  conf_loss=False, lambda_c=0.0)),
    ("3_normpix_conf_Bonly",   dict(norm_pix_loss=True,  conf_loss=True,  lambda_c=0.0)),
    ("4_normpix_conf_BC",      dict(norm_pix_loss=True,  conf_loss=True,  lambda_c=0.4)),
]


def fixed_batch():
    """One batch from the overfit set: sources [B,S,3,H,W], targets [B,T,3,H,W] in [0,1]."""
    ds = Hypersim_NVS(split="train", config_path=CONFIG, overfit=True, load_depth=False,
                      n_target_views=N_TARGET, max_views=MAX_VIEWS)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    items = [ds[i] for i in range(BATCH)]
    src = torch.stack([torch.as_tensor(np.asarray(it["s_images"])) for it in items]).float()
    tgt = torch.stack([torch.as_tensor(np.asarray(it["t_images"])) for it in items]).float()
    if src.max() > 1.5:                                   # uint8 -> [0, 1] if needed
        src, tgt = src / 255.0, tgt / 255.0
    return src.to(DEVICE), tgt.to(DEVICE)


class Probe:
    """Records the B mask and the B head output of the last forward pass."""
    def __init__(self, model):
        self.mask = None
        self.out = None
        orig_sample = model.sample_mask

        def sample_mask(N, P, device, mask_ratio=None, mask_type=None, ratios=None):
            m = orig_sample(N, P, device, mask_ratio=mask_ratio, mask_type=mask_type, ratios=ratios)
            if ratios is None and self.mask is None:      # first call of a forward = B mask
                self.mask = m.detach()
            return m
        model.sample_mask = sample_mask

        def hook(_, __, out):
            if self.out is None:                          # first call = masked pass
                self.out = out.detach()
        model.encoder_pred.register_forward_hook(hook)

    def reset(self):
        self.mask, self.out = None, None


def run_fixed(model, src, tgt):
    """Forward with every random choice frozen (num_clean, masks)."""
    random.seed(0); torch.manual_seed(0)
    return model(src, tgt)


def stats(model, probe, src):
    """pred_std vs target_std inside masked B patches, in the loss space."""
    out = probe.out.reshape(-1, probe.out.shape[-2], probe.out.shape[-1])
    pred = out[..., :model.patch_dim].float()
    S_m = out.shape[0] // src.shape[0]
    imgs = src[:, -S_m:].reshape(-1, *src.shape[2:])
    target = model.patchify(imgs.float())
    if model.norm_pix_loss:
        target = (target - target.mean(-1, keepdim=True)) / torch.sqrt(target.var(-1, keepdim=True) + 1e-6)
    else:
        pred = pred.sigmoid()
    m = probe.mask.reshape(-1).bool()
    return pred.reshape(-1, pred.shape[-1])[m].std(-1).mean().item(), \
           target.reshape(-1, target.shape[-1])[m].std(-1).mean().item()


def grad_norm(params):
    g = [p.grad.norm() for p in params if p.grad is not None]
    return torch.stack(g).norm().item() if g else 0.0


def save_b(model, src, tgt, path):
    model.eval()
    with torch.no_grad():
        _, _, _, _, b_masked, b_recon = run_fixed(model, src, tgt)
    model.train()
    S_m = b_masked.shape[1]
    gt = src[0, -S_m:]
    rows = [gt, b_masked[0], b_recon[0]]
    fig, ax = plt.subplots(3, S_m, figsize=(3 * S_m, 9), squeeze=False)
    for r, (name, ims) in enumerate(zip(["GT", "masked", "recon"], rows)):
        for c in range(S_m):
            ax[r, c].imshow(ims[c].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax[r, c].set_title(f"{name} B{c}"); ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    src, tgt = fixed_batch()
    print(f"batch: sources {tuple(src.shape)}  targets {tuple(tgt.shape)}  "
          f"range [{src.min().item():.2f}, {src.max().item():.2f}]")

    for name, kw in VARIANTS:
        torch.manual_seed(0)
        model = vista_v4_base(**kw).to(DEVICE).train()
        probe = Probe(model)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        print(f"\n=== {name}  {kw}")
        for step in range(1, STEPS + 1):
            probe.reset()
            loss, _, _, ld, _, _ = run_fixed(model, src, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 1 or step % LOG_EVERY == 0:
                p_std, t_std = stats(model, probe, src)
                print(f"step {step:4d}  loss {loss.item():8.4f}  mse_b {float(ld['mse_b']):.4f}  "
                      f"mse_c {float(ld['mse_c']):.4f}  pred_std {p_std:.3f}  target_std {t_std:.3f}  "
                      f"grad_head {grad_norm(model.encoder_pred.parameters()):.2e}  "
                      f"grad_enc {grad_norm(model.encoder.parameters()):.2e}")
            opt.step()
        save_b(model, src, tgt, os.path.join(OUT, f"{name}.png"))
        print(f"saved {OUT}/{name}.png")


if __name__ == "__main__":
    main()