#!/usr/bin/env python
"""
Evaluate novel-view-depth (NVD) probes on frozen SSL backbones: depth and point maps of the TARGET views only.

Each sample: S source views (images + GT cameras) go through the frozen encoder; the probe predicts the depth
of T target views from their GT cameras (K, world-to-camera) alone. Target images are never fed to the model.

Tasks (target views only)
  depth     : Pi^3 / CUT3R depth_evaluation (AbsRel, SqRel, RMSE, LogRMSE, delta1/2/3) on the predicted
              target depths. --depth_align per_seq: one alignment shared by all T targets of a sample;
              per_view: one per target image. The median GT/pred ratio is also reported (1.0 = metric probe).
  pointmap  : predicted target depth is unprojected with the GT target intrinsics and GT target poses
              (no camera prediction is involved), GT target depth likewise; the T target views of a sample
              form one point cloud. Acc / Comp / NC (Pi3 mv_recon metrics, Open3D normals) in GT units.
              --pm_align
                scale : one median depth scale per sample, no rotation / translation / ICP (default; the
                        cameras are GT, so only the global scale is free)
                pi3   : Pi3 mv_recon protocol as in eval_ssl.py (Umeyama Sim(3) + point-to-point ICP)
                none  : metric, no alignment at all

Outputs (all under --results_dir), same layout as eval_ssl.py
  results.pkl            persistent, append-only store (checkpoint fingerprint, settings, code hash, validation
                         report, metrics), updated after EACH dataset (atomic write + .bak + file lock)
  raw/<model>/...pkl     raw per-unit data -> metrics can be recomputed without the model (--recompute)
  vis/<model>/<dataset>/ --save_vis N: sources | GT | prediction | AbsRel map for the first N samples

Usage
  # --ckpt = train_nvd.py output folder under --ckpt_root (checkpoint-final.pth inside is used), or a .pth file.
  # --model defaults to the constructor string stored in the checkpoint's training args.
  python eval_nvd.py --model_name vista --ckpt vista_base_model_nvd_probe_all_datasets_scannetpp_model_224_x_224_30ep \
      --dataset "ScanNetPP_NVS(split='test', ..., n_target_views=2, max_views=6, load_depth=True)" \
      --results_dir results/nvd_eval
  python eval_nvd.py --model "mae_nvd_base_downstream()" --model_name mae --ckpt <mae_nvd_run> --dataset "..."
  python eval_nvd.py --show --results_dir results/nvd_eval
  python print_tables.py results/nvd_eval/results.pkl --models mae mum muskie vista
  python eval_nvd.py --recompute --results_dir results/nvd_eval

The checkpoint holds the WHOLE model (frozen encoder + probe), as train_nvd.py saves it, and is loaded strictly,
so no separate SSL checkpoint is needed.

Models: MAE, MuM, Muskie, Vista (base).
"""

import argparse
import contextlib
import datetime
import functools
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import socket
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

# Project
from datasets import get_data_loader  # noqa: E402
from models import (  # noqa: E402,F401  (used through eval(args.model))
    vista_nvd_base_downstream,
    mae_nvd_base_downstream,
    mum_nvd_base_downstream,
    muskie_nvd_base_downstream,
)

MODEL_NAMES = sorted(n for n in dir() if n.endswith("_downstream"))

# all NVD probe runs live in folders under this root; --ckpt <folder name> picks the final weights inside
CKPT_ROOT = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints"
CKPT_EXTS = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")

DISPLAY_NAMES = {
    "mae": "MAE",
    "mum": "MuM",
    "muskie": "Muskie",
    "vista": "Vista",
}

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("eval_nvd")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("NVD probe evaluation: target-view depth / point map")
    # model
    p.add_argument("--model", type=str, default=None,
                   help="Constructor as in train_nvd.py, e.g. 'vista_nvd_base_downstream()' or "
                        "vista_nvd_base_downstream (+ --model_kwargs). Default: the --model stored in the checkpoint")
    p.add_argument("--model_name", type=str,
                   help="Unique key in the results store, e.g. vista, mae (letters, digits, . _ + - only). "
                        "Default: the --ckpt run folder name")
    p.add_argument("--display_name", type=str, default=None,
                   help="Name shown in tables (raw LaTeX allowed). Default: DISPLAY_NAMES[--model_name]")
    p.add_argument("--description", type=str, default="", help="Free-text note stored with the record")
    p.add_argument("--model_kwargs", type=str, default="{}",
                   help="JSON dict of constructor kwargs (only with a bare constructor name in --model)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="train_nvd.py output folder name under --ckpt_root, a folder path, or a weights file "
                        "(e.g. <run>/checkpoint-best.pth). For a folder, checkpoint-final.pth wins")
    p.add_argument("--ckpt_root", type=str, default=CKPT_ROOT, help="Folder that holds all run folders")
    p.add_argument("--non_strict", action="store_true",
                   help="Allow missing / unexpected keys when loading the checkpoint (logged); default strict")
    p.add_argument("--amp", choices=list(AMP), default="fp32",
                   help="Autocast for the whole forward. fp32 = as in training (the encoder still runs its own "
                        "bf16 autocast inside encode_sources; the probe runs in fp32)")
    # data
    p.add_argument("--dataset", type=str, default="",
                   help="'+'-separated dataset strings for get_data_loader (needs load_depth=True)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=-1, help="Debug: stop after N batches per dataset")
    # eval settings
    p.add_argument("--tasks", type=str, default="depth,pointmap")
    p.add_argument("--depth_align", choices=["per_seq", "per_view"], default="per_seq",
                   help="per_seq: one alignment for all target views of a sample; per_view: one per target image")
    p.add_argument("--depth_align_method", choices=["scale", "scale_shift", "lstsq", "median", "metric"],
                   default="scale", help="Pi^3/CUT3R alignment: scale (IRLS), scale_shift (L1), lstsq, median, metric")
    p.add_argument("--depth_lr", type=float, default=1e-4, help="Adam lr for scale_shift alignment")
    p.add_argument("--depth_max_iters", type=int, default=1000)
    p.add_argument("--depth_max", type=float, default=None,
                   help="Ignore GT depth >= this and clip predictions to it")
    p.add_argument("--pm_align", choices=["scale", "pi3", "none"], default="scale",
                   help="Point-map alignment (cameras are always GT): scale = one median depth scale per sample; "
                        "pi3 = Umeyama Sim(3) + ICP as eval_ssl.py; none = metric")
    p.add_argument("--pm_max_sequences", type=int, default=None,
                   help="Score point maps on the first N samples only (faster). Depth uses every sample")
    p.add_argument("--pm_icp_threshold", type=float, default=0.1,
                   help="ICP threshold in GT units (only --pm_align pi3)")
    p.add_argument("--seed", type=int, default=0)
    # output
    p.add_argument("--results_dir", type=str, default="results/nvd_eval")
    p.add_argument("--store", type=str, default="results.pkl", help="Results store file inside --results_dir")
    p.add_argument("--save_vis", type=int, default=0, help="Save qualitative figures for the first N samples")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-evaluate even if a complete record with the same checkpoint + settings exists")
    p.add_argument("--show", action="store_true", help="Print what is in the store and exit")
    p.add_argument("--recompute", action="store_true",
                   help="Re-derive metrics from saved raw data with the current code (no model needed)")
    return p.parse_args()


# =============================================================================
# Model loading + forward
# =============================================================================
def resolve_ckpt(ckpt, root):
    """--ckpt -> weights file. Accepts a file, a folder path, or a folder name under `root`.
    In a folder: a file named final/last wins, else the highest step/epoch number in the name,
    else the newest file. Searches the folder itself first, then its subfolders."""
    import re
    if ckpt is None:
        return None
    for cand in (ckpt, osp.join(root, ckpt)):
        if osp.isfile(cand):
            return osp.abspath(cand)
        if osp.isdir(cand):
            folder = cand
            break
    else:
        raise SystemExit(f"--ckpt '{ckpt}' is neither a file nor a folder (also looked in {root})")

    files = [osp.join(folder, f) for f in os.listdir(folder) if f.endswith(CKPT_EXTS)]
    if not files:  # e.g. run/checkpoints/*.pth
        files = [osp.join(d, f) for d, _, fs in os.walk(folder) for f in fs if f.endswith(CKPT_EXTS)]
    if not files:
        raise SystemExit(f"no weights ({', '.join(CKPT_EXTS)}) found in {folder}")

    def stem(f):
        return osp.splitext(osp.basename(f))[0].lower()

    named = [f for f in files if re.search(r"(^|[-_.])(final|last|latest)([-_.]|$)", stem(f))]
    if named:
        named.sort(key=lambda f: ("final" not in stem(f), -os.path.getmtime(f)))  # final > last/latest > newest
        choice, why = named[0], "named final/last"
    else:
        numbered = [(int(re.findall(r"\d+", stem(f))[-1]), f) for f in files if re.findall(r"\d+", stem(f))]
        if numbered:
            choice, why = max(numbered)[1], "highest step/epoch"
        else:
            choice, why = max(files, key=os.path.getmtime), "newest file"
    others = sorted(osp.relpath(f, folder) for f in files if f != choice)
    logger.info(f"--ckpt {ckpt} -> {choice} ({why})"
                + (f"; other weights there: {others[:6]}{' ...' if len(others) > 6 else ''}" if others else ""))
    return osp.abspath(choice)


def ckpt_run_name(ckpt, root):
    """Default --model_name: the run folder name (or the file's parent folder)."""
    for cand in (ckpt, osp.join(root, ckpt)):
        if osp.isdir(cand):
            return osp.basename(osp.normpath(cand))
    return osp.basename(osp.dirname(osp.abspath(ckpt)))



@functools.lru_cache(maxsize=1)
def read_ckpt(path):
    """train_nvd.py checkpoint: {'model': full state dict, 'args': training Namespace, 'epoch': ...}."""
    return torch.load(path, map_location="cpu", weights_only=False)


def ckpt_train_args(path):
    ckpt = read_ckpt(path)
    return ckpt.get("args") if isinstance(ckpt, dict) else None


def model_constructor(spec):
    """'vista_nvd_base_downstream()' / 'vista_nvd_base_downstream' -> 'vista_nvd_base_downstream'."""
    return spec.split("(")[0].strip()


def load_model(args, device):
    kwargs = json.loads(args.model_kwargs)
    spec = args.model.strip()
    if spec.endswith(")"):                                       # 'vista_nvd_base_downstream()' as in train_nvd.py
        if kwargs:
            raise SystemExit("give either --model 'f(...)' or --model f with --model_kwargs, not both")
        model = eval(spec)
    else:
        model = eval(spec)(**kwargs)

    ckpt = read_ckpt(args.ckpt)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt:
                state = ckpt[k]
                break
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    msg = f"{len(missing)} missing, {len(unexpected)} unexpected keys"
    if (missing or unexpected) and not args.non_strict:
        raise SystemExit(f"{args.ckpt} does not match {spec}: {msg}\n  missing: {missing[:10]}\n"
                         f"  unexpected: {unexpected[:10]}\n(wrong --model? use --non_strict to load anyway)")
    logger.info(f"Loaded {spec} from {args.ckpt} (epoch {ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'}): {msg}")
    if missing:
        logger.warning(f"Missing (first 10): {missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected (first 10): {unexpected[:10]}")
    read_ckpt.cache_clear()                                      # free the CPU copy
    return model.to(device).eval()


def run_model(model, x, amp_dtype):
    """Same call as train_nvd.forward_and_loss. Returns predicted target depth (B, T, H, W), float32."""
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        out = model(x["src_images"], x["src_w2c"], x["tgt_K"], x["tgt_w2c"])
    depth = out["depth"].float()
    if depth.dim() == 5:
        depth = depth[..., 0]
    return depth


# =============================================================================
# Batch handling
# =============================================================================
def _tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        return torch.stack([_tensor(v) for v in x])
    raise TypeError(f"Cannot convert {type(x)} to tensor")


def to_4x4(T):
    if T.shape[-2] == 3:
        bottom = torch.zeros(*T.shape[:-2], 1, 4, dtype=T.dtype, device=T.device)
        bottom[..., 0, 3] = 1.0
        T = torch.cat([T, bottom], dim=-2)
    return T


def gather_nvd(batch, device):
    """
    Model inputs exactly as train_nvd.prepare_batch (sources: images + w2c; targets: K + w2c), plus GT target
    depth / mask. GT *_extrinsics are world-to-camera (OpenCV). Target images are not used.
    """
    for k in ("t_depths", "t_intrinsics", "t_extrinsics"):
        if k not in batch:
            raise KeyError(f"batch has no '{k}' (keys: {sorted(batch)}); NVD eval needs target views with "
                           "depth: set n_target_views >= 1 and load_depth=True in the dataset string")
    get = lambda k: _tensor(batch[k]).float().to(device, non_blocking=True)  # noqa: E731
    x = dict(src_images=get("s_images"), src_w2c=get("s_extrinsics"),
             tgt_K=get("t_intrinsics"), tgt_w2c=get("t_extrinsics"))
    B, T = x["tgt_K"].shape[:2]
    d = get("t_depths")
    d = d.reshape(B, T, *d.shape[-2:])
    if "t_depth_masks" in batch:
        m = _tensor(batch["t_depth_masks"]).to(device).reshape(B, T, *d.shape[-2:]).bool()
    else:
        m = d > 0
    x["tgt_depth"] = d
    x["tgt_mask"] = m & (d > 0) & torch.isfinite(d)
    return x


def scale_K(K, from_hw, to_hw):
    if tuple(from_hw) == tuple(to_hw):
        return K
    K = K.clone()
    sy, sx = to_hw[0] / from_hw[0], to_hw[1] / from_hw[1]
    K[..., 0, 0] *= sx
    K[..., 0, 2] *= sx
    K[..., 1, 1] *= sy
    K[..., 1, 2] *= sy
    return K

# =============================================================================
# Depth
# =============================================================================
# --- Copied from Pi^3 / CUT3R depth evaluation ------------------------------------------------
# Reference: https://github.com/CUT3R/CUT3R/blob/main/eval/monodepth/tools.py
# Only change: the custom mask is moved to the same device as `mask` before indexing
# (the original `.cpu()` breaks when the depth tensors live on the GPU).

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params
    predicted_aligned = s * predicted_depth + t
    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)


def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    from scipy.optimize import minimize

    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    initial_params = [s, t]  # s = 1, t = 0
    result = minimize(
        absolute_error_loss,
        initial_params,
        args=(predicted_depth_np, ground_truth_depth_np),
    )
    s, t = result.x
    return s, t


def absolute_value_scaling2(
    predicted_depth,
    ground_truth_depth,
    s_init=1.0,
    t_init=0.0,
    lr=1e-4,
    max_iters=1000,
    tol=1e-6,
):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()
        predicted_aligned = s * predicted_depth + t
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)
        loss = torch.sum(abs_error)
        loss.backward()
        optimizer.step()
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss.item()
    return s.detach().item(), t.detach().item()


def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map,
    with an option for least squares alignment. A 3D input (S, H, W) is flattened, so a
    whole sequence shares ONE alignment (video-depth protocol).
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if metric_scale:
        predicted_depth = predicted_depth
    elif align_with_lstsq:
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-8
            weights = 1.0 / abs_residuals
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth ** 2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        s = s.clamp(min=1e-3)
        s = s.detach()
        predicted_depth = s * predicted_depth
    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.to(mask.device)[mask]  # (original: custom_mask.cpu()[mask])
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()

    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25 ** 2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25 ** 3).float()).item()

    # Compute the depth error parity map
    if metric_scale:
        predicted_depth_original = predicted_depth_original
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
        )
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
        )
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
        )
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
        )

    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)

    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)

    num_valid_pixels = torch.sum(mask).item() if custom_mask is None else torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        (abs_rel, sq_rel, rmse, log_rmse, threshold_0, threshold_1, threshold_2, threshold_3) = (0,) * 8

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": threshold_0,
        "δ < 1.25": threshold_1,
        "δ < 1.25^2": threshold_2,
        "δ < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }
    return results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full

# --- end of Pi^3 / CUT3R code -------------------------------------------------------------------


PI3_KEYS = {"Abs Rel": "abs_rel", "Sq Rel": "sq_rel", "RMSE": "rmse", "Log RMSE": "log_rmse",
            "δ < 1.25": "delta1", "δ < 1.25^2": "delta2", "δ < 1.25^3": "delta3"}

ALIGN_KWARGS = {  # --depth_align_method -> depth_evaluation flags
    "scale": dict(align_with_scale=True),       # Pi^3 / CUT3R "scale" (IRLS, L1)
    "scale_shift": dict(align_with_lad2=True),  # Pi^3 / CUT3R "scale & shift" (L1, Adam)
    "lstsq": dict(align_with_lstsq=True),       # scale & shift, least squares
    "median": dict(),                            # median scaling (function default)
    "metric": dict(metric_scale=True),          # no alignment
}


def eval_depth_batch(pred, gt, mask, args):
    """
    pred/gt/mask: (B, S, H, W). per_seq -> one call per sequence on (S, H, W), i.e. a single
    alignment shared by all views (Pi^3 video-depth protocol); per_view -> one call per image.
    Returns a list of (results_dict, valid_pixels).
    """
    kw = dict(max_depth=args.depth_max, post_clip_max=args.depth_max, lr=args.depth_lr,
              max_iters=args.depth_max_iters, **ALIGN_KWARGS[args.depth_align_method])
    out = []
    for b in range(pred.shape[0]):
        units = [(pred[b], gt[b], mask[b])] if args.depth_align == "per_seq" else \
                [(pred[b, s][None], gt[b, s][None], mask[b, s][None]) for s in range(pred.shape[1])]
        for p, g, m in units:
            if (m & (g > 0)).sum() < 10:
                continue
            res, _, _, _ = depth_evaluation(p.float().contiguous(), g.float().contiguous(),
                                            custom_mask=m.bool().contiguous(), **kw)
            if res["valid_pixels"] > 0:
                out.append(res)
    return out


def summarize_depth(results):
    """Pixel-weighted average over sequences (as in Pi^3 / CUT3R)."""
    if not results:
        return {}
    w = np.array([r["valid_pixels"] for r in results], dtype=np.float64)
    summary = {ours: float(np.average([r[k] for r in results], weights=w)) for k, ours in PI3_KEYS.items()}
    summary["num_units"] = len(results)
    summary["valid_pixels"] = int(w.sum())
    return summary



# =============================================================================
# Point map (GT target cameras)
# =============================================================================
def unproject(depth, K, c2w):
    """depth (S,H,W), K (S,3,3), c2w (S,4,4) -> world points (S,H,W,3)."""
    S, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device),
                            indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).to(depth.dtype)
    rays = torch.einsum("sij,nj->sni", torch.linalg.inv(K.to(depth.dtype)), pix)
    cam = rays * depth.reshape(S, -1, 1)
    c2w = c2w.to(depth.dtype)
    world = torch.einsum("sij,snj->sni", c2w[:, :3, :3], cam) + c2w[:, None, :3, 3]
    return world.reshape(S, H, W, 3)

# --- Pi3 mv_recon/utils.py, verbatim ------------------------------------------------------------
# Reference: https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/utils.py
from scipy.spatial import cKDTree as KDTree


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


# --- end of Pi3 mv_recon/utils.py ------------------------------------------------------------------



def pointmap_metrics_gt_cameras(pred_depth, gt_depth, K, w2c, valid, align, icp_threshold):
    """
    One sample (its T target views form one point cloud). Predicted AND GT target depth are unprojected with the
    same GT target intrinsics / world-to-camera poses, so pred and GT points correspond pixel by pixel.
      align 'scale': pred depth *= median(GT) / median(pred) over the valid pixels of the sample (one scale)
      align 'pi3'  : Umeyama Sim(3) on the correspondences + Open3D point-to-point ICP (eval_ssl.py protocol)
      align 'none' : metric
    Then Open3D normals and Acc / Comp / NC on the valid pixels, in GT units.
    Inputs (torch, one sample): depth (T,H,W), K (T,3,3), w2c (T,4,4), valid (T,H,W) GT validity mask.
    """
    import open3d as o3d

    valid = valid & (gt_depth > 0) & torch.isfinite(pred_depth) & (pred_depth > 0)
    if valid.sum() < 10:
        return None
    pred_depth, gt_depth = pred_depth.double(), gt_depth.double()
    scale = 1.0
    if align == "scale":
        scale = (gt_depth[valid].median() / pred_depth[valid].median()).item()

    c2w = torch.linalg.inv(to_4x4(w2c.double()))
    pred_pts = unproject(pred_depth * scale, K.double(), c2w)[valid].cpu().numpy()
    gt_pts = unproject(gt_depth, K.double(), c2w)[valid].cpu().numpy()

    if align == "pi3":                                            # coarse Sim(3)
        c, R, t = umeyama(pred_pts.T, gt_pts.T)
        pred_pts = c * pred_pts @ R.T + t.T
        scale = float(c)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)

    if align == "pi3":                                            # ICP refinement
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd, pcd_gt, icp_threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd = pcd.transform(reg_p2p.transformation)

    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    pred_normal = np.asarray(pcd.normals)
    gt_normal = np.asarray(pcd_gt.normals)

    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    return {
        "acc": float(acc), "acc_med": float(acc_med),
        "comp": float(comp), "comp_med": float(comp_med),
        "nc": float((nc1 + nc2) / 2), "nc_med": float((nc1_med + nc2_med) / 2),
        "nc1": float(nc1), "nc1_med": float(nc1_med),
        "nc2": float(nc2), "nc2_med": float(nc2_med),
        "pm_scale": float(scale), "num_points": int(len(gt_pts)),
    }


# =============================================================================
# Qualitative figures
# =============================================================================
def save_vis(x, pred, b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = x["src_images"][b].float().cpu()
    gt, m = x["tgt_depth"][b].cpu(), x["tgt_mask"][b].cpu()
    pr = pred[b].float().cpu()
    S, T = src.shape[0], gt.shape[0]
    cols = max(S, 3)
    fig, ax = plt.subplots(1 + T, cols, figsize=(2.4 * cols, 2.4 * (1 + T)), squeeze=False)
    for a in ax.flat:
        a.axis("off")
    for s in range(S):
        ax[0, s].imshow(src[s].permute(1, 2, 0).clamp(0, 1).numpy())
        ax[0, s].set_title(f"source {s}", fontsize=8)
    for t in range(T):
        g, p, mm = gt[t], pr[t], m[t]
        if mm.sum() < 10:
            continue
        p = p * (g[mm].median() / p[mm].median())                    # median-aligned for display
        lo, hi = g[mm].quantile(0.02).item(), g[mm].quantile(0.98).item()
        err = torch.where(mm, (p - g).abs() / g.clamp_min(1e-6), torch.zeros_like(g))
        ax[1 + t, 0].imshow(torch.where(mm, g, torch.nan).numpy(), cmap="Spectral", vmin=lo, vmax=hi)
        ax[1 + t, 0].set_title(f"GT target {t}", fontsize=8)
        ax[1 + t, 1].imshow(p.numpy(), cmap="Spectral", vmin=lo, vmax=hi)
        ax[1 + t, 1].set_title("prediction (median-aligned)", fontsize=8)
        ax[1 + t, 2].imshow(err.numpy(), cmap="magma", vmin=0, vmax=0.25)
        ax[1 + t, 2].set_title(f"AbsRel {err[mm].mean().item():.3f}", fontsize=8)
    plt.tight_layout()
    os.makedirs(osp.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# Evaluation loop
# =============================================================================
@torch.no_grad()
def evaluate_dataset(model, loader, name, args, device):
    """Runs the model and returns RAW data only; metrics come from summarize_raw()."""
    tasks = set(args.tasks.split(","))
    amp_dtype = AMP[args.amp]
    raw = {"depth_units": [], "pose_seqs": [], "pointmap_units": [], "scale_ratios": [],
           "num_sequences": 0, "num_frames": 0, "nonfinite_pred_frames": 0, "pm_cameras": "gt",
           "num_source_views": [], "num_target_views": []}
    n_vis = 0

    for it, batch in enumerate(loader):
        if 0 <= args.max_batches <= it:
            break
        x = gather_nvd(batch, device)
        gt_d, gt_m = x["tgt_depth"], x["tgt_mask"]
        B, T, dh, dw = gt_d.shape
        H, W = x["src_images"].shape[-2:]                            # target K is given at the image size

        pred_d = run_model(model, x, amp_dtype)                      # (B, T, h, w)
        if pred_d.shape[-2:] != (dh, dw):
            pred_d = F.interpolate(pred_d.reshape(B * T, 1, *pred_d.shape[-2:]), size=(dh, dw),
                                   mode="bilinear", align_corners=False).reshape(B, T, dh, dw)
        raw["nonfinite_pred_frames"] += int((~torch.isfinite(pred_d)).flatten(2).any(-1).sum())

        # is the probe metric? median GT / pred per target view (1.0 = right scale)
        for b in range(B):
            for t in range(T):
                m = gt_m[b, t] & torch.isfinite(pred_d[b, t]) & (pred_d[b, t] > 0)
                if m.sum() >= 10:
                    raw["scale_ratios"].append(float(gt_d[b, t][m].median() / pred_d[b, t][m].median()))

        if "depth" in tasks:
            raw["depth_units"].extend(eval_depth_batch(pred_d, gt_d, gt_m, args))

        pm_left = None if args.pm_max_sequences is None else args.pm_max_sequences - raw["num_sequences"]
        if "pointmap" in tasks and (pm_left is None or pm_left > 0):
            K_d = scale_K(x["tgt_K"], (H, W), (dh, dw))
            for b in range(B if pm_left is None else min(B, pm_left)):
                res = pointmap_metrics_gt_cameras(pred_d[b], gt_d[b], K_d[b], x["tgt_w2c"][b], gt_m[b],
                                                  args.pm_align, args.pm_icp_threshold)
                if res is not None:
                    raw["pointmap_units"].append(res)

        for b in range(B):
            if n_vis >= args.save_vis:
                break
            save_vis(x, pred_d, b, osp.join(args.results_dir, "vis", args.model_name, name,
                                            f"{raw['num_sequences'] + b:05d}.png"))
            n_vis += 1

        raw["num_sequences"] += B
        raw["num_frames"] += B * T
        raw["num_source_views"].append(int(x["src_images"].shape[1]))
        raw["num_target_views"].append(int(T))
        if it % 50 == 0:
            msg = f"[{name}] batch {it}/{len(loader)}"
            if raw["depth_units"]:
                d = summarize_depth(raw["depth_units"])
                msg += f"  AbsRel {d['abs_rel']:.4f}  d1 {d['delta1']:.4f}"
            if raw["scale_ratios"]:
                msg += f"  gt/pred {np.median(raw['scale_ratios']):.3f}"
            if raw["pointmap_units"]:
                msg += f"  Acc {np.mean([u['acc'] for u in raw['pointmap_units']]):.4f}"
            logger.info(msg)
    return raw


def summarize_raw(raw, settings):
    """Raw data -> metrics. Shared by evaluation and --recompute, so both use identical code."""
    metrics = {}
    if raw["depth_units"]:
        metrics["depth"] = summarize_depth(raw["depth_units"])
        r = np.asarray(raw.get("scale_ratios", []), dtype=np.float64)
        if len(r):
            metrics["depth"]["scale_ratio_median"] = float(np.median(r))
            metrics["depth"]["scale_ratio_iqr"] = float(np.subtract(*np.percentile(r, [75, 25])))
    if raw["pointmap_units"]:
        units = raw["pointmap_units"]
        metrics["pointmap"] = {k: float(np.mean([u[k] for u in units])) for k in units[0]}
        metrics["pointmap"]["num_scenes"] = len(units)
    return metrics


# =============================================================================
# Validation: catch metric / protocol errors before they reach a table
# =============================================================================
METRIC_RANGES = {
    "depth": {"abs_rel": (0, None), "sq_rel": (0, None), "rmse": (0, None), "log_rmse": (0, None),
              "delta1": (0, 1), "delta2": (0, 1), "delta3": (0, 1)},
    "pointmap": {"acc": (0, None), "acc_med": (0, None), "comp": (0, None), "comp_med": (0, None),
                 "nc": (0, 1), "nc_med": (0, 1), "nc1": (0, 1), "nc1_med": (0, 1), "nc2": (0, 1),
                 "nc2_med": (0, 1)},
}


def validate(metrics, raw, settings, expected_sequences):
    """Returns (errors, warnings). Any error -> record is stored but marked incomplete."""
    errors, warnings = [], []
    tasks = settings["tasks"].split(",")
    n_seq = raw["num_sequences"]

    if n_seq == 0:
        errors.append("no sequences evaluated")
    if settings["max_batches"] >= 0:
        errors.append(f"debug run (--max_batches {settings['max_batches']})")
    if expected_sequences is not None and n_seq < expected_sequences:
        errors.append(f"partial run: {n_seq}/{expected_sequences} sequences")

    for task in tasks:
        if task not in METRIC_RANGES:
            errors.append(f"unknown task '{task}' (NVD eval has: {', '.join(METRIC_RANGES)})")
            continue
        if task not in metrics or not metrics[task]:
            errors.append(f"task '{task}' requested but produced no metrics")
            continue
        for k, (lo, hi) in METRIC_RANGES[task].items():
            if k not in metrics[task]:
                continue
            v = metrics[task][k]
            if not np.isfinite(v):
                errors.append(f"{task}.{k} is not finite ({v})")
            elif (lo is not None and v < lo - 1e-9) or (hi is not None and v > hi + 1e-9):
                errors.append(f"{task}.{k}={v:.4f} outside [{lo}, {hi}]")

    if raw["nonfinite_pred_frames"]:
        warnings.append(f"{raw['nonfinite_pred_frames']}/{raw['num_frames']} target views have NaN/Inf predicted depth")
    if len(set(raw.get("num_target_views", []))) > 1:
        warnings.append(f"number of target views varies across batches: {sorted(set(raw['num_target_views']))}")
    d = metrics.get("depth")
    if d:
        expected_units = n_seq if settings["depth_align"] == "per_seq" else raw["num_frames"]
        if d["num_units"] < 0.9 * expected_units:
            warnings.append(f"depth: only {d['num_units']}/{expected_units} units had valid GT")
        if not (d["delta1"] <= d["delta2"] + 1e-9 <= d["delta3"] + 2e-9):
            errors.append("depth: delta1 <= delta2 <= delta3 violated")
    p = metrics.get("pointmap")
    if p:
        pm_expected = n_seq if settings.get("pm_max_sequences") is None else min(n_seq, settings["pm_max_sequences"])
        if p["num_scenes"] < 0.9 * pm_expected:
            warnings.append(f"pointmap: only {p['num_scenes']}/{pm_expected} samples had enough valid points")
    return errors, warnings


# =============================================================================
# Results store: append-only history, atomic writes, .bak, file lock
# =============================================================================
STORE_SCHEMA = 1
try:
    CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]
except Exception:  # noqa: BLE001
    CODE_SHA = "unknown"

try:
    import fcntl
except ImportError:  # non-POSIX: no locking
    fcntl = None


def _atomic_pickle(obj, path, keep_backup=False):
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    if keep_backup and osp.exists(path):
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)  # atomic on POSIX: readers see the old or the new file, never half of one


def _read_store(path):
    for p in (path, path + ".bak"):
        if osp.exists(p):
            try:
                with open(p, "rb") as f:
                    store = pickle.load(f)
                if p != path:
                    logger.warning(f"{path} unreadable -> recovered from {p}")
                return store
            except Exception as e:  # noqa: BLE001
                logger.error(f"could not read {p}: {e}")
    if osp.exists(path) or osp.exists(path + ".bak"):
        raise RuntimeError(f"{path} exists but neither it nor its .bak can be read; refusing to "
                           "continue so the store is not overwritten. Inspect it manually.")
    return {"schema": STORE_SCHEMA, "runs": {}}


@contextlib.contextmanager
def locked_store(path, write=True):
    """Read-modify-write under an exclusive lock, so parallel jobs never lose each other's results."""
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "a") as lf:
        if fcntl is not None:
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            store = _read_store(path)
            yield store
            if write:
                _atomic_pickle(store, path, keep_backup=True)
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)


def ckpt_fingerprint(path):
    if not path:
        return {"path": None}
    if not osp.exists(path):
        return {"path": path, "missing": True}
    st = os.stat(path)
    return {"path": osp.abspath(path), "size": st.st_size, "mtime": st.st_mtime}



def eval_settings(args):
    """Everything that changes the numbers. Records are only comparable if these match."""
    return {
        "protocol": "nvd_targets_only",
        "tasks": args.tasks, "amp": args.amp,
        "model_class": args.model,
        "model_kwargs": args.model_kwargs, "seed": args.seed,
        "depth_align": args.depth_align, "depth_align_method": args.depth_align_method,
        "depth_max": args.depth_max, "depth_lr": args.depth_lr, "depth_max_iters": args.depth_max_iters,
        "pm_protocol": f"gt_target_cameras/{args.pm_align}", "pm_icp_threshold": args.pm_icp_threshold,
        "pm_max_sequences": args.pm_max_sequences,
        "max_batches": args.max_batches,
    }

def make_record(model_name, dataset, ckpt_fp, settings, raw, raw_file, expected_sequences, source,
                display_name=None, description=""):
    metrics = summarize_raw(raw, settings)
    errors, warnings = validate(metrics, raw, settings, expected_sequences)
    return {
        "model": model_name, "display_name": display_name, "description": description,
        "dataset": dataset, "source": source,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "ckpt": ckpt_fp, "settings": settings, "code_sha": CODE_SHA,
        "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "num_sequences": raw["num_sequences"], "expected_sequences": expected_sequences,
        "pm_cameras": raw["pm_cameras"],
        "complete": not errors, "errors": errors, "warnings": warnings,
        "metrics": metrics, "raw_file": raw_file,
    }


def append_record(store, rec):
    store["runs"].setdefault(rec["model"], {}).setdefault(rec["dataset"], []).append(rec)


def latest_record(history, allow_incomplete=False):
    for rec in reversed(history):
        if rec["complete"] or allow_incomplete:
            return rec
    return None


def find_matching(store, model, dataset, ckpt_fp, settings):
    """Latest complete record with the same checkpoint and settings (used to skip re-evaluation)."""
    for rec in reversed(store["runs"].get(model, {}).get(dataset, [])):
        if rec["complete"] and rec["ckpt"] == ckpt_fp and rec["settings"] == settings:
            return rec
    return None



def log_record(rec):
    status = "COMPLETE" if rec["complete"] else "INCOMPLETE"
    m = rec["metrics"]
    parts = []
    if "depth" in m:
        d = m["depth"]
        parts.append(f"AbsRel {d['abs_rel']:.4f} d1 {d['delta1']:.3f} RMSE {d['rmse']:.4f}"
                     + (f" gt/pred {d['scale_ratio_median']:.3f}" if "scale_ratio_median" in d else ""))
    if "pointmap" in m:
        p = m["pointmap"]
        parts.append(f"Acc {p['acc']:.4f} Comp {p['comp']:.4f} NC {p['nc']:.3f}")
    logger.info(f"[{rec['model']} / {rec['dataset']}] {status}  " + "  |  ".join(parts))
    for e in rec["errors"]:
        logger.error(f"[{rec['model']} / {rec['dataset']}] ERROR: {e}")
    for w in rec["warnings"]:
        logger.warning(f"[{rec['model']} / {rec['dataset']}] WARNING: {w}")


def show_store(args):
    path = osp.join(args.results_dir, args.store)
    if not osp.exists(path):
        print(f"No store at {path}")
        return
    with locked_store(path, write=False) as store:
        runs = store["runs"]
    print("Showing the record the tables use (latest complete one) for every (model, dataset).")
    print(f"{'model':<12} {'dataset':<20} {'status':<10} {'seqs':>11} {'AbsRel':>7} {'d1':>6} {'gt/pr':>6} "
          f"{'Acc':>7} {'Comp':>7} {'NC':>6} {'hist':>4} {'when':<19} flags")
    for model in sorted(runs):
        for ds in sorted(runs[model]):
            hist = runs[model][ds]
            used = latest_record(hist)
            rec = used if used is not None else hist[-1]
            m = rec["metrics"]
            f = lambda t, k, n=4: f"{m[t][k]:.{n}f}" if t in m and k in m[t] else "--"  # noqa: E731
            seqs = f"{rec['num_sequences']}/{rec['expected_sequences'] if rec['expected_sequences'] is not None else '?'}"
            flags = rec["errors"] + rec["warnings"]
            if used is not None and hist[-1] is not used:
                flags.append(f"newer run on {hist[-1]['time']} is flagged and ignored: "
                             + "; ".join(hist[-1]["errors"]))
            flags = "; ".join(flags)
            print(f"{model:<12} {ds:<20} {'ok' if rec['complete'] else 'FLAGGED':<10} {seqs:>11} "
                  f"{f('depth', 'abs_rel'):>7} {f('depth', 'delta1', 3):>6} {f('depth', 'scale_ratio_median', 3):>6} "
                  f"{f('pointmap', 'acc'):>7} {f('pointmap', 'comp'):>7} {f('pointmap', 'nc', 3):>6} "
                  f"{len(hist):>4} {rec['time']:<19} {flags}")

def recompute(args):
    path = osp.join(args.results_dir, args.store)
    only_ds = {d.split("(")[0].strip() for d in args.dataset.split("+") if d.strip()} if args.dataset else None
    with locked_store(path) as store:
        todo = []
        for model, per_ds in store["runs"].items():
            if args.model_name and model != args.model_name:
                continue
            for ds, hist in per_ds.items():
                if only_ds and ds not in only_ds:
                    continue
                with_raw = [r for r in hist if r.get("raw_file")]
                # recompute the record the tables use (latest complete); fall back to the latest one
                rec = next((r for r in reversed(with_raw) if r["complete"]), with_raw[-1] if with_raw else None)
                if rec is None:
                    continue
                if rec["code_sha"] == CODE_SHA and not args.overwrite:
                    logger.info(f"[{model} / {ds}] already on current code {CODE_SHA} -> nothing to recompute")
                    continue
                todo.append(rec)
        for rec in todo:
            raw_path = osp.join(args.results_dir, rec["raw_file"])
            with open(raw_path, "rb") as f:
                raw = pickle.load(f)
            new = make_record(rec["model"], rec["dataset"], rec["ckpt"], rec["settings"], raw,
                              rec["raw_file"], rec["expected_sequences"], source="recompute",
                              display_name=rec.get("display_name"), description=rec.get("description", ""))
            append_record(store, new)
            log_record(new)
    logger.info(f"Recomputed {len(todo)} record(s) with code {CODE_SHA}")



# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args.show:
        show_store(args)
        return
    if args.recompute:
        recompute(args)
        return

    if not (args.ckpt and args.dataset):
        raise SystemExit("--ckpt and --dataset are required unless --show/--recompute")
    if not args.model_name:                       # default key: the run folder name
        args.model_name = ckpt_run_name(args.ckpt, args.ckpt_root)
    args.ckpt = resolve_ckpt(args.ckpt, args.ckpt_root)
    if args.model is None:                        # the constructor string train_nvd.py was run with
        train_args = ckpt_train_args(args.ckpt)
        args.model = getattr(train_args, "model", None)
        if args.model is None:
            raise SystemExit(f"{args.ckpt} stores no training args: pass --model")
        logger.info(f"--model from the checkpoint's training args: {args.model}")
    if model_constructor(args.model) not in MODEL_NAMES:
        raise SystemExit(f"unknown --model '{args.model}'; choose from {MODEL_NAMES}")
    if args.display_name is None:
        args.display_name = DISPLAY_NAMES.get(args.model_name, args.model_name)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", args.model_name):
        raise SystemExit(f"--model_name '{args.model_name}' may only use letters, digits and . _ + - "
                         "(put spaces / LaTeX in --display_name instead)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store_file = osp.join(args.results_dir, args.store)
    ckpt_fp = ckpt_fingerprint(args.ckpt)
    if ckpt_fp.get("missing"):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    settings = eval_settings(args)

    datasets = [d.strip() for d in args.dataset.split("+") if d.strip() and d.strip() != "[None]"]
    with locked_store(store_file, write=False) as store:  # decide what to skip before loading the model
        todo = []
        for ds in datasets:
            name = ds.split("(")[0].strip()
            match = find_matching(store, args.model_name, name, ckpt_fp, settings)
            if match is not None and not args.overwrite:
                logger.info(f"[{args.model_name} / {name}] already evaluated on {match['time']} with the same "
                            "checkpoint + settings -> skipping (use --overwrite to re-run)")
                continue
            todo.append((ds, name))
    if not todo:
        logger.info("Nothing to evaluate.")
        return

    model = load_model(args, device)
    failures = []
    for ds, name in todo:
        try:
            logger.info(f"Evaluating {args.model_name} on {name}")
            loader = get_data_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_mem=True, shuffle=False, drop_last=False)
            try:
                expected = len(loader.dataset)
            except Exception:  # noqa: BLE001
                expected = None
            raw = evaluate_dataset(model, loader, name, args, device)

            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            raw_file = osp.join("raw", args.model_name, f"{name}__{stamp}__{os.getpid()}.pkl")
            _atomic_pickle(raw, osp.join(args.results_dir, raw_file))   # raw first, then the index
            rec = make_record(args.model_name, name, ckpt_fp, settings, raw, raw_file, expected, source="eval",
                              display_name=args.display_name, description=args.description)
            with locked_store(store_file) as store:                      # saved right after EACH dataset
                append_record(store, rec)
            log_record(rec)
        except Exception:  # noqa: BLE001 -- one broken dataset must not cost the others
            failures.append(name)
            logger.error(f"[{args.model_name} / {name}] FAILED:\n{traceback.format_exc()}")
        finally:
            torch.cuda.empty_cache()

    logger.info(f"Results stored in {store_file}. Tables: python print_tables.py {store_file}")
    if failures:
        logger.error(f"Failed datasets (nothing stored for them): {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()