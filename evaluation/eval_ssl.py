#!/usr/bin/env python
"""
Evaluate SSL backbones (with fine-tuned depth + camera heads) on multi-view data.

Tasks
  depth     : Pi^3 / CUT3R depth_evaluation (AbsRel, SqRel, RMSE, LogRMSE, delta1/2/3)
  camera    : Poincar3 relpose metrics (AUC@{3,5,15,20,30}, RRA / RTA @{5,15,20,30}, mAA30), mean over scenes
  pointmap  : Pi3 eval/mv_recon (umeyama Sim(3) + Open3D ICP, Open3D normals, Acc/Comp/NC in GT units, all points)

Outputs (all under --results_dir)
  results.pkl            persistent, append-only store: every (model, dataset) evaluation is kept
                         with checkpoint fingerprint, settings, code hash, validation report and
                         metrics. Updated after EACH dataset (atomic write + .bak + file lock).
  raw/<model>/...pkl     raw per-sequence data (depth per-unit metrics, predicted/GT poses,
                         point-map per-scene metrics) -> metrics can be recomputed without the model

Tables are NOT made here: run print_tables.py on results.pkl (works without torch / GPU).

Usage
  # evaluate (datasets already evaluated with the same checkpoint + settings are skipped)
  #   --ckpt = run folder name under --ckpt_root; the final weights inside it are used
  python eval_ssl.py --model vista_base_downstream --model_name vista --ckpt <vista_run_folder> \
      --dataset "Hypersim_NVS(...)+ScanNet_NVS(...)" --results_dir results/ssl_eval
  python eval_ssl.py --model mum_base_downstream --model_name mum_v2 --ckpt <mum_v2_run_folder> ...
  # Poincar3 (experiments/ffrecon checkpoint; architecture comes from the checkpoint itself)
  python eval_ssl.py --model poincar3 --model_name poincar3 --ckpt /path/to/ffrecon_run/checkpoint.pth ...
  #   --model_kwargs '{"use_ema": false}'  raw weights instead of EMA
  #   --model_kwargs '{"run_path": "..."}' adapter checkpoints: moved backbone pretraining run

  python eval_ssl.py --show --results_dir results/ssl_eval          # what is stored
  python print_tables.py results/ssl_eval/results.pkl --models mae mum mum_v2 muskie muskie_v2 vista
  python eval_ssl.py --recompute --results_dir results/ssl_eval     # re-derive metrics from raw data

LaTeX preamble needs:  \\usepackage{booktabs} \\usepackage{graphicx}

The two places you may need to adapt to your code:
  * load_model() / run_model() -> how your model is built / called and what keys it returns
    (Poincar3: poincar3_recon.load_recon_checkpoint; its forward returns the VGGT-style keys run_model reads)
  * gather_views()  -> batch key names (GT *_extrinsics are world-to-camera)
"""

import argparse
import contextlib
import datetime
import hashlib
import pickle
import shutil
import socket
import sys
import traceback
import json
import logging
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Project
from datasets import get_data_loader  # noqa: E402
from models import (  # noqa: E402,F401  (used through eval(args.model))
    mae_base_downstream, mae_large_downstream,
    mum_base_downstream, mum_large_downstream,
    muskie_base_downstream, muskie_large_downstream,
    vista_base_downstream, vista_large_downstream,
    vista_basev2_downstream, vista_basev3_downstream
)

# --model: any constructor imported above (built through eval(args.model) + state dict), or a model that
# loads itself from its checkpoint (CHECKPOINT_MODELS: architecture + weights both come from the file)
CHECKPOINT_MODELS = {"poincar3"}
MODEL_NAMES = sorted(n for n in dir() if n.endswith("_downstream")) + sorted(CHECKPOINT_MODELS)

# all fine-tuned runs live in folders under this root; --ckpt <folder name> picks the final weights inside
CKPT_ROOT = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints"
CKPT_EXTS = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")

# --model_name -> name shown in tables (used when --display_name is not given; raw LaTeX allowed)
DISPLAY_NAMES = {
    "mae": "MAE",
    "mum": "MuM",
    "mum_v2": "MuM-v2",
    "muskie": "Muskie",
    "muskie_v2": "Muskie-v2",
    "vista": "Vista",
    "vista_force": "Vista-Force",
    "poincar3": "Poincar3",
}


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("eval_ssl")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("SSL backbone evaluation: depth / camera / point map")
    # model
    p.add_argument("--model", type=str,
                   help="Constructor to build, e.g. vista_base_downstream, mum_large_downstream; or poincar3 "
                        "(experiments/ffrecon checkpoint, end-to-end or adapter)")
    p.add_argument("--model_name", type=str,
                   help="Unique key in the results store, e.g. mum, mum_v2, vista (letters, digits, . _ + - only). "
                        "Default: the --ckpt run folder name")
    p.add_argument("--display_name", type=str, default=None,
                   help="Name shown in tables (raw LaTeX allowed). Default: DISPLAY_NAMES[--model_name], "
                        "else the key. print_tables.py --rename overrides it")
    p.add_argument("--description", type=str, default="",
                   help='Free-text note stored with the record, e.g. "official weights" / "retrained on our data"')
    p.add_argument("--model_kwargs", type=str, default="{}",
                   help="JSON dict of constructor kwargs (poincar3: use_ema, run_path)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Run folder name under --ckpt_root (e.g. mum_v2_probe), a folder path, or a weights file. "
                        "For a folder, the final weights inside it are used (final/last file, else highest step)")
    p.add_argument("--ckpt_root", type=str, default=CKPT_ROOT, help="Folder that holds all run folders")
    p.add_argument("--amp", choices=list(AMP), default="bf16")
    # data
    p.add_argument("--dataset", type=str, default="", help="'+'-separated dataset strings for get_data_loader")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--views", choices=["all", "source", "target"], default="all",
                   help="Which views of each sample are fed to the model and evaluated")
    p.add_argument("--max_batches", type=int, default=-1, help="Debug: stop after N batches per dataset")
    # eval settings
    p.add_argument("--tasks", type=str, default="depth,camera,pointmap")
    p.add_argument("--depth_align", choices=["per_seq", "per_view"], default="per_seq",
                   help="per_seq: one alignment per sequence (Pi^3 video depth); per_view: one per image (monodepth)")
    p.add_argument("--depth_align_method", choices=["scale", "scale_shift", "lstsq", "median", "metric"],
                   default="scale", help="Pi^3/CUT3R alignment: scale (IRLS), scale_shift (L1), lstsq, median, metric")
    p.add_argument("--depth_lr", type=float, default=1e-4,
                   help="Adam lr for scale_shift alignment (Pi^3 default 1e-4; 1e-3 for NYU)")
    p.add_argument("--depth_max_iters", type=int, default=1000)
    p.add_argument("--depth_max", type=float, default=None,
                   help="Ignore GT depth >= this and clip predictions to it (Pi^3 uses 70 for Sintel/Bonn)")
    p.add_argument("--pose_group", choices=["scene", "sequence"], default="scene",
                   help="Pose metrics are averaged over scenes (Poincar3). 'scene' groups by batch "
                        "'scene_name'/'seq_name' when present; 'sequence' treats each sample as a scene")
    p.add_argument("--pm_max_sequences", type=int, default=None,
                   help="Score point maps on the first N sequences only "
                        "(faster). Depth and camera still use every sequence. Default: all")
    p.add_argument("--pm_icp_threshold", type=float, default=0.1,
                   help="Pi3 ICP threshold, in GT units (0.1 for metre datasets; Pi3 uses 100 for DTU's "
                        "millimetres). Acc/Comp are reported in GT units")
    p.add_argument("--seed", type=int, default=0)
    # output
    p.add_argument("--results_dir", type=str, default="results/ssl_eval")
    p.add_argument("--store", type=str, default="results.pkl", help="Results store file inside --results_dir")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-evaluate even if a complete record with the same checkpoint + settings exists "
                        "(the old record is kept in the history, never deleted)")
    p.add_argument("--show", action="store_true", help="Print what is in the store and exit")
    p.add_argument("--recompute", action="store_true",
                   help="Re-derive metrics from saved raw data with the current code (no model needed); "
                        "restrict with --model_name / --dataset")
    return p.parse_args()


# =============================================================================
def resolve_ckpt(ckpt, root):
    """--ckpt -> weights file. Accepts a file, folder, or folder name under root.

    In a folder, priority:
      1. best
      2. final
      3. last/latest
      4. highest step/epoch number
      5. newest file

    Searches the folder itself first, then subfolders if necessary.
    """
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
        raise SystemExit(
            f"--ckpt '{ckpt}' is neither a file nor a folder "
            f"(also looked in {root})"
        )

    files = [
        osp.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(CKPT_EXTS)
    ]

    if not files:
        files = [
            osp.join(d, f)
            for d, _, fs in os.walk(folder)
            for f in fs
            if f.endswith(CKPT_EXTS)
        ]

    if not files:
        raise SystemExit(
            f"no weights ({', '.join(CKPT_EXTS)}) found in {folder}"
        )

    def stem(f):
        return osp.splitext(osp.basename(f))[0].lower()

    # Prefer explicitly named best checkpoints.
    best = [f for f in files if re.search(r"(^|[-_.])best([-_.]|$)", stem(f))]

    if best:
        choice = max(best, key=os.path.getmtime)
        why = "named best"

    else:
        named = [
            f for f in files
            if re.search(r"(^|[-_.])(final|last|latest)([-_.]|$)", stem(f))
        ]

        if named:
            named.sort(
                key=lambda f: (
                    "final" not in stem(f),
                    -os.path.getmtime(f),
                )
            )
            choice = named[0]
            why = "named final/last"

        else:
            numbered = [
                (int(re.findall(r"\d+", stem(f))[-1]), f)
                for f in files
                if re.findall(r"\d+", stem(f))
            ]

            if numbered:
                choice = max(numbered)[1]
                why = "highest step/epoch"
            else:
                choice = max(files, key=os.path.getmtime)
                why = "newest file"

    others = sorted(
        osp.relpath(f, folder) for f in files if f != choice
    )

    logger.info(
        f"--ckpt {ckpt} -> {choice} ({why})"
        + (
            f"; other weights there: {others[:6]}"
            f"{' ...' if len(others) > 6 else ''}"
            if others else ""
        )
    )

    return osp.abspath(choice)

def ckpt_run_name(ckpt, root):
    """Default --model_name: the run folder name (or the file's parent folder)."""
    for cand in (ckpt, osp.join(root, ckpt)):
        if osp.isdir(cand):
            return osp.basename(osp.normpath(cand))
    return osp.basename(osp.dirname(osp.abspath(ckpt)))


def load_model(args, device):
    kwargs = json.loads(args.model_kwargs)
    if args.model == "poincar3":                                # architecture + weights from the checkpoint
        from models.poincar3.benchmarks.ffrecon.inference import load_recon_checkpoint
        input_size = kwargs.pop("input_size", None)            # eval-script option, not a loader kwarg
        model = load_recon_checkpoint(args.ckpt, **kwargs)      # use_ema / run_path from --model_kwargs
        model.input_size = input_size                           # e.g. 256: resize inputs for Poincar3
        logger.info(f"Loaded {type(model).__name__} from {args.ckpt} (kwargs {kwargs}, input_size {input_size})")
        return model.eval()

    model = eval(args.model)(**kwargs)                          # e.g. vista_base_downstream(**kwargs)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for k in ("model", "state_dict", "model_state_dict"):
                if k in ckpt:
                    state = ckpt[k]
                    break
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded {args.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            logger.warning(f"Missing (first 10): {missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected (first 10): {unexpected[:10]}")
    return model.to(device).eval()


def run_model(model, images, amp_dtype):
    """
    images: (B, S, 3, H, W) in [0, 1].
    Returns dict:
        depth : (B, S, h, w) float
        w2c   : (B, S, 4, 4) predicted world-to-camera (first view = world) or None
        K     : (B, S, 3, 3) predicted intrinsics at (H, W) or None
    Our downstream models: VGGT-style outputs 'depth' (B,S,H,W[,1]) and 'pose_enc' (B,S,9),
    or 'extrinsic' (B,S,3,4) w2c + 'intrinsic' (B,S,3,3).
    Poincar3 ReconModel: infer() on one sequence at a time, returns a ReconPrediction.
    """
    # ---------------------------------------------------------------
    # Poincar3 ReconModel: infer() takes one sequence (N, 3, H, W)
    # ---------------------------------------------------------------
    if hasattr(model, "infer"):
        B, S, _, H, W = images.shape
        size = getattr(model, "input_size", None)
        run_imgs = images
        if size is not None and (H, W) != (size, size):
            run_imgs = F.interpolate(images.flatten(0, 1), size=(size, size), mode="bilinear",
                                     align_corners=False).reshape(B, S, 3, size, size)
        rh, rw = run_imgs.shape[-2:]

        depths, w2cs, Ks = [], [], []
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            for b in range(B):
                pred = model.infer(run_imgs[b].clamp(0, 1))
                d = pred.depth.float()
                if d.dim() == 4:                                # (S, h, w, 1) -> (S, h, w)
                    d = d[..., 0]
                depths.append(d)
                w2cs.append(to_4x4(pred.extrinsics.float()))   # (S, 4, 4) cam-from-world (VGGT convention)
                Ks.append(scale_K(pred.intrinsics.float(), (rh, rw), (H, W)))   # back to input resolution
        return {"depth": torch.stack(depths), "w2c": torch.stack(w2cs), "K": torch.stack(Ks)}

    # ---------------------------------------------------------------
    # Our downstream models: forward returns a VGGT-style dict
    # ---------------------------------------------------------------
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        out = model(images)
    if isinstance(out, (tuple, list)):  # e.g. VGGT_Depth_vae returns (_, _, pred_dict)
        out = out[-1]

    depth = out["depth"].float()
    if depth.dim() == 5:
        depth = depth[..., 0]
    res = {"depth": depth, "w2c": None, "K": None}

    if "pose_enc" in out:
        from models.utils.pose_enc import pose_encoding_to_extri_intri

        H, W = images.shape[-2:]
        extri, intri = pose_encoding_to_extri_intri(out["pose_enc"].float(), image_size_hw=(H, W))
        res["w2c"] = to_4x4(extri.float())
        res["K"] = intri.float()
    elif "extrinsic" in out:                    # cameras decoded by the model itself
        res["w2c"] = to_4x4(out["extrinsic"].float())
        res["K"] = out["intrinsic"].float() if out.get("intrinsic") is not None else None
    return res
# =============================================================================
# Batch handling
# =============================================================================
_warned = set()


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


def gather_views(batch, views):
    """
    Concatenate views in the same order as probe training: target views first, then source
    (view 0 = reference frame for VGGT-style heads). GT poses: '*_extrinsics', world-to-camera.
    """
    prefixes = {"all": ["t_", "s_"], "source": ["s_"], "target": ["t_"]}[views]
    imgs, poses, Ks, depths, masks = [], [], [], [], []
    for p in prefixes:
        img = _tensor(batch[f"{p}images"]).float()
        B, S, _, H, W = img.shape
        imgs.append(img)
        poses.append(to_4x4(_tensor(batch[f"{p}extrinsics"]).float()))
        Ks.append(_tensor(batch[f"{p}intrinsics"]).float())

        if f"{p}depths" in batch:
            d = _tensor(batch[f"{p}depths"]).float()
            d = d.reshape(B, S, *d.shape[-2:])
            if f"{p}depth_masks" in batch:
                m = _tensor(batch[f"{p}depth_masks"]).reshape(B, S, *d.shape[-2:]).bool()
            else:
                m = d > 0
        else:
            if p not in _warned:
                logger.warning(f"No '{p}depths' in batch: these views are used for camera eval only")
                _warned.add(p)
            d = torch.zeros(B, S, H, W)
            m = torch.zeros(B, S, H, W, dtype=torch.bool)
        depths.append(d)
        masks.append(m)

    w2c = torch.cat(poses, 1)                      # GT extrinsics are world-to-camera
    return {
        "images": torch.cat(imgs, 1),
        "c2w": torch.linalg.inv(w2c),
        "w2c": w2c,
        "K": torch.cat(Ks, 1),
        "depth": torch.cat(depths, 1),
        "mask": torch.cat(masks, 1),
    }


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
# Camera
# =============================================================================
# --- Copied from Poincar3 relpose benchmark (follows VGGT evaluation/test_relpose.py) -----------
# Only addition: `mat_to_quat` is imported from VGGT (Poincar3 imports its own copy from
# poincar3.heads.rotation); a local fallback is used if vggt is not importable.
try:
    from models.utils.rotation import mat_to_quat
except ImportError:  # fallback: robust matrix -> quaternion (Shepperd); sign is irrelevant below
    def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
        m = matrix.reshape(-1, 3, 3)
        m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
        q_abs = torch.sqrt(torch.clamp(torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                                                    1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1), min=0))
        cands = torch.stack([
            torch.stack([q_abs[:, 0] ** 2, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
            torch.stack([m[:, 2, 1] - m[:, 1, 2], q_abs[:, 1] ** 2, m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0]], -1),
            torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] + m[:, 0, 1], q_abs[:, 2] ** 2, m[:, 2, 1] + m[:, 1, 2]], -1),
            torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 2, 1] + m[:, 1, 2], q_abs[:, 3] ** 2], -1),
        ], 1) / (2.0 * q_abs[..., None].clamp(min=0.1))
        best = q_abs.argmax(-1)
        q = cands[torch.arange(len(m)), best]
        return (q / q.norm(dim=-1, keepdim=True)).reshape(*matrix.shape[:-2], 4)


def build_pair_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    i1, i2 = torch.combinations(torch.arange(n), 2, with_replacement=False).unbind(-1)
    return i1, i2


def rotation_angle(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def compare_translation_by_angle(
    t_gt: torch.Tensor, t: torch.Tensor, eps: float = 1e-15, default_err: float = 1e6
) -> torch.Tensor:
    t = t / (torch.norm(t, dim=1, keepdim=True) + eps)
    t_gt = t_gt / (torch.norm(t_gt, dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt: torch.Tensor, tvec_pred: torch.Tensor, ambiguity: bool = True) -> torch.Tensor:
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi
    if ambiguity:  # essential-matrix-style translation-direction sign ambiguity
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    return rel_tangle_deg


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    R, T = se3[:, :3, :3], se3[:, :3, 3:]
    R_t = R.transpose(1, 2)
    top_right = -torch.bmm(R_t, T)
    inv = torch.eye(4, device=se3.device, dtype=se3.dtype)[None].repeat(len(R), 1, 1)
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = top_right
    return inv


def se3_to_relative_pose_error(
    pred_se3: torch.Tensor, gt_se3: torch.Tensor, num_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """pred_se3/gt_se3: `(N, 4, 4)`, cam-from-world (w2c). Returns per-pair
    `(rel_rangle_deg, rel_tangle_deg)` over every `C(N, 2)` frame pair."""
    i1, i2 = build_pair_index(num_frames)
    relative_gt = gt_se3[i1].bmm(closed_form_inverse_se3(gt_se3[i2]))
    relative_pred = pred_se3[i1].bmm(closed_form_inverse_se3(pred_se3[i2]))
    rel_rangle_deg = rotation_angle(relative_gt[:, :3, :3], relative_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_gt[:, :3, 3], relative_pred[:, :3, 3])
    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30) -> tuple[float, np.ndarray]:
    """AUC of the cumulative-accuracy curve over `max(r_error, t_error)`,
    binned at 1-degree resolution up to `max_threshold`."""
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(float) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized_histogram))), normalized_histogram


def calculate_rra_rta_maa(r_error: np.ndarray, t_error: np.ndarray,
                          thresholds=(5, 15, 20, 30)) -> dict[str, float]:
    """`RRA@tau`/`RTA@tau` (fraction of pairs under a tau-degree rotation/
    translation error) for every tau in `thresholds`, and `mAA30` (mean of the pointwise-min
    rotation/translation accuracy curves, thresholds 0-30 degrees).
    (Poincar3 reports tau = 15 only; the other thresholds use the identical formula.)"""
    num_pairs = len(r_error)
    if num_pairs == 0:
        out = {f"{m}@{t}": 0.0 for t in thresholds for m in ("RRA", "RTA")}
        out["mAA30"] = 0.0
        return out

    out = {}
    for tau in thresholds:
        out[f"RRA@{tau}"] = float(np.mean(r_error < tau))
        out[f"RTA@{tau}"] = float(np.mean(t_error < tau))

    max_threshold = 30
    bins = np.arange(max_threshold + 1)
    r_hist, _ = np.histogram(r_error, bins=bins)
    r_curve = np.cumsum(r_hist.astype(float) / num_pairs)
    t_hist, _ = np.histogram(t_error, bins=bins)
    t_curve = np.cumsum(t_hist.astype(float) / num_pairs)
    out["mAA30"] = float(np.mean(np.minimum(r_curve, t_curve)))
    return out

# --- end of Poincar3 code ------------------------------------------------------------------------


# thresholds (degrees) of the pose metrics
AUC_THRESHOLDS = (30, 20, 15, 5, 3)          # AUC of the max(rot, trans) error curve up to tau
RA_THRESHOLDS = (5, 15, 20, 30)              # RRA@tau / RTA@tau

POSE_KEYS = {f"AUC@{t}": f"auc{t}" for t in AUC_THRESHOLDS}
POSE_KEYS.update({f"{m}@{t}": f"{m.lower()}{t}" for t in RA_THRESHOLDS for m in ("RRA", "RTA")})
POSE_KEYS["mAA30"] = "maa30"


def batch_scene_names(batch, B):
    """Scene id per sample from the batch ('scene_name' / 'scene' / 'seq_name'), or None."""
    for k in ("scene_name", "scene", "seq_name"):
        if k in batch:
            names = batch[k]
            names = list(names) if isinstance(names, (list, tuple)) else [names] * B
            if len(names) == B:
                return [str(n) for n in names]
    return [None] * B


def camera_metrics(pose_seqs, group):
    """Poincar3 relpose metrics. Each sequence: pairwise errors from (S,4,4) w2c poses. Errors are pooled
    per scene, every scene gets AUC@{30,20,15,5,3} + RRA / RTA @{5,15,20,30} + mAA30, and the result is the
    mean across scenes."""
    scene_r, scene_t = defaultdict(list), defaultdict(list)
    for rec in pose_seqs:
        scene = rec["scene"] if (group == "scene" and rec["scene"] is not None) else f"seq{rec['seq_id']}"
        pred_se3 = torch.from_numpy(rec["pred_w2c"]).double()
        gt_se3 = torch.from_numpy(rec["gt_w2c"]).double()
        rel_r, rel_t = se3_to_relative_pose_error(pred_se3, gt_se3, pred_se3.shape[0])
        scene_r[scene].append(rel_r.numpy())
        scene_t[scene].append(rel_t.numpy())

    per_scene = []
    for scene in scene_r:
        r_errors, t_errors = np.concatenate(scene_r[scene]), np.concatenate(scene_t[scene])
        m = {f"AUC@{t}": calculate_auc_np(r_errors, t_errors, max_threshold=t)[0] for t in AUC_THRESHOLDS}
        m.update(calculate_rra_rta_maa(r_errors, t_errors, RA_THRESHOLDS))
        per_scene.append(m)
    if not per_scene:
        return {}

    out = {POSE_KEYS[k]: float(np.mean([m[k] for m in per_scene])) for k in per_scene[0]}
    all_t = np.concatenate([np.concatenate(t) for t in scene_t.values()])
    out["num_scenes"] = len(per_scene)
    out["num_pairs"] = int(len(all_t))
    out["degenerate_pairs"] = int((all_t > 1e3).sum())   # Poincar3 default_err (NaN/Inf translation)
    return out


# =============================================================================
# Point map
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


try:  # Poincar3's `.geometry.unproject_depth_map_to_point_map` (VGGT's)
    from models.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    def unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam):
        """Fallback with VGGT's semantics: depth (S,H,W[,1]), w2c extrinsics (S,3,4), K (S,3,3) -> (S,H,W,3)."""
        depth = torch.as_tensor(np.asarray(depth_map), dtype=torch.float64)
        depth = depth[..., 0] if depth.dim() == 4 else depth
        w2c = to_4x4(torch.as_tensor(np.asarray(extrinsics_cam), dtype=torch.float64))
        K = torch.as_tensor(np.asarray(intrinsics_cam), dtype=torch.float64)
        return unproject(depth, K, torch.linalg.inv(w2c)).numpy()


def pointmap_metrics(pred_depth, pred_K, pred_w2c, gt_depth, gt_K, gt_w2c, valid, icp_threshold):
    """
    One sequence, exactly as Pi3 eval/mv_recon (steps 5-10), on all valid points:
      pred points = unproject(pred depth, pred w2c, pred K)   GT points = unproject(GT depth, GT w2c, GT K)
      Umeyama Sim(3) -> keep valid points -> Open3D point-to-point ICP -> Open3D normals -> Acc / Comp / NC.
    Inputs (torch, one sequence): depth (S,H,W), K (S,3,3), w2c (S,4,4), valid (S,H,W) GT validity mask.
    """
    import open3d as o3d

    # breakpoint()

    pred_pts = np.asarray(unproject_depth_map_to_point_map(
        pred_depth[..., None].float().cpu().numpy(), pred_w2c[:, :3, :].float().cpu().numpy(),
        pred_K.float().cpu().numpy()))
    gt_pts = np.asarray(unproject_depth_map_to_point_map(
        gt_depth[..., None].float().cpu().numpy(), gt_w2c[:, :3, :].float().cpu().numpy(),
        gt_K.float().cpu().numpy()))
    assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} != GT shape {gt_pts.shape}"
    # non-finite predictions are dropped only so umeyama cannot fail on NaNs
    valid_mask = (valid & (gt_depth > 0)).cpu().numpy()
    if valid_mask.sum() < 10:
        return None

    # 5. coarse align
    c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
    pred_pts = c * np.einsum("nhwj, ij -> nhwi", pred_pts, R) + t.T

    # 6. filter invalid points
    pred_pts = pred_pts[valid_mask].reshape(-1, 3)
    gt_pts = gt_pts[valid_mask].reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)

    # 8. ICP align refinement
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, icp_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd = pcd.transform(reg_p2p.transformation)

    # 9. estimate normals
    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    pred_normal = np.asarray(pcd.normals)
    gt_normal = np.asarray(pcd_gt.normals)

    # 10. compute metrics
    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    return {
        "acc": float(acc), "acc_med": float(acc_med),
        "comp": float(comp), "comp_med": float(comp_med),
        "nc": float((nc1 + nc2) / 2), "nc_med": float((nc1_med + nc2_med) / 2),
        "nc1": float(nc1), "nc1_med": float(nc1_med),
        "nc2": float(nc2), "nc2_med": float(nc2_med),
    }


# =============================================================================
# Evaluation loop
# =============================================================================
@torch.no_grad()
def evaluate_dataset(model, loader, name, args, device):
    """Runs the model and returns RAW data only; metrics come from summarize_raw()."""
    tasks = set(args.tasks.split(","))
    amp_dtype = AMP[args.amp]
    rng = np.random.default_rng(args.seed)

    raw = {"depth_units": [], "pose_seqs": [], "pointmap_units": [],
           "num_sequences": 0, "num_frames": 0, "nonfinite_pred_frames": 0, "pm_cameras": None}

    for it, batch in enumerate(loader):
        if 0 <= args.max_batches <= it:
            break
        v = gather_views(batch, args.views)
        images = v["images"].to(device, non_blocking=True)
        gt_d, gt_m = v["depth"].to(device), v["mask"].to(device)
        gt_c2w, gt_w2c, gt_K = v["c2w"].to(device), v["w2c"].to(device), v["K"].to(device)
        B, S, _, H, W = images.shape
        dh, dw = gt_d.shape[-2:]

        out = run_model(model, images, amp_dtype)
        pred_d = out["depth"]
        if pred_d.shape[-2:] != (dh, dw):
            pred_d = F.interpolate(pred_d.reshape(B * S, 1, *pred_d.shape[-2:]), size=(dh, dw),
                                   mode="bilinear", align_corners=False).reshape(B, S, dh, dw)
        raw["nonfinite_pred_frames"] += int((~torch.isfinite(pred_d)).flatten(2).any(-1).sum())

        if "depth" in tasks:
            raw["depth_units"].extend(eval_depth_batch(pred_d, gt_d, gt_m, args))

        if "camera" in tasks and out["w2c"] is not None:
            names = batch_scene_names(batch, B)
            for b in range(B):
                raw["pose_seqs"].append({
                    "scene": names[b], "seq_id": raw["num_sequences"] + b,
                    "pred_w2c": out["w2c"][b].double().cpu().numpy(),   # (S, 4, 4) cam-from-world
                    "gt_w2c": gt_w2c[b].double().cpu().numpy(),
                })

        pm_left = None if args.pm_max_sequences is None else args.pm_max_sequences - raw["num_sequences"]
        if "pointmap" in tasks and (pm_left is None or pm_left > 0):
            gt_K_d = scale_K(gt_K, (H, W), (dh, dw))
            if out["w2c"] is not None and out["K"] is not None:
                pr_w2c = out["w2c"]
                pr_K = scale_K(out["K"], (H, W), (dh, dw))
                raw["pm_cameras"] = "pred"
            else:  # no camera head: unproject predicted depth with GT cameras
                pr_w2c, pr_K = gt_w2c, gt_K_d
                raw["pm_cameras"] = "gt"
            for b in range(B if pm_left is None else min(B, pm_left)):
                res = pointmap_metrics(pred_d[b], pr_K[b], pr_w2c[b], gt_d[b], gt_K_d[b], gt_w2c[b], gt_m[b],
                                       args.pm_icp_threshold)
                if res is not None:
                    raw["pointmap_units"].append(res)

        raw["num_sequences"] += B
        raw["num_frames"] += B * S
        if it % 50 == 0:
            msg = f"[{name}] batch {it}/{len(loader)}"
            if raw["depth_units"]:
                d = summarize_depth(raw["depth_units"])
                msg += f"  AbsRel {d['abs_rel']:.4f}  d1 {d['delta1']:.4f}"
            logger.info(msg)
    return raw


def summarize_raw(raw, settings):
    """Raw data -> metrics. Shared by evaluation and --recompute, so both use identical code."""
    metrics = {}
    if raw["depth_units"]:
        metrics["depth"] = summarize_depth(raw["depth_units"])
    if raw["pose_seqs"]:
        metrics["camera"] = camera_metrics(raw["pose_seqs"], settings["pose_group"])
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
    "camera": {k: (0, 1) for k in POSE_KEYS.values()},
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
        if task not in metrics or not metrics[task]:
            errors.append(f"task '{task}' requested but produced no metrics"
                          + (" (model returned no camera prediction?)" if task == "camera" else ""))
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
        warnings.append(f"{raw['nonfinite_pred_frames']}/{raw['num_frames']} frames have NaN/Inf predicted depth")
    d = metrics.get("depth")
    if d:
        expected_units = n_seq if settings["depth_align"] == "per_seq" else raw["num_frames"]
        if d["num_units"] < 0.9 * expected_units:
            warnings.append(f"depth: only {d['num_units']}/{expected_units} units had valid GT")
        if not (d["delta1"] <= d["delta2"] + 1e-9 <= d["delta3"] + 2e-9):
            errors.append("depth: delta1 <= delta2 <= delta3 violated")
    c = metrics.get("camera")
    if c:
        if c.get("degenerate_pairs"):
            warnings.append(f"camera: {c['degenerate_pairs']}/{c['num_pairs']} pairs hit the NaN/Inf default error")
        if c["num_scenes"] < 2:
            warnings.append(f"camera: only {c['num_scenes']} scene(s) -> averaging over scenes is degenerate")
    p = metrics.get("pointmap")
    if p:
        pm_expected = n_seq if settings.get("pm_max_sequences") is None else min(n_seq, settings["pm_max_sequences"])
        if p["num_scenes"] < 0.9 * pm_expected:
            warnings.append(f"pointmap: only {p['num_scenes']}/{pm_expected} scenes had enough valid points")
        if raw["pm_cameras"] == "gt" and "camera" in tasks:
            warnings.append("pointmap used GT cameras (no predicted cameras)")
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
        "tasks": args.tasks, "views": args.views, "amp": args.amp,
        "model_class": args.model,
        "model_kwargs": args.model_kwargs, "seed": args.seed,
        "depth_align": args.depth_align, "depth_align_method": args.depth_align_method,
        "depth_max": args.depth_max, "depth_lr": args.depth_lr, "depth_max_iters": args.depth_max_iters,
        "pose_group": args.pose_group,
        "pm_protocol": "pi3_mv_recon", "pm_icp_threshold": args.pm_icp_threshold,
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
        parts.append(f"AbsRel {m['depth']['abs_rel']:.4f} d1 {m['depth']['delta1']:.3f}")
    if "camera" in m:
        c = m["camera"]
        ra = " ".join(f"{c.get(f'rra{t}', float('nan')):.3f}/{c.get(f'rta{t}', float('nan')):.3f}"
                      for t in RA_THRESHOLDS)
        parts.append(f"AUC@30 {c['auc30']:.3f} AUC@20 {c.get('auc20', float('nan')):.3f}  "
                     f"RRA/RTA@{{{','.join(map(str, RA_THRESHOLDS))}}} {ra}")
    if "pointmap" in m:
        parts.append(f"Acc {m['pointmap']['acc']:.4f} Comp {m['pointmap']['comp']:.4f} NC {m['pointmap']['nc']:.3f}")
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
    print(f"{'model':<12} {'dataset':<20} {'status':<10} {'seqs':>11} {'AbsRel':>7} {'AUC@30':>7} "
          f"{'Acc':>7} {'hist':>4} {'when':<19} flags")
    for model in sorted(runs):
        for ds in sorted(runs[model]):
            hist = runs[model][ds]
            used = latest_record(hist)                    # the record the tables use
            rec = used if used is not None else hist[-1]
            m = rec["metrics"]
            f = lambda t, k: f"{m[t][k]:.4f}" if t in m and k in m[t] else "--"  # noqa: E731
            seqs = f"{rec['num_sequences']}/{rec['expected_sequences'] if rec['expected_sequences'] is not None else '?'}"
            flags = rec["errors"] + rec["warnings"]
            if used is not None and hist[-1] is not used:
                flags.append(f"newer run on {hist[-1]['time']} is flagged and ignored: "
                             + "; ".join(hist[-1]["errors"]))
            flags = "; ".join(flags)
            print(f"{model:<12} {ds:<20} {'ok' if rec['complete'] else 'FLAGGED':<10} {seqs:>11} "
                  f"{f('depth', 'abs_rel'):>7} {f('camera', 'auc30'):>7} {f('pointmap', 'acc'):>7} "
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

    if not (args.model and args.ckpt and args.dataset):
        raise SystemExit("--model, --ckpt and --dataset are required unless --show/--recompute")
    if not args.model_name:                       # default key: the run folder name
        args.model_name = ckpt_run_name(args.ckpt, args.ckpt_root)
    args.ckpt = resolve_ckpt(args.ckpt, args.ckpt_root)
    if args.model not in MODEL_NAMES:
        raise SystemExit(f"unknown --model '{args.model}'; choose from {MODEL_NAMES}")
    if args.display_name is None:
        args.display_name = DISPLAY_NAMES.get(args.model_name, args.model_name)
    import re
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