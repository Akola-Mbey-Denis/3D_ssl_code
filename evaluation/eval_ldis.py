#!/usr/bin/env python
"""
Evaluate layered-depth (LDI) probes on multi-view data and report occlusion metrics.

Companion of eval_ssl.py (same workflow: evaluate model by model into one append-only results store,
then make the tables from the store). Models are compared on the SAME views, GT LDIs and GT cameras.

Layer existence
  LDI models return `layered_depth` (B, S, N, H, W) and `layered_depth_stop`, the STOP INDEX per pixel (number of
  surfaces along the ray, 0..N), as logits with a class dim of size N+1 or as an index. Their layers are defined as
  layer k exists  <=>  k < stop. The GT stop index is the number of valid GT layers (as LaRI: stop_gt = sum_k mask_k).
  The predicted stop index is used ONLY for the IoU / Dice metrics. Every depth / distance metric (alignment, depth,
  ray, scene, per-view chamfer) selects the predicted points with the GT layer mask, as LaRI (and depth > 0).

Metrics (layer 0 = visible surface, layers 1.. = hidden surfaces, N <= 4)
  visible depth             (layer 0 only, every model incl. visible-only depth models; pixel-pooled)
    vis_abs_rel, vis_sq_rel, vis_rmse, vis_log_rmse, vis_d1, vis_d2, vis_d3
                            standard depth metrics on the pixels where GT layer 0 exists and the model predicts
                            depth > 0 (after --align). Meaningless for --layer0 gt and gt_l0 (they ARE the GT).
  stop index as layer masks (the only stop-index metrics; mask-only: no depth, no alignment; LaRI Table 6)
                            layer k exists iff stop > k, predicted vs GT, pooled over the pixels with >= 1 valid
                            GT surface (GT stop 0 = missing GT depth, excluded)
    stop_iou_L{k}, stop_dice_L{k}       IoU = TP / (TP + FP + FN), Dice = 2 TP / (2 TP + FP + FN) of layer k
    stop_miou, stop_mdice               mean over the layers present in the GT
    stop_miou_hidden, stop_mdice_hidden mean over the hidden layers (k >= 1); layer 0 is ~1 by construction
  per pixel, per layer      (pixels with any GT surface; pooled over all pixels of the dataset)
    abs_rel_L{k}, d1_L{k}   AbsRel / delta<1.25 of layer k on the pixels where GT layer k exists
    *_hidden                the same pooled over the hidden layers k >= 1
  per ray                   (index-free: the predicted hits of a pixel are matched one-to-one to its GT hits
                             along the ray, |d_pred - d_gt| <= tau, closest pairs first)
    ray_{p,r,f1}_all@tau    precision / recall / F1 of ALL hits (layer 0 + hidden)
    ray_{p,r,f1}_hidden@tau the same for the HIDDEN hits (layers >= 1)
  scene (all views of a sample lifted to 3D with the GT intrinsics + poses, metric, every cloud sampled to --num_eval_pts points as LaRI)
    chamfer                 0.5 * (mean pred->GT + mean GT->pred nearest-neighbour distance), metres;
                            F-scores use strict < tau (as LaRI)
    {p,r,f}_all@tau         point-cloud precision / recall / F-score of the WHOLE LDI (Tanks-and-Temples style)
    occ_recall@tau          share of OCCLUDED GT points (not visible in any input view) with a prediction within tau
    occ_precision@tau       share of predicted HIDDEN points (layers >= 1) with a GT surface within tau
    occ_f1@tau              harmonic mean of the two
    scene metrics are averaged over samples (as Pi3 / CUT3R mv_recon); tau = --taus (2 / 5 / 10 / 20 cm)
  point cloud, LaRI protocol (per view, --num_eval_pts points sampled with replacement from prediction and GT)
    cd_{layers}_{mask}      chamfer 0.5 * (mean pred->GT + mean GT->pred), metres
    fscore@tau_{layers}_{mask}  F-score (strict < tau, as LaRI)
                            layers = L0 (visible), L1, L2, L3 (each hidden layer), hidden (L1-L3), all (L0-L3)
                            mask   = gtmask: predicted points where the GT layer mask says the layer exists
                                             (as LaRI; the stop index is not used)
  stop index as masks       (LaRI Table 6; from the stop confusion, layer k exists iff stop > k)
    stop_iou_L{k}, stop_miou, stop_mdice, stop_miou_hidden

Layer-0 modes (--layer0)
  pred   the model's own layer 0 (default)
  gt     ORACLE layer 0: layer 0 is replaced by the GT visible depth (and its GT mask) before every metric, so the
         hidden-layer / occlusion numbers measure only how well the model completes the hidden layers
Baseline (--model gt_l0)
  no model: layer 0 = GT visible depth, no hidden layers. Its numbers are the "perfect visible depth, no amodal
  reasoning" reference (hidden F1 0, occluded recall only from lucky hits).

Visible-only models (no `layered_depth` output, e.g. a depth / point-map model) are evaluated as a single-layer
LDI: layer 0 = their depth where > 0, no hidden layers.

Usage
  # 1. evaluate (one call per model; datasets already evaluated with the same checkpoint + settings are skipped)
  python eval_ldi.py --model "mum_base_ldi(num_layers=4)" --model_name mum --ckpt <run> --dataset "..."
  python eval_ldi.py --model "mum_base_ldi(num_layers=4)" --model_name mum_oracleL0 --layer0 gt --ckpt <run> --dataset "..."
  python eval_ldi.py --model gt_l0 --model_name gt_l0 --dataset "..."          # visible-only reference
  #   --align          median_view (default): one scale per view; median: one per sample; none: metric.
  #                    scale = median(GT / pred) on the model's own layer 0, applied to all layers of that view /
  #                    sample. Mask-only metrics (occupancy, stop index, existence F1) ignore it.
  #   --overwrite      re-evaluate records made before the visible-depth metrics existed

  # 2. report (no GPU needed): Markdown to stdout + ldi_tables.md / ldi_tables.tex / ldi_results.csv
  python eval_ldi.py --report --models gt_l0 mum muskie vista poincar3

  # 3. prune the store (no GPU needed): drop old records, keep the newest --keep per (model, dataset)
  #    the latest COMPLETE record is always kept; a timestamped copy of the store is written first
  python eval_ldi.py --prune                            # every model / dataset, keep 1
  python eval_ldi.py --prune --models mum vista --keep 2

The places you may need to adapt:
  * run_model()      -> how a model is called / what it returns (LDI probes: layered_depth + layered_depth_stop)
  * prepare_batch()  -> batch key names (GT extrinsics are world-to-camera; any common world frame works)
"""

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import os.path as osp
import pickle
import re
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

try:                                    # fast exact nearest neighbours (LaRI uses pytorch3d too); optional
    from pytorch3d.ops import knn_points
except ImportError:
    knn_points = None

try:
    import fcntl
except ImportError:          # not on Linux: no file lock
    fcntl = None

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("eval_ldi")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CKPT_ROOT = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints"
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
STORE_SCHEMA = 2
try:
    CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]
except OSError:
    CODE_SHA = "unknown"

# --model_name -> name in the tables
DISPLAY_NAMES = {"mae": "MAE", "mum": "MuM", "muskie": "Muskie", "vista": "Vista", "vistav2": "Vista-v2",
                 "poincar3": "Poincar3", "gt_l0": "GT layer 0 only"}
DEFAULT_REPORT_MODELS = ["gt_l0", "mum", "muskie", "poincar3"]

from models import (  # noqa: E402,F401  (used through eval(args.model))
vista_base_ldi, mum_base_ldi, mae_base_ldi
)
# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("LDI evaluation")
    # model
    p.add_argument("--model", type=str,
                   help="constructor call from `models`, e.g. \"mum_base_ldi(num_layers=4)\", or gt_l0 (no model)")
    p.add_argument("--model_name", type=str, help="key in the results store (mum, muskie, poincar3, ...)")
    p.add_argument("--display_name", type=str, default=None, help="name in the tables (default: DISPLAY_NAMES)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="run folder (name under --ckpt_root or path) or weights file; none = constructor weights")
    p.add_argument("--ckpt_root", type=str, default=CKPT_ROOT)
    p.add_argument("--ckpt_file", type=str, default="checkpoint-final.pth", help="weights file inside a run folder")
    p.add_argument("--non_strict", action="store_true", help="load_state_dict(strict=False)")
    p.add_argument("--amp", choices=list(AMP), default="bf16")
    # data
    p.add_argument("--dataset", type=str, default="", help="'+'-separated dataset strings for get_data_loader")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=-1, help="debug: stop after N batches per dataset")
    # metrics
    p.add_argument("--num_layers", type=int, default=4, help="GT layers evaluated (layer 0 + hidden)")
    p.add_argument("--layer0", choices=["pred", "gt"], default="pred",
                   help="gt = oracle: replace layer 0 by the GT visible depth before all metrics")
    p.add_argument("--align", choices=["none", "median", "median_view"], default="median_view",
                   help="scale from layer 0 (median GT / pred), applied to all layers: median_view = one scale per "
                        "view (default), median = one per sample, none = metric")
    p.add_argument("--max_depth", type=float, default=16.0, help="predicted / GT depths above this are ignored")
    p.add_argument("--taus", type=str, default="0.02,0.05,0.10,0.20",
                   help="distance thresholds (m) for ray F1, scene F-score and occluded recall / precision / F1")
    p.add_argument("--occ_eps", type=float, default=0.03,
                   help="a GT hidden point closer than this to a visible (layer-0) GT point counts as visible")
    p.add_argument("--num_eval_pts", type=int, default=100000,
                   help="points sampled with replacement from every prediction / GT cloud, as LaRI (LaRI: 100000)")
    # store / report
    p.add_argument("--results_dir", type=str, default="results/ldi_eval")
    p.add_argument("--store", type=str, default="results_ldi.pkl")
    p.add_argument("--overwrite", action="store_true", help="evaluate again even if a matching record exists")
    p.add_argument("--report", action="store_true", help="make the tables from the store and exit")
    p.add_argument("--prune", action="store_true",
                   help="drop old records from the store (keeps the newest --keep per model/dataset) and exit")
    p.add_argument("--keep", type=int, default=1, help="prune: records kept per (model, dataset), >= 1")
    p.add_argument("--models", nargs="+", default=None,
                   help="report: model keys in row order; prune: only these models (default: all)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="report: dataset keys (default: all); prune: only these datasets (default: all)")
    p.add_argument("--allow_incomplete", action="store_true", help="report: also use incomplete records")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def taus(s):
    return [float(t) for t in s.split(",") if t.strip()]


def tau_key(t):
    return f"{round(t * 100):g}cm"


def eval_settings(args):
    """Everything that changes the numbers; records are comparable only if these match."""
    return {
        "model_class": args.model, "amp": args.amp, "num_layers": args.num_layers,
        "layer0": args.layer0, "align": args.align,
        "max_depth": args.max_depth, "taus": args.taus, "occ_eps": args.occ_eps,
        "max_batches": args.max_batches,
        "stop_protocol": "stop_logits_gt_valid",  # stop index from the stop logits; pixels with GT stop 0 excluded
        "align_protocol": "l0_all_layers",   # scale from layer 0, applied to all layers, before oracle L0
        "num_eval_pts": args.num_eval_pts, "pc_protocol": "lari_sampling_gtmask_all",
    }


# =============================================================================
# Model
# =============================================================================
def resolve_ckpt(ckpt, root, fname):
    if ckpt is None:
        return None
    for cand in (ckpt, osp.join(root, ckpt)):
        if osp.isfile(cand):
            return osp.abspath(cand)
        if osp.isdir(cand):
            for name in (fname, "checkpoint-final.pth", "checkpoint-best.pth", "checkpoint-last.pth"):
                if osp.isfile(osp.join(cand, name)):
                    return osp.abspath(osp.join(cand, name))
            raise SystemExit(f"no {fname} / checkpoint-final / -best / -last .pth in {cand}")
    raise SystemExit(f"--ckpt '{ckpt}' is neither a file nor a folder (also looked in {root})")


def ckpt_fingerprint(path):
    if not path:
        return {"path": None}
    st = os.stat(path)
    return {"path": path, "size": st.st_size, "mtime": st.st_mtime}


def load_model(args, ckpt_path, device):
    if args.model == "gt_l0":                                # reference: no network
        return None
    import models                                           # project models: any constructor in it can be used
    model = eval(args.model, vars(models))
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for k in ("model", "state_dict", "model_state_dict"):
                if k in ckpt:
                    state = ckpt[k]
                    break
        state = {k.removeprefix("module."): v for k, v in state.items()}
        msg = model.load_state_dict(state, strict=not args.non_strict)
        logger.info(f"Loaded {ckpt_path}: {msg}")
    return model.to(device).eval()


def _stop_from_output(out, N):
    """layered_depth_stop -> stop index (B, S, H, W) in [0, N].
    Accepts logits (B, S, N+1, H, W) or an index (B, S, H, W)."""
    s = out["layered_depth_stop"]
    if s.dim() == 5:                      # logits
        s = s.argmax(dim=2)
    return s.round().long().clamp(0, N)


@torch.no_grad()
def run_model(model, images, amp_dtype, inp=None):
    """images (B, S, 3, H, W) -> depth (B, S, N, H, W), stop (B, S, H, W) or None, layered (bool)."""
    if model is None:                                        # gt_l0: GT visible depth, no hidden layers
        return inp["ldi"][:, :, :1].clone(), inp["mask"][:, :, 0].long(), False
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        out = model(images=images)
    if "layered_depth" in out:
        depth = out["layered_depth"].float()                 # (B, S, N, H, W)
        return depth, _stop_from_output(out, depth.shape[2]), True
    return out["depth"].float()[:, :, None], None, False    # visible-only model: one layer

# =============================================================================
# Batch
# =============================================================================
def to_4x4(T):
    if T.shape[-2] == 4:
        return T
    pad = torch.zeros(*T.shape[:-2], 1, 4, dtype=T.dtype, device=T.device)
    pad[..., 0, 3] = 1
    return torch.cat([T, pad], dim=-2)


def bs(x):
    """(B, S, ...) tensor, or a collated per-view list [S x (B, ...)] -> (B, S, ...)."""
    return torch.stack(list(x), dim=1) if isinstance(x, (list, tuple)) else x


def prepare_batch(batch, device, num_layers):
    images = bs(batch["images"]).float().to(device)                             # (B, S, 3, H, W)
    ldi = bs(batch["ldis"])[:, :, :num_layers].float().to(device)               # (B, S, L, H, W)
    mask = bs(batch["ldis_mask"])[:, :, :num_layers].bool().to(device) if "ldis_mask" in batch else ldi > 0
    K = bs(batch["intrinsics"]).float().to(device)                              # (B, S, 3, 3)
    w2c = to_4x4(bs(batch["extrinsics"]).float().to(device))                    # (B, S, 4, 4)
    names = batch.get("seq_name", [f"sample{i}" for i in range(images.shape[0])])
    return dict(images=images, ldi=ldi, mask=mask & (ldi > 0), K=K, c2w=torch.linalg.inv(w2c.double()).float(),
                names=list(names))


# =============================================================================
# Metrics
# =============================================================================
def stop_index_from_masks(mask):
    """(..., N, H, W) bool layer masks -> (..., H, W) long stop index, as LaRI: number of valid layers."""
    return mask.long().sum(dim=-3)


def masks_from_stop_index(C, N):
    """(..., H, W) stop index -> (..., N, H, W) bool: layer k exists iff k < C."""
    return torch.arange(N, device=C.device).view(N, 1, 1) < C.unsqueeze(-3)


def new_accumulator(L, tau_list):
    z = lambda: np.zeros(L, dtype=np.float64)                                     # noqa: E731
    return {"abs_rel_sum": z(), "d1_sum": z(), "depth_n": z(),
            # visible depth (layer 0): pixel-pooled sums of the standard depth metrics
            "vis": dict(n=0.0, abs_rel=0.0, sq_rel=0.0, se=0.0, log_se=0.0, d1=0.0, d2=0.0, d3=0.0),
            # per tau: [pred hidden matched, pred hidden, gt hidden matched, gt hidden,
            #           pred all matched,    pred all,    gt all matched,    gt all]
            "ray": {tau_key(t): np.zeros(8) for t in tau_list},
            "stop_conf": None,                                                    # (C, C) confusion, rows = GT
            "scenes": [], "pc": [], "num_samples": 0, "num_layers_pred": None, "stop_source": None}


def greedy_ray_match(pd, pv, gd, gv, tau):
    """One-to-one matching of the hits of each ray, closest pairs first, pairs farther than tau excluded.
    pd/pv (P, N) predicted depths / validity, gd/gv (P, L) GT -> distance of each match (P, N), (P, L), inf if
    unmatched. Pairs are taken in increasing distance, so the matching at any tau' <= tau is exactly the matches
    with distance <= tau': run once at the largest tau."""
    P, N = pd.shape
    L = gd.shape[1]
    dist = (pd[:, :, None] - gd[:, None, :]).abs()                                 # (P, N, L)
    ok = pv[:, :, None] & gv[:, None, :] & (dist <= tau)
    dist = torch.where(ok, dist, torch.full_like(dist, float("inf"))).reshape(P, N * L)
    order = dist.argsort(dim=1)
    inf = float("inf")
    pm = torch.full((P, N), inf, device=pd.device)
    gm = torch.full((P, L), inf, device=pd.device)
    rows = torch.arange(P, device=pd.device)
    for s in range(N * L):
        idx = order[:, s]
        i, j = idx // L, idx % L
        d = dist[rows, idx]
        fin = torch.isfinite(d)
        if not fin.any():
            break                                                                   # sorted: the rest is inf too
        take = fin & torch.isinf(pm[rows, i]) & torch.isinf(gm[rows, j])
        pm[rows[take], i[take]] = d[take]
        gm[rows[take], j[take]] = d[take]
    return pm, gm


def unproject(depth, K, c2w):
    """depth (S, N, H, W), K (S, 3, 3), c2w (S, 4, 4) -> points (S, N, H, W, 3) in the world frame."""
    S, N, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1)                            # (H, W, 3)
    rays = torch.einsum("sij,hwj->shwi", torch.linalg.inv(K), pix)                  # (S, H, W, 3)
    cam = rays[:, None] * depth[..., None]                                          # (S, N, H, W, 3)
    return torch.einsum("sij,snhwj->snhwi", c2w[:, :3, :3], cam) + c2w[:, None, None, None, :3, 3]


def _f(p, r):
    if math.isnan(p) or math.isnan(r):
        return float("nan")
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def nn_dist_torch(src, dst, max_pairs=2 ** 28):
    """Nearest-neighbour distance of each src point to dst on the GPU: pytorch3d knn if installed, else chunked
    matmul cdist in fp32 with TF32 off, on centred points (error well below 1 mm at the distances we threshold)."""
    if knn_points is not None:
        return knn_points(src[None].float(), dst[None].float(), K=1).dists[0, :, 0].clamp_min(0).sqrt()
    c = dst.mean(0)
    src, dst = (src - c).float(), (dst - c).float()
    chunk = max(1, max_pairs // max(len(dst), 1))
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.cat([torch.cdist(s, dst).min(dim=1).values for s in src.split(chunk)])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def sample_pts(p, n):
    """LaRI uniform_sample_3dpts_with_interp: n points drawn uniformly with replacement."""
    return p[torch.randint(len(p), (n,), device=p.device)]


def cd_fscore(p, g, args):
    """LaRI chamfer_and_fscore on clouds already sampled to the same size: unsquared L2 nearest-neighbour
    distances, chamfer = (mean pred->GT + mean GT->pred) / 2, F-score with strict < tau.
    Returns chamfer, {tau: (precision, recall, F)}."""
    d_p, d_g = nn_dist_torch(p, g), nn_dist_torch(g, p)
    prf = {}
    for t in taus(args.taus):
        pr, rc = float((d_p < t).float().mean()), float((d_g < t).float().mean())
        prf[tau_key(t)] = (pr, rc, _f(pr, rc))
    return float(0.5 * (d_p.mean() + d_g.mean())), d_p, d_g, prf


def scene_metrics(pred_all, pred_hidden, gt_all, gt_visible, args):
    """All views of one sample fused in 3D (torch, metres). Every cloud is sampled to --num_eval_pts points with
    replacement, as LaRI (no voxels) -> chamfer, whole-LDI F-score, occluded-geometry recall / precision / F1."""
    n = args.num_eval_pts
    out = {"num_gt": len(gt_all), "num_pred": len(pred_all), "num_pred_hidden": len(pred_hidden)}
    nan_all = {f"{w}_all@{tau_key(t)}": float("nan") for t in taus(args.taus) for w in ("p", "r", "f")}
    if len(gt_all) == 0:
        return out | {"chamfer": float("nan"), "num_occluded": 0} | nan_all
    G = sample_pts(gt_all, n)
    if len(pred_all) == 0:
        out |= {"chamfer": float("nan")} | {k: 0.0 for k in nan_all}
        P = None
    else:
        P = sample_pts(pred_all, n)
        out["chamfer"], _, _, prf = cd_fscore(P, G, args)
        for k, (pr, rc, f) in prf.items():
            out[f"p_all@{k}"], out[f"r_all@{k}"], out[f"f_all@{k}"] = pr, rc, f
    # occluded GT = GT points farther than occ_eps from every visible GT point (full-resolution visible cloud)
    far = nn_dist_torch(G, gt_visible) > args.occ_eps if len(gt_visible) else torch.ones_like(G[:, 0], dtype=torch.bool)
    occ = G[far]
    out["num_occluded"] = len(occ)
    occ = sample_pts(occ, n) if len(occ) else occ
    Ph = sample_pts(pred_hidden, n) if len(pred_hidden) else pred_hidden
    d_occ = nn_dist_torch(occ, P) if (len(occ) and P is not None) else None
    d_hid = nn_dist_torch(Ph, G) if len(Ph) else None
    for t in taus(args.taus):
        k = tau_key(t)
        r = float("nan") if not len(occ) else 0.0 if d_occ is None else float((d_occ < t).float().mean())
        p = 0.0 if d_hid is None else float((d_hid < t).float().mean())       # no hidden prediction = 0
        out[f"occ_recall@{k}"], out[f"occ_precision@{k}"], out[f"occ_f1@{k}"] = r, p, _f(p, r)
    return out


def pc_metrics(pp, pg, pmask, gmask, args):
    """LaRI point-cloud metrics of one view. pp (N, H, W, 3), pg (L, H, W, 3), gmask (L, H, W),
    pmask = {"gtmask": (N, H, W)} predicted-point selection."""
    n = args.num_eval_pts
    out = {}
    subsets = [(f"L{k}", slice(k, k + 1)) for k in range(len(pg))] + [("hidden", slice(1, None)),
                                                                       ("all", slice(None))]
    for sub, sl in subsets:
        g = pg[sl][gmask[sl]]
        if len(g) == 0:                                                             # e.g. no hidden GT in this view
            continue
        g = sample_pts(g, n)
        for src, pm in pmask.items():
            p = pp[sl][pm[sl]]
            if len(p) == 0:                                                         # nothing predicted: F = 0
                out[f"cd_{sub}_{src}"] = float("nan")
                out.update({f"fscore@{tau_key(t)}_{sub}_{src}": 0.0 for t in taus(args.taus)})
                continue
            out[f"cd_{sub}_{src}"], _, _, prf = cd_fscore(sample_pts(p, n), g, args)
            out.update({f"fscore@{k}_{sub}_{src}": f for k, (_, _, f) in prf.items()})
    return out


@torch.no_grad()
def accumulate(acc, depth, stop, inp, args):
    """Add one batch to the accumulator. depth (B, S, N, H, W) at the GT resolution, stop (B, S, H, W) or None."""
    gt, gmask = inp["ldi"], inp["mask"]                                              # (B, S, L, H, W)
    B, S, L, H, W = gt.shape
    N = depth.shape[2]
    acc["num_layers_pred"] = N
    gmask = gmask & (gt <= args.max_depth)

    # ---------------- predicted existence: from the stop index (LDI models), else depth > 0 ----------------
    if stop is not None:
        pexist = masks_from_stop_index(stop, N)
        acc["stop_source"] = "stop"
    else:                                                                           # visible-only model
        pexist = depth > 0
        acc["stop_source"] = "depth"
    pexist = pexist & (depth > 0) & torch.isfinite(depth)                           # max_depth is applied to the GT

    # ---------------- stop index, before any oracle substitution ----------------
    # only pixels with a valid GT surface: GT stop 0 means missing GT depth here, not an empty ray
    stop_gt = stop_index_from_masks(gmask)                                          # (B, S, H, W) in [0, L]
    stop_pred = stop if stop is not None else stop_index_from_masks(pexist)
    C = max(L, N) + 1
    if acc["stop_conf"] is None:
        acc["stop_conf"] = np.zeros((C, C), dtype=np.float64)
    valid = stop_gt > 0
    idx = stop_gt[valid].clamp(0, C - 1) * C + stop_pred[valid].clamp(0, C - 1)
    acc["stop_conf"] += torch.bincount(idx, minlength=C * C).view(C, C).double().cpu().numpy()

    # ---------------- predicted points used by every distance metric: GT layer mask, as LaRI ----------------
    # the stop index is used only by the IoU / Dice above
    gsel = torch.zeros_like(pexist)
    n = min(N, L)
    gsel[:, :, :n] = gmask[:, :, :n]
    pexist = gsel & (depth > 0) & torch.isfinite(depth)

    # ---------------- scale alignment: per view (default) or per sample, shared by all layers ----------------
    # from the model's OWN layer 0, before any oracle substitution, so hidden layers keep their depth relative to the
    # visible surface
    if args.align in ("median", "median_view"):
        for b in range(B):
            for v in (range(S) if args.align == "median_view" else [slice(None)]):
                m = gmask[b, v, 0] & pexist[b, v, 0]
                s = (gt[b, v, 0][m] / depth[b, v, 0][m]).median() if m.any() else 1.0
                depth[b, v] *= s
                acc.setdefault("align_scales", []).append(float(s))

    # ---------------- oracle layer 0 (after alignment: hidden layers stay in the aligned model scale) ----------------
    if args.layer0 == "gt":
        depth = depth.clone()
        pexist = pexist.clone()
        depth[:, :, 0] = gt[:, :, 0]
        pexist[:, :, 0] = gmask[:, :, 0]

    valid_px = gmask[:, :, :1]                                                      # pixels with any GT surface

    # ---------------- visible depth (layer 0): full depth-metric set ----------------
    # pixels where GT layer 0 exists and the model predicts a finite depth > 0 (after alignment)
    d0, g0 = depth[:, :, 0], gt[:, :, 0]
    m0 = gmask[:, :, 0] & (d0 > 0) & torch.isfinite(d0)
    d, g = d0[m0].double(), g0[m0].double()
    if d.numel():
        r = torch.maximum(d / g, g / d)
        v = acc["vis"]
        v["n"] += d.numel()
        v["abs_rel"] += float(((d - g).abs() / g).sum())
        v["sq_rel"] += float(((d - g) ** 2 / g).sum())
        v["se"] += float(((d - g) ** 2).sum())
        v["log_se"] += float(((d.clamp_min(1e-5).log() - g.log()) ** 2).sum())
        v["d1"] += float((r < 1.25).sum())
        v["d2"] += float((r < 1.25 ** 2).sum())
        v["d3"] += float((r < 1.25 ** 3).sum())

    # ---------------- per pixel, per layer ----------------
    for k in range(min(L, N)):
        g = gmask[:, :, k] & valid_px[:, :, 0]
        m = g & (depth[:, :, k] > 0)
        d, gg = depth[:, :, k][m], gt[:, :, k][m]
        acc["abs_rel_sum"][k] += float(((d - gg).abs() / gg).sum())
        acc["d1_sum"][k] += float((torch.maximum(d / gg, gg / d) < 1.25).sum())
        acc["depth_n"][k] += float(m.sum())

    # ---------------- per ray (index-free matching): hidden hits and all hits ----------------
    vp = valid_px[:, :, 0].reshape(-1)
    pd = depth.permute(0, 1, 3, 4, 2).reshape(-1, N)[vp]
    pv = (pexist & valid_px).permute(0, 1, 3, 4, 2).reshape(-1, N)[vp]
    gd = gt.permute(0, 1, 3, 4, 2).reshape(-1, L)[vp]
    gv = gmask.permute(0, 1, 3, 4, 2).reshape(-1, L)[vp]
    pm_d, gm_d = greedy_ray_match(pd, pv, gd, gv, max(taus(args.taus)))
    for t in taus(args.taus):
        pm, gm = pm_d <= t, gm_d <= t
        acc["ray"][tau_key(t)] += np.array([
            float(pm[:, 1:].sum()), float(pv[:, 1:].sum()), float(gm[:, 1:].sum()), float(gv[:, 1:].sum()),
            float(pm.sum()), float(pv.sum()), float(gm.sum()), float(gv.sum())])

    # ---------------- scene: all views of each sample in 3D ----------------
    for b in range(B):
        pts_p = unproject(depth[b], inp["K"][b], inp["c2w"][b])                       # (S, N, H, W, 3)
        pts_g = unproject(gt[b], inp["K"][b], inp["c2w"][b])                          # (S, L, H, W, 3)
        pm_b = pexist[b] & valid_px[b]
        hidden = pm_b.clone()
        hidden[:, 0] = False
        rec = scene_metrics(
            pred_all=pts_p[pm_b], pred_hidden=pts_p[hidden],
            gt_all=pts_g[gmask[b]], gt_visible=pts_g[:, :1][gmask[b][:, :1]],
            args=args)
        rec["name"] = inp["names"][b]
        acc["scenes"].append(rec)
        # LaRI protocol, per view: predicted points selected by the stop index and by the GT mask
        ok = (depth[b] > 0) & torch.isfinite(depth[b])
        for v in range(S):
            pmask = {"gtmask": gmask[b, v, :N] & ok[v]}
            acc["pc"].append(pc_metrics(pts_p[v], pts_g[v], pmask, gmask[b, v], args))
    acc["num_samples"] += B


def stop_mask_scores(conf):
    """LaRI Table 6: IoU / Dice of the layer masks from the stop index (layer k exists iff class > k), from the
    stop confusion (rows = GT). Layers absent from the GT are skipped."""
    conf = np.asarray(conf, dtype=np.float64)
    out, iou, dice = {}, {}, {}
    for k in range(conf.shape[0] - 1):
        tp, fp, fn = conf[k + 1:, k + 1:].sum(), conf[:k + 1, k + 1:].sum(), conf[k + 1:, :k + 1].sum()
        if tp + fn == 0:
            continue
        iou[k], dice[k] = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
        out[f"stop_iou_L{k}"], out[f"stop_dice_L{k}"] = float(iou[k]), float(dice[k])
    for name, ks in (("", list(iou)), ("_hidden", [k for k in iou if k >= 1])):
        if ks:
            out[f"stop_miou{name}"] = float(np.mean([iou[k] for k in ks]))
            out[f"stop_mdice{name}"] = float(np.mean([dice[k] for k in ks]))
    return out


def summarize(acc):
    """Accumulator -> flat metric dict (the table values)."""
    L = len(acc["depth_n"])
    m = {}

    # stop index: IoU / Dice of the layer masks only
    conf = acc["stop_conf"]
    if conf is not None and conf.sum() > 0:
        m.update(stop_mask_scores(conf))

    # visible depth (layer 0)
    v = acc["vis"]
    if v["n"] > 0:
        n = v["n"]
        m.update(vis_abs_rel=v["abs_rel"] / n, vis_sq_rel=v["sq_rel"] / n,
                 vis_rmse=math.sqrt(v["se"] / n), vis_log_rmse=math.sqrt(v["log_se"] / n),
                 vis_d1=v["d1"] / n, vis_d2=v["d2"] / n, vis_d3=v["d3"] / n,
                 vis_num_pixels=int(n))

    for k in range(L):
        n = acc["depth_n"][k]
        m[f"abs_rel_L{k}"] = acc["abs_rel_sum"][k] / n if n > 0 else float("nan")
        m[f"d1_L{k}"] = acc["d1_sum"][k] / n if n > 0 else float("nan")
    h = slice(1, L)
    n = acc["depth_n"][h].sum()
    m["abs_rel_hidden"] = acc["abs_rel_sum"][h].sum() / n if n > 0 else float("nan")
    m["d1_hidden"] = acc["d1_sum"][h].sum() / n if n > 0 else float("nan")

    for k, c in acc["ray"].items():
        for name, (pm, pn, gm, gn) in (("hidden", c[:4]), ("all", c[4:])):
            p = pm / pn if pn > 0 else 0.0
            r = gm / gn if gn > 0 else float("nan")
            m[f"ray_p_{name}@{k}"], m[f"ray_r_{name}@{k}"], m[f"ray_f1_{name}@{k}"] = p, r, _f(p, r)

    keys = [k for k in (acc["scenes"][0] if acc["scenes"] else {}) if k != "name" and not k.startswith("num_")]
    for k in keys:
        vals = np.array([s[k] for s in acc["scenes"]], dtype=np.float64)
        m[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    pc_keys = sorted({k for s in acc["pc"] for k in s})                    # LaRI point clouds: mean over views
    for k in pc_keys:
        vals = np.array([s.get(k, np.nan) for s in acc["pc"]], dtype=np.float64)
        m[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    if acc.get("align_scales"):                                  # how far off the model's metric scale was
        s = np.asarray(acc["align_scales"], dtype=np.float64)
        m["align_scale_median"], m["align_scale_p10"], m["align_scale_p90"] = (
            float(np.median(s)), float(np.percentile(s, 10)), float(np.percentile(s, 90)))
    m["num_samples"] = acc["num_samples"]
    return m


# =============================================================================
# Results store (append-only, file-locked, atomic writes; same idea as eval_ssl.py)
# =============================================================================
def _atomic_pickle(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    if osp.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def _read_store(path):
    for p in (path, path + ".bak"):
        if osp.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:  # noqa: BLE001
                logger.error(f"could not read {p}: {e}")
    if osp.exists(path):
        raise RuntimeError(f"{path} exists but cannot be read; not overwriting it")
    return {"schema": STORE_SCHEMA, "runs": {}}


@contextlib.contextmanager
def locked_store(path, write=True):
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "a") as lf:
        if fcntl is not None:
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            store = _read_store(path)
            yield store
            if write:
                _atomic_pickle(store, path)
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)


def find_matching(store, model, dataset, ckpt_fp, settings):
    for rec in reversed(store["runs"].get(model, {}).get(dataset, [])):
        if rec["complete"] and rec["ckpt"] == ckpt_fp and rec["settings"] == settings:
            return rec
    return None


def prune(args, store_path):
    """Keep the newest --keep records of every (model, dataset) -- and always the latest COMPLETE one, so the
    tables never lose a result. Runs under the store lock; a timestamped copy of the store is written first."""
    if args.keep < 1:
        raise SystemExit("--keep must be >= 1")
    if not osp.exists(store_path):
        raise SystemExit(f"nothing to prune: {store_path} does not exist")
    with locked_store(store_path) as store:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safety = f"{store_path}.before_prune_{stamp}"
        with open(safety, "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
        removed = 0
        for model, per_ds in store["runs"].items():
            if args.models and model not in args.models:
                continue
            for ds, hist in per_ds.items():
                if args.datasets and ds not in args.datasets:
                    continue
                kept = hist[-args.keep:]
                last_complete = next((r for r in reversed(hist) if r["complete"]), None)
                if last_complete is not None and not any(r is last_complete for r in kept):
                    kept = [last_complete] + kept                    # newer runs were all flagged: keep the good one
                if len(kept) < len(hist):
                    logger.info(f"{model} / {ds}: {len(hist)} -> {len(kept)} records")
                removed += len(hist) - len(kept)
                per_ds[ds] = kept
    logger.info(f"pruned {removed} record(s); copy of the store before pruning: {safety}")


def dataset_key(ds):
    """'ScanNet_LDI(split='test', ...)' -> 'ScanNet_LDI/test' (table column / store key)."""
    import re
    name = ds.strip().split("(")[0].split("@")[-1].strip()
    split = re.search(r"split\s*=\s*['\"](\w+)['\"]", ds)
    return f"{name}/{split.group(1)}" if split else name


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_dataset(model, loader, args, device, max_batches=-1):
    amp_dtype = AMP[args.amp]
    acc = new_accumulator(args.num_layers, taus(args.taus))
    layered = None
    t0 = time.time()
    for i, batch in enumerate(loader):
        if 0 <= max_batches <= i:
            break
        inp = prepare_batch(batch, device, args.num_layers)
        depth, stop, layered = run_model(model, inp["images"], amp_dtype, inp)
        H, W = inp["ldi"].shape[-2:]
        B, S, N = depth.shape[:3]
        if depth.shape[-2:] != (H, W):                                                # e.g. a model at another resolution
            depth = F.interpolate(depth.flatten(0, 1), (H, W), mode="nearest").view(B, S, N, H, W)
        if stop is not None and stop.shape[-2:] != (H, W):
            stop = F.interpolate(stop.float().flatten(0, 1)[:, None], (H, W), mode="nearest")[:, 0] \
                .view(B, S, H, W).long()
        accumulate(acc, depth, stop, inp, args)
        if i % 20 == 0:
            logger.info(f"  batch {i}/{len(loader)}  {acc['num_samples']} samples  {time.time() - t0:.0f}s")
    return acc, layered


def main():
    args = parse_args()
    store_path = osp.join(args.results_dir, args.store)
    if args.report:
        return report(args, store_path)
    if args.prune:
        return prune(args, store_path)
    if not (args.model and args.model_name and args.dataset):
        raise SystemExit("--model, --model_name and --dataset are required (or use --report / --prune)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_ckpt(args.ckpt, args.ckpt_root, args.ckpt_file)
    ckpt_fp = ckpt_fingerprint(ckpt_path)
    settings = eval_settings(args)

    from datasets import get_data_loader
    datasets = [d.strip() for d in args.dataset.split("+") if d.strip()]
    todo = []
    with locked_store(store_path, write=False) as store:
        for ds in datasets:
            hit = find_matching(store, args.model_name, dataset_key(ds), ckpt_fp, settings)
            if hit and not args.overwrite:
                logger.info(f"skip {dataset_key(ds)}: already evaluated ({hit['time']}); --overwrite to redo")
            else:
                todo.append(ds)
    if not todo:
        return

    if knn_points is None:
        logger.info("pytorch3d not found: nearest neighbours use chunked cdist (slower; pip install pytorch3d)")
    model = load_model(args, ckpt_path, device)
    for ds in todo:
        key = dataset_key(ds)
        logger.info(f"=== {args.model_name} on {key} (layer0={args.layer0}, align={args.align}) ===")
        loader = get_data_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_mem=True, shuffle=False, drop_last=False)
        acc, layered = evaluate_dataset(model, loader, args, device, args.max_batches)
        metrics = summarize(acc)
        errors = []
        if acc["num_samples"] == 0:
            errors.append("no samples")
        if args.max_batches >= 0:
            errors.append(f"partial run (--max_batches {args.max_batches})")
        for k in ("abs_rel_L0", "chamfer", "vis_abs_rel"):
            if not np.isfinite(metrics.get(k, float("nan"))):
                errors.append(f"{k} is not finite")
        rec = {"model": args.model_name, "display_name": args.display_name, "dataset": key, "dataset_str": ds,
               "time": datetime.datetime.now().isoformat(timespec="seconds"), "ckpt": ckpt_fp,
               "settings": settings, "code_sha": CODE_SHA, "host": socket.gethostname(),
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "layered_model": layered,
               "num_layers_pred": acc["num_layers_pred"], "existence_from": acc["stop_source"],
               "stop_confusion": acc["stop_conf"], "complete": not errors, "errors": errors,
               "metrics": metrics, "per_sample": acc["scenes"]}
        with locked_store(store_path) as store:
            store["runs"].setdefault(args.model_name, {}).setdefault(key, []).append(rec)
        t_mid = tau_key(taus(args.taus)[min(2, len(taus(args.taus)) - 1)])
        logger.info(f"{'COMPLETE' if not errors else 'INCOMPLETE ' + str(errors)}  {key} "
                    f"[existence from {acc['stop_source']}, "
                    f"align scale median {metrics.get('align_scale_median', float('nan')):.3f}]: "
                    f"visible AbsRel {metrics.get('vis_abs_rel', float('nan')):.4f} "
                    f"RMSE {metrics.get('vis_rmse', float('nan')):.4f} "
                    f"d1 {metrics.get('vis_d1', float('nan')):.3f} | "
                    f"stop mIoU {metrics.get('stop_miou', float('nan')):.3f} mDice {metrics.get('stop_mdice', float('nan')):.3f} | "
                    f"AbsRel hidden {metrics['abs_rel_hidden']:.4f} | "
                    f"chamfer {metrics['chamfer']:.4f} m  F@{t_mid} {metrics.get(f'f_all@{t_mid}', float('nan')):.3f}  "
                    f"occF1@{t_mid} {metrics.get(f'occ_f1@{t_mid}', float('nan')):.3f} | "
                    f"stop mIoU hidden {metrics.get('stop_miou_hidden', float('nan')):.3f} | CD gtmask "
                    f"L0 {metrics.get('cd_L0_gtmask', float('nan')):.4f} "
                    f"hidden {metrics.get('cd_hidden_gtmask', float('nan')):.4f} m  "
                    f"({metrics['num_samples']} samples)")
    logger.info(f"store: {store_path}   tables: python {osp.basename(__file__)} --report --results_dir {args.results_dir}")


# =============================================================================
# Report
# =============================================================================
# ONE table: (key, header, better: +1 higher / -1 lower, format, scale, column group)
P = ("{:.1f}", 100)                                                     # shown in %
TABLES = {
    "ldi": ("Layered depth probing. Depth after per-view median scale alignment; hidden = layers 1-3. Occupancy: "
            "IoU / Dice of the hidden-layer masks from the predicted stop index. Chamfer (cm) and F-score: per view, "
            "predicted points selected by the GT layer mask (as LaRI). Occluded geometry: all views fused, GT points "
            "no input view sees, at 10 cm", [
        ("vis_abs_rel", "AbsRel", -1, "{:.3f}", 1, "Visible depth"), ("vis_d1", "δ1", +1, *P, "Visible depth"),
        ("abs_rel_hidden", "AbsRel", -1, "{:.3f}", 1, "Hidden depth"), ("d1_hidden", "δ1", +1, *P, "Hidden depth"),
        ("stop_miou_hidden", "mIoU", +1, *P, "Occupancy"), ("stop_mdice_hidden", "mDice", +1, *P, "Occupancy"),
        ("cd_L0_gtmask", "CD L0", -1, "{:.2f}", 100, "Chamfer / F-score"),
        ("cd_hidden_gtmask", "CD hid.", -1, "{:.2f}", 100, "Chamfer / F-score"),
        ("cd_all_gtmask", "CD all", -1, "{:.2f}", 100, "Chamfer / F-score"),
        ("fscore@10cm_hidden_gtmask", "F hid.", +1, *P, "Chamfer / F-score"),
        ("occ_precision@10cm", "P", +1, *P, "Occluded geometry"), ("occ_recall@10cm", "R", +1, *P, "Occluded geometry"),
        ("occ_f1@10cm", "F1", +1, *P, "Occluded geometry"),
    ]),
}

def _uses_gt_layer0(rec):
    """True for records whose layer 0 IS the GT (gt_l0 reference / --layer0 gt oracle)."""
    return rec["settings"].get("model_class") == "gt_l0" or rec["settings"].get("layer0") == "gt"


def report(args, store_path):
    with locked_store(store_path, write=False) as store:
        runs = store["runs"]
    if not runs:
        raise SystemExit(f"nothing in {store_path}")
    models = args.models or [m for m in DEFAULT_REPORT_MODELS if m in runs] or sorted(runs)
    missing = [m for m in models if m not in runs]
    if missing:
        logger.warning(f"no records for {missing} (stored: {sorted(runs)})")
        models = [m for m in models if m in runs]
    latest = {}
    for m in models:
        for ds, hist in runs[m].items():
            for rec in reversed(hist):
                if rec["complete"] or args.allow_incomplete:
                    latest[(m, ds)] = rec
                    break
    datasets = args.datasets or sorted({ds for (_, ds) in latest})
    settings = {json.dumps(r["settings"] | {"model_class": None, "layer0": None}, sort_keys=True)
                for r in latest.values()}
    if len(settings) > 1:
        logger.warning("the selected records were evaluated with DIFFERENT settings (thresholds / align / sampling)")
    no_vis = sorted({m for (m, _), r in latest.items() if not _uses_gt_layer0(r) and "vis_abs_rel" not in r["metrics"]})
    if no_vis:
        logger.warning(f"no visible-depth metrics for {no_vis}: re-evaluate them with --overwrite")

    def name(m):
        rec = next((r for (mm, _), r in latest.items() if mm == m), None)
        base = (rec or {}).get("display_name") or DISPLAY_NAMES.get(m, m)
        if rec and rec["settings"].get("layer0") == "gt" and m != "gt_l0":
            base += " (GT L0)"
        return base

    md, tex, rows = [], [], []
    for tkey, (title, cols) in TABLES.items():
        # drop columns no selected record has (e.g. stop classes beyond the GT range)
        cols = [c for c in cols if any(np.isfinite(r["metrics"].get(c[0], float("nan"))) for r in latest.values())]
        if not cols:
            continue
        t_models = models
        md.append(f"## {title}\n")
        # LaTeX header: column groups, then metric names
        groups = []
        for c in cols:
            if groups and groups[-1][0] == c[5]:
                groups[-1][1] += 1
            else:
                groups.append([c[5], 1])
        grp, rules, j = [], [], 2
        for g, w in groups:
            grp.append(rf"\multicolumn{{{w}}}{{c}}{{{g}}}")
            rules.append(rf"\cmidrule(lr){{{j}-{j + w - 1}}}")
            j += w
        arrow = lambda c: r"$\downarrow$" if c[2] < 0 else r"$\uparrow$"                # noqa: E731
        tex += [f"% {title}", r"\begin{table*}[t]\centering", r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
                " & " + " & ".join(grp) + r" \\", " ".join(rules),
                "Method & " + " & ".join(re.sub(r"δ(\d)", r"$\\delta_\1$", c[1]) + arrow(c) for c in cols) + r" \\"]

        def value(m, ds, c):
            rec = latest.get((m, ds))
            if rec is None or (c[0].startswith("vis_") and _uses_gt_layer0(rec)):  # layer 0 IS the GT: not scored
                return float("nan")
            return rec["metrics"].get(c[0], float("nan"))

        blocks = datasets + (["Avg."] if len(datasets) > 1 else [])
        for ds in blocks:
            md.append(f"**{ds}**\n")
            md.append("| Model | " + " | ".join(f"{c[5]}: {c[1]}" for c in cols) + " |")
            md.append("|---|" + "---|" * len(cols))
            ds_tex = ds.replace("_", r"\_")
            tex += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{ds_tex}}}}} \\"]
            if ds == "Avg.":                                     # mean over the datasets; needs every dataset
                vals = {m: [float(np.mean(v)) if np.all(np.isfinite(v := [value(m, d, c) for d in datasets]))
                            else float("nan") for c in cols] for m in t_models}
            else:
                vals = {m: [value(m, ds, c) for c in cols] for m in t_models}
            # rank on the PRINTED values: ties share the rank (all best bold, second-best distinct value underlined)
            ranks = []
            for j, c in enumerate(cols):
                shown = {m: float(c[3].format(vals[m][j] * c[4])) for m in t_models if np.isfinite(vals[m][j])}
                levels = sorted({v * c[2] for v in shown.values()}, reverse=True)
                ranks.append({m: levels.index(v * c[2]) for m, v in shown.items()} if len(levels) > 1 else {})
            for m in t_models:
                cells_md, cells_tex = [], []
                for j, c in enumerate(cols):
                    v = vals[m][j]
                    s = c[3].format(v * c[4]) if np.isfinite(v) else "–"
                    r = ranks[j].get(m)
                    cells_md.append(f"**{s}**" if r == 0 else s)
                    cells_tex.append(rf"\textbf{{{s}}}" if r == 0 else rf"\underline{{{s}}}" if r == 1 else s)
                    rows.append({"table": tkey, "dataset": ds, "model": m, "metric": c[0],
                                 "value": v if np.isfinite(v) else ""})
                md.append(f"| {name(m)} | " + " | ".join(cells_md) + " |")
                tex.append(f"{name(m)} & " + " & ".join(cells_tex).replace("_", r"\_") + r" \\")
            md.append("")
        tex += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{title}.}}", r"\end{table*}", ""]

    n = {ds: {m: latest[(m, ds)]["metrics"].get("num_samples") for m in models if (m, ds) in latest} for ds in datasets}
    md.append("Samples per dataset: " + "; ".join(f"{ds}: {v}" for ds, v in n.items()))
    vis_px = {ds: {m: latest[(m, ds)]["metrics"].get("vis_num_pixels") for m in models
                   if (m, ds) in latest and not _uses_gt_layer0(latest[(m, ds)])} for ds in datasets}
    md.append("Visible-depth pixels scored (should match across models): "
              + "; ".join(f"{ds}: {v}" for ds, v in vis_px.items()))
    text = "\n".join(md)
    print(text)
    os.makedirs(args.results_dir, exist_ok=True)
    with open(osp.join(args.results_dir, "ldi_tables.md"), "w") as f:
        f.write(text + "\n")
    with open(osp.join(args.results_dir, "ldi_tables.tex"), "w") as f:
        f.write("% needs \\usepackage{booktabs} \\usepackage{graphicx}\n" + "\n".join(tex))
    with open(osp.join(args.results_dir, "ldi_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "dataset", "model", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"wrote ldi_tables.md / ldi_tables.tex / ldi_results.csv to {args.results_dir}")


if __name__ == "__main__":
    main()