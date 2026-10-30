#!/usr/bin/env python
"""
Evaluate layered-depth (LDI) probes on multi-view data and report occlusion metrics.

Companion of eval_ssl.py (same workflow: evaluate model by model into one append-only results store,
then make the tables from the store). Models are compared on the SAME views, GT LDIs and GT cameras.

Metrics (layer 0 = visible surface, layers 1.. = hidden surfaces, N <= 4)
  per pixel, per layer      (pixels with any GT surface; pooled over all pixels of the dataset)
    exist_f1_L{k}           existence F1 of layer k (pred: sigmoid(exist_k) > --exist_threshold)
    abs_rel_L{k}, d1_L{k}   AbsRel / delta<1.25 of layer k on the pixels where GT layer k exists
    *_hidden                the same pooled over the hidden layers k >= 1
  per ray                   (index-free: the predicted hits of a pixel are matched one-to-one to its GT hits
                             along the ray, |d_pred - d_gt| <= tau, closest pairs first)
    ray_{p,r,f1}_hidden@tau precision / recall / F1 of the HIDDEN hits (layers >= 1), tau = --ray_taus
  scene (all views of a sample lifted to 3D with the GT intrinsics + poses, metric, voxel-downsampled)
    chamfer                 0.5 * (mean pred->GT + mean GT->pred nearest-neighbour distance), metres
    occ_recall@tau          share of OCCLUDED GT points (not visible in any input view) with a prediction within tau
    occ_precision@tau       share of predicted HIDDEN points (layers >= 1) with a GT surface within tau
    occ_f1@tau              harmonic mean of the two; tau = --scene_taus (5 / 10 cm)
    scene metrics are averaged over samples (as Pi3 / CUT3R mv_recon)

Visible-only models (no `layered_depth` output, e.g. a depth / point-map model) are evaluated as a single-layer
LDI: layer 0 = their depth where > 0, no hidden layers. Their hidden-layer / occlusion numbers are the
"no amodal reasoning" baseline (existence F1 0, AbsRel undefined, occluded recall from lucky hits only).

Usage
  # 1. evaluate (one call per model; datasets already evaluated with the same checkpoint + settings are skipped)
  python eval_ldi.py --model "mum_base_ldi(num_layers=4)" --model_name mum \
      --ckpt mum_base_model_downstream_ldi_scannet_matterport_arkitscenes_model_224_x_224_70ep \
      --dataset "ScanNet_LDI(split='test', config_path='configs/scannet_ldi.yaml') + NRGBD_LDI(...)"
  python eval_ldi.py --model "muskie_base_ldi(num_layers=4)" --model_name muskie --ckpt <muskie LDI run> --dataset "..."
  python eval_ldi.py --model "<poincar3 constructor>" --model_name poincar3 --ckpt <run or weights> --dataset "..."

  # 2. report (no GPU needed): Markdown to stdout + ldi_tables.md / ldi_tables.tex / ldi_results.csv
  python eval_ldi.py --report                                   # default models: mum muskie poincar3
  python eval_ldi.py --report --models mum muskie vista poincar3

The places you may need to adapt:
  * run_model()      -> how a model is called / what it returns (LDI probes: layered_depth + layered_depth_exist)
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
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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
STORE_SCHEMA = 1
try:
    CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]
except OSError:
    CODE_SHA = "unknown"

# --model_name -> name in the tables
DISPLAY_NAMES = {"mae": "MAE", "mum": "MuM", "muskie": "Muskie", "vista": "Vista", "poincar3": "Poincar3"}
DEFAULT_REPORT_MODELS = ["mum", "muskie", "poincar3"]


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("LDI evaluation")
    # model
    p.add_argument("--model", type=str, help="constructor call from `models`, e.g. \"mum_base_ldi(num_layers=4)\"")
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
    p.add_argument("--exist_threshold", type=float, default=0.5, help="sigmoid(exist logit) threshold")
    p.add_argument("--align", choices=["none", "median"], default="none",
                   help="none = metric; median = per-sample scale from layer 0 (pred and GT), applied to all layers")
    p.add_argument("--max_depth", type=float, default=16.0, help="predicted / GT depths above this are ignored")
    p.add_argument("--ray_taus", type=str, default="0.05,0.10", help="ray hit tolerances (m)")
    p.add_argument("--scene_taus", type=str, default="0.05,0.10", help="scene F-score / recall tolerances (m)")
    p.add_argument("--voxel", type=float, default=0.02, help="voxel size (m) for the scene point clouds")
    p.add_argument("--occ_eps", type=float, default=0.03,
                   help="a GT hidden point closer than this to a visible (layer-0) GT point counts as visible")
    # store / report
    p.add_argument("--results_dir", type=str, default="results/ldi_eval")
    p.add_argument("--store", type=str, default="results_ldi.pkl")
    p.add_argument("--overwrite", action="store_true", help="evaluate again even if a matching record exists")
    p.add_argument("--report", action="store_true", help="make the tables from the store and exit")
    p.add_argument("--models", nargs="+", default=None, help="report: model keys in row order")
    p.add_argument("--datasets", nargs="+", default=None, help="report: dataset keys (default: all)")
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
        "exist_threshold": args.exist_threshold, "align": args.align, "max_depth": args.max_depth,
        "ray_taus": args.ray_taus, "scene_taus": args.scene_taus, "voxel": args.voxel, "occ_eps": args.occ_eps,
        "max_batches": args.max_batches,
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


@torch.no_grad()
def run_model(model, images, amp_dtype):
    """
    images (B, S, 3, H, W) in [0, 1] ->
        depth (B, S, N, h, w) metres, exist_logits (B, S, N, h, w) or None (None = visible-only model)
    LDI probes return layered_depth / layered_depth_exist; any other model only needs `depth`.
    """
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        out = model(images=images)
    if isinstance(out, (tuple, list)):
        out = out[-1]
    if "layered_depth" in out:
        exist = out.get("layered_depth_exist")
        return out["layered_depth"].float(), (exist.float() if exist is not None else None), True
    depth = out["depth"].float()
    if depth.dim() == 5 and depth.shape[-1] == 1:           # (B, S, H, W, 1)
        depth = depth[..., 0]
    return depth[:, :, None], None, False                    # single layer, existence = depth > 0


# =============================================================================
# Batch
# =============================================================================
def _stack(x):
    """Collated per-view list [S x (B, ...)] -> (B, S, ...); tensors / arrays pass through."""
    if isinstance(x, (list, tuple)):
        return torch.stack([torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v) else v for v in x], dim=1)
    return torch.as_tensor(x)


def to_4x4(T):
    if T.shape[-2] == 4:
        return T
    pad = torch.zeros(*T.shape[:-2], 1, 4, dtype=T.dtype, device=T.device)
    pad[..., 0, 3] = 1
    return torch.cat([T, pad], dim=-2)


def prepare_batch(batch, device, num_layers):
    images = _stack(batch["images"]).float().to(device)
    if images.dim() == 4:
        images = images[:, None]
    ldi = _stack(batch["ldis"]).float().to(device)                       # (B, S, L, H, W)
    mask = _stack(batch["ldis_mask"]).bool().to(device) if "ldis_mask" in batch else ldi > 0
    if ldi.dim() == 4:
        ldi, mask = ldi[:, None], mask[:, None]
    ldi, mask = ldi[:, :, :num_layers], mask[:, :, :num_layers]
    K = _stack(batch["intrinsics"]).float().to(device)                    # (B, S, 3, 3)
    w2c = to_4x4(_stack(batch["extrinsics"]).float().to(device))           # (B, S, 4, 4)
    if K.dim() == 3:
        K, w2c = K[:, None], w2c[:, None]
    B = images.shape[0]
    names = batch.get("seq_name", [f"sample{i}" for i in range(B)])
    return dict(images=images, ldi=ldi, mask=mask & (ldi > 0), K=K, c2w=torch.linalg.inv(w2c.double()).float(),
                names=list(names))


# =============================================================================
# Metrics
# =============================================================================
def new_accumulator(L, ray_taus):
    z = lambda: np.zeros(L, dtype=np.float64)                                     # noqa: E731
    return {"tp": z(), "fp": z(), "fn": z(), "abs_rel_sum": z(), "d1_sum": z(), "depth_n": z(),
            "ray": {tau_key(t): np.zeros(4) for t in ray_taus},   # [pred hidden matched, pred hidden, gt hidden matched, gt hidden]
            "scenes": [], "num_samples": 0, "num_layers_pred": None}


def greedy_ray_match(pd, pv, gd, gv, tau):
    """One-to-one matching of the hits of each ray, closest pairs first.
    pd/pv (P, N) predicted depths / validity, gd/gv (P, L) GT -> matched flags (P, N), (P, L)."""
    P, N = pd.shape
    L = gd.shape[1]
    dist = (pd[:, :, None] - gd[:, None, :]).abs()                                 # (P, N, L)
    ok = pv[:, :, None] & gv[:, None, :] & (dist <= tau)
    dist = torch.where(ok, dist, torch.full_like(dist, float("inf"))).reshape(P, N * L)
    order = dist.argsort(dim=1)
    pm = torch.zeros(P, N, dtype=torch.bool, device=pd.device)
    gm = torch.zeros(P, L, dtype=torch.bool, device=pd.device)
    rows = torch.arange(P, device=pd.device)
    for s in range(N * L):
        idx = order[:, s]
        i, j = idx // L, idx % L
        take = torch.isfinite(dist[rows, idx]) & ~pm[rows, i] & ~gm[rows, j]
        pm[rows[take], i[take]] = True
        gm[rows[take], j[take]] = True
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


def voxel_down(p, voxel):
    """Mean point per occupied voxel. p (M, 3) numpy."""
    if len(p) == 0 or voxel <= 0:
        return p
    keys = np.floor(p / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, p)
    return out / counts[:, None]


def nn_dist(src, dst):
    from scipy.spatial import cKDTree
    if len(src) == 0:
        return np.zeros(0)
    if len(dst) == 0:
        return np.full(len(src), np.inf)
    return cKDTree(dst).query(src, k=1, workers=-1)[0]


def scene_metrics(pred_all, pred_hidden, gt_all, gt_visible, args):
    """Point clouds (numpy, metres) of one sample -> chamfer and occluded-geometry recall / precision / F1."""
    v = args.voxel
    P, Ph, G, Gv = (voxel_down(x, v) for x in (pred_all, pred_hidden, gt_all, gt_visible))
    occluded = G[nn_dist(G, Gv) > args.occ_eps] if len(Gv) else G                  # not visible in any input view
    acc, comp = nn_dist(P, G), nn_dist(G, P)
    out = {"chamfer": 0.5 * (acc.mean() + comp.mean()) if len(P) and len(G) else float("nan"),
           "num_gt": len(G), "num_occluded": len(occluded), "num_pred": len(P), "num_pred_hidden": len(Ph)}
    d_occ = nn_dist(occluded, P)
    d_hid = nn_dist(Ph, G)
    for t in taus(args.scene_taus):
        k = tau_key(t)
        r = float((d_occ <= t).mean()) if len(occluded) else float("nan")
        p = float((d_hid <= t).mean()) if len(Ph) else 0.0                         # no hidden prediction = 0
        out[f"occ_recall@{k}"] = r
        out[f"occ_precision@{k}"] = p
        out[f"occ_f1@{k}"] = 2 * p * r / (p + r) if (p + r) > 0 and not math.isnan(r) else (
            float("nan") if math.isnan(r) else 0.0)
    return out


@torch.no_grad()
def accumulate(acc, depth, exist_logits, inp, args):
    """Add one batch to the accumulator. depth (B, S, N, H, W) at the GT resolution."""
    gt, gmask = inp["ldi"], inp["mask"]                                              # (B, S, L, H, W)
    B, S, L, H, W = gt.shape
    N = depth.shape[2]
    acc["num_layers_pred"] = N
    pexist = (torch.sigmoid(exist_logits) > args.exist_threshold) if exist_logits is not None else depth > 0
    pexist = pexist & (depth > 0) & (depth <= args.max_depth) & torch.isfinite(depth)
    gmask = gmask & (gt <= args.max_depth)
    valid_px = gmask[:, :, :1]                                                      # pixels with any GT surface

    if args.align == "median":                                                      # per-sample scale from layer 0
        both = (gmask[:, :, 0] & pexist[:, :, 0]).flatten(1)
        ratio = (gt[:, :, 0] / depth[:, :, 0].clamp_min(1e-6)).flatten(1)
        s = torch.stack([ratio[b][both[b]].median() if both[b].any() else ratio.new_tensor(1.0) for b in range(B)])
        depth = depth * s.view(B, 1, 1, 1, 1)

    # ---------------- per pixel, per layer ----------------
    for k in range(L):
        g = gmask[:, :, k] & valid_px[:, :, 0]
        p = (pexist[:, :, k] & valid_px[:, :, 0]) if k < N else torch.zeros_like(g)
        acc["tp"][k] += float((p & g).sum())
        acc["fp"][k] += float((p & ~g).sum())
        acc["fn"][k] += float((~p & g).sum())
        if k < N:
            m = g & (depth[:, :, k] > 0)
            d, gg = depth[:, :, k][m], gt[:, :, k][m]
            acc["abs_rel_sum"][k] += float(((d - gg).abs() / gg).sum())
            acc["d1_sum"][k] += float((torch.maximum(d / gg, gg / d) < 1.25).sum())
            acc["depth_n"][k] += float(m.sum())

    # ---------------- per ray (hidden hits, index-free matching) ----------------
    vp = valid_px[:, :, 0].reshape(-1)
    pd = depth.permute(0, 1, 3, 4, 2).reshape(-1, N)[vp]
    pv = (pexist & valid_px).permute(0, 1, 3, 4, 2).reshape(-1, N)[vp]
    gd = gt.permute(0, 1, 3, 4, 2).reshape(-1, L)[vp]
    gv = gmask.permute(0, 1, 3, 4, 2).reshape(-1, L)[vp]
    for t in taus(args.ray_taus):
        pm, gm = greedy_ray_match(pd, pv, gd, gv, t)
        acc["ray"][tau_key(t)] += np.array([float(pm[:, 1:].sum()), float(pv[:, 1:].sum()),
                                            float(gm[:, 1:].sum()), float(gv[:, 1:].sum())])

    # ---------------- scene: all views of each sample in 3D ----------------
    for b in range(B):
        pts_p = unproject(depth[b], inp["K"][b], inp["c2w"][b])                       # (S, N, H, W, 3)
        pts_g = unproject(gt[b], inp["K"][b], inp["c2w"][b])                          # (S, L, H, W, 3)
        pm_b = pexist[b] & valid_px[b]
        hidden = pm_b.clone()
        hidden[:, 0] = False
        rec = scene_metrics(
            pred_all=pts_p[pm_b].double().cpu().numpy(),
            pred_hidden=pts_p[hidden].double().cpu().numpy(),
            gt_all=pts_g[gmask[b]].double().cpu().numpy(),
            gt_visible=pts_g[:, :1][gmask[b][:, :1]].double().cpu().numpy(),
            args=args)
        rec["name"] = inp["names"][b]
        acc["scenes"].append(rec)
    acc["num_samples"] += B


def summarize(acc):
    """Accumulator -> flat metric dict (the table values)."""
    L = len(acc["tp"])
    m = {}
    f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")  # noqa: E731
    for k in range(L):
        m[f"exist_f1_L{k}"] = f1(acc["tp"][k], acc["fp"][k], acc["fn"][k])
        n = acc["depth_n"][k]
        m[f"abs_rel_L{k}"] = acc["abs_rel_sum"][k] / n if n > 0 else float("nan")
        m[f"d1_L{k}"] = acc["d1_sum"][k] / n if n > 0 else float("nan")
    h = slice(1, L)
    m["exist_f1_hidden"] = f1(acc["tp"][h].sum(), acc["fp"][h].sum(), acc["fn"][h].sum())
    n = acc["depth_n"][h].sum()
    m["abs_rel_hidden"] = acc["abs_rel_sum"][h].sum() / n if n > 0 else float("nan")
    m["d1_hidden"] = acc["d1_sum"][h].sum() / n if n > 0 else float("nan")
    for k, (pm, pn, gm, gn) in acc["ray"].items():
        p = pm / pn if pn > 0 else 0.0
        r = gm / gn if gn > 0 else float("nan")
        m[f"ray_p_hidden@{k}"], m[f"ray_r_hidden@{k}"] = p, r
        m[f"ray_f1_hidden@{k}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    keys = [k for k in (acc["scenes"][0] if acc["scenes"] else {}) if k != "name" and not k.startswith("num_")]
    for k in keys:
        vals = np.array([s[k] for s in acc["scenes"]], dtype=np.float64)
        m[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
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
    acc = new_accumulator(args.num_layers, taus(args.ray_taus))
    layered = None
    t0 = time.time()
    for i, batch in enumerate(loader):
        if 0 <= max_batches <= i:
            break
        inp = prepare_batch(batch, device, args.num_layers)
        depth, exist, layered = run_model(model, inp["images"], amp_dtype)
        H, W = inp["ldi"].shape[-2:]
        if depth.shape[-2:] != (H, W):                                                # e.g. a model at another resolution
            B, S, N = depth.shape[:3]
            depth = F.interpolate(depth.flatten(0, 1), (H, W), mode="nearest").view(B, S, N, H, W)
            if exist is not None:
                exist = F.interpolate(exist.flatten(0, 1), (H, W), mode="nearest").view(B, S, N, H, W)
        accumulate(acc, depth, exist, inp, args)
        if i % 20 == 0:
            logger.info(f"  batch {i}/{len(loader)}  {acc['num_samples']} samples  {time.time() - t0:.0f}s")
    return acc, layered


def main():
    args = parse_args()
    store_path = osp.join(args.results_dir, args.store)
    if args.report:
        return report(args, store_path)
    if not (args.model and args.model_name and args.dataset):
        raise SystemExit("--model, --model_name and --dataset are required (or use --report)")

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

    model = load_model(args, ckpt_path, device)
    for ds in todo:
        key = dataset_key(ds)
        logger.info(f"=== {args.model_name} on {key} ===")
        loader = get_data_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_mem=True, shuffle=False, drop_last=False)
        acc, layered = evaluate_dataset(model, loader, args, device, args.max_batches)
        metrics = summarize(acc)
        errors = []
        if acc["num_samples"] == 0:
            errors.append("no samples")
        if args.max_batches >= 0:
            errors.append(f"partial run (--max_batches {args.max_batches})")
        for k in ("exist_f1_L0", "abs_rel_L0", "chamfer"):
            if not np.isfinite(metrics.get(k, float("nan"))):
                errors.append(f"{k} is not finite")
        rec = {"model": args.model_name, "display_name": args.display_name, "dataset": key, "dataset_str": ds,
               "time": datetime.datetime.now().isoformat(timespec="seconds"), "ckpt": ckpt_fp,
               "settings": settings, "code_sha": CODE_SHA, "host": socket.gethostname(),
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "layered_model": layered,
               "num_layers_pred": acc["num_layers_pred"], "complete": not errors, "errors": errors,
               "metrics": metrics, "per_sample": acc["scenes"]}
        with locked_store(store_path) as store:
            store["runs"].setdefault(args.model_name, {}).setdefault(key, []).append(rec)
        logger.info(f"{'COMPLETE' if not errors else 'INCOMPLETE ' + str(errors)}  {key}: "
                    f"F1 L0 {metrics['exist_f1_L0']:.3f} hidden {metrics['exist_f1_hidden']:.3f} | "
                    f"AbsRel L0 {metrics['abs_rel_L0']:.4f} hidden {metrics['abs_rel_hidden']:.4f} | "
                    f"chamfer {metrics['chamfer']:.4f} m  ({metrics['num_samples']} samples)")
    logger.info(f"store: {store_path}   tables: python {osp.basename(__file__)} --report --results_dir {args.results_dir}")


# =============================================================================
# Report
# =============================================================================
# (key, header, better: +1 higher / -1 lower, format, scale)
TABLES = {
    "layers": ("Per-layer: existence F1 and depth (layer 0 = visible, hidden = layers 1-3 pooled)", [
        ("exist_f1_L0", "F1 L0", +1, "{:.3f}", 1), ("exist_f1_hidden", "F1 hidden", +1, "{:.3f}", 1),
        ("abs_rel_L0", "AbsRel L0", -1, "{:.3f}", 1), ("abs_rel_hidden", "AbsRel hidden", -1, "{:.3f}", 1),
        ("d1_L0", "δ1 L0", +1, "{:.3f}", 1), ("d1_hidden", "δ1 hidden", +1, "{:.3f}", 1),
    ]),
    "occlusion": ("Occluded geometry: hidden-hit ray F1, scene-level occluded recall / F-score, overall Chamfer", [
        ("ray_f1_hidden@10cm", "Ray F1 hidden @10cm", +1, "{:.3f}", 1),
        ("occ_recall@5cm", "Occ. recall @5cm", +1, "{:.3f}", 1), ("occ_recall@10cm", "Occ. recall @10cm", +1, "{:.3f}", 1),
        ("occ_f1@5cm", "Occ. F @5cm", +1, "{:.3f}", 1), ("occ_f1@10cm", "Occ. F @10cm", +1, "{:.3f}", 1),
        ("chamfer", "Chamfer (cm)", -1, "{:.2f}", 100),
    ]),
    "per_layer": ("Per layer k: existence F1 / AbsRel / δ1", [
        (f"{m}_L{k}", f"{lab} L{k}", b, f, 1) for k in range(4)
        for m, lab, b, f in (("exist_f1", "F1", +1, "{:.3f}"), ("abs_rel", "AbsRel", -1, "{:.3f}"), ("d1", "δ1", +1, "{:.3f}"))
    ]),
}


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
    settings = {json.dumps(r["settings"] | {"model_class": None}, sort_keys=True) for r in latest.values()}
    if len(settings) > 1:
        logger.warning("the selected records were evaluated with DIFFERENT settings (thresholds / align / voxel)")

    def name(m):
        rec = next((r for (mm, _), r in latest.items() if mm == m), None)
        return (rec or {}).get("display_name") or DISPLAY_NAMES.get(m, m)

    md, tex, rows = [], [], []
    for tkey, (title, cols) in TABLES.items():
        md.append(f"## {title}\n")
        tex += [f"% {title}", r"\begin{table}[t]\centering\small", r"\resizebox{\linewidth}{!}{%",
                r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
                "Model & " + " & ".join(c[1].replace("δ1", r"$\delta_1$") for c in cols) + r" \\"]
        for ds in datasets:
            md.append(f"**{ds}**\n")
            md.append("| Model | " + " | ".join(c[1] for c in cols) + " |")
            md.append("|---|" + "---|" * len(cols))
            ds_tex = ds.replace("_", r"\_")
            tex += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{ds_tex}}}}} \\"]
            vals = {m: [latest[(m, ds)]["metrics"].get(c[0], float("nan")) if (m, ds) in latest else float("nan")
                        for c in cols] for m in models}
            # rank on the PRINTED values: ties share the rank (all best bold, second-best distinct value underlined)
            ranks = []
            for j, c in enumerate(cols):
                shown = {m: float(c[3].format(vals[m][j] * c[4])) for m in models if np.isfinite(vals[m][j])}
                levels = sorted({v * c[2] for v in shown.values()}, reverse=True)
                ranks.append({m: levels.index(v * c[2]) for m, v in shown.items()} if len(levels) > 1 else {})
            for m in models:
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
                tex.append(f"{name(m)} & " + " & ".join(cells_tex) + r" \\")
            md.append("")
        tex += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{title}.}}", r"\end{table}", ""]

    n = {ds: {m: latest[(m, ds)]["metrics"].get("num_samples") for m in models if (m, ds) in latest} for ds in datasets}
    md.append("Samples per dataset: " + "; ".join(f"{ds}: {v}" for ds, v in n.items()))
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