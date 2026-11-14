#!/usr/bin/env python
"""
Dump layered-depth (LDI) predictions for figures. Companion of dump_ssl.py (same naming) and eval_ldi.py (same model
loading, batch keys and scale alignment).

Per sample, every view and every layer k (layer 0 = visible surface, layers 1.. = hidden surfaces):
  <out_dir>/<model_name>/images/<model_name>_<name>.png                                   RGB of the view
  <out_dir>/<model_name>/depthmaps/<model_name>_<name>_L<k>_gt_depth.png                  GT layer k (black: no surface)
  <out_dir>/<model_name>/depthmaps/<model_name>_<name>_L<k>_pred_depth.png                predicted layer k, scale-aligned
  <out_dir>/<model_name>/normalmaps/<model_name>_<name>_L<k>_{gt,pred}_normal.png        surface normals of layer k
  <out_dir>/<model_name>/point_maps/<model_name>_per_view_<name>_gt_pointmap.glb          all GT layers of the view
  <out_dir>/<model_name>/point_maps/<model_name>_per_view_<name>_pred_pointmap.glb        all predicted layers of the view
  <out_dir>/<model_name>/point_maps/<model_name>_per_view_<name>_L<k>_{gt,pred}_pointmap.glb  one layer (--per_layer)
  <out_dir>/<model_name>/point_clouds/<model_name>_combined_<sample>_gt_pointcloud.glb    all views, all GT layers
  <out_dir>/<model_name>/point_clouds/<model_name>_combined_<sample>_pred_pointcloud.glb  all views, all predicted layers
  <out_dir>/<model_name>/arrays/<model_name>_<sample>_ldi.npz                             raw arrays (see save_arrays)
  <name> = <sample>_<image name>, <sample> = <dataset>_<index>_<sequence>

LDI -> depth maps
  Every LDI layer is a depth map (distance along the camera z axis, as in eval_ldi.py). The predicted LDI is aligned to
  the GT exactly as eval_ldi.py (--align median_view, default): one scale per view, s = median(GT / pred) over the pixels
  where GT layer 0 exists and the predicted layer 0 is finite and > 0, applied to ALL layers of that view, so the hidden
  layers keep their depth relative to the visible surface. --align median: one scale per sample; none: metric.

Predicted LDIs are NOT masked (--pred_mask none, default): every predicted layer is shown at every pixel, also where the
  GT has no surface or the predicted stop index says the layer does not exist. Only non-finite values are black, and
  point maps skip depths <= 0 (they cannot be unprojected). --pred_mask stop / gt masks them with the predicted stop
  index / the GT layer masks instead.
Colours: one depth range per view, from the GT depths of all its layers, shared by the GT and the prediction (and by
  every model dumped on the same sample, since the range comes from the GT only). Spectral_r, as dump_ssl.py.
LDI -> point maps: every layer of the GT and of the aligned prediction is unprojected with the GT cameras (LDI probes
  predict no cameras), so GT and prediction share one world frame.
  Point colours: layer 0 (visible surface) = the colour of the input image (GT colour, also for the prediction);
  hidden layers 1.. (occluded surfaces) = surface normal, coloured identically for GT and prediction:
    normal = cross(d/dx, d/dy) of the layer's own point map on the image grid (central differences, one-sided next to
    missing pixels), oriented towards the camera of the view, expressed in the world frame (--normal_frame world,
    default: a surface has the same colour in every view and in the combined cloud) or in the view's camera frame
    (--normal_frame camera), colour = (n + 1) / 2. Pixels without a normal (isolated points) are grey.

  python dump_ldi.py --model "vista_base_ldi(num_layers=4)" --model_name vista --ckpt <run> \
      --dataset "ScannetPP_LDI(split='val', max_views=4, config_path='...')" --indices 0,12,40
"""

import argparse
import os
import os.path as osp
import re

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torch.utils.data import DataLoader, Subset

CKPT_ROOT = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints"
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

NO_NORMAL_GREY = np.array([128, 128, 128], dtype=np.uint8)


# =============================================================================
# Helpers (same as dump_ssl.py / eval_ldi.py)
# =============================================================================
def resolve_ckpt(ckpt, root, fname):
    """File, folder, or folder name under root. In a folder: --ckpt_file > final > best > last (as eval_ldi.py)."""
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


def to_4x4(T):
    if T.shape[-2] == 4:
        return T
    pad = torch.zeros(*T.shape[:-2], 1, 4, dtype=T.dtype, device=T.device)
    pad[..., 0, 3] = 1
    return torch.cat([T, pad], dim=-2)


def bs(x):
    """(B, S, ...) tensor, or a collated per-view list [S x (B, ...)] -> (B, S, ...)."""
    if isinstance(x, (list, tuple)):
        return torch.stack([torch.as_tensor(v) for v in x], dim=1)
    return torch.as_tensor(x)


def rgb_uint8(img):
    """(3,H,W) image from the loader -> (H,W,3) uint8. Handles [0,1], [-1,1] (mean/std 0.5) and [0,255] images."""
    img = img.detach().float()
    if img.min() < -0.01:
        img = (img + 1) / 2
    elif img.max() > 1.5:
        img = img / 255
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def depth_png(depth, valid, vmin, vmax):
    """Spectral_r colormap (as dump_ssl.py), pixels where valid is False black."""
    import matplotlib
    t = np.clip((np.nan_to_num(depth) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    out = (matplotlib.colormaps["Spectral_r"](t)[..., :3] * 255).astype(np.uint8)
    out[~valid] = 0
    return out


def save_cloud(path_no_ext, pts, rgb):
    """Writes <path>.glb (as dump_ssl.py)."""
    if len(pts) == 0:
        return
    trimesh.PointCloud(vertices=pts.astype(np.float32), colors=rgb).export(path_no_ext + ".glb", file_type="glb")


def save_ply(path, pts, rgb):
    """Coloured point cloud as binary .ply (opens in MeshLab / CloudCompare / Open3D with its RGB)."""
    trimesh.PointCloud(vertices=pts.astype(np.float32), colors=rgb).export(path, file_type="ply")


def clean(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")[:80] or "x"


def first(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return x


def view_names(batch, n):
    """Image names from the dataset ('image_names'), else view00, view01, ..."""
    for key in ("image_names", "t_image_names", "s_image_names"):
        names = batch.get(key)
        if names is not None and len(names) == n:
            names = [clean(first(v)) for v in names]
            if len(set(names)) == n:
                return names
    return [f"view{i:02d}" for i in range(n)]


def iterate(dataset_str, indices, num_workers):
    """(sample name, batch of one sample). <sample> = <dataset>_<index>_<sequence>, as dump_ssl.py."""
    from datasets import get_data_loader
    base = get_data_loader(dataset_str, batch_size=1, num_workers=max(1, num_workers), pin_mem=False,
                           shuffle=False, drop_last=False)
    idxs = list(range(len(base.dataset))) if indices is None else [int(i) for i in indices.split(",")]
    loader = DataLoader(Subset(base.dataset, idxs), batch_size=1, shuffle=False,
                        num_workers=max(1, num_workers), collate_fn=base.collate_fn)
    tag = clean(dataset_str.split("(")[0].split("@")[-1].strip().replace("_LDI", "").replace("_NVS", ""))
    for idx, batch in zip(idxs, loader):
        scene = clean(first(batch.get("seq_name", batch.get("scene_name", "scene"))))
        yield f"{tag}_{idx:05d}_{scene}", batch


# =============================================================================
# Model (as eval_ldi.py)
# =============================================================================
def load_model(model_str, ckpt_path, device, non_strict=False):
    import models                                           # any constructor in `models`, e.g. vista_base_ldi(...)
    model = eval(model_str, vars(models))
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for k in ("model", "state_dict", "model_state_dict"):
                if k in ckpt:
                    state = ckpt[k]
                    break
        state = {k.removeprefix("module."): v for k, v in state.items()}
        print(f"loaded {ckpt_path}: {model.load_state_dict(state, strict=not non_strict)}")
    return model.to(device).eval()


def stop_from_output(out, N):
    """layered_depth_stop -> stop index (B, S, H, W) in [0, N]; logits (B, S, N+1, H, W) or an index."""
    s = out["layered_depth_stop"]
    if s.dim() == 5:
        s = s.argmax(dim=2)
    return s.round().long().clamp(0, N)


@torch.no_grad()
def predict(model, batch, device, num_layers, amp_dtype, max_depth):
    """One sample -> dict of (S, ...) tensors: GT LDI + masks, predicted LDI + stop index, cameras, images."""
    images = bs(batch["images"]).float().to(device)                                   # (1, S, 3, H, W)
    gt = bs(batch["ldis"])[:, :, :num_layers].float().to(device)                      # (1, S, L, H, W)
    gmask = bs(batch["ldis_mask"])[:, :, :num_layers].bool().to(device) if "ldis_mask" in batch else gt > 0
    gmask = gmask & (gt > 0) & torch.isfinite(gt) & (gt <= max_depth)
    K = bs(batch["intrinsics"]).float().to(device)                                    # (1, S, 3, 3)
    c2w = torch.linalg.inv(to_4x4(bs(batch["extrinsics"]).float().to(device)).double()).float()

    with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype != torch.float32):
        out = model(images=images)
    if "layered_depth" not in out:
        raise RuntimeError("model returns no 'layered_depth' (not an LDI probe)")
    pred = out["layered_depth"].float()                                               # (1, S, N, H, W)
    B, S, N = pred.shape[:3]
    stop = stop_from_output(out, N) if "layered_depth_stop" in out else None

    H, W = gt.shape[-2:]
    if pred.shape[-2:] != (H, W):                                                     # as eval_ldi.py
        pred = F.interpolate(pred.flatten(0, 1), (H, W), mode="nearest").view(B, S, N, H, W)
    if stop is not None and stop.shape[-2:] != (H, W):
        stop = F.interpolate(stop.float().flatten(0, 1)[:, None], (H, W), mode="nearest")[:, 0].view(B, S, H, W).long()

    return dict(images=images[0], gt=gt[0], gmask=gmask[0], K=K[0], c2w=c2w[0], pred=pred[0],
                stop=None if stop is None else stop[0])


# =============================================================================
# LDI -> aligned depth maps / point maps
# =============================================================================
def align_ldi(pred, gt, gmask, mode):
    """Scale alignment of eval_ldi.py: s = median(GT / pred) on layer 0 (GT layer 0 exists, prediction finite and > 0),
    applied to every layer. pred (S, N, H, W), gt / gmask (S, L, H, W). mode: median_view (one scale per view),
    median (one per sample), none. -> aligned pred (unmasked), scales (S,)."""
    pred = pred.clone()
    S = pred.shape[0]
    ok0 = gmask[:, 0] & torch.isfinite(pred[:, 0]) & (pred[:, 0] > 0)
    scales = torch.ones(S, device=pred.device)
    if mode == "median_view":
        for v in range(S):
            if ok0[v].any():
                scales[v] = (gt[v, 0][ok0[v]] / pred[v, 0][ok0[v]]).median()
    elif mode == "median":
        if ok0.any():
            scales[:] = (gt[:, 0][ok0] / pred[:, 0][ok0]).median()
    pred *= scales.view(S, 1, 1, 1)
    return pred, scales


def pred_masks(pred, gmask, stop, how):
    """Which predicted pixels are shown. none: every finite value (default); stop: layer k where k < stop;
    gt: where the GT layer exists. -> (S, N, H, W) bool."""
    S, N = pred.shape[:2]
    finite = torch.isfinite(pred)
    if how == "none":
        return finite
    if how == "stop":
        if stop is None:
            raise SystemExit("--pred_mask stop: the model returns no 'layered_depth_stop'")
        return finite & (torch.arange(N, device=pred.device).view(1, N, 1, 1) < stop[:, None])
    m = torch.zeros_like(finite)
    n = min(N, gmask.shape[1])
    m[:, :n] = gmask[:, :n]
    return finite & m


def unproject(depth, K, c2w):
    """depth (S, N, H, W), K (S, 3, 3), c2w (S, 4, 4) -> world points (S, N, H, W, 3) numpy (as eval_ldi.py)."""
    S, N, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1)
    rays = torch.einsum("sij,hwj->shwi", torch.linalg.inv(K), pix)
    cam = rays[:, None] * torch.nan_to_num(depth)[..., None]
    pts = torch.einsum("sij,snhwj->snhwi", c2w[:, :3, :3], cam) + c2w[:, None, None, None, :3, 3]
    return pts.cpu().numpy()


def _grid_diff(P, V, axis):
    """Derivative of the point map P (N, H, W, 3) along image axis 1 (rows) or 2 (columns): forward difference where
    the next pixel is valid, else backward difference, else NaN. V (N, H, W) valid pixels."""
    d = np.full_like(P, np.nan)
    a = [slice(None)] * 3
    lo, hi = list(a), list(a)
    lo[axis], hi[axis] = slice(None, -1), slice(1, None)
    lo, hi = tuple(lo), tuple(hi)
    step = P[hi] - P[lo]
    ok = V[hi] & V[lo]
    bwd = np.full_like(P, np.nan)
    bwd[hi] = np.where(ok[..., None], step, np.nan)                 # P[i] - P[i-1] stored at i
    fwd = np.full_like(P, np.nan)
    fwd[lo] = np.where(ok[..., None], step, np.nan)                 # P[i+1] - P[i] stored at i
    d = np.where(np.isfinite(fwd).all(-1, keepdims=True), fwd, bwd)
    return d


def layer_normals(P, V, c2w, frame):
    """Surface normals of every layer of one view. P (N, H, W, 3) world points, V (N, H, W) valid, c2w (4, 4).
    -> unit normals (N, H, W, 3) oriented towards the camera, in the world or camera frame, and ok (N, H, W)."""
    P = np.where(V[..., None], P, np.nan).astype(np.float64)
    n = np.cross(_grid_diff(P, V, 2), _grid_diff(P, V, 1))          # d/dx x d/dy
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = V & np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-12)
    n = np.where(ok[..., None], n / np.where(norm > 0, norm, 1), 0.0)
    to_cam = c2w[:3, 3] - np.nan_to_num(P)                          # towards the camera centre
    n = np.where(((n * to_cam).sum(-1, keepdims=True) < 0), -n, n)
    if frame == "camera":
        n = n @ c2w[:3, :3]                                         # world -> camera: R_c2w^T n
    return n, ok


def normal_colors(n, ok):
    """(..., 3) unit normals -> uint8 colours (n + 1) / 2; grey where there is no normal."""
    col = ((np.clip(n, -1, 1) + 1) / 2 * 255 + 0.5).astype(np.uint8)
    col[~ok] = NO_NORMAL_GREY
    return col


def point_colors(rgb_v, P, V, c2w, frame):
    """Colours of the points of one view: layer 0 = input-image (GT) colour, hidden layers = surface normal.
    rgb_v (H, W, 3), P (N, H, W, 3), V (N, H, W) -> colours (N, H, W, 3), normals (N, H, W, 3), normal ok (N, H, W)."""
    n, ok = layer_normals(P, V, c2w, frame)
    cols = normal_colors(n, ok)
    cols[0] = rgb_v
    return cols, n, ok


# =============================================================================
# Dump
# =============================================================================
def dump_sample(root, model_name, sample, names, rgb, gt, gmask, pred, pshow, gt_pts, pred_pts, c2w,
                normal_frame="world", per_layer=False):
    """rgb (S,H,W,3) uint8; gt / gmask (S,L,H,W); pred / pshow (S,N,H,W) aligned prediction and shown pixels;
    *_pts (S,·,H,W,3) world points; c2w (S,4,4) GT cameras. Every file name starts with <model_name>_."""
    for d in ("images", "depthmaps", "normalmaps", "point_maps", "point_clouds"):
        os.makedirs(osp.join(root, d), exist_ok=True)
    S, L = gt.shape[:2]
    N = pred.shape[1]
    all_gt, all_pred = [], []
    for v, n in enumerate(names):
        name = f"{model_name}_{sample}_{n}"
        Image.fromarray(rgb[v]).save(osp.join(root, "images", f"{name}.png"))

        # one colour range per view, from the GT depths of all its layers (same for every model on this sample)
        g = gt[v][gmask[v]]
        if g.size:
            lo, hi = float(g.min()), float(g.max())
        else:
            p = pred[v][pshow[v]]
            lo, hi = (float(np.percentile(p, 1)), float(np.percentile(p, 99))) if p.size else (0.0, 1.0)
        for k in range(L):
            Image.fromarray(depth_png(gt[v, k], gmask[v, k], lo, hi)).save(
                osp.join(root, "depthmaps", f"{name}_L{k}_gt_depth.png"))
        for k in range(N):
            Image.fromarray(depth_png(pred[v, k], pshow[v, k], lo, hi)).save(
                osp.join(root, "depthmaps", f"{name}_L{k}_pred_depth.png"))

        # LDI -> point maps: layer 0 coloured with the image, hidden layers with their surface normals (same for GT / pred)
        gm = gmask[v]
        pm = pshow[v] & (pred[v] > 0)                                   # depth <= 0 cannot be unprojected
        gcol, gn, gok = point_colors(rgb[v], gt_pts[v], gm, c2w[v], normal_frame)
        pcol, pn, pok = point_colors(rgb[v], pred_pts[v], pm, c2w[v], normal_frame)
        for k in range(L):
            Image.fromarray(np.where(gok[k][..., None], normal_colors(gn[k], gok[k]), 0).astype(np.uint8)).save(
                osp.join(root, "normalmaps", f"{name}_L{k}_gt_normal.png"))
        for k in range(N):
            Image.fromarray(np.where(pok[k][..., None], normal_colors(pn[k], pok[k]), 0).astype(np.uint8)).save(
                osp.join(root, "normalmaps", f"{name}_L{k}_pred_normal.png"))
        if per_layer:
            for k in range(L):
                save_cloud(osp.join(root, "point_maps", f"{model_name}_per_view_{sample}_{n}_L{k}_gt_pointmap"),
                           gt_pts[v, k][gm[k]], gcol[k][gm[k]])
            for k in range(N):
                save_cloud(osp.join(root, "point_maps", f"{model_name}_per_view_{sample}_{n}_L{k}_pred_pointmap"),
                           pred_pts[v, k][pm[k]], pcol[k][pm[k]])
        save_cloud(osp.join(root, "point_maps", f"{model_name}_per_view_{sample}_{n}_gt_pointmap"), gt_pts[v][gm], gcol[gm])
        save_cloud(osp.join(root, "point_maps", f"{model_name}_per_view_{sample}_{n}_pred_pointmap"), pred_pts[v][pm], pcol[pm])
        all_gt.append((gt_pts[v][gm], gcol[gm]))
        all_pred.append((pred_pts[v][pm], pcol[pm]))

    # combined point maps of the sample (all views, all layers)
    combined = {}
    for who, parts in (("gt", all_gt), ("pred", all_pred)):
        combined[who] = (np.concatenate([q for q, _ in parts]), np.concatenate([c for _, c in parts]))
        save_cloud(osp.join(root, "point_clouds", f"{model_name}_combined_{sample}_{who}_pointcloud"), *combined[who])
    return combined


def save_arrays(root, model_name, sample, names, r, pred_aligned, pshow, scales):
    """Raw arrays for custom figures: GT LDI + masks, unmasked aligned prediction, shown-pixel mask, stop index,
    alignment scales, cameras."""
    os.makedirs(osp.join(root, "arrays"), exist_ok=True)
    np.savez_compressed(
        osp.join(root, "arrays", f"{model_name}_{sample}_ldi.npz"),
        view_names=np.array(names), gt_ldi=r["gt"].cpu().numpy(), gt_mask=r["gmask"].cpu().numpy(),
        pred_ldi_aligned=pred_aligned.cpu().numpy(), pred_shown=pshow.cpu().numpy(),
        pred_stop=(r["stop"].cpu().numpy() if r["stop"] is not None else np.zeros(0)),
        align_scales=scales.cpu().numpy(), K=r["K"].cpu().numpy(), c2w=r["c2w"].cpu().numpy())


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser("Dump LDI predictions")
    p.add_argument("--model", required=True, help="constructor call, e.g. \"vista_base_ldi(num_layers=4)\"")
    p.add_argument("--model_name", required=True, help="output folder and file-name prefix, e.g. vista")
    p.add_argument("--ckpt", default=None, help="run folder (name under --ckpt_root or path) or .pth file")
    p.add_argument("--ckpt_root", default=CKPT_ROOT)
    p.add_argument("--ckpt_file", default="checkpoint-final.pth", help="weights file inside a run folder")
    p.add_argument("--non_strict", action="store_true", help="load_state_dict(strict=False)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--indices", default=None, help="comma-separated dataset indices (default: all)")
    p.add_argument("--num_layers", type=int, default=4, help="GT layers dumped (layer 0 + hidden)")
    p.add_argument("--align", choices=["none", "median", "median_view"], default="median_view",
                   help="as eval_ldi.py: scale from layer 0, applied to all layers")
    p.add_argument("--pred_mask", choices=["none", "stop", "gt"], default="none",
                   help="none: predicted LDIs not masked (default); stop: predicted stop index; gt: GT layer masks")
    p.add_argument("--normal_frame", choices=["world", "camera"], default="world",
                   help="frame of the surface normals that colour the hidden layers (same for GT and prediction)")
    p.add_argument("--per_layer", action="store_true", help="also write one point map per layer and view")
    p.add_argument("--max_depth", type=float, default=16.0, help="GT depths above this count as no surface")
    p.add_argument("--amp", choices=list(AMP), default="bf16")
    p.add_argument("--no_arrays", action="store_true", help="do not write the .npz arrays")
    p.add_argument("--out_dir", default="results/figures/ldi")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, resolve_ckpt(args.ckpt, args.ckpt_root, args.ckpt_file), device, args.non_strict)
    root = osp.join(args.out_dir, args.model_name)
    for i, (sample, batch) in enumerate(iterate(args.dataset, args.indices, args.num_workers)):
        r = predict(model, batch, device, args.num_layers, AMP[args.amp], args.max_depth)
        pred, scales = align_ldi(r["pred"], r["gt"], r["gmask"], args.align)
        pshow = pred_masks(pred, r["gmask"], r["stop"], args.pred_mask)
        names = view_names(batch, pred.shape[0])
        rgb = np.stack([rgb_uint8(im) for im in r["images"]])
        combined = dump_sample(root, args.model_name, sample, names, rgb,
                    gt=r["gt"].cpu().numpy(), gmask=r["gmask"].cpu().numpy(),
                    pred=pred.cpu().numpy(), pshow=pshow.cpu().numpy(),
                    gt_pts=unproject(r["gt"], r["K"], r["c2w"]), pred_pts=unproject(pred, r["K"], r["c2w"]),
                    c2w=r["c2w"].cpu().numpy(), normal_frame=args.normal_frame, per_layer=args.per_layer)
        if not args.no_arrays:
            save_arrays(root, args.model_name, sample, names, r, pred, pshow, scales)
        print(f"{sample}  align scales {np.round(scales.cpu().numpy(), 3)}")

        # debug: first example, all views and layers, GT vs aligned prediction in the current directory (open in MeshLab)
        if i == 0:
            save_ply("./gt_pointmap.ply", *combined["gt"])
            save_ply("./pred_pointmap.ply", *combined["pred"])
            print(f"debug: wrote ./gt_pointmap.ply ({len(combined['gt'][0])} pts) and "
                  f"./pred_pointmap.ply ({len(combined['pred'][0])} pts) for {sample}")
            breakpoint()
    print("done ->", root)


if __name__ == "__main__":
    main()