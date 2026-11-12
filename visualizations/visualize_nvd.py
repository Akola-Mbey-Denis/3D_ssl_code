#!/usr/bin/env python
"""
Dump NVD-probe predictions for figures (target views only): images, GT / predicted depth maps, per-view point
maps and merged point clouds, GT and prediction.

Standalone (no import of eval_nvd.py); model loading / forward / batch handling are the same as in eval_nvd.py, so
the dumped predictions are exactly the ones the metrics are computed on. The probe predicts the depth of the T target views from the source views and the
target cameras. Each predicted view is aligned to its own GT view (default: IRLS scale per view) and unprojected with
the GT camera of that view, so every predicted point map lands on the GT point map of the same view; the merged
point cloud of a sample is the union of these per-view aligned point maps.

Layout: one folder per model, identical file names in every model folder
  <out_dir>/<model_name>/
      images/        <name>.png                                    input / target RGB of the view
      depthmaps/     <name>_gt_depth.png    <name>_pred_depth.png  colourised, ONE range for GT and prediction
                     <name>_error.png                              AbsRel of the aligned prediction (0 .. --err_max)
      point_maps/    <name>_gt_pointmap.glb     <name>_pred_pointmap.glb        individual views (GT world frame)
      point_clouds/  <sample>_gt_pointcloud.glb <sample>_pred_pointcloud.glb    merged: union of the views above
      metadata.json  per sample: views, alignment (scale / shift / Sim(3)), per-view AbsRel / delta1, cameras,
                     colour range, GT checksum
  <name>   = <sample>_<view>       e.g. ScanNetPP_00025_scene0011-00_tgt00
  <sample> = <dataset>_<idx:05d>_<scene>, <view> = tgt00, tgt01, ... (src00 ... for source images)
  (if the batch carries image paths, <view> is replaced by the image file stem: <sample>_<stem>)

Consistency across models
  * same --dataset string + same --indices -> same samples / views (the loader is not shuffled); every run
    compares its GT checksum per sample with the other model folders in --out_dir and warns on a mismatch.
  * colour range: one global [min, max] over the GT depth and the aligned predicted depth of the sample
    (--range_scope sample) or view (view), used for GT and prediction alike in the depth maps.
    --range percentile: 2nd-98th percentile instead.
  * point maps / point clouds are coloured with the RGB image of the view; the predicted points are the aligned
    predicted depth unprojected with the GT cameras, so they overlap the GT points.
  * point maps and point clouds are in the GT world frame; --opengl (default) flips y/z so they open upright in
    GLB viewers.

Usage
  for m in vista mae mum muskie; do
    python dump_nvd.py --model_name $m --ckpt <${m}_nvd_run> \
        --dataset "ScanNetPP_NVS(split='test', ..., n_target_views=2, max_views=6, load_depth=True)" \
        --indices 0,10,25 --out_dir results/figures/nvd
  done
"""

import argparse
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("dump_nvd")


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("Dump NVD-probe target-view depth maps, point maps and point clouds for figures")
    # model (same meaning as in the eval scripts)
    p.add_argument("--model", type=str, default=None,
                   help="Constructor, e.g. 'vista_nvd_base_downstream()'. Default: the one stored in the checkpoint")
    p.add_argument("--model_name", type=str, required=True,
                   help="Folder name of this model in the dump, e.g. vista, mae, mum, muskie")
    p.add_argument("--model_kwargs", type=str, default="{}")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--ckpt_root", type=str, default=None, help="Default: CKPT_ROOT of the eval script")
    p.add_argument("--non_strict", action="store_true", help="Allow missing / unexpected keys")
    p.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="fp32",
                   help="fp32 = as trained (the encoder runs its own bf16 autocast inside)")
    # data
    p.add_argument("--dataset", type=str, required=True, help="'+'-separated dataset strings for get_data_loader")
    p.add_argument("--indices", type=str, default=None, help="Comma-separated dataset indices, e.g. 0,10,25")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Without --indices: start + k * stride, k < N. Default: every sample of the dataset")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sequential", action="store_true",
                   help="Iterate the full loader in order and keep the selected indices (slower; use if your "
                        "dataset cannot be indexed by a plain int)")
    p.add_argument("--img_norm", choices=["none", "imagenet"], default="none",
                   help="How the loader normalises images (to recover RGB for inputs / point colours)")
    # alignment
    p.add_argument("--align", choices=["scale", "median", "lstsq", "none"], default="scale",
                   help="Per-view depth alignment to GT: scale = IRLS L1 scale, median, lstsq = scale + shift, none")
    # rendering
    p.add_argument("--cmap", type=str, default="Spectral", help="Matplotlib colormap for depth")
    p.add_argument("--colorize", choices=["depth", "disparity"], default="depth")
    p.add_argument("--range", choices=["minmax", "percentile"], default="minmax",
                   help="Depth colour range: global min / max of GT and aligned prediction together, or their "
                        "2nd-98th percentile (robust to outliers). The same range is used for GT and prediction")
    p.add_argument("--range_scope", choices=["sample", "view"], default="sample",
                   help="Range over all views of the sample or per view")
    p.add_argument("--err_max", type=float, default=0.25, help="AbsRel value shown as full colour")
    p.add_argument("--pred_mask", choices=["gt", "all"], default="gt",
                   help="Predicted points: only at GT-valid pixels (comparable to GT) or every pixel")
    p.add_argument("--pc_stride", type=int, default=1, help="Keep every k-th pixel in x and y for point clouds")
    p.add_argument("--opengl", type=int, default=1, help="1: flip y/z (OpenCV -> OpenGL) for GLB viewers")
    # output
    p.add_argument("--out_dir", type=str, default="results/figures/nvd",
                   help="Root; files go to <out_dir>/<model_name>/{images,depthmaps,point_clouds,point_maps}")
    return p.parse_args()


# =============================================================================
# Small utilities
# =============================================================================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_rgb(img, norm):
    """(3,H,W) model-space image -> (H,W,3) float in [0,1]."""
    img = img.detach().float().cpu()
    if norm == "imagenet":
        img = img * IMAGENET_STD + IMAGENET_MEAN
    elif img.min() < -0.01:                       # [-1, 1] images
        img = (img + 1) / 2
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def save_png(arr, path):
    from PIL import Image
    os.makedirs(osp.dirname(path), exist_ok=True)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def colorize(depth, mask, vmin, vmax, cmap, mode="depth", invalid=(255, 255, 255)):
    """depth (H,W) numpy -> (H,W,3) uint8. mask=None colours every finite pixel."""
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    d = depth.astype(np.float64)
    if mode == "disparity":
        d, vmin, vmax = 1.0 / np.maximum(d, 1e-6), 1.0 / max(vmax, 1e-6), 1.0 / max(vmin, 1e-6)
    t = np.clip((d - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    out = (cm(t)[..., :3] * 255).astype(np.uint8)
    ok = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        ok &= mask
    out[~ok] = invalid
    return out


def error_map(pred, gt, mask, vmax, cmap="magma"):
    import matplotlib
    err = np.where(mask, np.abs(pred - gt) / np.maximum(gt, 1e-6), 0.0)
    out = (matplotlib.colormaps[cmap](np.clip(err / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    out[~mask] = 255
    return out


def write_glb(path, points, colors):
    """Point cloud -> .glb with trimesh. points (N,3) float, colors (N,3) uint8 RGB."""
    import trimesh
    os.makedirs(osp.dirname(path), exist_ok=True)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    col = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    col = np.concatenate([col, np.full((len(col), 1), 255, np.uint8)], 1)          # RGBA
    trimesh.PointCloud(vertices=pts, colors=col).export(path, file_type="glb")


def clean(s, n=60):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")[:n] or "x"


def scene_name(batch):
    for k in ("scene_name", "scene", "seq_name", "label"):
        if k in batch:
            v = batch[k]
            v = v[0] if isinstance(v, (list, tuple)) else v
            if isinstance(v, (list, tuple)):          # per-view names
                v = v[0]
            return clean(v)
    return "scene"


def image_stems(batch, prefix, n):
    """File stems of the view images if the batch carries paths ('<prefix>image_paths', ...), else None."""
    for k in ("image_paths", "img_paths", "paths", "image_names", "img_names", "names"):
        v = batch.get(prefix + k)
        if v is None:
            continue
        v = v[0] if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (list, tuple)) else v
        if isinstance(v, (list, tuple)) and len(v) == n:
            v = [x[0] if isinstance(x, (list, tuple)) else x for x in v]    # default collate: [(path,), ...]
            return [clean(osp.splitext(osp.basename(str(x)))[0], 80) for x in v]
    return None


def view_names(batch, prefix, n):
    """Image file stems when the batch has paths (and they are unique), else tgt00.. / src00.."""
    stems = image_stems(batch, prefix, n)
    role = "tgt" if prefix == "t_" else "src"
    return stems if stems and len(set(stems)) == n else [f"{role}{i:02d}" for i in range(n)]


# =============================================================================
# Alignment
# =============================================================================
def fit_alignment(pred, gt, method):
    """1-D tensors of valid pixels -> (scale, shift) so that scale * pred + shift ~ gt."""
    if method == "none" or len(pred) < 10:
        return 1.0, 0.0
    pred, gt = pred.double(), gt.double()
    if method == "median":
        return float(gt.median() / pred.median()), 0.0
    if method == "lstsq":
        A = torch.stack([pred, torch.ones_like(pred)], 1)
        sol = torch.linalg.lstsq(A, gt[:, None]).solution[:, 0]
        return float(sol[0]), float(sol[1])
    # 'scale': IRLS (Weiszfeld) L1 scale, as depth_evaluation(align_with_scale=True)
    s = gt.mean() / pred.mean()
    for _ in range(10):
        w = 1.0 / ((s * pred - gt).abs() + 1e-8)
        s = (w * pred * gt).sum() / (w * pred ** 2).sum()
    return float(s.clamp(min=1e-3)), 0.0


def align_depths(pred, gt, mask, method):
    """pred/gt/mask (V,H,W) -> aligned pred (V,H,W), list of (scale, shift) per view. Every view is aligned to its
    own GT view, so each predicted point map lands on the GT point map of the same view."""
    ok = mask & torch.isfinite(pred) & (pred > 0)
    params = [fit_alignment(pred[v][ok[v]], gt[v][ok[v]], method) for v in range(pred.shape[0])]
    out = torch.stack([pred[v] * s + t for v, (s, t) in enumerate(params)])
    return out, params


def unproject(depth, K, c2w):
    """depth (V,H,W), K (V,3,3), c2w (V,4,4) -> world points (V,H,W,3)."""
    V, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device),
                            indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).to(depth.dtype)
    rays = torch.einsum("vij,nj->vni", torch.linalg.inv(K.to(depth.dtype)), pix)
    cam = rays * depth.reshape(V, -1, 1)
    c2w = c2w.to(depth.dtype)
    return (torch.einsum("vij,vnj->vni", c2w[:, :3, :3], cam) + c2w[:, None, :3, 3]).reshape(V, H, W, 3)


def to_4x4(T):
    if T.shape[-2] == 3:
        bottom = torch.zeros(*T.shape[:-2], 1, 4, dtype=T.dtype, device=T.device)
        bottom[..., 0, 3] = 1.0
        T = torch.cat([T, bottom], dim=-2)
    return T


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
# Model loading / forward / batch handling (same as eval_nvd.py, kept here so this script is standalone)
# =============================================================================
from models import (  # noqa: E402,F401  (used through eval(args.model))
    vista_nvd_base_downstream,
    mae_nvd_base_downstream,
    mum_nvd_base_downstream,
    muskie_nvd_base_downstream,
)

MODEL_NAMES = sorted(n for n in dir() if n.endswith("_downstream"))
CKPT_ROOT = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints"
CKPT_EXTS = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_ckpt(ckpt, root):
    """--ckpt -> weights file: a file, a folder path, or a folder name under `root`. In a folder: a file named
    final/last wins, else the highest step/epoch number, else the newest file."""
    for cand in (ckpt, osp.join(root, ckpt)):
        if osp.isfile(cand):
            return osp.abspath(cand)
        if osp.isdir(cand):
            folder = cand
            break
    else:
        raise SystemExit(f"--ckpt '{ckpt}' is neither a file nor a folder (also looked in {root})")
    files = [osp.join(folder, f) for f in os.listdir(folder) if f.endswith(CKPT_EXTS)]
    if not files:
        files = [osp.join(d, f) for d, _, fs in os.walk(folder) for f in fs if f.endswith(CKPT_EXTS)]
    if not files:
        raise SystemExit(f"no weights ({', '.join(CKPT_EXTS)}) found in {folder}")

    def stem(f):
        return osp.splitext(osp.basename(f))[0].lower()

    named = [f for f in files if re.search(r"(^|[-_.])(final|last|latest)([-_.]|$)", stem(f))]
    if named:
        named.sort(key=lambda f: ("final" not in stem(f), -os.path.getmtime(f)))
        choice = named[0]
    else:
        numbered = [(int(re.findall(r"\d+", stem(f))[-1]), f) for f in files if re.findall(r"\d+", stem(f))]
        choice = max(numbered)[1] if numbered else max(files, key=os.path.getmtime)
    logger.info(f"--ckpt {ckpt} -> {choice}")
    return osp.abspath(choice)


def model_constructor(spec):
    """'vista_nvd_base_downstream()' / 'vista_nvd_base_downstream' -> 'vista_nvd_base_downstream'."""
    return spec.split("(")[0].strip()


def load_model(args, device):
    """train_nvd.py checkpoint = whole model (frozen encoder + probe) + training args. --model defaults to the
    constructor string stored in the checkpoint. Loaded strictly unless --non_strict."""
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.model is None:
        args.model = getattr(ckpt.get("args") if isinstance(ckpt, dict) else None, "model", None)
        if args.model is None:
            raise SystemExit(f"{args.ckpt} stores no training args: pass --model")
        logger.info(f"--model from the checkpoint's training args: {args.model}")
    if model_constructor(args.model) not in MODEL_NAMES:
        raise SystemExit(f"unknown --model '{args.model}'; choose from {MODEL_NAMES}")
    kwargs = json.loads(args.model_kwargs)
    spec = args.model.strip()
    if spec.endswith(")"):
        if kwargs:
            raise SystemExit("give either --model 'f(...)' or --model f with --model_kwargs, not both")
        model = eval(spec)
    else:
        model = eval(spec)(**kwargs)

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
    logger.info(f"Loaded {spec} from {args.ckpt}: {msg}")
    return model.to(device).eval()


def run_model(model, x, amp_dtype):
    """Same call as train_nvd.py. Returns predicted target depth (B, T, H, W), float32."""
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        out = model(x["src_images"], x["src_w2c"], x["tgt_K"], x["tgt_w2c"])
    depth = out["depth"].float()
    return depth[..., 0] if depth.dim() == 5 else depth


def _tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        return torch.stack([_tensor(v) for v in x])
    raise TypeError(f"Cannot convert {type(x)} to tensor")


def gather_nvd(batch, device):
    """Model inputs as train_nvd.prepare_batch (sources: images + w2c; targets: K + w2c) + GT target depth / mask."""
    for k in ("t_depths", "t_intrinsics", "t_extrinsics"):
        if k not in batch:
            raise KeyError(f"batch has no '{k}': set n_target_views >= 1 and load_depth=True in the dataset string")
    get = lambda k: _tensor(batch[k]).float().to(device, non_blocking=True)  # noqa: E731
    x = dict(src_images=get("s_images"), src_w2c=get("s_extrinsics"),
             tgt_K=get("t_intrinsics"), tgt_w2c=get("t_extrinsics"))
    B, T = x["tgt_K"].shape[:2]
    d = get("t_depths")
    d = d.reshape(B, T, *d.shape[-2:])
    m = (_tensor(batch["t_depth_masks"]).to(device).reshape(B, T, *d.shape[-2:]).bool()
         if "t_depth_masks" in batch else d > 0)
    x["tgt_depth"] = d
    x["tgt_mask"] = m & (d > 0) & torch.isfinite(d)
    return x


# =============================================================================
# Prediction for one sample
# =============================================================================
def build(args, device):
    """Returns predict_fn. predict_fn(batch) -> dict of per-sample tensors (leading batch dim 1 removed):
         names [V], rgb (V,H,W,3) numpy, gt_depth/gt_mask (V,h,w), K (V,3,3) at (h,w), w2c (V,4,4) GT,
         pred_depth (V,h,w), pred_w2c / pred_K (V,4,4)/(V,3,3) at (h,w) or None, inputs {name: rgb}
    """
    args.ckpt_root = args.ckpt_root or CKPT_ROOT
    args.ckpt = resolve_ckpt(args.ckpt, args.ckpt_root)
    model = load_model(args, device)                  # also fills --model from the checkpoint if not given
    amp = AMP[args.amp]

    @torch.no_grad()
    def predict(batch):
        x = gather_nvd(batch, device)
        pred = run_model(model, x, amp)[0]
        d, m = x["tgt_depth"][0], x["tgt_mask"][0]
        H, W = x["src_images"].shape[-2:]
        T = d.shape[0]
        if pred.shape[-2:] != d.shape[-2:]:
            pred = F.interpolate(pred[:, None], size=d.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        names = view_names(batch, "t_", T)
        src_names = view_names(batch, "s_", x["src_images"].shape[1])
        inputs = {n: to_rgb(im, args.img_norm) for n, im in zip(src_names, x["src_images"][0])}
        rgb = None
        if "t_images" in batch:
            timg = _tensor(batch["t_images"]).float()[0]
            rgb = [to_rgb(im, args.img_norm) for im in timg]
            inputs.update({n: r for n, r in zip(names, rgb)})
        return dict(names=names, rgb=rgb, gt_depth=d, gt_mask=m, K=scale_K(x["tgt_K"][0], (H, W), d.shape[-2:]),
                    w2c=to_4x4(x["tgt_w2c"][0]), pred_depth=pred, pred_w2c=None, pred_K=None, inputs=inputs)
    return predict


# =============================================================================
# Dump one sample
# =============================================================================
def points_and_colors(pts, mask, rgb_v, depth_v, vmin, vmax, args):
    """pts (H,W,3) numpy, mask (H,W) bool -> (N,3) float32, (N,3) uint8 with pixel stride and colour choice."""
    s = args.pc_stride
    sel = np.zeros_like(mask)
    sel[::s, ::s] = True
    sel &= mask
    p = pts[sel]
    if rgb_v is not None:                               # colour = RGB of the pixel
        c = rgb_v
        if c.shape[:2] != mask.shape:
            c = F.interpolate(torch.from_numpy(c).permute(2, 0, 1)[None], size=mask.shape, mode="bilinear",
                              align_corners=False)[0].permute(1, 2, 0).numpy()
        c = (np.clip(c[sel], 0, 1) * 255).astype(np.uint8)
    else:                                               # no image in the batch: depth colormap
        c = colorize(depth_v, None, vmin, vmax, args.cmap, args.colorize)[sel]
    if args.opengl:
        p = p * np.array([1.0, -1.0, -1.0])
    return p.astype(np.float32), c


def multiview_consistency(depth, mask, K, w2c, max_pts=4000):
    """Median relative depth error when the GT points of view i are projected into view j and compared with the
    GT depth of view j (all ordered pairs). ~0.01-0.05 = cameras and depths agree; >> 0.1 = they do not."""
    V, H, W = depth.shape
    if V < 2:
        return None
    pts = unproject(depth, K, torch.linalg.inv(w2c))
    errs = []
    for i in range(V):
        P = pts[i][mask[i]]
        P = P[:: max(1, len(P) // max_pts)]
        for j in range(V):
            if i == j or len(P) == 0:
                continue
            cam = P @ w2c[j, :3, :3].T + w2c[j, :3, 3]
            z = cam[:, 2]
            uv = cam @ K[j].T
            u, v = uv[:, 0] / uv[:, 2].clamp(min=1e-9), uv[:, 1] / uv[:, 2].clamp(min=1e-9)
            ok = (z > 1e-3) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            if ok.sum() < 50:
                continue
            ui, vi = u[ok].round().long(), v[ok].round().long()
            dj, mj = depth[j][vi, ui], mask[j][vi, ui]
            if mj.sum() < 50:
                continue
            errs.append(((z[ok][mj] - dj[mj]).abs() / dj[mj]))
    return float(torch.cat(errs).median()) if errs else None


_POSE_WARNED = set()


def gt_world_to_camera(depth, mask, K, ext):
    """Returns the GT world-to-camera poses to unproject with, and a report. Keeps the poses as given (w2c, as the
    eval scripts assume) unless their inverse makes the GT depth of the views agree much better."""
    as_given, inverted = ext, torch.linalg.inv(ext)
    e_given = multiview_consistency(depth, mask, K, as_given)
    e_inv = multiview_consistency(depth, mask, K, inverted)
    use = "c2w" if (e_given is not None and e_inv is not None and e_inv < 0.5 * e_given) else "w2c"
    if use == "c2w" and "c2w" not in _POSE_WARNED:
        logger.warning(f"GT *_extrinsics look camera-to-world (reprojection error {e_given:.3f} as w2c vs "
                       f"{e_inv:.3f} inverted): inverting them for the point maps / clouds")
        _POSE_WARNED.add("c2w")
    err = e_inv if use == "c2w" else e_given
    if err is not None and err > 0.1 and "bad" not in _POSE_WARNED:
        logger.warning(f"GT views do not agree (median reprojection depth error {err:.3f}): check K / extrinsics "
                       "of the dataset; GT point clouds of different views will not overlap")
        _POSE_WARNED.add("bad")
    report = {"used": use, "reproj_err_as_w2c": e_given, "reproj_err_inverted": e_inv}
    return (inverted if use == "c2w" else as_given), report


def gt_checksum(gt_np, gm_np, w2c):
    return hashlib.sha1(np.round(np.where(gm_np, gt_np, 0), 4).astype(np.float32).tobytes()
                        + w2c.float().cpu().numpy().round(4).tobytes()).hexdigest()[:16]


def dump_sample(out, idx, sample, dataset_str, root, args):
    """Writes every file of one sample into root/{images,depthmaps,point_clouds,point_maps}; returns its metadata."""
    views = out["names"]
    V = len(views)
    names = [f"{sample}_{v}" for v in views]
    gt_d, gt_m = out["gt_depth"].float(), out["gt_mask"].bool() & (out["gt_depth"] > 0)
    pred_raw = out["pred_depth"].float()
    K, w2c = out["K"].double(), out["w2c"].double()
    w2c, pose_check = gt_world_to_camera(gt_d.double(), gt_m, K, w2c)
    c2w = torch.linalg.inv(w2c)
    gt_np, gm_np = gt_d.cpu().numpy(), gt_m.cpu().numpy()
    dirs = {k: osp.join(root, k) for k in ("images", "depthmaps", "point_clouds", "point_maps")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ---- aligned predicted depth ----
    pred_al, params = align_depths(pred_raw, gt_d, gt_m, args.align)     # per view
    pred_np = pred_al.cpu().numpy()

    # ---- point maps (world frame of the GT cameras) ----
    gt_pts = unproject(gt_d.double(), K, c2w).cpu().numpy()
    pr_pts = unproject(pred_al.double(), K, c2w).cpu().numpy()
    pred_ok = np.isfinite(pr_pts).all(-1) & np.isfinite(pred_np) & (pred_np > 0)
    pred_ok = pred_ok & gm_np if args.pred_mask == "gt" else pred_ok

    # ---- colour range: ONE [min, max] shared by GT and prediction (aligned) of the sample / view ----
    ranges = []
    for v in range(V):
        sel = slice(v, v + 1) if args.range_scope == "view" else slice(None)
        vals = np.concatenate([gt_np[sel][gm_np[sel]], pred_np[sel][pred_ok[sel]]])
        if not vals.size:
            ranges.append((0.1, 10.0))
        elif args.range == "minmax":
            ranges.append((float(vals.min()), float(vals.max())))
        else:
            ranges.append((float(np.percentile(vals, 2)), float(np.percentile(vals, 98))))

    # ---- images (sources / inputs too) ----
    for v, im in out["inputs"].items():
        save_png(im, osp.join(dirs["images"], f"{sample}_{v}.png"))

    rgb = out["rgb"] or [None] * V
    merged = {"gt": ([], []), "pred": ([], [])}
    per_view = []
    for v, n in enumerate(names):
        lo, hi = ranges[v]
        # depth maps
        save_png(colorize(gt_np[v], gm_np[v], lo, hi, args.cmap, args.colorize),
                 osp.join(dirs["depthmaps"], f"{n}_gt_depth.png"))
        save_png(colorize(pred_np[v], None, lo, hi, args.cmap, args.colorize),
                 osp.join(dirs["depthmaps"], f"{n}_pred_depth.png"))
        m = gm_np[v] & np.isfinite(pred_np[v]) & (pred_np[v] > 0)
        save_png(error_map(pred_np[v], gt_np[v], m, args.err_max), osp.join(dirs["depthmaps"], f"{n}_error.png"))
        # point clouds
        for who, pts, mask, dep in (("gt", gt_pts, gm_np, gt_np), ("pred", pr_pts, pred_ok, pred_np)):
            p, c = points_and_colors(pts[v], mask[v], rgb[v], dep[v], lo, hi, args)
            write_glb(osp.join(dirs["point_maps"], f"{n}_{who}_pointmap.glb"), p, c)       # individual view
            merged[who][0].append(p)
            merged[who][1].append(c)
        g, q = gt_np[v][m], pred_np[v][m]
        per_view.append({"view": views[v], "name": n, "scale": params[v][0], "shift": params[v][1],
                         "abs_rel": float(np.mean(np.abs(q - g) / g)) if m.any() else None,
                         "delta1": float(np.mean(np.maximum(q / g, g / q) < 1.25)) if m.any() else None,
                         "num_pixels": int(m.sum()), "depth_range": ranges[v],
                         "K": out["K"][v].cpu().tolist(), "w2c": w2c[v].cpu().tolist()})
    for who, (ps, cs) in merged.items():
        write_glb(osp.join(dirs["point_clouds"], f"{sample}_{who}_pointcloud.glb"), np.concatenate(ps),
                  np.concatenate(cs))

    valid = [r for r in per_view if r["abs_rel"] is not None]
    return {"sample": sample, "dataset": dataset_str, "index": idx, "views": per_view,
            "gt_checksum": gt_checksum(gt_np, gm_np, w2c), "gt_pose_check": pose_check,
            "abs_rel_mean": float(np.mean([r["abs_rel"] for r in valid])) if valid else None}


def check_against_other_models(out_dir, model_name, samples):
    """Warn if another model folder has the same sample with different GT (different data sampling)."""
    for other in sorted(os.listdir(out_dir)) if osp.isdir(out_dir) else []:
        path = osp.join(out_dir, other, "metadata.json")
        if other == model_name or not osp.isfile(path):
            continue
        theirs = {s["sample"]: s["gt_checksum"] for s in json.load(open(path)).get("samples", [])}
        bad = [s["sample"] for s in samples if s["sample"] in theirs and theirs[s["sample"]] != s["gt_checksum"]]
        if bad:
            logger.warning(f"GT differs from model '{other}' for {len(bad)} sample(s), e.g. {bad[:3]}: "
                           "use the same --dataset / --indices for every model")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", args.model_name):
        raise SystemExit("--model_name: letters, digits and . _ + - only")
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict = build(args, device)
    from datasets import get_data_loader

    root = osp.join(args.out_dir, args.model_name)
    meta_path = osp.join(root, "metadata.json")
    meta = json.load(open(meta_path)) if osp.isfile(meta_path) else {"samples": []}
    meta.update({"model_name": args.model_name, "model": args.model, "ckpt": args.ckpt, "task": "nvd",
                 "align": args.align, "align_scope": "per_view", "pred_mask": args.pred_mask,
                 "range": args.range, "range_scope": args.range_scope, "cmap": args.cmap, "colorize": args.colorize,
                 "opengl_point_clouds": bool(args.opengl), "pc_stride": args.pc_stride})
    by_name = {s["sample"]: s for s in meta["samples"]}

    for ds in [d.strip() for d in args.dataset.split("+") if d.strip() and d.strip() != "[None]"]:
        name = ds.split("(")[0].strip()
        nw = max(1, args.num_workers)        # get_data_loader sets prefetch_factor, which needs num_workers > 0
        base = get_data_loader(ds, batch_size=1, num_workers=nw, pin_mem=False, shuffle=False, drop_last=False)
        n = len(base.dataset)
        idxs = ([int(i) for i in args.indices.split(",") if i.strip()] if args.indices
                else list(range(args.start, n, args.stride))[:args.num_samples])
        idxs = [i for i in idxs if 0 <= i < n]
        logger.info(f"[{args.model_name} / {name}] {len(idxs)} samples -> {root}")
        if args.sequential:
            wanted, last = set(idxs), max(idxs)
            full = get_data_loader(ds, batch_size=1, num_workers=nw, pin_mem=False, shuffle=False,
                                   drop_last=False)
            stream = ((i, b) for i, b in itertools.takewhile(lambda ib: ib[0] <= last, enumerate(full))
                      if i in wanted)
        else:
            loader = DataLoader(Subset(base.dataset, idxs), batch_size=1, shuffle=False,
                                num_workers=nw, collate_fn=base.collate_fn)
            stream = zip(idxs, loader)
        ds_tag = clean(name.replace("_NVS", ""), 30)
        for idx, batch in stream:
            out = predict(batch)
            sample = f"{ds_tag}_{idx:05d}_{scene_name(batch)}"
            rec = dump_sample(out, idx, sample, ds, root, args)
            by_name[sample] = rec
            ar = rec["abs_rel_mean"]
            logger.info(f"  {sample}  views {out['names']}  " + (f"AbsRel {ar:.4f}" if ar is not None else "no valid GT"))

    meta["samples"] = sorted(by_name.values(), key=lambda s: s["sample"])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    check_against_other_models(args.out_dir, args.model_name, meta["samples"])
    logger.info(f"done -> {root}")


if __name__ == "__main__":
    main()