#!/usr/bin/env python
"""
Dump SSL-downstream predictions (depth + camera heads) for figures.

Per sample, all views (targets first, then sources, as in eval_ssl.py):
  <out_dir>/<model_name>/images/<name>.png                      RGB of the view
  <out_dir>/<model_name>/depthmaps/<name>_gt_depth.png          GT depth
  <out_dir>/<model_name>/depthmaps/<name>_pred_depth.png        predicted depth (scale-aligned to GT, same colours)
  <out_dir>/<model_name>/point_maps/<name>_gt_pointmap.glb      GT point map of the view
  <out_dir>/<model_name>/point_maps/<name>_pred_pointmap.glb    predicted point map of the view (Sim(3)-aligned to GT)
  <out_dir>/<model_name>/point_clouds/<sample>_gt_pointcloud.glb    all GT point maps of the sample
  <out_dir>/<model_name>/point_clouds/<sample>_pred_pointcloud.glb  all aligned predicted point maps of the sample
  <name> = <sample>_<image name>, <sample> = <dataset>_<index>_<sequence>

Depth maps: predicted depth scale-aligned to GT per view (masked IRLS).
Point maps: predicted depth unprojected with the predicted cameras, then moved onto the GT point maps (GT depth + GT
cameras) with one Umeyama Sim(3) per sample; the combined cloud is all views of the sample together.
Every point cloud is written as .glb and .ply. Points are coloured with the RGB image.

  python dump_ssl.py --model vista_base_downstream --model_name vista --ckpt <run> \
      --dataset "ScanNetPP_NVS(split='vis', ..., load_depth=True)"
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


# =============================================================================
# Helpers
# =============================================================================
def resolve_ckpt(ckpt, root=CKPT_ROOT):
    """File, folder, or folder name under root. In a folder: checkpoint-final > *last* > newest .pth."""
    for cand in (ckpt, osp.join(root, ckpt)):
        if osp.isfile(cand):
            return cand
        if osp.isdir(cand):
            files = sorted(f for f in os.listdir(cand) if f.endswith((".pth", ".pt")))
            for key in ("final", "last", "latest"):
                hit = [f for f in files if key in f]
                if hit:
                    return osp.join(cand, hit[0])
            if files:
                return max((osp.join(cand, f) for f in files), key=osp.getmtime)
    raise SystemExit(f"no checkpoint found for --ckpt {ckpt}")


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.stack([to_tensor(v) for v in x])


def to_4x4(T):
    if T.shape[-2] == 3:
        bottom = torch.zeros(*T.shape[:-2], 1, 4, dtype=T.dtype, device=T.device)
        bottom[..., 0, 3] = 1.0
        T = torch.cat([T, bottom], dim=-2)
    return T


def unproject(depth, K, w2c):
    """depth (V,H,W), K (V,3,3), w2c (V,4,4) -> world points (V,H,W,3), numpy float64."""
    depth, K, c2w = depth.double(), K.double(), torch.linalg.inv(to_4x4(w2c.double()))
    V, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).double()
    cam = torch.einsum("vij,nj->vni", torch.linalg.inv(K), pix) * depth.reshape(V, -1, 1)
    world = torch.einsum("vij,vnj->vni", c2w[:, :3, :3], cam) + c2w[:, None, :3, 3]
    return world.reshape(V, H, W, 3).cpu().numpy()


def umeyama(X, Y, with_scale=True):
    """Sim(3) (or rigid if with_scale=False) with c * R @ X + t ~ Y; X, Y (3, N) (Pi3 mv_recon/utils.py)."""
    mu_x, mu_y = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    var_x = np.square(X - mu_x).sum(0).mean()
    cov = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x if with_scale else 1.0
    R = U @ S @ VH
    return c, R, mu_y - c * R @ mu_x


def align_depth_with_scale(pred, gt, mask, num_iters=10, eps=1e-8):
    """Per view, masked IRLS (L1) scale: s * pred ~ gt on the valid pixels. pred/gt/mask (V,H,W) -> aligned (V,H,W)."""
    V = pred.shape[0]
    p, g, m = pred.reshape(V, -1).double(), gt.reshape(V, -1).double(), mask.reshape(V, -1).double()
    s = (m * p * g).sum(-1, keepdim=True) / ((m * p ** 2).sum(-1, keepdim=True) + eps)
    for _ in range(num_iters):
        w = m / ((s * p - g).abs() + eps)
        s = (w * p * g).sum(-1, keepdim=True) / ((w * p ** 2).sum(-1, keepdim=True) + eps)
    return (s.clamp(min=1e-3) * p).reshape(pred.shape).float()


def rgb_uint8(img):
    """(3,H,W) image from the loader -> (H,W,3) uint8. Handles [0,1], [-1,1] (mean/std 0.5) and [0,255] images."""
    img = img.detach().float()
    if img.min() < -0.01:                 # normalised with mean 0.5 / std 0.5
        img = (img + 1) / 2
    elif img.max() > 1.5:                 # 0..255
        img = img / 255
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def depth_png(depth, valid, vmin, vmax):
    """Spectral_r colormap, invalid pixels black."""
    import matplotlib
    t = np.clip((depth - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    out = (matplotlib.colormaps["Spectral_r"](t)[..., :3] * 255).astype(np.uint8)
    out[~valid] = 0
    return out


def save_glb(path, pts, rgb):
    trimesh.PointCloud(vertices=pts.astype(np.float32), colors=rgb).export(path, file_type="glb")


def save_ply(path, pts, rgb):
    """Coloured point cloud as binary .ply (opens in MeshLab / CloudCompare / Open3D with its RGB)."""
    trimesh.PointCloud(vertices=pts.astype(np.float32), colors=rgb).export(path, file_type="ply")


def save_cloud(path_no_ext, pts, rgb):
    """Writes <path>.glb and <path>.ply."""
    save_glb(path_no_ext + ".glb", pts, rgb)
    save_ply(path_no_ext + ".ply", pts, rgb)


def clean(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")[:80] or "x"


def first(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return x


def view_names(batch, prefix, n):
    """Image names from the dataset ('<prefix>image_names'), else tgt00.. / src00.."""
    names = batch.get(prefix + "image_names")
    if names is not None and len(names) == n:
        names = [clean(first(v)) for v in names]
        if len(set(names)) == n:
            return names
    return [f"{'tgt' if prefix == 't_' else 'src'}{i:02d}" for i in range(n)]


def dump_sample(root, sample, names, rgb, gt_depth, gt_mask, pred_depth, gt_pts, pred_pts):
    """Writes images, GT / predicted depth, per-view point maps and the combined point map of one sample.
    rgb (V,H,W,3) uint8, depths (V,H,W), gt_mask (V,H,W) bool, *_pts (V,H,W,3) in the GT world frame."""
    for d in ("images", "depthmaps", "point_maps", "point_clouds"):
        os.makedirs(osp.join(root, d), exist_ok=True)
    all_gt, all_pred = [], []
    for v, n in enumerate(names):
        name = f"{sample}_{n}"
        m = gt_mask[v]
        pred_ok = m & np.isfinite(pred_pts[v]).all(-1)
        Image.fromarray(rgb[v]).save(osp.join(root, "images", f"{name}.png"))

        # depth: prediction already scale-aligned to GT; one [min, max] for GT and prediction
        p = pred_depth[v]
        vals = np.concatenate([gt_depth[v][m], p[m]])
        lo, hi = (vals.min(), vals.max()) if vals.size else (0.0, 1.0)
        Image.fromarray(depth_png(gt_depth[v], m, lo, hi)).save(osp.join(root, "depthmaps", f"{name}_gt_depth.png"))
        Image.fromarray(depth_png(p, m, lo, hi)).save(
            osp.join(root, "depthmaps", f"{name}_pred_depth.png"))


        # per-view point maps, coloured with the RGB image
        save_cloud(osp.join(root, "point_maps", f"per_view_{name}_gt_pointmap"), gt_pts[v][m], rgb[v][m])
        save_cloud(osp.join(root, "point_maps", f"per_view_{name}_pred_pointmap"), pred_pts[v][pred_ok], rgb[v][pred_ok])
        all_gt.append((gt_pts[v][m], rgb[v][m]))
        all_pred.append((pred_pts[v][pred_ok], rgb[v][pred_ok]))

    # combined point maps of the sample (all views)
    for who, parts in (("gt", all_gt), ("pred", all_pred)):
        save_cloud(osp.join(root, "point_clouds", f"combined_{sample}_{who}_pointcloud"),
                 np.concatenate([q for q, _ in parts]), np.concatenate([c for _, c in parts]))


def iterate(dataset_str, indices, num_workers):
    from datasets import get_data_loader
    base = get_data_loader(dataset_str, batch_size=1, num_workers=max(1, num_workers), pin_mem=False,
                           shuffle=False, drop_last=False)
    idxs = list(range(len(base.dataset))) if indices is None else [int(i) for i in indices.split(",")]
    loader = DataLoader(Subset(base.dataset, idxs), batch_size=1, shuffle=False,
                        num_workers=max(1, num_workers), collate_fn=base.collate_fn)
    tag = clean(dataset_str.split("(")[0].replace("_NVS", ""))
    for idx, batch in zip(idxs, loader):
        scene = clean(first(batch.get("seq_name", batch.get("scene_name", "scene"))))
        yield f"{tag}_{idx:05d}_{scene}", batch


from models import (  # noqa: F401  (used through eval(args.model))
    mae_base_downstream, mae_large_downstream,
    mum_base_downstream, mum_large_downstream,
    muskie_base_downstream, muskie_large_downstream,
    vista_base_downstream, vista_large_downstream,
    vista_basev2_downstream, vista_basev3_downstream,
)


# =============================================================================
# Model
# =============================================================================
def load_model(name, ckpt, device):
    model = eval(name)()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model.to(device).eval()

@torch.no_grad()
def predict(model, batch, device):
    imgs, Ks, w2cs, depths, masks, names = [], [], [], [], [], []
    for p in ("t_", "s_"):                                   # targets first, as eval_ssl.py
        if p + "images" not in batch:
            continue
        img = to_tensor(batch[p + "images"]).float()          # (1, V, 3, H, W)
        V = img.shape[1]
        d = to_tensor(batch[p + "depths"]).float().reshape(1, V, *img.shape[-2:])
        m = to_tensor(batch[p + "depth_masks"]).reshape(1, V, *img.shape[-2:]).bool() \
            if p + "depth_masks" in batch else d > 0
        imgs.append(img)
        Ks.append(to_tensor(batch[p + "intrinsics"]).float())
        w2cs.append(to_4x4(to_tensor(batch[p + "extrinsics"]).float()))
        depths.append(d)
        masks.append(m & (d > 0) & torch.isfinite(d))
        names += view_names(batch, p, V)
    images = torch.cat(imgs, 1).to(device)
    K, w2c = torch.cat(Ks, 1)[0].to(device), torch.cat(w2cs, 1)[0].to(device)
    gt_depth, gt_mask = torch.cat(depths, 1)[0].to(device), torch.cat(masks, 1)[0].to(device)
    H, W = images.shape[-2:]

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(images)
    if isinstance(out, (tuple, list)):
        out = out[-1]
    pred = out["depth"].float()[0]
    pred = pred[..., 0] if pred.dim() == 4 else pred
    if pred.shape[-2:] != (H, W):
        pred = F.interpolate(pred[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]

    # predicted cameras
    if "pose_enc" in out:
        from models.utils.pose_enc import pose_encoding_to_extri_intri
        extri, intri = pose_encoding_to_extri_intri(out["pose_enc"].float(), image_size_hw=(H, W))
        pred_K, pred_w2c = intri[0].float(), to_4x4(extri[0].float())
    elif "extrinsic" in out:
        pred_w2c = to_4x4(out["extrinsic"][0].float())
        pred_K = out["intrinsic"][0].float() if out.get("intrinsic") is not None else K
    else:
        raise RuntimeError("model returns no camera prediction ('pose_enc' or 'extrinsic')")

    gm = gt_mask.cpu().numpy()

    # 1) depth for dumping: per-view scale alignment to GT
    pred_depth_aligned = align_depth_with_scale(pred, gt_depth, gt_mask)

    # 2) point maps: GT depth + GT cameras; predicted depth + predicted cameras,
    #    then one Umeyama Sim(3) (all views of the sample) onto the GT point map
    gt_pts = unproject(gt_depth, K, w2c)                       # (V,H,W,3) GT world
    pred_pts = unproject(pred, pred_K, pred_w2c)               # (V,H,W,3) predicted world
    ok = gm
    c, R, t = umeyama(pred_pts[ok].T, gt_pts[ok].T)
    pred_pts = c * pred_pts @ R.T + t.T

    rgb = np.stack([rgb_uint8(im) for im in images[0]])

    # debug: valid points of all views, GT vs prediction (open both in MeshLab)
    save_ply(path="./pred_pointmap.ply", pts=pred_pts[gm].reshape(-1, 3), rgb=rgb[gm].reshape(-1, 3))
    save_ply(path="./gt_pointmap.ply", pts=gt_pts[gm].reshape(-1, 3), rgb=rgb[gm].reshape(-1, 3))

    # camera centres relative to view 0 (pred view 0 is ~identity, so these are in pred world units)
    cp = -(pred_w2c[:, :3, :3].transpose(1, 2) @ pred_w2c[:, :3, 3:]).squeeze(-1)
    gw = w2c @ torch.linalg.inv(w2c[0])
    cg = -(gw[:, :3, :3].transpose(1, 2) @ gw[:, :3, 3:]).squeeze(-1)
    print("pred centre dist", cp.norm(dim=-1).cpu().numpy().round(3))
    print("GT   centre dist", cg.norm(dim=-1).cpu().numpy().round(3))
    print("pred median depth %.3f   GT median depth %.3f" % (pred[gt_mask].median(), gt_depth[gt_mask].median()))
    print("depth ratio GT/pred %.3f   translation ratio GT/pred %.3f   Umeyama c %.3f" % (
        (gt_depth[gt_mask] / pred[gt_mask]).median(), cg.norm(dim=-1)[1:].mean() / cp.norm(dim=-1)[1:].mean(), c))
    # breakpoint()

    return names, rgb, gt_depth.cpu().numpy(), gm, pred_depth_aligned.cpu().numpy(), gt_pts, pred_pts


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="e.g. vista_base_downstream")
    p.add_argument("--model_name", required=True, help="output folder name, e.g. vista")
    p.add_argument("--ckpt", required=True, help="run folder (name under CKPT_ROOT or path) or .pth file")
    p.add_argument("--dataset", required=True)
    p.add_argument("--indices", default=None, help="comma-separated dataset indices (default: all)")
    p.add_argument("--out_dir", default="results/figures/ssl")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, resolve_ckpt(args.ckpt), device)
    root = osp.join(args.out_dir, args.model_name)
    for sample, batch in iterate(args.dataset, args.indices, args.num_workers):
        names, rgb, gt_depth, gt_mask, pred_depth_aligned, gt_pts, pred_pts =  predict(model, batch, device)
        dump_sample(root, sample, names=names, rgb=rgb, gt_depth=gt_depth, gt_mask=gt_mask,
                    pred_depth=pred_depth_aligned, gt_pts=gt_pts, pred_pts=pred_pts)
        print(sample)
    print("done ->", root)


if __name__ == "__main__":
    main()