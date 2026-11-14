#!/usr/bin/env python
"""
Feature PCA of the encoder patch tokens (companion of dump_ssl.py: same data loading, view order and file naming).

For every sample, the patch tokens of ALL its views are projected onto their first 3 principal components (one PCA per
sample and model), and the 3 components are shown as RGB. If the features are consistent across views, the same
surface gets the same colour in every view.

Per sample (views: targets first, then sources, as dump_ssl.py / eval_ssl.py):
  <out_dir>/<model_name>/images/<model_name>_<name>.png                 RGB of the view
  <out_dir>/<model_name>/feature_pca/<model_name>_<name>_pca.png        PCA colours, patch grid upsampled (nearest)
  <out_dir>/<model_name>/feature_pca/<model_name>_<name>_pca_smooth.png PCA colours, bilinear upsampling
  <out_dir>/<model_name>/feature_pca/<model_name>_<sample>_pca_grid.png  all views: RGB row + PCA row
  <out_dir>/<model_name>/point_clouds/<model_name>_combined_<sample>_pca_pointcloud.glb
                                                                        GT depth + GT cameras coloured with the PCA
                                                                        (--pointcloud; needs load_depth=True)
  <out_dir>/<model_name>/arrays/<model_name>_<sample>_pca.npz          tokens' PCA coordinates, components, variance
  <name> = <sample>_<image name>, <sample> = <dataset>_<index>_<sequence>

Where the tokens come from (the same encoder call as each downstream model's forward, no head is run)
  MAE        model.encoder.forward_features(images)                       per block (B, S, 1 + P, D), prefix 1
  MuM        model.encoder.forward_encoder(x=images, return_all_blocks=True)  per block, prefix model.patch_start_idx
  Muskie     model.encoder.get_latent_embeddings(images=images)           per block, prefix = register tokens
  Vista v2   model.encoder(images=images, return_intermediate_outputs=True) -> (per block, patch_start_idx);
             tokens are [frame, global] concatenated (2 D)
  Poincar3   model._taps(images resized to 256 x 256)                     (frame, inter-frame) pair per block,
             prefix model.encoder.patch_token_start; the PCA images are resized back to the input resolution
  --layer picks the block (default -1: last block; for Poincar3 -1 = inter-frame output of the last block).
  The patch tokens are the last P = (H / patch) * (W / patch) tokens of each view, after the model's prefix
  (cls / register / storage tokens).

PCA details
  tokens of all views -> optional L2 normalisation (--l2norm) -> centred -> top-3 right singular vectors.
  Each component is mapped to [0, 1] with its 1st / 99th percentile over all views of the sample, so every view of the
  sample shares one colour scale. Component signs are fixed (largest |loading| positive) so reruns give the same colours.
  The PCA is fitted per model: colours are comparable across the views of one model, not between models.

  python -m visualizations.visualize_feature_pca --model_name vista --model vista_basev2_downstream --ckpt <run> \
      --dataset "ScanNetPP_NVS(split='vis', ..., load_depth=True)" --indices 0,3 --out_dir ./figures/feature_pca
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
# Helpers (as dump_ssl.py)
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


def rgb_uint8(img):
    """(3,H,W) image from the loader -> (H,W,3) uint8. Handles [0,1], [-1,1] (mean/std 0.5) and [0,255] images."""
    img = img.detach().float()
    if img.min() < -0.01:
        img = (img + 1) / 2
    elif img.max() > 1.5:
        img = img / 255
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def save_glb(path, pts, rgb):
    trimesh.PointCloud(vertices=pts.astype(np.float32), colors=rgb).export(path, file_type="glb")


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


# =============================================================================
# Model + token capture
# =============================================================================
def load_model(name, ckpt, device):
    import models                                            # any constructor in `models`, e.g. vista_basev2_downstream
    model = eval(name if "(" in name else name + "()", vars(models))
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded {ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model.to(device).eval()


def _per_view(tok, S):
    """(B, S, T, D) or (B*S, T, D) with B = 1 -> (S, T, D)."""
    tok = tok.float()
    if tok.dim() == 4:
        tok = tok[0]
    elif tok.shape[0] != S:
        tok = tok.reshape(S, -1, tok.shape[-1])
    if tok.shape[0] != S:
        raise RuntimeError(f"token tensor {tuple(tok.shape)} does not match {S} views")
    return tok


@torch.no_grad()
def encoder_patch_tokens(model, images, layer=-1):
    """images (1, S, 3, H, W) -> patch tokens (S, P, D), patch grid (h, w), backbone name.
    Calls the encoder exactly as the downstream model's forward does (no head)."""
    S, H, W = images.shape[1], images.shape[-2], images.shape[-1]
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda)
    enc = getattr(model, "encoder", None)
    with amp:
        if hasattr(model, "_taps"):                                         # Poincar3_LDI (frozen Poincar3)
            size = model.input_size
            x = model._resize(images, (size, size)).clamp(0, 1)
            taps, psi = model._taps(x)
            blocks, kind, (H, W) = taps, "poincar3", (size, size)
        elif hasattr(enc, "forward_features"):                              # MAE
            blocks, psi, kind = enc.forward_features(images), 1, "mae"
        elif hasattr(enc, "forward_encoder"):                               # MuM
            blocks, psi, kind = enc.forward_encoder(x=images, return_all_blocks=True), model.patch_start_idx, "mum"
        elif hasattr(enc, "get_latent_embeddings"):                         # Muskie
            blocks, psi, kind = enc.get_latent_embeddings(images=images), model.patch_start_idx, "muskie"
        else:                                                               # Vista v2 (EncoderV2)
            blocks, psi = enc(images=images, return_intermediate_outputs=True)
            kind = "vista"
    patch = getattr(enc, "patch_size", None) or model.patch_size
    patch = patch[0] if isinstance(patch, (tuple, list)) else int(patch)
    h, w = H // patch, W // patch
    tok = _per_view(blocks[layer], S)
    T, P = tok.shape[1], h * w
    if T - psi != P:                                                        # prefix not as expected: patches come last
        print(f"  note: {T} tokens, prefix {psi}, {P} patches expected; using the last {P} tokens")
        psi = T - P
    return tok[:, psi:], (h, w), kind


# =============================================================================
# PCA
# =============================================================================
def feature_pca(tok, l2norm=False, k=3):
    """tok (S, P, D) -> PCA coordinates (S, P, k) in [0, 1], components (k, D), explained variance ratio (k,)."""
    X = tok.reshape(-1, tok.shape[-1]).double()
    if l2norm:
        X = F.normalize(X, dim=-1)
    X = X - X.mean(0, keepdim=True)
    U, Sv, Vh = torch.linalg.svd(X, full_matrices=False)
    comps = Vh[:k]
    # deterministic signs: largest |loading| of every component positive
    signs = torch.sign(comps[torch.arange(k), comps.abs().argmax(1)])
    comps = comps * signs[:, None]
    Y = X @ comps.T                                                                     # (S*P, k)
    lo = torch.quantile(Y, 0.01, dim=0)
    hi = torch.quantile(Y, 0.99, dim=0)
    Y = ((Y - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
    var = (Sv ** 2) / (Sv ** 2).sum()
    return Y.reshape(*tok.shape[:2], k).float(), comps.float(), var[:k].float()


def pca_images(Y, h, w, H, W):
    """(S, h*w, 3) in [0, 1] -> nearest and bilinear (S, H, W, 3) uint8 images."""
    g = Y.reshape(-1, h, w, 3).permute(0, 3, 1, 2)
    near = F.interpolate(g, size=(H, W), mode="nearest")
    smooth = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    to8 = lambda t: (t.permute(0, 2, 3, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)  # noqa: E731
    return to8(near), to8(smooth)


# =============================================================================
# Per sample
# =============================================================================
def collect_views(batch, device):
    """Targets first, then sources (as dump_ssl.py). -> images (1,V,3,H,W), names, GT depth/mask, K, w2c (or None)."""
    imgs, names, depths, masks, Ks, w2cs = [], [], [], [], [], []
    for p in ("t_", "s_"):
        if p + "images" not in batch:
            continue
        img = to_tensor(batch[p + "images"]).float()
        V = img.shape[1]
        imgs.append(img)
        names += view_names(batch, p, V)
        if p + "depths" in batch:
            d = to_tensor(batch[p + "depths"]).float().reshape(1, V, *img.shape[-2:])
            m = to_tensor(batch[p + "depth_masks"]).reshape(1, V, *img.shape[-2:]).bool() \
                if p + "depth_masks" in batch else d > 0
            depths.append(d)
            masks.append(m & (d > 0) & torch.isfinite(d))
            Ks.append(to_tensor(batch[p + "intrinsics"]).float())
            w2cs.append(to_4x4(to_tensor(batch[p + "extrinsics"]).float()))
    images = torch.cat(imgs, 1).to(device)
    geo = None
    if len(depths) == len(imgs) and depths:
        geo = (torch.cat(depths, 1)[0], torch.cat(masks, 1)[0], torch.cat(Ks, 1)[0], torch.cat(w2cs, 1)[0])
    return images, names, geo


@torch.no_grad()
def run_sample(model, batch, device, args):
    images, names, geo = collect_views(batch, device)
    H, W = images.shape[-2], images.shape[-1]
    tok, (h, w), kind = encoder_patch_tokens(model, images, args.layer)
    Y, comps, var = feature_pca(tok, l2norm=args.l2norm)
    near, smooth = pca_images(Y, h, w, H, W)
    rgb = np.stack([rgb_uint8(im) for im in images[0]])
    return dict(names=names, rgb=rgb, near=near, smooth=smooth, Y=Y.cpu().numpy(), comps=comps.cpu().numpy(),
                var=var.cpu().numpy(), geo=geo, grid=(h, w), kind=kind)


def dump_sample(root, model_name, sample, r, pointcloud=False, save_arrays=True):
    for d in ("images", "feature_pca"):
        os.makedirs(osp.join(root, d), exist_ok=True)
    for v, n in enumerate(r["names"]):
        name = f"{model_name}_{sample}_{n}"
        Image.fromarray(r["rgb"][v]).save(osp.join(root, "images", f"{name}.png"))
        Image.fromarray(r["near"][v]).save(osp.join(root, "feature_pca", f"{name}_pca.png"))
        Image.fromarray(r["smooth"][v]).save(osp.join(root, "feature_pca", f"{name}_pca_smooth.png"))
    # overview: RGB row + PCA row, all views of the sample
    grid = np.concatenate([np.concatenate(list(r["rgb"]), 1), np.concatenate(list(r["near"]), 1)], 0)
    Image.fromarray(grid).save(osp.join(root, "feature_pca", f"{model_name}_{sample}_pca_grid.png"))

    if pointcloud:
        if r["geo"] is None:
            print("  --pointcloud: no GT depth in the batch (load_depth=True?), skipped")
        else:
            d, m, K, w2c = r["geo"]
            pts = unproject(d, K, w2c)
            m = m.cpu().numpy()
            os.makedirs(osp.join(root, "point_clouds"), exist_ok=True)
            save_glb(osp.join(root, "point_clouds", f"{model_name}_combined_{sample}_pca_pointcloud.glb"),
                     pts[m], r["smooth"][m])
    if save_arrays:
        os.makedirs(osp.join(root, "arrays"), exist_ok=True)
        np.savez_compressed(osp.join(root, "arrays", f"{model_name}_{sample}_pca.npz"),
                            view_names=np.array(r["names"]), pca=r["Y"], components=r["comps"],
                            explained_variance=r["var"], grid=np.array(r["grid"]))


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser("Feature PCA of encoder patch tokens")
    p.add_argument("--model", required=True, help="constructor, e.g. vista_basev2_downstream or \"mum_base_downstream()\"")
    p.add_argument("--model_name", required=True, help="output folder and file-name prefix, e.g. vista")
    p.add_argument("--ckpt", default=None, help="run folder (name under CKPT_ROOT or path) or .pth file")
    p.add_argument("--dataset", required=True)
    p.add_argument("--indices", default=None, help="comma-separated dataset indices (default: all)")
    p.add_argument("--layer", type=int, default=-1, help="encoder block whose tokens are used (default: last)")
    p.add_argument("--l2norm", action="store_true", help="L2-normalise the tokens before the PCA")
    p.add_argument("--pointcloud", action="store_true", help="also write the GT point cloud coloured with the PCA")
    p.add_argument("--no_arrays", action="store_true")
    p.add_argument("--out_dir", default="results/figures/feature_pca")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, resolve_ckpt(args.ckpt) if args.ckpt else None, device)
    root = osp.join(args.out_dir, args.model_name)
    for sample, batch in iterate(args.dataset, args.indices, args.num_workers):
        r = run_sample(model, batch, device, args)
        dump_sample(root, args.model_name, sample, r, pointcloud=args.pointcloud, save_arrays=not args.no_arrays)
        print(f"{sample}  [{r['kind']}, grid {r['grid'][0]}x{r['grid'][1]}]  explained variance of PC1-3: "
              f"{np.round(r['var'], 3)}")
    print("done ->", root)


if __name__ == "__main__":
    main()