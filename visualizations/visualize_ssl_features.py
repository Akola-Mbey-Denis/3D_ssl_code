#!/usr/bin/env python
"""
Feature PCA of the encoder patch tokens (companion of dump_ssl.py: same data loading, view order and file naming).

For every sample, the patch tokens of ALL its views are projected onto their first 3 principal components (one PCA per
sample and model), and the 3 components are shown as RGB. If the features are consistent across views, the same
surface gets the same colour in every view.

One encoder pass per sample; one PCA per layer, chosen automatically: MAE / MuM -> the blocks their DPT head reads
(2, 5, 8, 11), Vista / Muskie -> blocks 0, 2, 3, 5, Poincar3 -> 4, 11, 17, 23. Tokens are always L2-normalised before the PCA.
Files of layer L go to <out_dir>/<model_name>/L<L>/ with the prefix <model_name>_L<L>, plus one overview per sample:
  <out_dir>/<model_name>/feature_pca/<model_name>_<sample>_layers_<L1>_<L2>...png   RGB row + one PCA row per layer

Per sample and layer (views: targets first, then sources, as dump_ssl.py / eval_ssl.py):
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
  MuM        model.encoder.forward_encoder(x=images, return_all_blocks=True)
             -> {block: tokens}, prefix model.patch_start_idx
  Muskie     model.encoder.get_latent_embeddings(images=images)           per block, prefix = register tokens
  Vista v2   model.encoder(images=images, return_intermediate_outputs=True) -> (per block, patch_start_idx);
             tokens are [frame, global] concatenated (2 D)
  Poincar3   model._taps(images resized to 256 x 256)                     (frame, inter-frame) pair per block,
             prefix model.encoder.patch_token_start; the PCA images are resized back to the input resolution
  Layers are chosen automatically (see above); Poincar3's taps are (frame, inter-frame) pairs, 2 entries per block.
  The patch tokens are the last P = (H / patch) * (W / patch) tokens of each view, after the model's prefix
  (cls / register / storage tokens).

PCA details
  tokens of all views -> L2 normalisation -> centred -> top-3 right singular vectors. Vista uses its fused [frame | global] token.

View drift (always, when the batch has GT cameras, i.e. load_depth=True)
  For every pair of views: GT relative rotation / camera-centre distance vs the distance of the views' mean tokens
  (relative to the within-view spread) and the distance of their mean PCA colours. Pearson / Spearman correlations are printed and saved to
  <out_dir>/<model_name>/<model_name>_view_drift.txt (+ per-pair CSV).
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
            for key in ("best", "final", "last"):
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


def dpt_layers(model):
    """Encoder blocks the model's DPT head reads (MAE / MuM: intermediate_layer_idx), else [-1]."""
    for head_name in ("dpt_head", "head", "depth_head"):
        head = getattr(model, head_name, None)
        idx = getattr(head, "intermediate_layer_idx", None)
        if idx is not None:
            return [int(i) for i in idx]
    return [-1]


MULTIVIEW_LAYERS = [0, 2, 3, 5]          # Vista / Muskie: fixed set of blocks
POINCAR3_LAYERS = [4, 11, 17, 23]        # Poincar3: fixed set of taps


def auto_layers(model):
    """Blocks to visualise, chosen from the model: MAE / MuM -> their DPT layers (2, 5, 8, 11);
    Vista / Muskie -> 0, 2, 3, 5; Poincar3 -> 4, 11, 17, 23."""
    cls = type(model).__name__.lower()
    if hasattr(model, "_taps"):
        return list(POINCAR3_LAYERS)
    if cls.startswith(("mae", "mum")):
        return dpt_layers(model)
    return list(MULTIVIEW_LAYERS)


@torch.no_grad()
def encoder_patch_tokens(model, images, layers=(-1,)):
    """images (1, S, 3, H, W) -> {layer: patch tokens (S, P, D)}, patch grid (h, w), backbone name.
    Calls the encoder once, exactly as the downstream model's forward does (no head)."""
    S, H, W = images.shape[1], images.shape[-2], images.shape[-1]
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda)
    enc = getattr(model, "encoder", None)
    cls = type(model).__name__.lower()                                      # dispatch on the downstream class
    with amp:
        if hasattr(model, "_taps"):                                         # Poincar3_LDI (frozen Poincar3)
            size = model.input_size
            x = model._resize(images, (size, size)).clamp(0, 1)
            taps, psi = model._taps(x)
            blocks, kind, (H, W) = taps, "poincar3", (size, size)
        elif cls.startswith("mum"):                                         # MuM (check before MAE: its encoder
            blocks = enc.forward_encoder(x=images, return_all_blocks=True)  # also has forward_features)
            psi, kind = model.patch_start_idx, "mum"
        elif cls.startswith("mae"):                                         # MAE
            blocks, psi, kind = enc.forward_features(images), 1, "mae"
        elif cls.startswith("muskie"):                                      # Muskie
            blocks, psi, kind = enc.get_latent_embeddings(images=images), model.patch_start_idx, "muskie"
        elif cls.startswith("vista"):                                       # Vista v2 (EncoderV2)
            blocks, psi = enc(images=images, return_intermediate_outputs=True)
            kind = "vista"
        else:
            raise ValueError(f"unknown downstream model class {type(model).__name__}")
    patch = getattr(enc, "patch_size", None) or model.patch_size
    patch = patch[0] if isinstance(patch, (tuple, list)) else int(patch)
    h, w = H // patch, W // patch
    def _desc(x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        if isinstance(x, dict):
            return {k: _desc(v) for k, v in list(x.items())[:3]} | ({"...": len(x)} if len(x) > 3 else {})
        if isinstance(x, (list, tuple)):
            return [_desc(v) for v in x[:2]] + ([f"... {len(x)} items"] if len(x) > 2 else [])
        return type(x).__name__
    n_blocks = len(blocks)
    toks = {}
    for layer in layers:
        if isinstance(blocks, dict):                                        # MuM: {block_idx: tokens}
            keys = sorted(blocks)
            key = keys[layer] if layer < 0 or layer not in blocks else layer
            out = blocks[key]
        else:
            if not -n_blocks <= layer < n_blocks:
                raise ValueError(f"layer {layer} out of range: the {kind} encoder returned {n_blocks} blocks")
            out = blocks[layer]
        if isinstance(out, (tuple, list)):                                  # (tokens, ...) per block
            out = out[0]
        tok = _per_view(out, S)
        T, P = tok.shape[1], h * w
        start = psi
        if T - start != P:                                                  # prefix not as expected: patches come last
            print(f"  note: layer {layer}: {T} tokens, prefix {psi}, {P} patches expected; using the last {P}")
            start = T - P
        toks[layer] = tok[:, start:]
    print(f"  [{kind}] {n_blocks} blocks, using layers {list(toks)}")
    return toks, (h, w), kind


# =============================================================================
# PCA
# =============================================================================
def feature_pca(tok, l2norm=False, k=3, center_views=False):
    """tok (S, P, D) -> PCA coordinates (S, P, k) in [0, 1], components (k, D), explained variance ratio (k,)."""
    X = tok.double()
    if l2norm:
        X = F.normalize(X, dim=-1)
    if center_views:                                  # remove each view's mean token (per-view offset)
        X = X - X.mean(1, keepdim=True)
    X = X.reshape(-1, X.shape[-1])
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
def run_sample(model, batch, device, args, layers):
    """One encoder call, one PCA per layer. -> {layer: result dict}."""
    images, names, geo = collect_views(batch, device)
    H, W = images.shape[-2], images.shape[-1]
    toks, (h, w), kind = encoder_patch_tokens(model, images, layers)
    rgb = np.stack([rgb_uint8(im) for im in images[0]])
    res = {}
    for layer, tok in toks.items():
        Y, comps, var = feature_pca(tok, l2norm=args.l2norm, center_views=args.center_views)
        near, smooth = pca_images(Y, h, w, H, W)
        res[layer] = dict(names=names, rgb=rgb, near=near, smooth=smooth, Y=Y.cpu().numpy(),
                          comps=comps.cpu().numpy(), var=var.cpu().numpy(), geo=geo, grid=(h, w), kind=kind,
                          drift=view_drift(tok, Y, geo, l2norm=args.l2norm),
                          corr=correspondences(tok, geo, (h, w), l2norm=args.l2norm))
    return res


def dump_layer_grid(root, model_name, sample, res):
    """RGB row + one smooth-PCA row per layer, all views side by side (like the Muskie appendix figure)."""
    os.makedirs(osp.join(root, "feature_pca"), exist_ok=True)
    first_r = next(iter(res.values()))
    rows = [np.concatenate(list(first_r["rgb"]), 1)] + [np.concatenate(list(r["smooth"]), 1) for r in res.values()]
    Image.fromarray(np.concatenate(rows, 0)).save(
        osp.join(root, "feature_pca", f"{model_name}_{sample}_layers_{'_'.join(str(l) for l in res)}.png"))


# =============================================================================
# Per-view drift vs camera motion (always on when the batch has GT cameras)
# =============================================================================
def view_drift(tok, Y, geo, l2norm=False):
    """For every pair of views (i, j): GT relative rotation (deg) and camera-centre distance, and how far apart
    the two views are in feature space (distance of the mean patch tokens / within-view spread) and in PCA colour (distance of
    the mean PCA colours). Returns a list of rows, or [] without GT cameras."""
    if geo is None:
        return []
    w2c = geo[3].double().cpu()
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    centres = -(R.transpose(1, 2) @ t[..., None])[..., 0]                         # camera centres (world)
    X = tok.double()
    if l2norm:
        X = F.normalize(X, dim=-1)
    mu = X.mean(1).cpu()                                                            # (S, D) mean token per view
    spread = (X - X.mean(1, keepdim=True)).pow(2).sum(-1).mean().sqrt().cpu()       # typical within-view deviation
    col = Y.mean(1).double().cpu()                                                  # (S, 3) mean PCA colour per view
    rows = []
    S = mu.shape[0]
    for i in range(S):
        for j in range(i + 1, S):
            cos = ((torch.trace(R[i] @ R[j].T) - 1) / 2).clamp(-1, 1)
            rows.append(dict(i=i, j=j, rot_deg=float(torch.rad2deg(torch.acos(cos))),
                             trans=float((centres[i] - centres[j]).norm()),
                             feat_dist=float((mu[i] - mu[j]).norm() / spread),
                             color_dist=float((col[i] - col[j]).norm())))
    return rows


def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def drift_summary(rows, model_name, root):
    """Pearson / Spearman correlation of feature and colour distances with GT camera motion; written to CSV + txt."""
    if not rows:
        print("no GT cameras in the batches (load_depth=True?): drift analysis skipped")
        return
    import csv
    keys = ["sample", "i", "j", "rot_deg", "trans", "feat_dist", "color_dist"]
    with open(osp.join(root, f"{model_name}_view_drift.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in keys[3:]}
    lines = [f"{model_name}: {len(rows)} view pairs",
             f"  mean feature distance (mean-token gap / spread): {a['feat_dist'].mean():.4f}",
             f"  mean PCA colour distance:                        {a['color_dist'].mean():.4f}"]
    for d in ("feat_dist", "color_dist"):
        for g in ("rot_deg", "trans"):
            pear = np.corrcoef(a[d], a[g])[0, 1]
            spear = np.corrcoef(_rank(a[d]), _rank(a[g]))[0, 1]
            lines.append(f"  corr({d:10s}, {g:7s})  pearson {pear:+.3f}  spearman {spear:+.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(osp.join(root, f"{model_name}_view_drift.txt"), "w") as f:
        f.write(txt + "\n")



# =============================================================================
# Correspondence analysis (always on when the batch has GT depth + cameras)
# =============================================================================
DISP_BINS = [0, 1, 2, 4, 8, 1e9]                       # pixel displacement of a correspondence, in patches


def _patch_depth(depth, mask, h, w):
    """depth/mask (S,H,W) -> median valid depth per patch (S,h,w) (0 where < half the patch is valid)."""
    S, H, W = depth.shape
    if H % h == 0 and W % w == 0:
        d = depth.reshape(S, h, H // h, w, W // w).permute(0, 1, 3, 2, 4).reshape(S, h, w, -1)
        m = mask.reshape(S, h, H // h, w, W // w).permute(0, 1, 3, 2, 4).reshape(S, h, w, -1)
        dm = torch.where(m, d, torch.full_like(d, float("nan")))
        med = dm.nanmedian(-1).values
        ok = m.float().mean(-1) >= 0.5
        return torch.where(ok & torch.isfinite(med), med, torch.zeros_like(med))
    ys = ((torch.arange(h) + 0.5) * H / h).long().clamp(max=H - 1)
    xs = ((torch.arange(w) + 0.5) * W / w).long().clamp(max=W - 1)
    d, m = depth[:, ys][:, :, xs], mask[:, ys][:, :, xs]
    return torch.where(m, d, torch.zeros_like(d))


def correspondences(tok, geo, grid, l2norm=True, occl_tol=0.05):
    """GT patch correspondences between every ordered pair of views (reprojection with GT depth + cameras, occlusion-
    checked), and for each: pixel displacement (patches), feature distance, and whether nearest-neighbour matching finds
    it. Returns dict of flat numpy arrays, or None without GT."""
    if geo is None:
        return None
    depth, mask, K, w2c = (g.double().cpu() if g.dtype != torch.bool else g.cpu() for g in geo)
    S, H, W = depth.shape
    h, w = grid
    X = tok.double().cpu()
    if l2norm:
        X = F.normalize(X, dim=-1)
    pd = _patch_depth(depth, mask, h, w)                                            # (S,h,w)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    u0, v0 = (xs + 0.5) * W / w, (ys + 0.5) * H / h                                 # patch centres in pixels
    pix = torch.stack([u0, v0, torch.ones_like(u0)], -1).reshape(-1, 3)              # (P,3)
    c2w = torch.linalg.inv(w2c)
    out = {k: [] for k in ("disp", "dist", "rel", "nn_exact", "nn_1", "rot_deg")}
    for i in range(S):
        z = pd[i].reshape(-1)
        valid_i = z > 0
        cam = (torch.linalg.inv(K[i]) @ pix.T).T * z[:, None]
        world = (c2w[i, :3, :3] @ cam.T).T + c2w[i, :3, 3]
        for j in range(S):
            if j == i:
                continue
            cj = (w2c[j, :3, :3] @ world.T).T + w2c[j, :3, 3]
            zj = cj[:, 2]
            uv = (K[j] @ cj.T).T
            uj, vj = uv[:, 0] / uv[:, 2].clamp_min(1e-9), uv[:, 1] / uv[:, 2].clamp_min(1e-9)
            inside = valid_i & (zj > 1e-6) & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
            if inside.sum() == 0:
                continue
            ui, vi = uj.clamp(0, W - 1).long(), vj.clamp(0, H - 1).long()
            dj = depth[j, vi, ui]
            vis = inside & mask[j, vi, ui] & ((zj - dj).abs() / dj.clamp_min(1e-6) < occl_tol)
            if vis.sum() == 0:
                continue
            src = vis.nonzero()[:, 0]
            tr, tc = (vj[src] * h / H).long().clamp(0, h - 1), (uj[src] * w / W).long().clamp(0, w - 1)
            tgt = tr * w + tc
            sim = X[i, src] @ X[j].T                                                 # (n, P) cosine
            d_all = 1 - sim                                                          # cosine distance
            d_gt = d_all[torch.arange(len(src)), tgt]
            nn = sim.argmax(1)
            nr, nc = nn // w, nn % w
            out["disp"].append((torch.stack([uj[src] - u0.reshape(-1)[src], vj[src] - v0.reshape(-1)[src]], -1)
                                .norm(dim=-1) / (W / w)).numpy())
            out["dist"].append(d_gt.numpy())
            out["rel"].append((d_gt / d_all.mean(1).clamp_min(1e-9)).numpy())        # vs a random token of view j
            out["nn_exact"].append((nn == tgt).numpy())
            out["nn_1"].append((((nr - tr).abs() <= 1) & ((nc - tc).abs() <= 1)).numpy())
            cos = ((torch.trace(w2c[i, :3, :3] @ w2c[j, :3, :3].T) - 1) / 2).clamp(-1, 1)
            out["rot_deg"].append(np.full(len(src), float(torch.rad2deg(torch.acos(cos)))))
    if not out["disp"]:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def corr_summary(per_layer, model_name, root):
    """Per layer: matching accuracy and feature distance of GT correspondences, binned by pixel displacement; Spearman of
    distance with displacement and with rotation. Writes <root>/correspondence/<model>_correspondence.{txt,csv,png}."""
    per_layer = {l: {k: np.concatenate([c[k] for c in cs]) for k in cs[0]} for l, cs in per_layer.items() if cs}
    if not per_layer:
        print("no GT depth + cameras in the batches (load_depth=True?): correspondence analysis skipped")
        return
    import csv
    out = osp.join(root, "correspondence")
    os.makedirs(out, exist_ok=True)
    edges = DISP_BINS
    labels = [f"{edges[b]}-{edges[b + 1]}" if edges[b + 1] < 1e8 else f">{edges[b]}" for b in range(len(edges) - 1)]
    lines, rows = [f"{model_name}: GT patch correspondences (occlusion-checked), displacement in patches"], []
    for l, a in per_layer.items():
        sp_disp = np.corrcoef(_rank(a["rel"]), _rank(a["disp"]))[0, 1]
        sp_rot = np.corrcoef(_rank(a["rel"]), _rank(a["rot_deg"]))[0, 1]
        lines.append(f"  L{l}: {len(a['disp'])} correspondences | NN match acc exact {a['nn_exact'].mean():.3f}  "
                     f"within 1 patch {a['nn_1'].mean():.3f} | rel. distance {a['rel'].mean():.3f} | "
                     f"spearman(rel dist, displacement) {sp_disp:+.3f}  (rel dist, rotation) {sp_rot:+.3f}")
        b = np.digitize(a["disp"], edges[1:-1])
        for k, lab in enumerate(labels):
            sel = b == k
            if sel.sum() == 0:
                continue
            row = dict(layer=l, disp_bin=lab, n=int(sel.sum()), rel_dist=float(a["rel"][sel].mean()),
                       cos_dist=float(a["dist"][sel].mean()), acc_exact=float(a["nn_exact"][sel].mean()),
                       acc_1patch=float(a["nn_1"][sel].mean()))
            rows.append(row)
            lines.append(f"      disp {lab:>6s}: n {row['n']:6d}  rel dist {row['rel_dist']:.3f}  "
                         f"acc exact {row['acc_exact']:.3f}  acc 1-patch {row['acc_1patch']:.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(osp.join(out, f"{model_name}_correspondence.txt"), "w") as f:
        f.write(txt + "\n")
    with open(osp.join(out, f"{model_name}_correspondence.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for l in per_layer:
        rs = [r for r in rows if r["layer"] == l]
        x = [labels.index(r["disp_bin"]) for r in rs]
        ax[0].plot(x, [r["rel_dist"] for r in rs], "o-", label=f"block {l}")
        ax[1].plot(x, [r["acc_1patch"] for r in rs], "o-", label=f"block {l}")
    for a_, t in zip(ax, ["feature distance of GT match / random token", "NN matching acc. (within 1 patch)"]):
        a_.set_xticks(range(len(labels))); a_.set_xticklabels(labels)
        a_.set_xlabel("pixel displacement of the correspondence (patches)"); a_.set_title(t, fontsize=10)
        a_.grid(alpha=0.3)
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    fig.suptitle(model_name); fig.tight_layout()
    fig.savefig(osp.join(out, f"{model_name}_correspondence.png"), dpi=150); plt.close(fig)


def dump_sample(root, model_name, sample, r, pointcloud=False, save_arrays=True):
    for d in ("images", "feature_pca"):
        os.makedirs(osp.join(root, d), exist_ok=True)
    for v, n in enumerate(r["names"]):
        name = f"{model_name}_{sample}_{n}"
        Image.fromarray(r["rgb"][v]).save(osp.join(root, "images", f"{name}.png"))
        Image.fromarray(r["near"][v]).save(osp.join(root, "feature_pca", f"{name}_pca.png"))
        Image.fromarray(r["smooth"][v]).save(osp.join(root, "feature_pca", f"{name}_pca_smooth.png"))

        # breakpoint()
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
    # if save_arrays:
    #     os.makedirs(osp.join(root, "arrays"), exist_ok=True)
    #     np.savez_compressed(osp.join(root, "arrays", f"{model_name}_{sample}_pca.npz"),
    #                         view_names=np.array(r["names"]), pca=r["Y"], components=r["comps"],
    #                         explained_variance=r["var"], grid=np.array(r["grid"]))


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
    p.add_argument("--pointcloud", action="store_true", help="also write the GT point cloud coloured with the PCA")
    p.add_argument("--no_arrays", action="store_true")
    p.add_argument("--out_dir", default="results/figures/feature_pca")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, resolve_ckpt(args.ckpt) if args.ckpt else None, device)
    args.l2norm, args.center_views = True, False          # always L2-normalised tokens, no per-view centring
    layers = auto_layers(model)
    print(f"layers: {layers}")
    root = osp.join(args.out_dir, args.model_name)
    drift_rows = {l: [] for l in layers}
    corr_rows = {l: [] for l in layers}
    for sample, batch in iterate(args.dataset, args.indices, args.num_workers):
        res = run_sample(model, batch, device, args, layers)
        for layer, r in res.items():
            name = f"{args.model_name}_L{layer}"                          # <out_dir>/<model_name>/L<layer>/...
            dump_sample(osp.join(root, f"L{layer}"), name, sample, r, pointcloud=args.pointcloud,
                        save_arrays=not args.no_arrays)
            drift_rows[layer] += [dict(sample=sample, **d) for d in r["drift"]]
            if r["corr"] is not None:
                corr_rows[layer].append(r["corr"])
            print(f"{sample}  [{r['kind']} L{layer}, grid {r['grid'][0]}x{r['grid'][1]}]  "
                  f"explained variance of PC1-3: {np.round(r['var'], 3)}")
        dump_layer_grid(root, args.model_name, sample, res)
    for layer in layers:
        lroot = osp.join(root, f"L{layer}")
        os.makedirs(lroot, exist_ok=True)
        drift_summary(drift_rows[layer], f"{args.model_name}_L{layer}", lroot)
    corr_summary(corr_rows, args.model_name, root)
    print("done ->", root)


if __name__ == "__main__":
    main()