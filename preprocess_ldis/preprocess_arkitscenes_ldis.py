"""
ARKitScenes mesh LDIs for EVERY frame of the dumped ARKitScenes (prepare_arkitscenes.py), at 224 x 224
(same output as dump_ldis.py / dump_ldis_matterport.py).

Input (the dump):
    <dump_root>/<fold>/<video_id>/rgb/<frame>.(png|jpg)      fold = Training / Validation
                                  depth/<frame>.npy           metres, 0 = invalid
                                  cameras.npz                 per-frame names, K and RT_c2w (OpenCV cam2world)
                                  mesh.ply                    the scan's mesh, same world frame as the poses
An ARKitScenes mesh covers one scan of one room / apartment, so the whole mesh is ray-cast for every view
(no per-view face subset as in Matterport).

Every frame is cropped / resized to 224 x 224 exactly like the dataloaders (crop around the principal point,
then resize, aug_crop=-1) -> K at 224, then one ray per pixel centre, all mesh hits in [min_depth, max_depth],
back faces dropped from the hidden layers (as the ARKitScenes LDI script), consecutive layers >= min_sep apart,
first num_layers kept.

Output, one file per frame (same name as the frame):
    <out_root>/LDIs/train/<video_id>/<frame>.npz   Training scans   (out_root defaults to dump_root)
    <out_root>/LDIs/val/<video_id>/<frame>.npz     Validation scans
        ldi  (224, 224, 4) float16   layer k = k-th surface along the pixel's ray, near -> far, 0 = no surface
                                     z-depth in metres by default (--ldi_type ray_distance for ray distances)
    A frame whose .npz already exists is skipped (re-runs only compute the missing frames; the mesh is
    loaded only if a scan still has missing frames).

Train the LDI probes with aug_crop=-1 (no random scale jitter), so the image crop and the LDI line up.
Logs are only printed (no files): [SKIP] / [WARN] per frame, one summary line per scan.

Usage:
    python dump_ldis_arkitscenes.py --dump_root /scratch/dma9300/preprocessed_arkitscenes --start_idx 0 --end_idx 200
"""

import argparse
import os
import os.path as osp

import numpy as np
import PIL
import trimesh
from PIL import Image
from tqdm import tqdm
from trimesh.ray.ray_pyembree import RayMeshIntersector

import utils.cropping as cropping


# ======================================================================
# Loading
# ======================================================================
def load_image(file_path):
    return np.array(Image.open(file_path).convert("RGB"))


def load_depth(depth_path):
    """Depth in metres (.npy), non-finite -> 0."""
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def rotate_inverse(se3):
    assert se3.shape == (4, 4), f"Expected (4, 4), got {se3.shape}"
    R, T = se3[:3, :3], se3[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T
    return out


def crop_resize(image, depthmap, intrinsics, resolution, aug_crop=-1, mask=None, rng=None):
    """The dataloaders' _crop_resize_if_necessary (crop around the principal point, then resize)."""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, (l, t, r, b), mask=mask)

    W, H = image.size
    assert resolution[0] >= resolution[1]
    if H > 1.1 * W:
        resolution = resolution[::-1]
    elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
        if rng.integers(2):
            resolution = resolution[::-1]

    target_resolution = np.array(resolution)
    if aug_crop > 1:
        target_resolution += rng.integers(0, aug_crop)
    image, depthmap, intrinsics, mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, mask=mask)

    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask=mask)
    return image, depthmap, intrinsics2, mask


# ======================================================================
# Ray generation + mesh intersection (from the ScanNet mesh-LDI script)
# ======================================================================
def generate_rays(K, RT_w2c, H, W):
    """One ray per pixel: origins (H*W, 3), unit dirs (H*W, 3) in world frame."""
    RT_c2w = rotate_inverse(RT_w2c)
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).T
    dirs_cam = (np.linalg.inv(K) @ pixels).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    R, T = RT_c2w[:3, :3], RT_c2w[:3, 3]
    ray_dirs = dirs_cam @ R.T
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_origins = np.broadcast_to(T, ray_dirs.shape).copy()
    return ray_origins, ray_dirs


def generate_layered_depths(mesh, ray_origins, ray_dirs, H, W, max_layers=4, min_depth=0.001, max_depth=16.0,
                            front_only=False, min_sep=16.0 / 256, intersector=None):
    """
    LDI by intersecting one ray per pixel with the mesh (trimesh + embree, all hits per ray).

    Hits outside [min_depth, max_depth] are dropped and the rest are sorted by distance.
    With front_only, hidden layers keep only front-facing hits (surfaces the ray enters);
    the first hit is always kept, so layer 1 is the visible surface.
    Consecutive kept layers are at least min_sep apart along the ray; the nearer hit wins.

    Returns
        dist   (H, W, L) float32  ray distance, 0 = none
        facing (H, W, L) int8     +1 front face, -1 back face, 0 none
    """
    L = max_layers
    n = H * W
    dist = np.zeros((n, L), np.float32)
    facing = np.zeros((n, L), np.int8)

    if intersector is None:
        intersector = RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=True)

    if len(locations) > 0:
        depths = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        valid = (depths >= min_depth) & (depths <= max_depth)
        depths, index_ray, index_tri = depths[valid], index_ray[valid], index_tri[valid]

        # sort hits by ray, then by distance
        order = np.lexsort((depths, index_ray))
        depths, index_ray, index_tri = depths[order], index_ray[order], index_tri[order]

        is_front = (mesh.face_normals[index_tri] * ray_dirs[index_ray]).sum(-1) < 0

        if front_only:
            is_first = np.r_[True, index_ray[1:] != index_ray[:-1]]
            keep_hit = is_first | is_front
            depths, index_ray, is_front = depths[keep_hit], index_ray[keep_hit], is_front[keep_hit]

        # minimum separation between consecutive kept layers (measured to the previous KEPT hit)
        if min_sep > 0 and len(depths) > 0:
            first = np.searchsorted(index_ray, index_ray, side="left")
            rank = np.arange(len(index_ray)) - first
            last_kept = np.full(n, -np.inf)
            keep_sep = np.zeros(len(depths), bool)
            for j in range(rank.max() + 1):
                idx = np.nonzero(rank == j)[0]
                ok = depths[idx] - last_kept[index_ray[idx]] >= min_sep
                keep_sep[idx[ok]] = True
                last_kept[index_ray[idx[ok]]] = depths[idx[ok]]
            depths, index_ray, is_front = depths[keep_sep], index_ray[keep_sep], is_front[keep_sep]

        # layer k of a ray = its k-th remaining hit
        first = np.searchsorted(index_ray, index_ray, side="left")
        layer = np.arange(len(index_ray)) - first
        keep = layer < L
        r, k = index_ray[keep], layer[keep]
        dist[r, k] = depths[keep]
        facing[r, k] = np.where(is_front[keep], 1, -1)

    return dist.reshape(H, W, L), facing.reshape(H, W, L)


def ray_distance_to_depth(dist, K):
    """Ray distances (H, W, L) -> camera-frame z-depths; 0 (no hit) stays 0."""
    H, W = dist.shape[:2]
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    return (dist / np.linalg.norm(rays, axis=-1)[..., None]).astype(np.float32)


def layer1_agreement(ldi_depth, target_depth, tol=0.05, rel_tol=0.02):
    """How well mesh layer 1 matches the depth map (both z-depths)."""
    m = (ldi_depth[..., 0] > 0) & (target_depth > 0)
    if not m.any():
        return {"pixels": 0}
    gt = target_depth[m]
    err = np.abs(ldi_depth[..., 0][m] - gt)
    thr = np.maximum(tol, rel_tol * gt)            # 5 cm, or 2% of the GT depth if larger
    return {
        "pixels": int(m.sum()),
        "coverage": float(m.mean()),
        "median_abs_err_m": float(np.median(err)),
        "frac_within_tol": float((err < thr).mean()),
    }


# ======================================================================
# Scenes
# ======================================================================
FOLD_DIR = {"Training": "train", "Validation": "val"}        # dump fold -> folder inside LDIs/


def list_scenes(dump_root, folds=("Training", "Validation")):
    """Scans "<fold>/<video_id>" whose dump finished (cameras.npz present), sorted."""
    scenes = []
    for fold in folds:
        fold_dir = osp.join(dump_root, fold)
        if not osp.isdir(fold_dir):
            continue
        scenes += [f"{fold}/{v}" for v in sorted(os.listdir(fold_dir))
                   if osp.exists(osp.join(fold_dir, v, "cameras.npz"))]
    return scenes


def _first(keys, candidates):
    return next((c for c in candidates if c in keys), None)


def load_cameras(scene_dir):
    """
    cameras.npz -> {frame stem: {"K": (3, 3), "RT_c2w": (4, 4)}}.
    Accepts per-frame arrays (frame names + K + RT_c2w, under the usual key names) or one pickled dict
    {frame name: {"K", "RT_c2w"}} as in the ScanNet cameras.pkl. Anything else raises with the keys found.
    """
    path = osp.join(scene_dir, "cameras.npz")
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    if len(keys) == 1:                                               # one pickled dict
        obj = data[next(iter(keys))]
        if obj.dtype == object and obj.shape == ():
            return {frame_key(strip_image_ext(name)): {"K": np.asarray(c["K"]), "RT_c2w": np.asarray(c["RT_c2w"])}
                    for name, c in obj.item().items()}

    k_names = _first(keys, ("frame_ids", "frame_names", "names", "image_names", "images", "frames", "rgb", "timestamps"))
    k_K = _first(keys, ("K", "intrinsics", "Ks", "intrinsic"))
    k_P = _first(keys, ("c2w", "RT_c2w", "cam2world", "poses", "extrinsics", "extrinsic"))
    if None in (k_names, k_K, k_P):
        raise ValueError(f"{path}: keys {sorted(keys)} -- set the frame-name / K / RT_c2w keys in load_cameras()")

    ids = data[k_names]
    Ks, Ps = np.asarray(data[k_K], np.float64), np.asarray(data[k_P], np.float64)
    if Ks.ndim == 2:                                                 # one K for every frame
        Ks = np.broadcast_to(Ks, (len(ids), 3, 3))
    if not (len(ids) == len(Ks) == len(Ps)):
        raise ValueError(f"{path}: {len(ids)} {k_names}, {len(Ks)} {k_K}, {len(Ps)} {k_P}")
    return {frame_key(strip_image_ext(n)): {"K": Ks[i], "RT_c2w": Ps[i]}
            for i, n in enumerate(ids)}


def strip_image_ext(name):
    """Base name without an image extension; other dots stay ("41069021_305.244.png" -> "41069021_305.244")."""
    name = osp.basename(str(name))
    stem, ext = osp.splitext(name)
    return stem if ext.lower() in (".png", ".jpg", ".jpeg") else name


def frame_key(name):
    """
    Key that matches a frame id from cameras.npz to its rgb file stem: "<video_id>_<timestamp>", "<timestamp>"
    and the timestamp as a number (so "305.240", "305.24" and 305.24 all match) give the same key.
    """
    tail = name.rsplit("_", 1)[-1]
    try:
        return f"{float(tail):.4f}"
    except ValueError:
        return name


def load_mesh(path, flip_faces=False):
    mesh = trimesh.load(path, process=False, force="mesh")
    if len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    if flip_faces:
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[:, ::-1], process=False)
    return mesh, RayMeshIntersector(mesh)



def plot_gt_depth_and_ldi(gt_depth, ldi_depth, save_path="ldis_gt_depthmap.png", vmin=0.0, vmax=16.0,
                          cmap="Spectral_r", title=None):
    """
    GT depth and every LDI layer side by side, on one shared color scale.

    gt_depth  (H, W)    z-depth, 0 = invalid
    ldi_depth (H, W, L) z-depth per layer, 0 = no hit
    Invalid pixels / missing hits are left blank.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = ldi_depth.shape[-1]
    panels = [("GT depth", gt_depth)] + [(f"LDI layer {k + 1}", ldi_depth[..., k]) for k in range(L)]

    fig, axs = plt.subplots(1, L + 1, figsize=(3.5 * (L + 1), 3.8), constrained_layout=True)
    for ax, (name, d) in zip(axs, panels):
        valid = np.isfinite(d) & (d > 0)
        im = ax.imshow(np.ma.masked_where(~valid, d), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} ({valid.mean():.0%})")
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label="depth (m)")
    if title is not None:
        fig.suptitle(title)

    os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def process_scene(scene, args):
    scene_dir = osp.join(args.dump_root, scene)
    fold, video_id = scene.split("/")
    out_dir = osp.join(args.out_root, "LDIs", FOLD_DIR.get(fold, fold), video_id)   # LDIs/train|val/<video_id>/<frame>.npz
    mesh_path = osp.join(args.mesh_dir or args.dump_root, scene, args.mesh_name)
    if not osp.exists(mesh_path):
        print(f"[MESH] {scene}: no mesh at {mesh_path}, skipping the scan")
        return "no_mesh"

    cameras = load_cameras(scene_dir)
    rgb_dir = osp.join(scene_dir, "rgb")
    frames = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not args.overwrite:                                           # only the frames without an LDI yet
        frames = [f for f in frames if not osp.exists(osp.join(out_dir, f"{osp.splitext(f)[0]}.npz"))]
    if not frames:
        return "all frames already done"
    os.makedirs(out_dir, exist_ok=True)

    mesh, intersector = None, None                                   # loaded only if a frame needs it
    rng = np.random.default_rng(seed=42)                             # unused for square 224 x 224 crops
    n_written, agree, hidden, n_layers = 0, [], [], []
    for frame in tqdm(frames, desc=scene, leave=False):
        stem = osp.splitext(frame)[0]
        out_path = osp.join(out_dir, f"{stem}.npz")
        cam = cameras.get(frame_key(stem))
        if cam is None:
            print(f"[SKIP] {scene}/{frame}: not in cameras.npz")
            continue
        c2w = np.asarray(cam["RT_c2w"], np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        if not np.isfinite(c2w).all():
            print(f"[SKIP] {scene}/{frame}: invalid pose")
            continue

        image = load_image(osp.join(rgb_dir, frame))
        depth = load_depth(osp.join(scene_dir, "depth", f"{stem}.npy"))
        K = np.asarray(cam["K"], np.float64)
        _, depth, K, _ = crop_resize(image, depth, K, (args.resolution, args.resolution),
                                     aug_crop=-1, mask=(depth > 0).astype(np.int32), rng=rng)
        depth = np.asarray(depth, np.float32)
        H, W = depth.shape

        if mesh is None:
            mesh, intersector = load_mesh(mesh_path, flip_faces=args.flip_faces)

        ray_origins, ray_dirs = generate_rays(K=K, RT_w2c=rotate_inverse(c2w), H=H, W=W)
        ldi_dist, facing = generate_layered_depths(
            mesh, ray_origins, ray_dirs, H, W, max_layers=args.num_layers, min_depth=args.min_depth,
            max_depth=args.max_depth, front_only=not args.keep_back_faces, min_sep=args.min_sep, intersector=intersector)
        ldi_depth = ray_distance_to_depth(ldi_dist, K)

        
        # plot_gt_depth_and_ldi(gt_depth = depth, ldi_depth = ldi_depth, save_path=f"./ldis_gt_depthmap-{stem}.png", vmin=0.0, vmax=16.0, cmap="Spectral_r", title=None)
        # breakpoint()


        ldi = ldi_depth if args.ldi_type == "z_depth" else ldi_dist
        ldi = np.where(np.isfinite(ldi) & (ldi > 0), ldi, 0.0).astype(np.float16)   # invalid / no surface = 0
        tmp = out_path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, ldi=ldi)
        os.replace(tmp, out_path)                                     # no half-written files
        n_written += 1

        target = depth.copy()
        target[target > args.max_depth] = 0.0
        stats = layer1_agreement(ldi_depth, target)
        has1 = facing[..., 0] != 0
        stats["frac_layer1_front_facing"] = float((facing[..., 0][has1] == 1).mean()) if has1.any() else 0.0
        stats["layers_per_pixel_mean"] = float((ldi_dist > 0).sum(-1).mean())
        stats["frac_pixels_with_hidden_layer"] = float((ldi_dist[..., 1] > 0).mean())
        agree.append(stats.get("frac_within_tol", np.nan))
        hidden.append(stats["frac_pixels_with_hidden_layer"])
        n_layers.append(stats["layers_per_pixel_mean"])
        if stats.get("frac_within_tol", 1.0) < 0.5:
            print(f"[WARN] {scene}/{stem}: only {stats['frac_within_tol']:.0%} of layer 1 agrees with the depth "
                  f"map -- check pose convention / depth units / mesh frame")
        if stats["frac_layer1_front_facing"] < 0.5 and has1.any():
            print(f"[WARN] {scene}/{stem}: layer 1 is mostly back-facing -- inverted face winding? (--flip_faces)")

    if n_written == 0:
        return "0 written"
    return (f"{n_written} written | layer 1 within tol of depth: {np.nanmean(agree):.1%} | "
            f"pixels with a hidden layer: {np.mean(hidden):.1%} | layers/pixel: {np.mean(n_layers):.2f}")


def get_args():
    p = argparse.ArgumentParser(description="ARKitScenes mesh LDIs (224 x 224, 4 layers, float16) for every frame of the dump")
    p.add_argument("--dump_root", type=str, default="/scratch/dma9300/preprocessed_arkitscenes",
                   help="dumped ARKitScenes: <dump_root>/<fold>/<video_id>/{rgb, depth, cameras.npz, mesh.ply}")
    p.add_argument("--folds", type=str, nargs="+", default=["Training", "Validation"])
    p.add_argument("--out_root", type=str, default=None, help="LDIs go to <out_root>/LDIs/<scene>/ (default: dump_root)")
    p.add_argument("--mesh_dir", type=str, default=None, help="meshes at <mesh_dir>/<fold>/<video_id>/<mesh_name> (default: dump_root)")
    p.add_argument("--mesh_name", type=str, default="mesh.ply", help="mesh file name inside each scene folder")
    p.add_argument("--flip_faces", action="store_true", help="invert the face winding (if layer 1 is mostly back-facing)")
    p.add_argument("--keep_back_faces", action="store_true",
                   help="keep back-facing hits in the hidden layers (default: dropped, as the ARKitScenes script)")
    p.add_argument("--start_idx", type=int, default=0, help="first scan (index in the sorted scan list)")
    p.add_argument("--end_idx", type=int, default=None, help="end scene, exclusive (default: all)")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ldi_type", choices=["z_depth", "ray_distance"], default="z_depth",
                   help="z_depth: same quantity as the depth maps (default); ray_distance: as the ARKitScenes LDI script")
    p.add_argument("--min_depth", type=float, default=0.10)
    p.add_argument("--max_depth", type=float, default=16.0)
    p.add_argument("--min_sep", type=float, default=16.0 / 256.0,
                   help="minimum ray-distance separation between consecutive layers (m); 0 = off")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.out_root = args.out_root or args.dump_root
    scenes = list_scenes(args.dump_root, folds=args.folds)
    end = len(scenes) if args.end_idx is None else min(args.end_idx, len(scenes))
    todo = scenes[args.start_idx:end+1]
    print(f"{len(scenes)} scans in {args.dump_root}; processing [{args.start_idx}, {end}) = {len(todo)} scans "
          f"-> {osp.join(args.out_root, 'LDIs')}")

    for scene in tqdm(todo, desc="scans"):
        status = process_scene(scene, args)
        tqdm.write(f"{scene}: {status}")