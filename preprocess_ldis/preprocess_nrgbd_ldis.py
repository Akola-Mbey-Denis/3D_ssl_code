"""
Mesh LDIs for EVERY valid frame of NRGBD, at 224 x 224 (same settings as dump_ldis.py for ScanNet / Matterport3D /
ARKitScenes).

Input (raw NRGBD, read as in nrgbd_overlaps.py):
    <root>/<scene>/images/img<i>.png        RGB, 640 x 480
                   depth/depth<i>.png       uint16 millimetres (-> metres, 0 = invalid)
                   poses.txt                4 lines per frame, OpenGL cam2world (frame i = block i); 'nan' = invalid frame
                   gt_mesh.ply              full ground-truth mesh (same world frame as poses.txt)
    Intrinsics are the same for every scene: fx = fy = 554.2562584220408, cx = 320, cy = 240.
    The mesh is read from <root>/<scene>/<mesh_name>, or from <root>/meshes/<scene>/<mesh_name> if not there.
    Use the FULL gt_mesh.ply: gt_mesh_culled.ply has the unobserved geometry removed (exactly the hidden surfaces an
    LDI needs), neural_rgbd.ply is a reconstruction, not ground truth.

Every frame is cropped / resized to 224 x 224 exactly like the dataloaders (crop around the principal point, then
resize, aug_crop=-1) -> K at 224, then one ray per pixel is cast into the mesh (trimesh + embree, all hits), hits
outside [min_depth, max_depth] dropped, consecutive layers >= min_sep apart, first num_layers kept.

Output, one file per frame, named like the frame:
    <out_root>/LDIs/<scene>/img<i>.npz       (out_root defaults to root)
        ldi  (224, 224, 4) float16   layer k = k-th surface along the pixel's ray, near -> far, 0 = no surface
                                     z-depth in metres by default (--ldi_type ray_distance for ray distances)
    Frames with a nan pose are skipped. A frame whose .npz already exists is skipped (re-runs only compute the
    missing frames; the mesh is loaded only if a scene still has missing frames).

Back faces: hidden layers keep only front-facing hits by default (as for Matterport3D / ARKitScenes; NRGBD meshes are
clean synthetic meshes); --keep_back_faces keeps them (as the ScanNet default).

Logs are only printed (no files): [SKIP] / [WARN] per frame, one summary line per scene. Layer 1 should agree with
the NRGBD depth maps almost everywhere (they were rendered from this mesh); a low agreement means a pose / mesh-frame
problem, not noisy data.

Usage:
    python dump_ldis_nrgbd.py --root /scratch/dma9300/nrgbd/NRGBD/ --start_idx 0 --end_idx 10   # scenes 0..9
    python dump_ldis_nrgbd.py --root ... --scenes breakfast_room --max_frames 5 --plot --save_ply   # quick check
    (--save_ply: <plot_dir>/<scene>/<frame>_ldi.ply, visible layer in image colours, hidden layers coloured by
     their surface normal (n + 1) / 2, all views in the same world frame -> open several frames together)
"""

import argparse
import os
import os.path as osp
import re

import imageio.v2 as imageio
import numpy as np
import PIL
import trimesh
from PIL import Image
from tqdm import tqdm

try:                                                       # embree: fast, all hits per ray
    from trimesh.ray.ray_pyembree import RayMeshIntersector
except ImportError:                                        # pure-python fallback (slow, same results)
    from trimesh.ray.ray_triangle import RayMeshIntersector
    print("[WARN] pyembree not available: using trimesh's slow ray tracer")

import utils.cropping as cropping


NRGBD_K = np.array([[554.2562584220408, 0.0, 320.0],
                    [0.0, 554.2562584220408, 240.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)


# ======================================================================
# NRGBD reading (from nrgbd_overlaps.py)
# ======================================================================
def load_image(file_path):
    return np.array(Image.open(file_path).convert("RGB"))


def load_depth(path):
    """NRGBD depth PNG (uint16, millimetres) -> metres, float32 (H, W), 0 = invalid (> 10 m or < 1 mm)."""
    depth = np.nan_to_num(imageio.imread(path).astype(np.float32), nan=0.0) / 1000.0
    depth[depth > 10] = 0.0
    depth[depth < 1e-3] = 0.0
    return depth


def load_poses(path):
    """poses.txt: 4 lines (one 4x4 OpenGL cam2world) per frame -> [4x4 array or None (nan = invalid frame)]."""
    with open(path, "r") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    poses = []
    for i in range(0, len(lines), 4):
        block = lines[i:i + 4]
        if any("nan" in line.lower() for line in block):
            poses.append(None)
        else:
            poses.append(np.array([[float(x) for x in line.split()] for line in block], dtype=np.float64))
    return poses


def frame_index(name):
    """img123.png -> 123 (numeric order: img0, img1, ..., img10, not img0, img1, img10, ...)."""
    return int(re.findall(r"\d+", osp.basename(name))[-1])


def opengl_to_opencv(c2w_gl):
    """OpenGL cam2world -> OpenCV cam2world (flip the camera y and z axes)."""
    c2w = c2w_gl.copy()
    c2w[:3, 1:3] *= -1.0
    return c2w


# ======================================================================
# Crop / resize (the dataloaders' _crop_resize_if_necessary, aug_crop=-1)
# ======================================================================
def crop_resize(image, depthmap, intrinsics, resolution, aug_crop=-1, mask=None, rng=None):
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
# Ray generation + mesh intersection (same as dump_ldis.py)
# ======================================================================
def generate_rays(K, c2w, H, W):
    """One ray per pixel: origins (H*W, 3), unit dirs (H*W, 3) in the world frame (OpenCV cam2world)."""
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).T
    dirs_cam = (np.linalg.inv(K) @ pixels).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    ray_dirs = dirs_cam @ c2w[:3, :3].T
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_origins = np.broadcast_to(c2w[:3, 3], ray_dirs.shape).copy()
    return ray_origins, ray_dirs


def generate_layered_depths(mesh, ray_origins, ray_dirs, H, W, max_layers=4, min_depth=0.001, max_depth=16.0,
                            front_only=True, min_sep=16.0 / 256, intersector=None, return_normals=False):
    """
    LDI by intersecting one ray per pixel with the mesh (all hits per ray).

    Hits outside [min_depth, max_depth] are dropped and the rest are sorted by distance.
    With front_only, hidden layers keep only front-facing hits (surfaces the ray enters);
    the first hit is always kept, so layer 1 is the visible surface.
    Consecutive kept layers are at least min_sep apart along the ray; the nearer hit wins.

    Returns
        dist   (H, W, L) float32  ray distance, 0 = none
        facing (H, W, L) int8     +1 front face, -1 back face, 0 none
        normal (H, W, L, 3) float32 world-frame face normal of each hit (only with return_normals)
    """
    L = max_layers
    n = H * W
    dist = np.zeros((n, L), np.float32)
    facing = np.zeros((n, L), np.int8)
    normal = np.zeros((n, L, 3), np.float32)

    if intersector is None:
        intersector = RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=True)

    if len(locations) > 0:
        depths = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        valid = (depths >= min_depth) & (depths <= max_depth)
        depths, index_ray, index_tri = depths[valid], index_ray[valid], index_tri[valid]

        order = np.lexsort((depths, index_ray))                       # by ray, then by distance
        depths, index_ray, index_tri = depths[order], index_ray[order], index_tri[order]

        is_front = (mesh.face_normals[index_tri] * ray_dirs[index_ray]).sum(-1) < 0

        if front_only:
            is_first = np.r_[True, index_ray[1:] != index_ray[:-1]]
            keep_hit = is_first | is_front
            depths, index_ray, is_front, index_tri = (depths[keep_hit], index_ray[keep_hit], is_front[keep_hit],
                                                      index_tri[keep_hit])

        if min_sep > 0 and len(depths) > 0:                            # separation to the previous KEPT hit
            first = np.searchsorted(index_ray, index_ray, side="left")
            rank = np.arange(len(index_ray)) - first
            last_kept = np.full(n, -np.inf)
            keep_sep = np.zeros(len(depths), bool)
            for j in range(rank.max() + 1):
                idx = np.nonzero(rank == j)[0]
                ok = depths[idx] - last_kept[index_ray[idx]] >= min_sep
                keep_sep[idx[ok]] = True
                last_kept[index_ray[idx[ok]]] = depths[idx[ok]]
            depths, index_ray, is_front, index_tri = (depths[keep_sep], index_ray[keep_sep], is_front[keep_sep],
                                                      index_tri[keep_sep])

        first = np.searchsorted(index_ray, index_ray, side="left")     # layer k of a ray = its k-th remaining hit
        layer = np.arange(len(index_ray)) - first
        keep = layer < L
        r, k = index_ray[keep], layer[keep]
        dist[r, k] = depths[keep]
        facing[r, k] = np.where(is_front[keep], 1, -1)
        normal[r, k] = mesh.face_normals[index_tri[keep]]

    if return_normals:
        return dist.reshape(H, W, L), facing.reshape(H, W, L), normal.reshape(H, W, L, 3)
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
    return {"pixels": int(m.sum()), "coverage": float(m.mean()), "median_abs_err_m": float(np.median(err)),
            "frac_within_tol": float((err < thr).mean())}


def plot_gt_depth_and_ldi(gt_depth, ldi_depth, save_path, vmin=0.0, vmax=10.0, cmap="Spectral_r", title=None):
    """GT depth and every LDI layer side by side, on one shared colour scale (blank = invalid / no hit)."""
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


def save_ldi_ply(ldi_depth, normals, rgb, K, c2w, ply_path, color_first="rgb"):
    """
    LDI of one frame as a world-frame point cloud (.ply):
        layer 1 (visible)   image colours (color_first="rgb") or its surface normal (color_first="normal")
        layers 2.. (hidden) surface normal, mapped to colours as (n + 1) / 2 -- shows the orientation of the
                            surfaces behind the visible one
    ldi_depth (H, W, L) z-depth, normals (H, W, L, 3) world-frame normals, rgb (H, W, 3) uint8, c2w OpenCV.
    """
    H, W, L = ldi_depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T              # (H, W, 3), z = 1
    pts, cols = [], []
    for k in range(L):
        m = ldi_depth[..., k] > 0
        if not m.any():
            continue
        cam = rays[m] * ldi_depth[..., k][m][:, None]
        pts.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
        if k == 0 and color_first == "rgb":
            cols.append(rgb[m].astype(np.float32) / 255.0)
        else:
            n = normals[..., k, :][m]
            n = n * np.where((n * (cam @ c2w[:3, :3].T)).sum(-1, keepdims=True) > 0, -1.0, 1.0)  # face the camera
            cols.append((n + 1.0) / 2.0)
    if not pts:
        return 0
    pts, cols = np.concatenate(pts), np.clip(np.concatenate(cols), 0, 1)
    os.makedirs(osp.dirname(ply_path) or ".", exist_ok=True)
    trimesh.PointCloud(pts, colors=(cols * 255).astype(np.uint8)).export(ply_path)
    return len(pts)


# ======================================================================
# Scenes
# ======================================================================
def list_scenes(root):
    """Scene folders with images/ and poses.txt, sorted ('meshes', 'LDIs', ... are not scenes)."""
    return [s for s in sorted(os.listdir(root))
            if osp.isdir(osp.join(root, s, "images")) and osp.exists(osp.join(root, s, "poses.txt"))]


def find_mesh(root, scene, mesh_name):
    for path in (osp.join(root, scene, mesh_name), osp.join(root, "meshes", scene, mesh_name)):
        if osp.exists(path):
            return path
    return None


def load_mesh(path, flip_faces=False):
    mesh = trimesh.load(path, process=False, force="mesh")
    if len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    if flip_faces:
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[:, ::-1], process=False)
    return mesh, RayMeshIntersector(mesh)


def process_scene(scene, args):
    scene_dir = osp.join(args.root, scene)
    out_dir = osp.join(args.out_root, "LDIs", scene)
    mesh_path = find_mesh(args.root, scene, args.mesh_name)
    if mesh_path is None:
        print(f"[MESH] {scene}: no {args.mesh_name} in {scene_dir} or {osp.join(args.root, 'meshes', scene)}")
        return "no_mesh"

    rgb_dir = osp.join(scene_dir, "images")
    frames = sorted((f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))), key=frame_index)
    poses = load_poses(osp.join(scene_dir, "poses.txt"))
    if len(frames) != len(poses):
        print(f"[WARN] {scene}: {len(frames)} images but {len(poses)} poses -- frames are matched by their index")
    n_invalid = sum(1 for f in frames if frame_index(f) >= len(poses) or poses[frame_index(f)] is None)
    frames = [f for f in frames if frame_index(f) < len(poses) and poses[frame_index(f)] is not None]
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not args.overwrite:                                           # only the frames without an LDI yet
        frames = [f for f in frames if not osp.exists(osp.join(out_dir, f"{osp.splitext(f)[0]}.npz"))]
    if not frames:
        return f"all frames already done ({n_invalid} invalid-pose frames skipped)"
    os.makedirs(out_dir, exist_ok=True)

    mesh, intersector = None, None                                   # loaded only if a frame needs it
    rng = np.random.default_rng(seed=42)                             # unused for square 224 x 224 crops
    n_written, agree, hidden, n_layers = 0, [], [], []
    for frame in tqdm(frames, desc=scene, leave=False):
        stem = osp.splitext(frame)[0]
        out_path = osp.join(out_dir, f"{stem}.npz")
        i = frame_index(frame)
        depth_path = osp.join(scene_dir, "depth", f"depth{i}.png")
        if not osp.exists(depth_path):
            print(f"[SKIP] {scene}/{frame}: no {depth_path}")
            continue
        c2w = opengl_to_opencv(poses[i])                             # OpenCV cam2world, same world frame as the mesh

        image = load_image(osp.join(rgb_dir, frame))
        depth = load_depth(depth_path)
        image, depth, K, _ = crop_resize(image, depth, NRGBD_K.copy(), (args.resolution, args.resolution),
                                         aug_crop=-1, mask=(depth > 0).astype(np.int32), rng=rng)
        depth = np.asarray(depth, np.float32)
        K = np.asarray(K, np.float64)
        H, W = depth.shape

        if mesh is None:
            mesh, intersector = load_mesh(mesh_path, flip_faces=args.flip_faces)

        ray_origins, ray_dirs = generate_rays(K=K, c2w=c2w, H=H, W=W)
        ldi_dist, facing, normals = generate_layered_depths(
            mesh, ray_origins, ray_dirs, H, W, max_layers=args.num_layers, min_depth=args.min_depth,
            max_depth=args.max_depth, front_only=not args.keep_back_faces, min_sep=args.min_sep,
            intersector=intersector, return_normals=True)
        ldi_depth = ray_distance_to_depth(ldi_dist, K)

        if args.save_ply:
            save_ldi_ply(ldi_depth, normals, np.asarray(image), K, c2w,
                         osp.join(args.plot_dir, scene, f"{stem}_ldi.ply"), color_first=args.ply_first_layer)

        if args.plot:
            plot_gt_depth_and_ldi(depth, ldi_depth, osp.join(args.plot_dir, scene, f"{stem}_gt_depth_and_ldi.png"),
                                  title=f"{scene}/{stem}")

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
        agree.append(stats.get("frac_within_tol", np.nan))
        hidden.append(float((ldi_dist[..., 1] > 0).mean()))
        n_layers.append(float((ldi_dist > 0).sum(-1).mean()))
        if stats.get("frac_within_tol", 1.0) < 0.5:
            print(f"[WARN] {scene}/{stem}: only {stats['frac_within_tol']:.0%} of layer 1 agrees with the depth "
                  f"map (median error {stats.get('median_abs_err_m', float('nan')):.3f} m) -- check the pose "
                  f"convention / mesh frame / mesh file")
        if stats["frac_layer1_front_facing"] < 0.5 and has1.any():
            print(f"[WARN] {scene}/{stem}: layer 1 is mostly back-facing -- inverted face winding? (--flip_faces)")

    if n_written == 0:
        return "0 written"
    return (f"{n_written} written ({n_invalid} invalid-pose frames skipped) | layer 1 within tol of depth: "
            f"{np.nanmean(agree):.1%} | pixels with a hidden layer: {np.mean(hidden):.1%} | "
            f"layers/pixel: {np.mean(n_layers):.2f}")


def get_args():
    p = argparse.ArgumentParser(description="Mesh LDIs (224 x 224, 4 layers, float16) for every valid NRGBD frame")
    p.add_argument("--root", type=str, default="/scratch/dma9300/nrgbd/NRGBD/", help="NRGBD root, one folder per scene")
    p.add_argument("--out_root", type=str, default=None, help="LDIs go to <out_root>/LDIs/<scene>/ (default: root)")
    p.add_argument("--mesh_name", type=str, default="gt_mesh.ply", help="full GT mesh (not gt_mesh_culled.ply)")
    p.add_argument("--flip_faces", action="store_true", help="invert the face winding (if layer 1 is mostly back-facing)")
    p.add_argument("--keep_back_faces", action="store_true",
                   help="keep back-facing hits in the hidden layers (default: front-facing only, as Matterport / ARKit)")
    p.add_argument("--scenes", nargs="+", default=None, help="only these scenes (default: all, or --start_idx / --end_idx)")
    p.add_argument("--start_idx", type=int, default=0, help="first scene (index in the sorted scene list)")
    p.add_argument("--end_idx", type=int, default=None, help="end scene, exclusive (default: all)")
    p.add_argument("--max_frames", type=int, default=None, help="debug: only the first N valid frames per scene")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ldi_type", choices=["z_depth", "ray_distance"], default="z_depth")
    p.add_argument("--min_depth", type=float, default=0.001)
    p.add_argument("--max_depth", type=float, default=16.0)
    p.add_argument("--min_sep", type=float, default=16.0 / 256.0,
                   help="minimum ray-distance separation between consecutive layers (m); 0 = off")
    p.add_argument("--plot", action="store_true", help="also save GT depth + LDI layer figures (for checking)")
    p.add_argument("--save_ply", action="store_true",
                   help="also save each frame's LDI as a world-frame point cloud: hidden layers coloured by surface normal")
    p.add_argument("--ply_first_layer", choices=["rgb", "normal"], default="rgb",
                   help="colour of the visible layer in the .ply (image colours or surface normal)")
    p.add_argument("--plot_dir", type=str, default="./ldi_check_nrgbd", help="figures / .ply files go here")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.out_root = args.out_root or args.root
    scenes = list_scenes(args.root)
    if args.scenes:
        missing = [s for s in args.scenes if s not in scenes]
        if missing:
            raise SystemExit(f"unknown scenes {missing}; available: {scenes}")
        todo = args.scenes
    else:
        end = len(scenes) if args.end_idx is None else min(args.end_idx, len(scenes))
        todo = scenes[args.start_idx:end]
    print(f"{len(scenes)} scenes in {args.root}; processing {len(todo)}: {todo} -> {osp.join(args.out_root, 'LDIs')}")

    for scene in tqdm(todo, desc="scenes"):
        status = process_scene(scene, args)
        tqdm.write(f"{scene}: {status}")