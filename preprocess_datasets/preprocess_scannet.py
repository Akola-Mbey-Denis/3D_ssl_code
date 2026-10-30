"""
ScanNet -> frames at the depth-map resolution (640x480), color aligned to the depth camera, one folder per scene.
Same output format as dump_hypersim_crops.py, so the overlap / sequence / LMDB scripts can be reused.

Input: every scene folder in --scans_root (train and val alike, split files are not used):
  <scans_root>/<scene>/color/<i>.jpg, depth/<i>.png (uint16 mm), pose/<i>.txt, intrinsic/intrinsic_depth.txt
  (output of the .sens reader; the .sens files are not read here). Scenes that are not extracted are skipped.

Steps per scene:
  1. check every frame: finite pose, finite intrinsics, >= MIN_VALID_FRAC pixels with depth > MIN_DEPTH
  2. keep up to --max_frames (1000) good frames, evenly spaced over the video
  3. color (1296x968) is resized (Lanczos) to the depth-map size (640x480) and uses the depth intrinsics
     (same as CUT3R / standard ScanNet practice). Depth is NOT resized: everything is dumped at the
     depth-map size with K = K_depth (pass --short_side 518 to resize like the other datasets)
  4. dump (depth converted to meters):
       <out_root>/<scene>/rgb/<frame>.jpg        rgb, 640x480 = depth size   (frame = 6-digit original index)
       <out_root>/<scene>/depth/<frame>.npy      float32, meters, 0 = invalid
       <out_root>/<scene>/mesh.ply               copy of <scene>_vh_clean_2.ply (same world frame as RT_c2w)
       <out_root>/<scene>/cameras.pkl            written last (its presence = scene done):
            {"<frame>.jpg": {"K": (3,3) at dumped resolution (OpenCV),
                             "RT_c2w": (4,4) OpenCV camera-to-world,
                             "hw": dumped (H, W),
                             "depth": "<frame>.npy",
                             "src_rgb": original color path}}
  5. at the end, --scene_file is written: {scene: [dumped rgb paths]} for the overlap script

Usage:
  python -m preprocess_datasets.preprocess_scannet                                # all scenes, one after another
  python -m preprocess_datasets.preprocess_scannet --start_idx 0 --end_idx 10      # scenes 0..9 (sorted by name)
"""
import os
import shutil
import argparse
import pickle as pkl

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

SHORT_SIDE = 0            # 0 = keep the depth-map size (640x480); e.g. 518 to resize like the other datasets
MAX_FRAMES = 1000
CAMERAS_FILE = "cameras.pkl"
MIN_VALID_FRAC = 0.5      # fraction of pixels with depth > MIN_DEPTH (ScanNet depth has holes on glass / far walls)
MIN_DEPTH = 0.10          # meters
DEPTH_SHIFT = 1000.0      # ScanNet depth png / .sens: millimeters

SCANS_ROOT = "/projects/rps/dff6142/fouheylab/shared_datasets/scannetv2_best/scans"
OUT_ROOT = "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_scannet"
SCENE_FILE = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/scannet_low_res_overlap_scene_files.pkl"


# ---------------------------------------------------------------------------
# frame sources: extracted folders (reader output) or the .sens file
# ---------------------------------------------------------------------------

class ExtractedScene:
    """color/<i>.jpg, depth/<i>.png, pose/<i>.txt, intrinsic/intrinsic_depth.txt written by the .sens reader."""

    def __init__(self, scene_dir):
        self.dir = scene_dir
        self.K_depth = np.loadtxt(os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt"))[:3, :3]
        self.indices = sorted(int(f[:-4]) for f in os.listdir(os.path.join(scene_dir, "pose")) if f.endswith(".txt"))

    def pose(self, i):
        return np.loadtxt(os.path.join(self.dir, "pose", f"{i}.txt")).reshape(4, 4)

    def depth(self, i):
        """meters, 0 = invalid"""
        d = cv2.imread(os.path.join(self.dir, "depth", f"{i}.png"), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / DEPTH_SHIFT

    def color(self, i):
        return Image.open(os.path.join(self.dir, "color", f"{i}.jpg")).convert("RGB")

    def src(self, i):
        return os.path.join(self.dir, "color", f"{i}.jpg")

    def iterate(self, indices):
        for i in indices:
            yield i, self.color(i), self.depth(i), self.pose(i)


def missing_inputs(scans_root, scene):
    """Reason the scene cannot be read (None if everything needed is there)."""
    scene_dir = os.path.join(scans_root, scene)
    for sub in ("color", "depth", "pose", "intrinsic"):
        if not os.path.isdir(os.path.join(scene_dir, sub)):
            return f"not extracted (missing {sub}/)"
    if not os.path.isfile(os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt")):
        return "missing intrinsic/intrinsic_depth.txt"
    return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def is_valid(depth, K, RT_c2w):
    return (
        np.mean(depth > MIN_DEPTH) >= MIN_VALID_FRAC
        and abs(K[0, 1]) < 1e-6                 # no skew
        and np.all(np.isfinite(K))
        and np.all(np.isfinite(RT_c2w))         # ScanNet marks lost tracking with -inf poses
    )


def good_frames(src, max_frames=MAX_FRAMES):
    """Check every frame, keep up to max_frames good ones evenly spaced over the video."""
    K = src.K_depth
    good = []
    for i in tqdm(src.indices, desc=f"{os.path.basename(src.dir)} [check]", leave=False):
        if not all(os.path.isfile(os.path.join(src.dir, sub, f"{i}{ext}"))
                   for sub, ext in (("color", ".jpg"), ("depth", ".png"))):
            continue                                                        # incomplete frame
        pose = src.pose(i)
        if np.all(np.isfinite(pose)) and is_valid(src.depth(i), K, pose):     # skip depth read for lost poses
            good.append(i)
    n_good = len(good)
    if n_good > max_frames:
        good = [good[j] for j in np.linspace(0, n_good - 1, max_frames, dtype=int)]
    return good, n_good


def resize_short_side(image, depth, K, short_side=SHORT_SIDE):
    """
    Resize so min(H, W) == short_side, keep aspect ratio, using DUSt3R's
    cropping.rescale_image_depthmap (image: Lanczos / bicubic if upscaling, depth: nearest,
    K via camera_matrix_of_crop). force=True so lower-resolution frames are upscaled too.
    """
    import utils.cropping as cropping                      # only needed when resizing
    out = cropping.rescale_image_depthmap(image, depth, K, (short_side, short_side), force=True)
    image, depth, K = out[0], out[1], out[2]
    return image, np.asarray(depth, np.float32), np.asarray(K, np.float32)


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------

def dump_scene(job):
    scene, scans_root, out_root, max_frames, jpg_quality, short_side = job
    scene_dir = os.path.join(out_root, scene)
    if os.path.exists(os.path.join(scene_dir, CAMERAS_FILE)):
        return f"{scene}: done, skipping"

    reason = missing_inputs(scans_root, scene)
    if reason:
        return f"{scene}: SKIPPED ({reason})"
    src = ExtractedScene(os.path.join(scans_root, scene))
    K_depth = np.asarray(src.K_depth, np.float64)
    if not (np.all(np.isfinite(K_depth)) and abs(K_depth[0, 1]) < 1e-6):
        return f"{scene}: SKIPPED (bad depth intrinsics)"

    frames, n_good = good_frames(src, max_frames)
    if not frames:
        return f"{scene}: SKIPPED (no good frames out of {len(src.indices)})"

    for sub in ("rgb", "depth"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)

    cameras = {}
    for i, color, depth, pose in tqdm(src.iterate(frames), total=len(frames),
                                      desc=f"{scene} [dump]", leave=False):
        color = color.resize(depth.shape[::-1], resample=Image.Resampling.LANCZOS)   # color -> depth size, uses K_depth
        if short_side:
            image, depth, K = resize_short_side(color, depth, K_depth.copy(), short_side)
        else:                                                       # keep the depth-map size
            image, depth, K = color, depth.astype(np.float32), K_depth.astype(np.float32)

        name = f"{i:06d}"
        image.save(os.path.join(scene_dir, "rgb", name + ".jpg"), quality=jpg_quality)
        np.save(os.path.join(scene_dir, "depth", name + ".npy"), depth)
        cameras[name + ".jpg"] = {
            "K": np.asarray(K, np.float32),                           # (3,3) at dumped resolution, OpenCV
            "RT_c2w": np.asarray(pose, np.float32),                   # (4,4) OpenCV camera-to-world (mesh frame)
            "hw": tuple(int(x) for x in depth.shape),                 # dumped (H, W)
            "depth": name + ".npy",
            "src_rgb": src.src(i),
        }

    copy_mesh(scans_root, scene, scene_dir)

    tmp = os.path.join(scene_dir, CAMERAS_FILE + ".tmp")
    with open(tmp, "wb") as f:
        pkl.dump(cameras, f)
    os.replace(tmp, os.path.join(scene_dir, CAMERAS_FILE))        # written last = scene done
    return f"{scene}: {len(src.indices)} frames, {n_good} good -> {len(cameras)} dumped"


def copy_mesh(scans_root, scene, scene_dir):
    """Copy <scene>_vh_clean_2.ply to <scene_dir>/mesh.ply (replaces an old symlink). Same world frame as RT_c2w."""
    mesh = os.path.join(scans_root, scene, f"{scene}_vh_clean_2.ply")
    dst = os.path.join(scene_dir, "mesh.ply")
    if not os.path.exists(mesh):
        print(f"{scene}: no mesh {mesh}")
        return False
    if os.path.islink(dst):
        os.remove(dst)
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(mesh):
        shutil.copyfile(mesh, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    return True


def write_scene_file(out_root, scene_file):
    """{scene: [dumped rgb paths, sorted]} for every finished scene (input of the overlap script)."""
    scene_files = {}
    for scene in sorted(os.listdir(out_root)):
        cam_file = os.path.join(out_root, scene, CAMERAS_FILE)
        if os.path.exists(cam_file):
            with open(cam_file, "rb") as f:
                names = sorted(pkl.load(f))
            scene_files[scene] = [os.path.join(out_root, scene, "rgb", n) for n in names]
    os.makedirs(os.path.dirname(scene_file), exist_ok=True)
    with open(scene_file, "wb") as f:
        pkl.dump(scene_files, f)
    print(f"wrote {scene_file}: {len(scene_files)} scenes, {sum(map(len, scene_files.values()))} frames")


# ---------------------------------------------------------------------------
# sanity check: fused point cloud from the dumped files (same as the Hypersim script)
# ---------------------------------------------------------------------------

def save_fused_ply(out_root, scene, n_frames=10, max_points_per_frame=50000):
    import open3d as o3d

    scene_dir = os.path.join(out_root, scene)
    with open(os.path.join(scene_dir, CAMERAS_FILE), "rb") as f:
        cams = pkl.load(f)
    rgb_names = sorted(cams)
    sel = np.linspace(0, len(rgb_names) - 1, min(n_frames, len(rgb_names))).astype(int)

    pts_all, col_all = [], []
    for i in sel:
        rgb_name = rgb_names[i]
        K, RT_c2w = cams[rgb_name]["K"], cams[rgb_name]["RT_c2w"]
        d = np.load(os.path.join(scene_dir, "depth", cams[rgb_name]["depth"])).astype(np.float32)
        rgb = np.array(Image.open(os.path.join(scene_dir, "rgb", rgb_name))).astype(np.float32) / 255.0
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        m = d > 0
        pix = np.stack([u[m], v[m], np.ones(m.sum())], 0)
        pc = (np.linalg.inv(K) @ pix) * d[m]
        pw = (RT_c2w @ np.vstack([pc, np.ones(pc.shape[1])]))[:3].T
        step = max(1, len(pw) // max_points_per_frame)
        pts_all.append(pw[::step])
        col_all.append(rgb[m][::step])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_all))
    out = os.path.join(out_root, f"{scene}_fused.ply")
    o3d.io.write_point_cloud(out, pcd)
    print(f"wrote {out} from {len(sel)} frames")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dump ScanNet frames at the depth-map size, color aligned to depth.")
    ap.add_argument("--scans_root", default=SCANS_ROOT, help="folder with one sub-folder per scene")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--scene_file", default=SCENE_FILE, help="output: {scene: [dumped rgb paths]}")
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=None, help="exclusive; default = all scenes")
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--short_side", type=int, default=SHORT_SIDE,
                    help="0 (default): dump at the depth-map size 640x480; >0: resize so the short side is this")
    ap.add_argument("--copy_meshes_only", action="store_true",
                    help="only copy the meshes into already dumped scenes (no frame dumping)")
    ap.add_argument("--ply_scene", default=None, help="write a fused point cloud of this scene for a visual check")
    args = ap.parse_args()

    scenes = sorted(s for s in os.listdir(args.scans_root) if os.path.isdir(os.path.join(args.scans_root, s)))
    scenes = scenes[args.start_idx:args.end_idx]
    os.makedirs(args.out_root, exist_ok=True)
    print(f"{len(scenes)} scenes -> {args.out_root}")

    if args.copy_meshes_only:
        done = [s for s in scenes if os.path.isdir(os.path.join(args.out_root, s))]
        n = sum(copy_mesh(args.scans_root, s, os.path.join(args.out_root, s)) for s in tqdm(done))
        print(f"copied meshes into {n} / {len(done)} dumped scenes")
        raise SystemExit

    jobs = [(s, args.scans_root, args.out_root, args.max_frames, args.jpg_quality, args.short_side) for s in scenes]
    for job in tqdm(jobs, desc="scenes"):
        tqdm.write(dump_scene(job))

    write_scene_file(args.out_root, args.scene_file)
    if args.ply_scene:
        save_fused_ply(args.out_root, args.ply_scene)