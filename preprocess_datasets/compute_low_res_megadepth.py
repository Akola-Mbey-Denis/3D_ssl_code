"""
MegaDepth -> resized frames (short side 518, aspect kept, no crop), one folder per scene key.

Same pipeline as dump_hypersim_crops.py, with MegaDepth's loading (h5 depth, camera_matrices.json,
undistorted K via cv2.getOptimalNewCameraMatrix) taken from the MegaDepth overlap script.

Steps:
  1. scene files: if --scene_file exists, load it; otherwise collect_scene_files groups every
     image by scene key ("<scene>_<denseX>", e.g. "5018_dense0"), checks depth / camera / image
     of EVERY file, keeps up to 1000 good files per key (evenly spaced) and saves --scene_file
  2. dump: resize each file so the short side is 518 and dump it
     (frames are re-checked as a safety net)

  <out_root>/<key>/rgb/<name>.jpg      rgb, short side 518 (portrait images stay portrait)
  <out_root>/<key>/depth/<name>.npy    depth, float32, MegaDepth SfM units (NOT metric), 0 = invalid
  <out_root>/<key>/cameras.pkl         one per key, written last (its presence = scene done):
                                       {"<name>.jpg": {"K":       (3,3) at dumped resolution (OpenCV),
                                                       "RT_c2w":  (4,4) OpenCV camera-to-world,
                                                       "hw":      dumped (H, W),
                                                       "depth":   "<name>.npy",
                                                       "src_rgb": original image path},
                                        ...}

Usage:
  python dump_megadepth_crops.py                              # first scene key (builds the scene file if missing)
  python dump_megadepth_crops.py --start_idx 0 --end_idx 50   # scene keys 0..49
"""

import os
import argparse
import pickle as pkl
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import utils.cropping as cropping

# --- reuse your functions (rename module to your MegaDepth overlap script's filename) ---
from .megadepth_overlaps import (
    load_depth,
    load_camera_matrices,
    is_valid_image,
)

SHORT_SIDE = 518
CAMERAS_FILE = "cameras.pkl"
MIN_VALID_FRAC = 0.6      # same as the overlap script: fraction of pixels with depth > MIN_DEPTH
MIN_DEPTH = 0.10

RGB_ROOT = "/projects/rps/dff6142/fouheylab/shared_datasets/MegaDepth/phoenix/S6/zl548/MegaDepth_v1/"
POSE_ROOT = "/projects/rps/dff6142/fouheylab/shared_datasets/MegaDepth/pose_data/"
IMG_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# MegaDepth paths and cameras
# ---------------------------------------------------------------------------

def pose_path_for(rgb_path):
    """<root>/<scene>/<denseX>/imgs/<img> -> <POSE_ROOT>/<scene>/sparse/manhattan/<X>/camera_matrices.json"""
    p = Path(rgb_path)
    main_dir = p.parents[2].name                  # '0000'
    subdir = p.parents[1].name.replace("dense", "")   # 'dense0' -> '0'
    return os.path.join(POSE_ROOT, main_dir, "sparse/manhattan", subdir, "camera_matrices.json")


def depth_path_for(rgb_path):
    """<...>/imgs/<name>.jpg -> <...>/depths/<name>.h5"""
    return os.path.splitext(rgb_path.replace("/imgs/", "/depths/"))[0] + ".h5"


_camera_cache = {}


def get_camera_info(pose_path):
    """camera_matrices.json is shared by every image of a dense folder: load it once."""
    if pose_path not in _camera_cache:
        _camera_cache[pose_path] = load_camera_matrices(pose_path)
    return _camera_cache[pose_path]


def load_depth_and_camera(rgb_path):
    """
    Depth (0 = invalid), undistorted K for the image as stored, cam-to-world (OpenCV).
    Same steps as the overlap script.
    """
    pose_path = pose_path_for(rgb_path)
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"missing {pose_path}")
    camera_dict = get_camera_info(pose_path)[os.path.basename(rgb_path)]

    depth, mask = load_depth(depth_path_for(rgb_path))
    depth = np.asarray(depth, np.float32)
    depth[~mask | ~np.isfinite(depth)] = 0.0

    K_pre = np.array(camera_dict["K"], dtype=np.float32)
    distortion = np.array(camera_dict["distortion"], dtype=np.float32)
    imsize_pre = tuple(int(x) for x in camera_dict["size"])
    H, W = depth.shape                                   # depth has the image's size
    K = cv2.getOptimalNewCameraMatrix(
        K_pre, distortion, imsize_pre, alpha=0,
        newImgSize=(W, H), centerPrincipalPoint=True,
    )[0]

    RT_w2c = np.array(camera_dict["RT"], dtype=np.float64)
    if RT_w2c.shape == (3, 4):
        RT_w2c = np.vstack([RT_w2c, [0, 0, 0, 1]])
    RT_c2w = np.linalg.inv(RT_w2c)

    return depth, np.asarray(K, np.float64), RT_c2w


def is_valid(depth, K, RT_c2w):
    return (
        np.mean(depth > MIN_DEPTH) >= MIN_VALID_FRAC
        and abs(K[0, 1]) < 1e-6                 # no skew
        and np.all(np.isfinite(K))
        and np.all(np.isfinite(RT_c2w))
    )


# ---------------------------------------------------------------------------
# scene files
# ---------------------------------------------------------------------------

def collect_scene_files(root_dir: str, max_images: int = 1000, min_images: int = 30) -> Dict[str, List[str]]:
    """
    Collect image file paths for MegaDepth structure and keep up to `max_images` GOOD files per key.

    1. group every image by key "<scene>_<denseX>" (same as the overlap script)
    2. check every image: pose file + camera entry + depth load, valid depth fraction,
       finite / skew-free K, finite pose, not mostly black (is_valid_image)
    3. discard the key if fewer than `min_images` good files remain (not stored)
    4. if more than `max_images` good files remain, keep `max_images` evenly spaced ones

    Returns:
        Dict[str, List[str]]
            key   -> "5018_dense0"
            value -> list of min_images..max_images good image paths
    """
    scene_files = {}

    # iterate over scene folders like 5018, 0000, etc.
    for scene in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        # iterate over dense folders like dense0
        for dense_subdir in sorted(os.listdir(scene_path)):
            imgs_path = os.path.join(scene_path, dense_subdir, "imgs")
            if not os.path.isdir(imgs_path):
                continue  # skip if no imgs folder

            image_files = []
            for root, _, files in os.walk(imgs_path):
                for f in files:
                    if f.lower().endswith(IMG_EXTS):
                        image_files.append(os.path.join(root, f))
            image_files = sorted(image_files)

            key = f"{scene}_{dense_subdir}"

            # too few images even before checking: can never reach min_images, skip the checks
            if len(image_files) < min_images:
                print(f"{key}: {len(image_files)} files < {min_images}, discarded")
                continue

            # check every image, then subsample the good ones
            good = []
            for p in tqdm(image_files, desc=f"{key} [check]"):
                try:
                    depth, K, RT = load_depth_and_camera(p)
                except Exception as e:
                    continue                     # no pose file / no camera entry / unreadable depth
                if is_valid(depth, K, RT) and is_valid_image(p):
                    good.append(p)

            # too few good images: discard the key, do not store it
            if len(good) < min_images:
                print(f"{key}: {len(image_files)} files -> {len(good)} good < {min_images}, discarded")
                continue

            if len(good) > max_images:
                indices = np.linspace(0, len(good) - 1, max_images, dtype=int)
                good = [good[i] for i in indices]

            print(f"{key}: {len(image_files)} files -> {len(good)} good kept")
            scene_files[key] = good

    return scene_files


def build_scene_file(scene_file, rgb_root=RGB_ROOT, max_images=1000):
    """Check every MegaDepth image once and save key -> up to max_images good files."""
    all_scene_files = collect_scene_files(root_dir=rgb_root, max_images=max_images)
    tmp = scene_file + ".tmp"
    with open(tmp, "wb") as f:
        pkl.dump(all_scene_files, f)
    os.replace(tmp, scene_file)
    n = sum(len(v) for v in all_scene_files.values())
    print(f"wrote {scene_file}: {len(all_scene_files)} scene keys, {n} good files")


def load_scene_file(scene_file, min_files=30):
    """Load the scene file and drop scenes with fewer than `min_files` files."""
    with open(scene_file, "rb") as f:
        scene_files = pkl.load(f)

    kept = {k: v for k, v in scene_files.items() if len(v) >= min_files}

    n_dropped = len(scene_files) - len(kept)
    if n_dropped:
        print(f"load_scene_file: dropped {n_dropped} scenes with < {min_files} files "
              f"({len(kept)} of {len(scene_files)} kept)")
    return kept

# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------

def resize_short_side(image, depth, K, short_side=SHORT_SIDE):
    """
    Resize so min(H, W) == short_side, keep aspect ratio (works for portrait and landscape),
    using DUSt3R's cropping.rescale_image_depthmap. Image: Lanczos (bicubic if upscaling),
    depth: nearest, K via camera_matrix_of_crop. force=True so smaller images are upscaled too.
    """
    out = cropping.rescale_image_depthmap(
        image, depth, K, (short_side, short_side), force=True)
    image, depth, K = out[0], out[1], out[2]
    return image, np.asarray(depth, np.float32), np.asarray(K, np.float32)


def frame_name(rgb_path):
    """Original MegaDepth image name without extension (unique within a dense folder)."""
    return os.path.splitext(os.path.basename(rgb_path))[0]


def dump_scene(key, rgb_paths, out_root, jpg_quality=95):
    """Resize and dump the files of one scene key; re-check each frame as a safety net."""
    scene_dir = os.path.join(out_root, key)                # folder named after the key
    for sub in ("rgb", "depth"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)

    cameras = {}                                            # rgb file name -> camera
    n_invalid = 0
    for p in tqdm(rgb_paths, desc=f"{key} [dump]"):
        name = frame_name(p)
        rgb_name = name + ".jpg"
        assert rgb_name not in cameras, f"duplicate frame name {rgb_name} in {key}"

        try:
            depth, K, RT = load_depth_and_camera(p)
        except Exception as e:
            print(f"  unreadable {p}: {e}")
            n_invalid += 1
            continue
        if not is_valid(depth, K, RT):
            n_invalid += 1
            continue

        image = Image.open(p).convert("RGB")
        assert image.size == depth.shape[::-1], f"image/depth size mismatch for {p}"
        image, depth, K = resize_short_side(image, depth, K)

        image.save(os.path.join(scene_dir, "rgb", rgb_name), quality=jpg_quality)
        np.save(os.path.join(scene_dir, "depth", name + ".npy"), depth)

        cameras[rgb_name] = {
            "K": np.asarray(K, np.float32),                 # (3,3) at dumped resolution, OpenCV
            "RT_c2w": np.asarray(RT, np.float32),           # (4,4) OpenCV camera-to-world
            "hw": tuple(int(x) for x in depth.shape),       # dumped (H, W)
            "depth": name + ".npy",                         # file in depth/
            "src_rgb": p,                                   # original image path
        }

    print(f"{key}: {len(rgb_paths)} frames -> {len(cameras)} dumped, {n_invalid} invalid")
    if not cameras:
        return

    # one camera file per key, written last: its presence marks the scene as done
    tmp = os.path.join(scene_dir, CAMERAS_FILE + ".tmp")
    with open(tmp, "wb") as f:
        pkl.dump(cameras, f)
    os.replace(tmp, os.path.join(scene_dir, CAMERAS_FILE))


def run_dump(args):
    """Dump scene keys [start_idx, end_idx) and return the scene keys in that slice."""
    # scene files: reuse the saved list if it exists, otherwise build it (checks every file)
    if not os.path.exists(args.scene_file):
        build_scene_file(args.scene_file)
    scene_files = load_scene_file(args.scene_file)

    # breakpoint()


    slice_names = list(scene_files.keys())[args.start_idx:args.end_idx + 1]   # end_idx exclusive

    for scene in slice_names:
        if os.path.exists(os.path.join(args.out_root, scene, CAMERAS_FILE)):
            print(f"{scene}: done, skipping")
            continue
        dump_scene(scene, scene_files[scene], args.out_root, args.jpg_quality)
    return slice_names


# ---------------------------------------------------------------------------
# sanity check: fused point cloud from the dumped files
# ---------------------------------------------------------------------------

def save_fused_ply(out_root, key, n_frames=10, max_points_per_frame=50000):
    """
    Unproject n_frames evenly spaced dumped frames of `key` with their K and RT_c2w and save one
    colored point cloud to <out_root>/<key>_fused.ply. Consistent depth / K / poses give one
    clean reconstruction; ghosted copies point at a pose or intrinsics problem.
    """
    import open3d as o3d

    scene_dir = os.path.join(out_root, key)
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
        u, v = np.meshgrid(np.arange(W), np.arange(H))     # OpenCV: pixel centers at integers
        m = d > 0
        pix = np.stack([u[m], v[m], np.ones(m.sum())], 0)
        pc = (np.linalg.inv(K) @ pix) * d[m]               # camera frame
        pw = (RT_c2w @ np.vstack([pc, np.ones(pc.shape[1])]))[:3].T   # world frame

        step = max(1, len(pw) // max_points_per_frame)
        pts_all.append(pw[::step])
        col_all.append(rgb[m][::step])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_all))
    out = os.path.join(out_root, f"{key}_fused.ply")
    o3d.io.write_point_cloud(out, pcd)
    print(f"wrote {out} from {len(sel)} frames")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dump MegaDepth frames resized to short side 518.")
    ap.add_argument("--out_root", type=str,
                    default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_megadepth")
    ap.add_argument("--scene_file", type=str,
                    default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/megadepth_low_res_good_scene_files_1000.pkl",
                    help="Pickle with key -> up to 1000 good files (built automatically if missing)")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=1, help="exclusive; default = first scene only")
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--ply_frames", type=int, default=10,
                    help="frames fused into the point cloud of the slice's first scene; 0 disables it")
    args = ap.parse_args()

    slice_names = run_dump(args)

    # # point cloud check on the first scene of this slice
    # if args.ply_frames > 0 and slice_names:
    #     first = slice_names[0]
    #     if os.path.exists(os.path.join(args.out_root, first, CAMERAS_FILE)):
    #         save_fused_ply(args.out_root, first, n_frames=args.ply_frames)
    #     else:
    #         print(f"{first}: not dumped (no valid frames?), no point cloud")