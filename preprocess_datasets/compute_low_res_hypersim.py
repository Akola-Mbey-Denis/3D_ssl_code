"""
Hypersim -> resized frames (short side 518, aspect kept, no crop), one folder per scene.

Steps:
  1. scene files: if --scene_file exists, load it; otherwise collect_scene_files groups every
     file by scene, checks depth / intrinsics / pose of EVERY file, keeps up to 1000 good
     files per scene (evenly spaced) and saves it to --scene_file
  2. dump: resize each file so the short side is 518 and dump it
     (frames are re-checked as a safety net)

  frame name = original Hypersim name prefixed with its camera folder, e.g.
    ai_001_001-cam_00__point_12_view_0_domain_rgb

  <out_root>/<scene>/rgb/<name>.jpg
  <out_root>/<scene>/depth/<name>.npy   float32, meters (planar z), 0 = invalid
  <out_root>/<scene>/cameras.pkl        one per scene, written last (its presence = scene done):
                                        {"<name>.jpg": {"K":       (3,3) at dumped resolution (OpenCV),
                                                        "RT_c2w":  (4,4) OpenCV camera-to-world,
                                                        "hw":      dumped (H, W),
                                                        "depth":   "<name>.npy",
                                                        "src_rgb": original rgb path},
                                         ...}

Usage:
  python dump_hypersim_crops.py                              # first scene (builds the scene file if missing)
  python dump_hypersim_crops.py --start_idx 0 --end_idx 50   # scenes 0..49
"""

import os
import argparse
import pickle as pkl
from collections import defaultdict
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import utils.cropping as cropping

# --- reuse your functions (rename module to your overlap script's filename) ---
from .hypersim_overlaps import (
    rgb_to_depth_path,
    rgb_to_pointinfo_path,
    load_depth_distance,
    load_camera_params,
    view_camera_parameters,
    hypersim_distance_to_depth,
    load_all_hypersim_cameras,
)

SHORT_SIDE = 518
CAMERAS_FILE = "cameras.pkl"
MIN_VALID_FRAC = 0.5      # fraction of pixels with depth > 0.1 m
MIN_DEPTH = 0.10

RGB_ROOT = "/scratch/dma9300/hypersim/rgb/hypersim"
CSV_FILE = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocess_datasets/camera_parameters_hypersim.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def collect_scene_files(root_dir: str, cam_settings_all: Dict, max_images: int = 1000) -> Dict[str, List[str]]:
    """
    Collects file paths for each unique scene, merging all sub-scene directories,
    and keeps up to `max_images` GOOD files per scene.
    Subdirectories are processed in sorted order.

    1. group every file by scene (same as before: 'ai_055_009-cam_00' -> 'ai_055_009')
    2. check every file of the scene: depth + camera load, valid depth fraction,
       no skew / finite intrinsics, finite pose (see is_valid)
    3. if more than `max_images` good files remain, keep `max_images` evenly spaced ones

    Args:
        root_dir (str): Root directory containing subdirectories like 'ai_055_009-cam_00'.
        cam_settings_all (dict): scene_name -> Hypersim camera settings (load_all_hypersim_cameras).

    Returns:
        Dict[str, List[str]]:
            scene_name -> [up to max_images good file paths, sorted by subdir]
            (scenes without camera settings or without any good file are left out)
    """
    scene_files = defaultdict(list)

    # Sort the subdirectories
    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Extract unique scene name
        scene_name = subdir.split('-')[0]

        # Loop through files in the subdirectory and append
        for file_name in sorted(os.listdir(subdir_path)):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path):
                scene_files[scene_name].append(file_path)

    # Check every file, then subsample the good ones
    good_scene_files = {}
    for scene_name, files in scene_files.items():
        if scene_name not in cam_settings_all:
            print(f"{scene_name}: no camera settings, skipping")
            continue

        good = []
        for file_path in tqdm(files, desc=f"{scene_name} [check]"):
            try:
                depth, K, RT = load_depth_and_camera(file_path, cam_settings_all[scene_name])
            except Exception as e:
                print(f"  unreadable {file_path}: {e}")
                continue
            if is_valid(depth, K, RT):
                good.append(file_path)

        if len(good) > max_images:
            idxs = np.linspace(0, len(good) - 1, max_images, dtype=int)
            good = [good[i] for i in idxs]

        print(f"{scene_name}: {len(files)} files -> {len(good)} good kept")
        if good:
            good_scene_files[scene_name] = good

    return good_scene_files


def load_depth_and_camera(rgb_path, cam_settings):
    """Planar depth in meters (0 = invalid), K at original resolution, cam-to-world (OpenCV)."""
    distance, mask = load_depth_distance(rgb_to_depth_path(rgb_path))
    cam = load_camera_params(rgb_to_pointinfo_path(rgb_path))
    RT_c2w, K = view_camera_parameters(cam, cam_settings=cam_settings)
    K = np.asarray(K, np.float64)
    depth = hypersim_distance_to_depth(
        distance, width=cam["width"], height=cam["height"],
        focal=(K[0, 0] + K[1, 1]) / 2,
    )
    depth[~mask | ~np.isfinite(depth)] = 0.0
    return depth, K, np.asarray(RT_c2w, np.float64)


def is_valid(depth, K, RT_c2w):
    return (
        np.mean(depth > MIN_DEPTH) >= MIN_VALID_FRAC
        and abs(K[0, 1]) < 1e-6                 # no skew
        and np.all(np.isfinite(K))
        and np.all(np.isfinite(RT_c2w))
    )


def resize_short_side(image, depth, K, short_side=SHORT_SIDE):
    """
    Resize so min(H, W) == short_side, keep aspect ratio, using DUSt3R's
    cropping.rescale_image_depthmap: scale = max(out / in), so asking for
    (short_side, short_side) scales the short side to short_side.
    Image: Lanczos (bicubic if upscaling), depth: nearest, K via camera_matrix_of_crop.
    force=True so lower-resolution images are upscaled too.
    """
    out = cropping.rescale_image_depthmap(
        image, depth, K, (short_side, short_side), force=True)
    image, depth, K = out[0], out[1], out[2]
    return image, np.asarray(depth, np.float32), np.asarray(K, np.float32)


def frame_name(rgb_path):
    """
    Original Hypersim name, prefixed with its camera folder so names stay unique within a scene:
      .../ai_001_001-cam_00/point_12_view_0_domain_rgb.png -> ai_001_001-cam_00__point_12_view_0_domain_rgb
    """
    cam_dir = os.path.basename(os.path.dirname(rgb_path))
    stem = os.path.splitext(os.path.basename(rgb_path))[0]
    return f"{cam_dir}__{stem}"


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------

def dump_scene(scene, rgb_paths, cam_settings, out_root, jpg_quality=95):
    """Check each frame from collect_scene_files, resize and dump the valid ones."""
    scene_dir = os.path.join(out_root, scene)
    for sub in ("rgb", "depth"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)

    cameras = {}                                            # rgb file name -> camera
    n_invalid = 0
    for p in tqdm(rgb_paths, desc=f"{scene} [dump]"):
        name = frame_name(p)
        rgb_name = name + ".jpg"
        assert rgb_name not in cameras, f"duplicate frame name {rgb_name} in {scene}"

        try:
            depth, K, RT = load_depth_and_camera(p, cam_settings)
        except Exception as e:
            print(f"  unreadable {p}: {e}")
            n_invalid += 1
            continue
        if not is_valid(depth, K, RT):                      # bad intrinsics / pose / depth
            n_invalid += 1
            continue

        image = Image.open(p).convert("RGB")
        image, depth, K = resize_short_side(image, depth, K)

        image.save(os.path.join(scene_dir, "rgb", rgb_name), quality=jpg_quality)
        np.save(os.path.join(scene_dir, "depth", name + ".npy"), depth)

        cameras[rgb_name] = {
            "K": np.asarray(K, np.float32),                 # (3,3) at dumped resolution, OpenCV
            "RT_c2w": np.asarray(RT, np.float32),           # (4,4) OpenCV camera-to-world
            "hw": tuple(int(x) for x in depth.shape),       # dumped (H, W)
            "depth": name + ".npy",                         # file in depth/
            "src_rgb": p,                                   # original rgb path
        }

    print(f"{scene}: {len(rgb_paths)} frames -> {len(cameras)} dumped, {n_invalid} invalid")
    if not cameras:
        return

    # one camera file per scene, written last: its presence marks the scene as done
    tmp = os.path.join(scene_dir, CAMERAS_FILE + ".tmp")
    with open(tmp, "wb") as f:
        pkl.dump(cameras, f)
    os.replace(tmp, os.path.join(scene_dir, CAMERAS_FILE))


def build_scene_file(scene_file, cam_settings_all, rgb_root=RGB_ROOT, max_images=1000):
    """Check every Hypersim file once and save scene -> up to max_images good files."""
    all_scene_files = collect_scene_files(root_dir=rgb_root, cam_settings_all=cam_settings_all,
                                          max_images=max_images)
    tmp = scene_file + ".tmp"
    with open(tmp, "wb") as f:
        pkl.dump(all_scene_files, f)
    os.replace(tmp, scene_file)
    n = sum(len(v) for v in all_scene_files.values())
    print(f"wrote {scene_file}: {len(all_scene_files)} scenes, {n} good files")


def load_scene_file(scene_file):
    with open(scene_file, "rb") as f:
        return pkl.load(f)


def run_dump(args):
    """Dump scenes [start_idx, end_idx) and return the scene names in that slice."""
    cams = load_all_hypersim_cameras(CSV_FILE)

    # scene files: reuse the saved list if it exists, otherwise build it (checks every file)
    if not os.path.exists(args.scene_file):
        build_scene_file(args.scene_file, cams)
    scene_files = load_scene_file(args.scene_file)
    # breakpoint()

    slice_names = list(scene_files.keys())[args.start_idx:args.end_idx + 1]   # end_idx inclusive

    for scene in slice_names:
        if os.path.exists(os.path.join(args.out_root, scene, CAMERAS_FILE)):
            print(f"{scene}: done, skipping")
            continue
        if scene not in cams:
            print(f"{scene}: no camera settings, skipping")
            continue
        dump_scene(scene, scene_files[scene], cams[scene], args.out_root, args.jpg_quality)
    return slice_names


# ---------------------------------------------------------------------------
# sanity check: fused point cloud from the dumped files
# ---------------------------------------------------------------------------

def save_fused_ply(out_root, scene, n_frames=10, max_points_per_frame=50000):
    """
    Unproject n_frames evenly spaced dumped frames of `scene` with their K and RT_c2w and
    save one colored point cloud to <out_root>/<scene>_fused.ply.
    If depth, K and poses are consistent, the frames form one clean scene; misaligned or
    duplicated copies point at a bad pose convention or a translation unit mismatch.
    """
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
    out = os.path.join(out_root, f"{scene}_fused.ply")
    o3d.io.write_point_cloud(out, pcd)
    print(f"wrote {out} from {len(sel)} frames")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dump Hypersim frames resized to short side 518.")
    ap.add_argument("--out_root", type=str,
                    default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_hypersim/")
    ap.add_argument("--scene_file", type=str,
                    default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/hypersim_low_res_good_scene_files_1000.pkl",
                    help="Pickle with scene -> up to 1000 good files (built automatically if missing)")
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