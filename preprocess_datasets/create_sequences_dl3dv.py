import os
import json
import argparse
import pickle as pkl
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm


# ----------------------------------------------------------------------
# Source selection (DL3DV is video: frames are in temporal order)
# ----------------------------------------------------------------------
MAX_SOURCES = 10       # sources per target
FRAME_WINDOW = 30      # sources come from frames within +-FRAME_WINDOW of the target
MAX_ANGLE = 60.0       # max rotation angle (degrees) between target and source
MAX_DIST = 1.0         # max camera distance, normalized scene units (mean camera distance to centroid = 1)
NUM_SCENES = {"train": 800, "val": 100, "test": 100}   # disjoint scene splits
SPLIT_SEED = 0


def rgb_to_transforms_path(rgb_path: str) -> str:
    """<scene>/images_8/frame_00001.png -> <scene>/transforms.json"""
    return os.path.join(os.path.dirname(os.path.dirname(rgb_path)), "transforms.json")


def load_dl3dv_cameras(list_rgbs: List[str]):
    """
    Read DL3DV transforms.json (nerfstudio / OpenGL RT_c2w) and return, for every image in list_rgbs
    that has a pose in the json (same order; images without a pose are dropped):
        img_paths, K (3x3, for the image as stored on disk), RT_c2w (4x4, OpenCV).
    """
    with open(rgb_to_transforms_path(list_rgbs[0]), "r") as f:
        meta = json.load(f)

    # json frames by image file name (json paths say images/..., files are in images_8/)
    frame_by_name = {os.path.basename(fr["file_path"]): fr for fr in meta["frames"]}

    # Real resolution on disk (intrinsics in the json may be for a different resolution)
    W_img, H_img = Image.open(list_rgbs[0]).size

    img_paths, K_cache, RT_cache = [], [], []
    for rgb_path in list_rgbs:
        fr = frame_by_name.get(os.path.basename(rgb_path))
        if fr is None:
            continue   # no camera parameters -> not usable
        w, h = fr.get("w", meta["w"]), fr.get("h", meta["h"])
        fx, fy = fr.get("fl_x", meta["fl_x"]), fr.get("fl_y", meta["fl_y"])
        cx, cy = fr.get("cx", meta["cx"]), fr.get("cy", meta["cy"])

        # json resolution -> image on disk
        sx, sy = W_img / w, H_img / h
        K = np.array([[fx * sx, 0, cx * sx],
                      [0, fy * sy, cy * sy],
                      [0, 0, 1]], dtype=np.float32)

        # OpenGL (x right, y up, z back) -> OpenCV (x right, y down, z forward)
        RT_c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        RT_c2w[2, :] *= -1
        RT_c2w = RT_c2w[np.array([1, 0, 2, 3]), :]
        RT_c2w[0:3, 1:3] *= -1

        img_paths.append(rgb_path)
        K_cache.append(K)
        RT_cache.append(RT_c2w)

    return img_paths, K_cache, RT_cache


def pairwise_pose_distances(RT_cache):
    """
    Rotation angle (degrees) and camera-center distance (normalized scene units) between all views.
    Distances are divided by the mean camera distance to the trajectory centroid (COLMAP scale is arbitrary).
    """
    RTs = np.stack(RT_cache).astype(np.float64)     # (N, 4, 4)
    Rs = RTs[:, :3, :3]
    centers = RTs[:, :3, 3]

    # angle(R_i, R_j) = arccos((trace(R_i^T R_j) - 1) / 2)
    trace = np.einsum("iab,jab->ij", Rs, Rs)
    angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))

    scale = np.linalg.norm(centers - centers.mean(0), axis=1).mean()
    if scale < 1e-8:
        scale = 1.0
    dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1) / scale

    return angle, dist


def make_scene_splits(scene_dict, split_file):
    """
    Randomly assign scenes to disjoint train / val / test sets (NUM_SCENES each) and dump the
    scene names to `split_file` (json). If the file exists it is loaded instead, so all splits
    are always built from the same assignment.
    """
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            splits = json.load(f)
    else:
        all_scenes = sorted(scene_dict.keys())
        need = sum(NUM_SCENES.values())
        if len(all_scenes) < need:
            print(f"Warning: only {len(all_scenes)} scenes for {need} requested; train gets the rest.")

        perm = np.random.default_rng(SPLIT_SEED).permutation(len(all_scenes))
        shuffled = [all_scenes[i] for i in perm]

        # val and test first (fixed size), train after them
        n_val, n_test = NUM_SCENES["val"], NUM_SCENES["test"]
        splits = {
            "val": sorted(shuffled[:n_val]),
            "test": sorted(shuffled[n_val:n_val + n_test]),
            "train": sorted(shuffled[n_val + n_test:n_val + n_test + NUM_SCENES["train"]]),
        }

        with open(split_file, "w") as f:
            json.dump(splits, f, indent=2)
        for name, scenes in splits.items():
            with open(split_file.replace(".json", f"_{name}.txt"), "w") as f:
                f.write("\n".join(scenes) + "\n")

    # no scene in more than one split
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (train & val) and not (train & test) and not (val & test), "scene splits overlap"

    print({k: len(v) for k, v in splits.items()}, "->", split_file)
    return splits


def build_sequences(scene_dict, scene_ids, split="train", target_stride=1):
    """
    Build target / source sequences for DL3DV (images + cameras only, no depth, no overlap matrix).

    For each target frame (every `target_stride`-th frame), candidate sources are frames within
    +-FRAME_WINDOW that satisfy angle <= MAX_ANGLE and distance <= MAX_DIST. Up to MAX_SOURCES are
    kept, spread evenly over the candidates (small and large baselines, both sides of the target).

    Returns:
        dict: {sceneid__targetid: {"target": {...}, "sources": [...]}}
    """
    results = {}

    for scene_id in tqdm(scene_ids, desc=f"Processing DL3DV {split}", leave=False):
        try:
            img_paths, K_cache, RT_cache = load_dl3dv_cameras(scene_dict[scene_id])
        except Exception as e:
            print(f"Skipping {scene_id}: {e}")
            continue

        num_images = len(img_paths)
        if num_images < 2:
            continue

        angle, dist = pairwise_pose_distances(RT_cache)

        for target_idx in range(0, num_images, target_stride):
            lo = max(0, target_idx - FRAME_WINDOW)
            hi = min(num_images, target_idx + FRAME_WINDOW + 1)

            candidates = np.array([
                j for j in range(lo, hi)
                if j != target_idx
                and angle[target_idx, j] <= MAX_ANGLE
                and dist[target_idx, j] <= MAX_DIST
            ], dtype=int)

            if len(candidates) == 0:
                continue


            # breakpoint()
            # Spread the selection evenly over the candidates (sorted by frame index)
            if len(candidates) > MAX_SOURCES:
                pick = np.round(np.linspace(0, len(candidates) - 1, MAX_SOURCES)).astype(int)
                candidates = candidates[pick]

            sources = []
            for src_idx in candidates:
             
                sources.append({
                    "image": img_paths[src_idx],
                    "K": K_cache[src_idx],
                    "extrinsics": RT_cache[src_idx],
                    "frame_gap": int(src_idx - target_idx),
                    "angle": float(angle[target_idx, src_idx]),           # relative rotation, degrees
                    "distance": float(dist[target_idx, src_idx]),         # camera-center dis
                })

            key_name = f"{scene_id}__{target_idx:05d}"
            results[key_name] = {
                "target": {
                    "image": img_paths[target_idx],
                    "K": K_cache[target_idx],
                    "extrinsics": RT_cache[target_idx],
                },
                "sources": sources,
            }

    return results


def collect_scene_files(root_dir: str, image_dir: str = "images_8") -> Dict[str, List[str]]:
    """
    DL3DV layout: <root_dir>/<scene_hash>/{transforms.json, images_8/*.png}
    Returns: scene_name ("<scene_hash>") -> sorted list of all image paths of the scene
    """
    scene_files = {}
    for scene in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene)
        rgb_dir = os.path.join(scene_dir, image_dir)
        if not (os.path.isdir(rgb_dir) and os.path.exists(os.path.join(scene_dir, "transforms.json"))):
            continue
        files = sorted(
            os.path.join(rgb_dir, f)
            for f in os.listdir(rgb_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if len(files) > 0:
            scene_files[scene] = files
    return scene_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", help="train, val or test")
    parser.add_argument("--root_dir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dl3dv_datasets/5K/", help="Subset folder containing the scene hash folders")
    parser.add_argument("--scene_file", type=str, default=None, help="Pickle file containing scene files (default: dl3dv_scene_files_<subset>.pkl)")
    parser.add_argument("--target_stride", type=int, default=1, help="Use every k-th frame as a target")
    args = parser.parse_args()

    subset = os.path.basename(os.path.normpath(args.root_dir))   # e.g. 5K
    preprocessed_dir = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data"
    if args.scene_file is None:
        args.scene_file = f"{preprocessed_dir}/dl3dv_scene_files_{subset}.pkl"

    if not os.path.exists(args.scene_file):
        with open(args.scene_file, "wb") as f:
            pkl.dump(collect_scene_files(root_dir=args.root_dir), f)

    with open(args.scene_file, "rb") as f:
        all_scene_dict = pkl.load(f)

    # Disjoint train / val / test scenes (created once, then reused)
    split_file = f"{preprocessed_dir}/dl3dv_{subset}_scene_splits.json"
    splits = make_scene_splits(all_scene_dict, split_file)

    results = build_sequences(all_scene_dict, splits[args.split], split=args.split, target_stride=args.target_stride)

    save_path = f"{preprocessed_dir}/preprocessed_dl3dv_{subset}_{args.split}_set.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(results, f)

    print(f"\n Saved {len(results)} sequences to {save_path}")