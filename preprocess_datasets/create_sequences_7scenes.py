"""
7-Scenes target / source sequences from the overlap matrices of sevenscenes_overlaps.py
(same output format as the ARKitScenes / ScanNet sequence scripts).

Inputs (all from sevenscenes_overlaps.py, nothing is recomputed)
    scene_file       "<scene>_<seq>" -> rgb paths, the exact list the overlaps were computed on
                     (row / col i of the matrices <-> scene_files[scene][i])
    overlap_basedir  <overlap_basedir>/<scene>_<seq>/camera_overlap.npy, camera_occlusion.npy
                     frames with too little valid depth have -1 rows / cols -> never a target or source
    raw 7-Scenes     frame-XXXXXX.color.png, frame-XXXXXX.depth.proj.png (uint16 mm, registered to RGB),
                     frame-XXXXXX.pose.txt (4 x 4 camera-to-world); K: fx = fy = 525, cx = 320, cy = 240

Output
    {ROOT}/preprocessed_7scenes_3d_{split}_set.pkl
    {"<scene>_<seq>__frame-XXXXXX": {"target": {"image", "depth", "K", "extrinsics"},
                                     "sources": [{"image", "depth", "K", "extrinsics", "overlap", "occlusion"}]}}
    K is the raw 640 x 480 intrinsics and extrinsics the raw camera-to-world pose (the dataloader crops, as for
    the other datasets). "depth" is the raw .depth.proj.png: load it as millimetres -> meters (load_depth in
    sevenscenes_overlaps.py), not as a .npy.

Splits: the official 7-Scenes split, <root_dir>/<scene>/{TrainSplit,TestSplit}.txt ("sequence1", ...);
split="all" keeps every sequence.

    python sevenscenes_sequences.py --split test
"""
import argparse
import os
import pickle as pkl
import re

import numpy as np
from tqdm import tqdm

SEVEN_SCENES_K = np.array([[525.0, 0.0, 320.0],
                           [0.0, 525.0, 240.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32)          # fx, fy, cx, cy of the RGB camera
SPLIT_FILES = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}


def rgb_to_depth_path(rgb_path):
    """frame-XXXXXX.color.png -> frame-XXXXXX.depth.proj.png (depth registered to the RGB camera)."""
    return rgb_path.replace(".color.png", ".depth.proj.png")


def rgb_to_pose_path(rgb_path):
    """frame-XXXXXX.color.png -> frame-XXXXXX.pose.txt (4 x 4 camera-to-world)."""
    return rgb_path.replace(".color.png", ".pose.txt")


def frame_name(rgb_path):
    """frame-XXXXXX.color.png -> frame-XXXXXX"""
    return os.path.basename(rgb_path).split(".")[0]


def split_sequences(root_dir, scene, split):
    """
    Sequence folders ("seq-01", ...) of `scene` in the official split, from <root_dir>/<scene>/<Split>.txt
    (lines "sequence1", ...). None = keep every sequence (split="all").
    """
    if split == "all":
        return None
    path = os.path.join(root_dir, scene, SPLIT_FILES[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: use --split all, or point --root_dir to the raw 7-Scenes")
    with open(path) as f:
        numbers = [int(m.group(1)) for line in f if (m := re.search(r"(\d+)", line))]
    return {f"seq-{n:02d}" for n in numbers}


def scene_and_seq(scene_id, scene_names):
    """"<scene>_<seq>" -> (scene, seq), e.g. "redkitchen_seq-03" -> ("redkitchen", "seq-03")."""
    scene, seq = scene_id.rsplit("_", 1)
    assert scene in scene_names, f"unexpected scene id {scene_id}"
    return scene, seq


def build_overlap_sequences(scene_dict, root_dir, overlap_basedir, overlap_thres=0.010, max_sources=100, split="test"):
    """
    One sequence per valid target frame: sources = the other frames of the same sequence with
    overlap >= overlap_thres, sorted by overlap (occlusion as tie-breaker), at most max_sources.
    """
    results = {}
    scene_names = {s.rsplit("_", 1)[0] for s in scene_dict}
    split_cache = {}
    counts = {"sequences": 0, "no_overlaps": 0, "targets": 0, "no_sources": 0}

    for scene_id in tqdm(sorted(scene_dict), desc=f"7-Scenes {split}"):
        scene, seq = scene_and_seq(scene_id, scene_names)
        if scene not in split_cache:
            split_cache[scene] = split_sequences(root_dir, scene, split)
        if split_cache[scene] is not None and seq not in split_cache[scene]:
            continue

        overlaps_dir = os.path.join(overlap_basedir, scene_id)
        if not os.path.exists(os.path.join(overlaps_dir, "camera_overlap.npy")):
            print(f"[SKIP] {scene_id}: no overlaps in {overlaps_dir}")
            counts["no_overlaps"] += 1
            continue
        counts["sequences"] += 1

        # ---------------- overlaps ----------------
        camera_overlap_matrix = np.load(os.path.join(overlaps_dir, "camera_overlap.npy"))
        occlusion_matrix = np.load(os.path.join(overlaps_dir, "camera_occlusion.npy"))
        scene_files = scene_dict[scene_id]
        num_images = len(scene_files)
        assert camera_overlap_matrix.shape == (num_images, num_images), \
            f"{scene_id}: overlap matrix {camera_overlap_matrix.shape} vs {num_images} images"

        # ---------------- cameras ----------------
        valid_frames = np.diag(camera_overlap_matrix) > 0                        # -1 rows: too little valid depth
        RT_list = [np.loadtxt(rgb_to_pose_path(p)).astype(np.float32) if valid_frames[i] else None
                   for i, p in enumerate(scene_files)]
        depth_paths = [rgb_to_depth_path(p) for p in scene_files]
        view_names = [frame_name(p) for p in scene_files]

        def view(idx):
            return {"image": scene_files[idx], "depth": depth_paths[idx],
                    "K": SEVEN_SCENES_K.copy(), "extrinsics": RT_list[idx]}

        # ---------------- sequences ----------------
        for target_idx in np.flatnonzero(valid_frames):
            counts["targets"] += 1
            overlap_scores = camera_overlap_matrix[target_idx]                  # (N,)
            occlusion_scores = occlusion_matrix[target_idx]                     # (N,)

            # sources: overlap above the threshold (excludes the -1 frames), not the target itself
            valid_indices = np.where(overlap_scores >= overlap_thres)[0]
            valid_indices = valid_indices[valid_indices != target_idx]
            if len(valid_indices) == 0:
                counts["no_sources"] += 1
                continue

            # sort by overlap (primary), occlusion as tie-breaker; keep top-k
            sorted_indices = valid_indices[np.lexsort((-occlusion_scores[valid_indices],
                                                       -overlap_scores[valid_indices]))]
            sources = [{**view(src_idx),
                        "occlusion": float(occlusion_scores[src_idx]),
                        "overlap": float(overlap_scores[src_idx])}
                       for src_idx in sorted_indices[:max_sources]]

            results[f"{scene_id}__{view_names[target_idx]}"] = {"target": view(target_idx), "sources": sources}

    print(f"[SUMMARY] {split}: {len(results)} sequences from {counts['sequences']} 7-Scenes sequences "
          f"({counts['targets']} valid targets, {counts['no_sources']} without sources, "
          f"{counts['no_overlaps']} sequences without overlaps)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"],
                        help="official 7-Scenes split (TrainSplit.txt / TestSplit.txt), or all sequences")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/7scenes_overlap_scene_files_1000.pkl",
                        help="Scene files the overlaps were computed on (sevenscenes_overlaps.py --scene_file)")
    parser.add_argument("--root_dir", type=str, default="/scratch/dma9300/7scenes/",
                        help="Raw 7-Scenes folder (for the split files)")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/7scenes_overlaps/",
                        help="Overlap matrices, one folder per sequence")
    parser.add_argument("--overlap_thres", type=float, default=0.010, help="Minimum source overlap")
    parser.add_argument("--max_sources", type=int, default=100, help="Sources kept per target")
    parser.add_argument("--save_dir", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data")
    args = parser.parse_args()

    with open(args.scene_file, "rb") as f:
        all_scene_dict = pkl.load(f)

    results = build_overlap_sequences(scene_dict=all_scene_dict, root_dir=args.root_dir,
                                      overlap_basedir=args.overlap_basedir, overlap_thres=args.overlap_thres,
                                      max_sources=args.max_sources, split=args.split)

    save_path = f"{args.save_dir}/preprocessed_7scenes_3d_{args.split}_set.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\n Saved {len(results)} sequences to {save_path}")