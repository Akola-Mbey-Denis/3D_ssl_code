
"""
Create NRGBD target/source sequences from precomputed camera overlap matrices.

Inputs
------
scene_file:
    Pickle produced by nrgbd_overlaps.py:
        scene_name -> {
            "img_paths": [RGB paths],
            "poses": [4x4 OpenGL camera-to-world poses]
        }

overlap_basedir/<scene_name>/:
    camera_overlap.npy
    camera_occlusion.npy
    frame_names.pkl (optional; filenames corresponding to matrix rows/columns)

Raw NRGBD data:
    <root_dir>/<scene>/images/img<i>.png
    <root_dir>/<scene>/depth/depth<i>.png
    <root_dir>/<scene>/poses.txt

Output
------
<save_dir>/preprocessed_nrgbd_3d_<split>_set.pkl

{
    "<scene>__img<i>": {
        "target": {
            "image", "depth", "K", "extrinsics"
        },
        "sources": [
            {
                "image", "depth", "K", "extrinsics",
                "overlap", "occlusion"
            }
        ]
    }
}

K and depth correspond to the original 640x480 RGB/depth images.
extrinsics is the OpenCV camera-to-world matrix.

Splits:
    all: all scenes
    train/val/test: optional scene-level split files:
        <root_dir>/train_scenes.txt
        <root_dir>/val_scenes.txt
        <root_dir>/test_scenes.txt

If a split file does not exist, an explicit error is raised.

Usage:
    python nrgbd_sequences.py --split all
    python nrgbd_sequences.py --split test
"""

import argparse
import os
import pickle as pkl
import re

import numpy as np
from tqdm import tqdm


NRGBD_K = np.array(
    [
        [554.2562584220408, 0.0, 320.0],
        [0.0, 554.2562584220408, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

SPLIT_FILES = {
    "train": "train_scenes.txt",
    "val": "val_scenes.txt",
    "test": "test_scenes.txt",
}


def frame_index(path):
    """Extract the numeric frame index from img123.png."""
    matches = re.findall(r"\d+", os.path.basename(path))
    if not matches:
        raise ValueError(f"Cannot extract frame index from {path}")
    return int(matches[-1])


def rgb_to_depth_path(rgb_path):
    """<scene>/images/img<i>.png -> <scene>/depth/depth<i>.png."""
    scene_dir = os.path.dirname(os.path.dirname(rgb_path))
    return os.path.join(
        scene_dir, "depth", f"depth{frame_index(rgb_path)}.png"
    )


def frame_name(rgb_path):
    """img123.png -> img123."""
    return os.path.splitext(os.path.basename(rgb_path))[0]


def load_poses(path):
    """
    Load NRGBD poses.txt.

    Returns:
        poses: list of 4x4 OpenGL camera-to-world matrices.
        valids: list of booleans indicating valid poses.
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 4 != 0:
        raise ValueError(
            f"{path}: expected four nonempty lines per pose, "
            f"got {len(lines)} lines"
        )

    poses = []
    valids = []

    for i in range(0, len(lines), 4):
        block = lines[i:i + 4]

        if any("nan" in line.lower() for line in block):
            poses.append(np.eye(4, dtype=np.float32))
            valids.append(False)
            continue

        pose = np.asarray(
            [[float(x) for x in line.split()] for line in block],
            dtype=np.float32,
        )

        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            poses.append(np.eye(4, dtype=np.float32))
            valids.append(False)
            continue

        poses.append(pose)
        valids.append(True)

    return poses, valids


def opengl_to_opencv_c2w(pose):
    """
    Convert an OpenGL camera-to-world pose to OpenCV camera-to-world.

    Flip the camera y and z axes by right-multiplying the pose
    by diag(1, -1, -1, 1).
    """
    pose = np.asarray(pose, dtype=np.float32).copy()
    pose[:, 1:3] *= -1.0
    return pose


def load_scene_file(scene_file):
    """Load the scene mapping and normalize entries."""
    with open(scene_file, "rb") as f:
        scene_dict = pkl.load(f)

    return scene_dict


def get_scene_names(root_dir, scene_dict, split):
    """
    Return the scene names for the requested split.

    The split files are optional dataset conventions, not official
    NRGBD splits. For split='all', keep all scenes.
    """
    if split == "all":
        return set(scene_dict.keys())

    split_path = os.path.join(root_dir, SPLIT_FILES[split])

    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. NRGBD does not necessarily provide "
            "standard train/val/test scene splits. Create the scene list "
            "or use --split all."
        )

    with open(split_path, "r") as f:
        names = {
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }

    unknown = names - set(scene_dict.keys())
    if unknown:
        print(
            f"[WARN] {len(unknown)} scenes in {split_path} "
            "are absent from scene_file."
        )

    return names & set(scene_dict.keys())


def load_matrix(path, scene_name, num_frames):
    """Load and validate an overlap or occlusion matrix."""
    matrix = np.load(path)

    if matrix.shape != (num_frames, num_frames):
        raise ValueError(
            f"{scene_name}: {os.path.basename(path)} has shape "
            f"{matrix.shape}, expected {(num_frames, num_frames)}"
        )

    return matrix.astype(np.float32, copy=False)



def build_overlap_sequences(
    scene_dict,
    root_dir,
    overlap_basedir,
    overlap_thres=0.010,
    max_sources=100,
    split="all",
):
    """
    One sequence per valid NRGBD target frame.

    Matrix row/column i corresponds directly to scene_dict[scene_name]
    ["img_paths"][i]. No frame_names.pkl is read or required.

    Sources are other valid frames whose target-to-source overlap meets
    overlap_thres, sorted by descending overlap and then occlusion.
    """
    results = {}
    selected_scenes = get_scene_names(root_dir, scene_dict, split)

    counts = {
        "scenes": 0,
        "no_overlaps": 0,
        "targets": 0,
        "no_sources": 0,
        "invalid_frames": 0,
    }

    for scene_name in tqdm(sorted(selected_scenes), desc=f"NRGBD {split}"):
        scene_entry = scene_dict[scene_name]
        overlaps_dir = os.path.join(overlap_basedir, scene_name)

        overlap_path = os.path.join(overlaps_dir, "camera_overlap.npy")
        occlusion_path = os.path.join(overlaps_dir, "camera_occlusion.npy")

        if not (
            os.path.isfile(overlap_path)
            and os.path.isfile(occlusion_path)
        ):
            print(f"[SKIP] {scene_name}: overlap matrices missing")
            counts["no_overlaps"] += 1
            continue

        # Preserve the exact ordering used to compute the matrices.
        img_paths = list(scene_entry["img_paths"])
        poses = scene_entry["poses"]
        num_frames = len(img_paths)

        if len(poses) != num_frames:
            print(
                f"[SKIP] {scene_name}: {num_frames} images vs "
                f"{len(poses)} poses"
            )
            continue

        camera_overlap = load_matrix(
            overlap_path, scene_name, num_frames
        )
        camera_occlusion = load_matrix(
            occlusion_path, scene_name, num_frames
        )

        # Convert OpenGL camera-to-world poses to OpenCV camera-to-world.
        RT_list = []
        valid_pose = np.zeros(num_frames, dtype=bool)

        for i, pose in enumerate(poses):
            pose = np.asarray(pose, dtype=np.float32)

            is_valid = (
                pose.shape == (4, 4)
                and np.isfinite(pose).all()
            )
            valid_pose[i] = is_valid
            RT_list.append(
                opengl_to_opencv_c2w(pose) if is_valid else None
            )

        # As in the 7-Scenes example, reject frames with invalid overlap
        # rows/columns. The diagonal must indicate a valid frame.
        valid_overlap = (
            np.isfinite(np.diag(camera_overlap))
            & (np.diag(camera_overlap) > 0)
            & np.all(np.isfinite(camera_overlap), axis=1)
            & np.all(camera_overlap >= 0, axis=1)
            & np.all(camera_overlap >= 0, axis=0)
        )

        depth_paths = [rgb_to_depth_path(path) for path in img_paths]
        view_names = [frame_name(path) for path in img_paths]

        file_valid = np.asarray(
            [
                os.path.isfile(img_paths[i])
                and os.path.isfile(depth_paths[i])
                for i in range(num_frames)
            ],
            dtype=bool,
        )

        valid_frames = valid_pose & valid_overlap & file_valid

        counts["scenes"] += 1
        counts["invalid_frames"] += int((~valid_frames).sum())

        def view(idx):
            return {
                "image": img_paths[idx],
                "depth": depth_paths[idx],
                "K": NRGBD_K.copy(),
                "extrinsics": RT_list[idx].copy(),
            }

        for target_idx in np.flatnonzero(valid_frames):
            counts["targets"] += 1

            overlap_scores = camera_overlap[target_idx]
            occlusion_scores = camera_occlusion[target_idx]

            # Exclude the target itself and every invalid source.
            valid_indices = np.flatnonzero(
                valid_frames
                & np.isfinite(overlap_scores)
                & np.isfinite(occlusion_scores)
                & (overlap_scores >= overlap_thres)
            )
            valid_indices = valid_indices[valid_indices != target_idx]

            if len(valid_indices) == 0:
                counts["no_sources"] += 1
                continue

            # Descending overlap, with descending occlusion as tie-breaker.
            sorted_indices = valid_indices[
                np.lexsort(
                    (
                        -occlusion_scores[valid_indices],
                        -overlap_scores[valid_indices],
                    )
                )
            ]

            sources = [
                {
                    **view(src_idx),
                    "overlap": float(overlap_scores[src_idx]),
                    "occlusion": float(occlusion_scores[src_idx]),
                }
                for src_idx in sorted_indices[:max_sources]
            ]

            key = f"{scene_name}__{view_names[target_idx]}"
            results[key] = {
                "target": view(target_idx),
                "sources": sources,
            }

    print(
        f"[SUMMARY] {split}: {len(results)} target sequences "
        f"from {counts['scenes']} scenes "
        f"({counts['targets']} valid targets, "
        f"{counts['no_sources']} without sources, "
        f"{counts['invalid_frames']} invalid frames, "
        f"{counts['no_overlaps']} scenes without overlap matrices)"
    )

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Scene-level split, or all scenes",
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        default=(
            "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/nrgbd_overlaps/nrgbd_scene_files.pkl"
        ),
        help="Scene mapping created by nrgbd_overlaps.py",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/scratch/dma9300/nrgbd/NRGBD/",
        help="Raw NRGBD dataset root",
    )
    parser.add_argument(
        "--overlap_basedir",
        type=str,
        default=(
            "/projects/rps/dff6142/fouheylab/shared_datasets/"
            "dma9300/dataset_overlaps/nrgbd_overlaps/"
        ),
        help="Directory containing precomputed overlap matrices",
    )
    parser.add_argument(
        "--overlap_thres",
        type=float,
        default=0.010,
        help="Minimum directional source overlap",
    )
    parser.add_argument(
        "--max_sources",
        type=int,
        default=100,
        help="Maximum sources per target",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=(
            "/projects/rps/dff6142/fouheylab/dma9300/"
            "3d_ssl_project/preprocessed_data"
        ),
        help="Output directory",
    )

    args = parser.parse_args()

    if args.max_sources < 1:
        parser.error("--max_sources must be >= 1")

    if args.overlap_thres < 0:
        parser.error("--overlap_thres must be >= 0")

    all_scene_dict = load_scene_file(args.scene_file)

    results = build_overlap_sequences(
        scene_dict=all_scene_dict,
        root_dir=args.root_dir,
        overlap_basedir=args.overlap_basedir,
        overlap_thres=args.overlap_thres,
        max_sources=args.max_sources,
        split=args.split,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(
        args.save_dir,
        f"preprocessed_nrgbd_3d_{args.split}_set.pkl",
    )

    with open(save_path, "wb") as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(results)} sequences to {save_path}")
