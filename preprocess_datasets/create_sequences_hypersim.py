import csv
import os
import argparse
import pickle as pkl

import numpy as np
from tqdm import tqdm


def load_split_ids(csv_path):
    split_dict = {"train": [], "val": [], "test": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)   # header row: id,train,val,test
        for row in reader:
            for split in ("train", "val", "test"):
                if row[split] == "1":
                    split_dict[split].append(row["id"])
    return split_dict


def build_overlap_sequences(split_csv_file, scene_dict, dump_root, overlap_basedir,
                            overlap_thres=0.010, max_sources=100, split="train"):
    """
    Build target / source sequences from the dumped scenes and their overlap matrices.

    Everything is already preprocessed:
      <dump_root>/<scene>/rgb/<name>.jpg
      <dump_root>/<scene>/depth/<name>.npy
      <dump_root>/<scene>/cameras.pkl     {"<name>.jpg": {"K", "RT_c2w" (OpenCV), "hw", "depth", "src_rgb"}}
      <overlap_basedir>/<scene>/camera_overlap.npy, camera_occlusion.npy
    Row / column i of the matrices <-> scene_dict[scene][i] (same list the overlaps were computed from).
    Frames were validated by the dump, so no image / depth checks here.
    """
    results = {}
    scene_ids = sorted(load_split_ids(split_csv_file)[split])

    for scene_id in tqdm(scene_ids, desc=f"Processing hypersim {split}"):
        scene_dir = os.path.join(dump_root, scene_id)
        overlaps_dir = os.path.join(overlap_basedir, scene_id)

        if scene_id not in scene_dict or not os.path.exists(os.path.join(overlaps_dir, "camera_overlap.npy")):
            continue

        # ---------------- overlaps ----------------
        camera_overlap_matrix = np.load(os.path.join(overlaps_dir, "camera_overlap.npy"))
        occlusion_matrix = np.load(os.path.join(overlaps_dir, "camera_occlusion.npy"))

        scene_files = scene_dict[scene_id]
        num_images = len(scene_files)
        assert camera_overlap_matrix.shape == (num_images, num_images), \
            f"{scene_id}: overlap matrix {camera_overlap_matrix.shape} vs {num_images} images"

        # ---------------- cameras (already processed) ----------------
        with open(os.path.join(scene_dir, "cameras.pkl"), "rb") as f:
            cameras = pkl.load(f)

        cams = [cameras[os.path.basename(p)] for p in scene_files]
        K_list = [np.asarray(c["K"], np.float32) for c in cams]
        RT_list = [np.asarray(c["RT_c2w"], np.float32) for c in cams]
        depth_paths = [os.path.join(scene_dir, "depth", c["depth"]) for c in cams]
        view_names = [os.path.splitext(os.path.basename(p))[0] for p in scene_files]

        # ---------------- sequences ----------------
        for target_idx in range(num_images):
            overlap_scores = camera_overlap_matrix[target_idx]      # (N,)
            occlusion_scores = occlusion_matrix[target_idx]         # (N,)

            # sources: overlap above the threshold, not the target itself
            valid_indices = np.where(overlap_scores >= overlap_thres)[0]
            valid_indices = valid_indices[valid_indices != target_idx]
            if len(valid_indices) == 0:
                continue

            # sort by overlap (primary), occlusion as tie-breaker; keep top-k
            sorted_indices = valid_indices[np.lexsort((-occlusion_scores[valid_indices],
                                                       -overlap_scores[valid_indices]))]
            selected_sources = sorted_indices[:max_sources]

            sources = [{
                "image": scene_files[src_idx],
                "depth": depth_paths[src_idx],
                "K": K_list[src_idx],
                "extrinsics": RT_list[src_idx],
                "occlusion": float(occlusion_scores[src_idx]),
                "overlap": float(overlap_scores[src_idx]),
            } for src_idx in selected_sources]

            results[f"{scene_id}__{view_names[target_idx]}"] = {
                "target": {
                    "image": scene_files[target_idx],
                    "depth": depth_paths[target_idx],
                    "K": K_list[target_idx],
                    "extrinsics": RT_list[target_idx],
                },
                "sources": sources,
            }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", help="dataset split")
    parser.add_argument("--scene_splits_csv", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/occ_project/occ/preprocess_datasets/hypersim_splits.csv", help="CSV file containing scene splits")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/hypersim_low_res_overlap_scene_files.pkl", help="Scene files of the dumped dataset (same file used for the overlaps)")
    parser.add_argument("--dump_root", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_hypersim/", help="Dumped dataset: <scene>/{rgb, depth, cameras.pkl}")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/low_res_hypersim_overlaps/", help="Overlap matrices, one folder per scene")
    args = parser.parse_args()

    with open(args.scene_file, "rb") as f:
        all_scene_dict = pkl.load(f)

    results = build_overlap_sequences(split_csv_file=args.scene_splits_csv, scene_dict=all_scene_dict,
                                      dump_root=args.dump_root, overlap_basedir=args.overlap_basedir,
                                      split=args.split)

    save_path = f"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/preprocessed_low_res_hypersim_3d_{args.split}_set.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\n Saved {len(results)} sequences to {save_path}")