import os
import argparse
import pickle as pkl

import numpy as np
from tqdm import tqdm


SPLIT_TO_FOLD = {"train": "Training", "val": "Validation"}      # ARKitScenes folds written by prepare_arkitscenes.py


def build_overlap_sequences(scene_dict, dump_root, overlap_basedir,
                            overlap_thres=0.010, max_sources=100, split="train"):
    """
    Build target / source sequences from the dumped ARKitScenes captures and their overlap matrices.

    Everything is already preprocessed (prepare_arkitscenes.py, arkitscenes_overlaps.py):
      <dump_root>/<fold>/<video_id>/rgb/<video_id>_<ts>.jpg
      <dump_root>/<fold>/<video_id>/depth/<video_id>_<ts>.npy        meters, 0 = invalid
      <dump_root>/<fold>/<video_id>/cameras.npz                      frame_ids, K (N,3,3), c2w (N,4,4) OpenCV
      <overlap_basedir>/<fold>/<video_id>/camera_overlap.npy, camera_occlusion.npy
    A scene is "<fold>/<video_id>"; the split is the fold (train = Training, val = Validation).
    Row / column i of the matrices <-> scene_dict[scene][i] (same list the overlaps were computed from).
    Frames were validated by the dump, so no image / depth checks here.
    """
    results = {}
    fold = SPLIT_TO_FOLD[split]
    scene_ids = sorted(s for s in scene_dict if s.split("/")[0] == fold)

    for scene_id in tqdm(scene_ids, desc=f"Processing arkitscenes {split}"):
        scene_dir = os.path.join(dump_root, scene_id)
        overlaps_dir = os.path.join(overlap_basedir, scene_id)

        if not os.path.exists(os.path.join(overlaps_dir, "camera_overlap.npy")):
            continue

        # ---------------- overlaps ----------------
        camera_overlap_matrix = np.load(os.path.join(overlaps_dir, "camera_overlap.npy"))
        occlusion_matrix = np.load(os.path.join(overlaps_dir, "camera_occlusion.npy"))

        scene_files = scene_dict[scene_id]
        num_images = len(scene_files)
        assert camera_overlap_matrix.shape == (num_images, num_images), \
            f"{scene_id}: overlap matrix {camera_overlap_matrix.shape} vs {num_images} images"

        # ---------------- cameras (already processed) ----------------
        cameras = np.load(os.path.join(scene_dir, "cameras.npz"))
        video_id = os.path.basename(os.path.normpath(scene_dir))
        frame_index = {str(fid): i for i, fid in enumerate(cameras["frame_ids"])}   # <ts> -> row of K / c2w

        view_names = [os.path.splitext(os.path.basename(p))[0] for p in scene_files]    # <video_id>_<ts>
        rows = [frame_index[name[len(video_id) + 1:]] for name in view_names]
        K_list = [np.asarray(cameras["K"][r], np.float32) for r in rows]
        RT_list = [np.asarray(cameras["c2w"][r], np.float32) for r in rows]
        depth_paths = [os.path.join(scene_dir, "depth", name + ".npy") for name in view_names]

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
    parser.add_argument("--split", type=str, default="train", choices=list(SPLIT_TO_FOLD), help="dataset split")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/arkitscenes_overlap_scene_files.pkl", help="Scene files of the dumped dataset (same file used for the overlaps)")
    parser.add_argument("--dump_root", type=str, default="/scratch/dma9300/preprocessed_arkitscenes", help="Dumped dataset: <fold>/<video_id>/{rgb, depth, cameras.npz}")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/arkitscenes_overlaps/", help="Overlap matrices, one folder per scene")
    args = parser.parse_args()

    with open(args.scene_file, "rb") as f:
        all_scene_dict = pkl.load(f)

    results = build_overlap_sequences(scene_dict=all_scene_dict, dump_root=args.dump_root,
                                      overlap_basedir=args.overlap_basedir, split=args.split)

    save_path = f"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/preprocessed_arkitscenes_3d_{args.split}_set.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\n Saved {len(results)} sequences to {save_path}")