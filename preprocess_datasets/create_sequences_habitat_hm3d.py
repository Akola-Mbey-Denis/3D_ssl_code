import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # must be set before cv2 is imported

import json
import argparse
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from tqdm import tqdm

# Each worker thread decodes one file at a time; keep OpenCV's own pool off
# so threads don't oversubscribe the CPUs.
cv2.setNumThreads(1)

from .paths import list_scenes_available  # noqa: F401  (kept from original)

OVERLAP_ROOT = "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/habitat_overlaps"
SAVE_DIR = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data"


# ----------------------------------------------------------------------------
# Per-view checks (same logic as the original functions)
# ----------------------------------------------------------------------------

def load_camera_params(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    RT_c2w = np.array(data["extrinsics"], dtype=np.float32)
    K = np.array(data["intrinsics"], dtype=np.float32)
    return K, RT_c2w


def load_exr_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise IOError(f"Could not read depth: {depth_path}")
    return depth.astype(np.float32, copy=False)


def is_valid_image(image_path, valid_pixels_frac=0.05, reduced=True):
    # cv2 (libjpeg-turbo) decodes faster than PIL; channel order is irrelevant here.
    # reduced=True decodes the JPEG at 1/4 resolution directly in the decoder
    # (DCT scaling): several times faster, and the black-pixel *fraction* is
    # essentially unchanged. Use reduced=False for the exact full-res check.
    flag = cv2.IMREAD_REDUCED_COLOR_4 if reduced else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)
    if image is None:
        raise IOError(f"Could not read image: {image_path}")
    black = image.max(axis=2) < 5          # all channels < 5
    return black.mean() < valid_pixels_frac


def has_sufficient_valid_depth(depth, threshold=0.10, min_fraction=0.6):
    return (depth > threshold).mean() >= min_fraction


def is_wall_depthmap(depth, tol=0.05, wall_frac_thresh=0.4):
    d = depth[depth > 0.1]
    if d.size == 0:
        return True
    med = np.median(d)
    return (np.abs(d - med) < tol * med).mean() >= wall_frac_thresh


def is_view_ok(img_path, depth_path, reduced_image=True):
    """All checks for one view. Cheapest-to-fail checks first; stops at first failure."""
    depth = load_exr_depth(depth_path)
    if not has_sufficient_valid_depth(depth):
        return False
    if is_wall_depthmap(depth):
        return False
    return is_valid_image(img_path, reduced=reduced_image)   # JPEG only decoded if depth passed


def _check_one(args):
    """One view: checks + camera. Pure function of its files (safe to run in a thread)."""
    img_path, depth_path, cam_path, reduced_image = args
    ok = is_view_ok(img_path, depth_path, reduced_image)
    K, RT = load_camera_params(cam_path)
    return ok, K, RT


def compute_view_info(img_paths, depth_paths, cam_paths, cache_file=None,
                      num_threads=16, reduced_image=True):
    if cache_file and os.path.exists(cache_file):
        c = np.load(cache_file)
        return c["ok"], list(c["K"]), list(c["RT"])

    n = len(img_paths)
    jobs = [(img_paths[i], depth_paths[i], cam_paths[i], reduced_image) for i in range(n)]

    # The per-view work is file I/O + cv2/numpy decoding, all of which release the
    # GIL, so threads overlap the (network-filesystem) latency. Each job only reads
    # its own files and returns a value -- no shared state is written.
    # num_threads=1 runs it sequentially. map() keeps the original order.
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            out = list(tqdm(ex.map(_check_one, jobs), total=n,
                            leave=False, desc="Checking views"))
    else:
        out = [_check_one(j) for j in tqdm(jobs, leave=False, desc="Checking views")]

    ok = np.fromiter((o for o, _, _ in out), dtype=bool, count=n)
    K_list = [k for _, k, _ in out]
    RT_list = [rt for _, _, rt in out]

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, ok=ok, K=np.stack(K_list), RT=np.stack(RT_list))
    return ok, K_list, RT_list


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_overlap_sequences(
    sequences_root,
    overlap_threshold=0.010,
    max_sources=100,
    split="train",
    cache_dir=None,
    num_threads=16,
    reduced_image=True,
):
    results = {}

    seq_dirs = sorted(
        d for d in os.listdir(sequences_root)
        if os.path.isdir(os.path.join(sequences_root, d))
    )

    for scene_name in tqdm(seq_dirs, desc="Processing sequences"):
        scene_path = os.path.join(sequences_root, scene_name)
        scene_id = scene_name.split("-")[-1]
        base_path = f"{scene_path}/{scene_id}.basis"

        overlap_path = f"{OVERLAP_ROOT}/{scene_name}/camera_overlap.npy"
        occ_path = f"{OVERLAP_ROOT}/{scene_name}/camera_occlusion.npy"
        if not (os.path.exists(overlap_path) and os.path.exists(occ_path)):
            continue

        overlap_matrix = np.load(overlap_path)
        occ_matrix = np.load(occ_path)
        n = occ_matrix.shape[0]

        img_paths = [f"{base_path}/rgbs/{i:05d}.jpeg" for i in range(n)]
        depth_paths = [f"{base_path}/depths/{i:05d}_depth.exr" for i in range(n)]
        cam_paths = [f"{base_path}/cameras/{i:05d}_cam.json" for i in range(n)]

        cache_file = f"{cache_dir}/{scene_name}_views.npz" if cache_dir else None
        view_ok, K_list, RT_list = compute_view_info(
            img_paths, depth_paths, cam_paths, cache_file,
            num_threads=num_threads, reduced_image=reduced_image,
        )

        # ------------------------------------------------------------
        # Target loop — only over views that passed all checks
        # ------------------------------------------------------------
        for t in np.flatnonzero(view_ok).tolist():
            overlaps = overlap_matrix[t]
            occ = occ_matrix[t]

            cand = np.flatnonzero(overlaps >= overlap_threshold)
            cand = cand[cand != t]
            if cand.size == 0:
                continue

            # overlap desc (primary), occlusion desc (tie-breaker)
            order = np.lexsort((-occ[cand], -overlaps[cand]))
            selected = cand[order][:max_sources]
            # Same as original: take top-k first, then drop invalid views
            selected = selected[view_ok[selected]]
            if selected.size == 0:
                continue

            sources = [
                {
                    "image": img_paths[s],
                    "depth": depth_paths[s],
                    "K": K_list[s],
                    "extrinsics": RT_list[s],
                    "occlusion": o,
                    "overlap": v,
                }
                for s, o, v in zip(selected.tolist(),
                                   occ[selected].tolist(),
                                   overlaps[selected].tolist())
            ]

            results[f"{scene_id}_{t:05d}"] = {
                "target": {
                    "image": img_paths[t],
                    "depth": depth_paths[t],
                    "K": K_list[t],
                    "extrinsics": RT_list[t],
                },
                "sources": sources,
            }

    save_path = f"{SAVE_DIR}/preprocessed_habitat_3d_{split}_set.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"\n PREPROCESSING COMPLETE — Saved {len(results)} sequences to {save_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_root", type=str,
                        default="/scratch/dma9300/preprocessed_habitat_matterport3d/hm3d/")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="cache per-view checks per scene; a rerun (or a resumed job) skips finished scenes' I/O")
    parser.add_argument("--num_threads", type=int, default=16,
                        help="threads for per-view file I/O (1 = sequential)")
    parser.add_argument("--exact_image_check", action="store_true",
                        help="decode JPEGs at full resolution for the black-pixel check (slower)")
    args = parser.parse_args()

    build_overlap_sequences(
        sequences_root=f"{args.dataset_root}/{args.split}",
        split=args.split,
        cache_dir=args.cache_dir,
        num_threads=args.num_threads,
        reduced_image=not args.exact_image_check,
    )