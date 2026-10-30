"""
Dump Hypersim at 518x518 so it matches the precomputed overlaps.

Reuses the exact functions from the overlap script (same scene list, same
crop call, same per-frame rng seed, same validity filter), so frame index i
here == row/col i in camera_overlap.npy / camera_occlusion.npy.

Per scene, writes:
    <out_root>/<scene>/rgb/000123.jpg
    <out_root>/<scene>/depth/000123.png      uint16, depth_m * DEPTH_SCALE, 0 = invalid
    <out_root>/<scene>/scene.pkl

scene.pkl:
{
  "scene": str,
  "dataset": "hypersim",
  "metric": True,
  "resolution": (518, 518),
  "depth_scale": DEPTH_SCALE,               # depth_m = png / depth_scale
  "frames": [                               # length N = len(scene_files[scene])
     {"idx": i, "src_rgb": original path,
      "valid": bool,                        # False -> was skipped in overlap computation
      "rgb": "rgb/000123.jpg" | None,
      "depth": "depth/000123.png" | None,
      "K": (3,3) float32 | None,            # at 518x518
      "E": (3,4) float32 | None},           # OpenCV world-to-camera
     ...],
  "overlap_dir": path to camera_overlap.npy / camera_occlusion.npy,
  "neighbors": {i: [(j, score), ...]}       # top-K sources covering target i,
                                            # score = overlap[j,i] * (1 - occlusion[j,i])
}
"""

import os
import argparse
import pickle as pkl

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- import from your overlap script (rename module if needed) ---
from compute_hypersim_overlaps import (
    rgb_to_depth_path,
    rgb_to_pointinfo_path,
    load_depth_distance,
    load_image,
    load_camera_params,
    view_camera_parameters,
    hypersim_distance_to_depth,
    _crop_resize_if_necessary,
    has_sufficient_valid_depth,
    load_all_hypersim_cameras,
    compute_frustum_overlap,
)

DEPTH_SCALE = 1000.0  # mm in uint16 -> max 65.5 m, fine for Hypersim
RES = (518, 518)

OVERLAP_BASEDIR = "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/hypersim_overlaps/"
CSV_FILE = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocess_datasets/camera_parameters_hypersim.csv"


def process_frame(rgb_path, cam_settings):
    """Identical to the overlap script's per-frame path."""
    depth_distance, mask = load_depth_distance(rgb_to_depth_path(rgb_path))
    rgb_image = load_image(rgb_path)
    cam = load_camera_params(rgb_to_pointinfo_path(rgb_path))
    RT_c2w, K = view_camera_parameters(cam, cam_settings=cam_settings)

    depth = hypersim_distance_to_depth(
        depth_distance, width=cam["width"], height=cam["height"],
        focal=(K[0, 0] + K[1, 1]) / 2,
    )

    rng = np.random.default_rng(seed=42)  # re-created per frame, same as overlap script
    image_new, depth_new, K_new, mask_new = _crop_resize_if_necessary(
        image=rgb_image, depthmap=depth, intrinsics=K,
        mask=mask.astype(np.int32), resolution=RES, rng=rng, info=None,
    )

    valid = has_sufficient_valid_depth(depth_new, threshold=0.10, min_fraction=0.5) \
        and abs(K_new[0, 1]) == 0

    depth_new = np.asarray(depth_new, dtype=np.float32).copy()
    depth_new[~(np.asarray(mask_new) > 0)] = 0.0
    return image_new, depth_new, K_new.astype(np.float32), RT_c2w.astype(np.float32), valid


def c2w_to_w2c_3x4(RT_c2w):
    R = RT_c2w[:3, :3]
    t = RT_c2w[:3, 3]
    E = np.zeros((3, 4), dtype=np.float32)
    E[:, :3] = R.T
    E[:, 3] = -R.T @ t
    return E


def build_neighbors(overlap_dir, valid_mask, top_k):
    ov = np.load(os.path.join(overlap_dir, "camera_overlap.npy"))
    oc = np.load(os.path.join(overlap_dir, "camera_occlusion.npy"))
    # ov[a, b] = fraction of b's surfaces landing in a.
    # Coverage of target t by source s = ov[s, t] * (1 - oc[s, t])  -> transpose.
    covis = (ov * (1.0 - oc)).T            # covis[t, s]
    covis[ov.T < 0] = -1.0                 # invalid pairs stay -1
    np.fill_diagonal(covis, -1.0)

    neighbors = {}
    for t in np.where(valid_mask)[0]:
        row = covis[t]
        cand = np.where((row > 0) & valid_mask)[0]
        cand = cand[np.argsort(-row[cand])][:top_k]
        neighbors[int(t)] = [(int(s), float(row[s])) for s in cand]
    return neighbors, ov, oc


def verify(scene_dir, frames, ov, n_pairs=20, seed=0):
    """Recompute overlap for random valid pairs from the DUMPED files."""
    rng = np.random.default_rng(seed)
    idx = [f["idx"] for f in frames if f["valid"]]
    if len(idx) < 2:
        return
    errs = []
    for _ in range(n_pairs):
        i, j = rng.choice(idx, 2, replace=False)
        fi, fj = frames[i], frames[j]
        di = np.array(Image.open(os.path.join(scene_dir, fi["depth"]))).astype(np.float32) / DEPTH_SCALE
        dj = np.array(Image.open(os.path.join(scene_dir, fj["depth"]))).astype(np.float32) / DEPTH_SCALE

        def c2w(E):
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = E[:, :3].T
            M[:3, 3] = -E[:, :3].T @ E[:, 3]
            return M

        r = compute_frustum_overlap(
            K1=fi["K"], K2=fj["K"], Rt1_c2w=c2w(fi["E"]), Rt2_c2w=c2w(fj["E"]),
            depth1=di, depth2=dj, mask1=di > 0, mask2=dj > 0,
        )
        errs.append(abs(r["overlap1"] - ov[i, j]))
    errs = np.array(errs)
    print(f"  verify: mean |Δoverlap| = {errs.mean():.4f}, max = {errs.max():.4f}")
    if errs.max() > 0.05:
        print("  WARNING: dumped data does not match stored overlaps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_file", default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/hypersim_scene_files_1000.pkl")
    p.add_argument("--out_root", required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None, help="exclusive")
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    with open(args.scene_file, "rb") as f:
        scene_files = pkl.load(f)       # never regenerate: indices must match overlaps
    cam_settings_all = load_all_hypersim_cameras(CSV_FILE)

    names = list(scene_files.keys())[args.start_idx:args.end_idx]
    for scene in names:
        scene_dir = os.path.join(args.out_root, scene)
        out_pkl = os.path.join(scene_dir, "scene.pkl")
        overlap_dir = os.path.join(OVERLAP_BASEDIR, scene)
        if os.path.exists(out_pkl):
            print(f"{scene}: done, skipping")
            continue
        if not os.path.exists(os.path.join(overlap_dir, "camera_overlap.npy")):
            print(f"{scene}: no overlaps, skipping")
            continue

        os.makedirs(os.path.join(scene_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)

        frames = []
        for i, rgb_path in enumerate(tqdm(scene_files[scene], desc=scene)):
            img, depth, K, RT_c2w, valid = process_frame(rgb_path, cam_settings_all[scene])
            fr = {"idx": i, "src_rgb": rgb_path, "valid": bool(valid),
                  "rgb": None, "depth": None, "K": None, "E": None}
            if valid:
                rgb_rel = f"rgb/{i:06d}.jpg"
                dep_rel = f"depth/{i:06d}.png"
                img.save(os.path.join(scene_dir, rgb_rel), quality=args.jpg_quality)
                d16 = np.clip(np.round(depth * DEPTH_SCALE), 0, 65535).astype(np.uint16)
                Image.fromarray(d16).save(os.path.join(scene_dir, dep_rel))
                fr.update(rgb=rgb_rel, depth=dep_rel, K=K, E=c2w_to_w2c_3x4(RT_c2w))
            frames.append(fr)

        valid_mask = np.array([f["valid"] for f in frames])
        neighbors, ov, oc = build_neighbors(overlap_dir, valid_mask, args.top_k)

        # consistency: frames valid now must be exactly those valid in the overlap run
        ov_valid = np.diag(ov) == 1.0
        if not np.array_equal(ov_valid, valid_mask):
            print(f"  WARNING {scene}: {np.sum(ov_valid != valid_mask)} frames disagree on validity")

        with open(out_pkl, "wb") as f:
            pkl.dump({
                "scene": scene, "dataset": "hypersim", "metric": True,
                "resolution": RES, "depth_scale": DEPTH_SCALE,
                "frames": frames, "overlap_dir": overlap_dir,
                "neighbors": neighbors,
            }, f)

        if args.verify:
            verify(scene_dir, frames, ov)


if __name__ == "__main__":
    main()