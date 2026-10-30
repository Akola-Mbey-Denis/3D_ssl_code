"""
Map-free (train scenes) -> target / source sequences from the provided SfM co-visibility overlaps.

Input (per training scene, as released):
  <root>/train/<scene>/seq0/frame_00000.jpg, seq1/frame_00000.jpg, ...
  <root>/train/<scene>/intrinsics.txt    "seq0/frame_00000.jpg fx fy cx cy W H"
  <root>/train/<scene>/poses.txt         "seq0/frame_00000.jpg qw qx qy qz tx ty tz"   world -> camera (OpenCV)
  <root>/train/<scene>/overlaps.npz      idxs (P, 4) = seq_A, frame_A, seq_B, frame_B ;  overlaps (P,) in [0, 1]

Only the training scenes have overlaps (and poses for both sequences), so the Map-free TRAINING scenes
(460) are split 80 / 10 / 10 into train / val / test by scene (no scene in two splits), saved once and reused.

Targets come from seq0, sources from seq1 (--target_seq / --source_seq):
For every target frame of seq0 (every --target_stride-th frame):
  sources = seq1 frames of the same scene with
            min_overlap <= overlap(target, source) <= max_overlap, sorted by overlap, top --max_sources
  Only inter-sequence pairs (seq0, seq1) are used; a pair given as (seq1, seq0) is flipped.

Output: <save_dir>/preprocessed_mapfree_{train,val,test}_set.pkl
  {"<scene>__seq0_<frame>": {
      "target":  {"image", "depth": None, "K" (3,3), "extrinsics" (4,4) OpenCV camera-to-world},
      "sources": [{"image", "depth": None, "K", "extrinsics", "overlap", "occlusion": None, "seq", "frame"}, ...]}}
  Map-free has no ground-truth depth, so depth is None (like DL3DV).

Usage:
  python preprocess_mapfree_sequences.py            # reads /projects/rps/dff6142/fouheylab/shared_datasets/mapfree/train/s*/
"""

import os
import json
import argparse
import pickle as pkl
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

MAPFREE_ROOT = "/projects/rps/dff6142/fouheylab/shared_datasets/mapfree"
SAVE_DIR = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data"
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 0


# ---------------------------------------------------------------------------
# reading a scene
# ---------------------------------------------------------------------------

def frame_key(seq, frame):
    return f"seq{int(seq)}/frame_{int(frame):05d}.jpg"


def load_intrinsics(scene_dir):
    """{"seq0/frame_00000.jpg": K (3,3)}"""
    Ks = {}
    with open(os.path.join(scene_dir, "intrinsics.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            fx, fy, cx, cy = map(float, p[1:5])
            Ks[p[0]] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    return Ks


def load_poses(scene_dir):
    """{"seq0/frame_00000.jpg": RT_c2w (4,4)} from world->camera quaternion + translation."""
    poses = {}
    with open(os.path.join(scene_dir, "poses.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) < 8 or p[0].startswith("#"):
                continue
            qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
            if not np.all(np.isfinite([qw, qx, qy, qz, tx, ty, tz])):
                continue
            R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()        # scipy: (x, y, z, w)
            t_w2c = np.array([tx, ty, tz])
            RT = np.eye(4, dtype=np.float32)
            RT[:3, :3] = R_w2c.T
            RT[:3, 3] = -R_w2c.T @ t_w2c
            poses[p[0]] = RT
    return poses


def load_overlaps(scene_dir):
    f = np.load(os.path.join(scene_dir, "overlaps.npz"), allow_pickle=True)
    return np.asarray(f["idxs"]).astype(np.int64), np.asarray(f["overlaps"]).astype(np.float32)


# ---------------------------------------------------------------------------
# split (by scene, over the Map-free training scenes): 80 / 10 / 10
# ---------------------------------------------------------------------------

def make_scene_splits(scenes, split_file):
    if os.path.exists(split_file):
        with open(split_file) as f:
            splits = json.load(f)
    else:
        scenes = sorted(scenes)
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(scenes))
        shuffled = [scenes[i] for i in perm]
        n_val = int(round(SPLIT_FRACTIONS["val"] * len(scenes)))
        n_test = int(round(SPLIT_FRACTIONS["test"] * len(scenes)))
        splits = {
            "train": sorted(shuffled[n_val + n_test:]),
            "val": sorted(shuffled[:n_val]),
            "test": sorted(shuffled[n_val:n_val + n_test]),
        }
        with open(split_file, "w") as f:
            json.dump(splits, f, indent=2)
        for name, lst in splits.items():
            with open(split_file.replace(".json", f"_{name}.txt"), "w") as f:
                f.write("\n".join(lst) + "\n")
    sets = {k: set(v) for k, v in splits.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), \
        "scene splits overlap"
    print({k: len(v) for k, v in splits.items()}, "->", split_file)
    return splits


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------

def build_scene_sequences(scene, scene_dir, min_overlap, max_overlap, max_sources, target_stride,
                          target_seq=0, source_seq=1):
    Ks = load_intrinsics(scene_dir)
    poses = load_poses(scene_dir)
    idxs, overlaps = load_overlaps(scene_dir)

    # only inter-sequence pairs: target from target_seq, source from source_seq (flip (src, tgt) pairs)
    neighbours = defaultdict(list)                                  # target key -> [(overlap, source key)]
    keep = (overlaps >= min_overlap) & (overlaps <= max_overlap)
    for (sa, fa, sb, fb), ov in zip(idxs[keep], overlaps[keep]):
        if sa == target_seq and sb == source_seq:
            neighbours[frame_key(sa, fa)].append((float(ov), frame_key(sb, fb)))
        elif sa == source_seq and sb == target_seq:
            neighbours[frame_key(sb, fb)].append((float(ov), frame_key(sa, fa)))

    def view(key):
        return {"image": os.path.join(scene_dir, key), "depth": None, "K": Ks[key], "extrinsics": poses[key]}

    def usable(key):
        return key in Ks and key in poses and os.path.exists(os.path.join(scene_dir, key))

    results = {}
    for target in sorted(neighbours):
        seq, frame = int(target[3]), int(target.split("frame_")[1][:5])
        if frame % target_stride != 0 or not usable(target):
            continue
        best = {}
        for ov, src in neighbours[target]:                          # dedupe (keep the highest score)
            if src != target and usable(src) and ov > best.get(src, -1.0):
                best[src] = ov
        if not best:
            continue
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:max_sources]
        sources = []
        for src, ov in ranked:
            s = view(src)
            # breakpoint()
            s.update({"overlap": ov, "occlusion": None,
                      "seq": int(src[3]), "frame": int(src.split("frame_")[1][:5])})
            sources.append(s)

        results[f"{scene}__seq{seq}_{frame:05d}"] = {"target": view(target), "sources": sources}
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Map-free sequences from the provided overlap scores (training scenes).")
    ap.add_argument("--root", default=MAPFREE_ROOT, help="Map-free root containing train/<scene>/ (e.g. train/s00229)")
    ap.add_argument("--save_dir", default=SAVE_DIR)
    ap.add_argument("--min_overlap", type=float, default=0.10)
    ap.add_argument("--max_overlap", type=float, default=1.00,
                    help="e.g. 0.9 to drop near-duplicate neighbouring video frames")
    ap.add_argument("--max_sources", type=int, default=100)
    ap.add_argument("--target_stride", type=int, default=1, help="use every n-th frame of a sequence as a target")
    ap.add_argument("--target_seq", type=int, default=0, help="targets come from this sequence")
    ap.add_argument("--source_seq", type=int, default=1, help="sources come from this sequence")
    args = ap.parse_args()

    train_root = os.path.join(args.root, "train")
    scenes = sorted(s for s in os.listdir(train_root)
                    if os.path.isfile(os.path.join(train_root, s, "overlaps.npz")))
    print(f"{len(scenes)} training scenes with overlaps.npz in {train_root}")

    os.makedirs(args.save_dir, exist_ok=True)
    splits = make_scene_splits(scenes, os.path.join(args.save_dir, "mapfree_scene_splits.json"))

    for split in ("train", "val", "test"):
        results = {}
        for scene in tqdm(splits[split], desc=f"mapfree {split}"):
            results.update(build_scene_sequences(scene, os.path.join(train_root, scene), args.min_overlap,
                                                 args.max_overlap, args.max_sources, args.target_stride,
                                                 args.target_seq, args.source_seq))
        n_src = [len(v["sources"]) for v in results.values()]
        out = os.path.join(args.save_dir, f"preprocessed_mapfree_3d_{split}_set.pkl")
        with open(out, "wb") as f:
            pkl.dump(results, f)
        print(f"{split}: {len(results)} sequences, sources/target median {int(np.median(n_src)) if n_src else 0} -> {out}")