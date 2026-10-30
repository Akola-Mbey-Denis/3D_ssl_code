import os
import random
import shutil
import argparse
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor

import io

import cv2
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

import utils.cropping as cropping      # run from the project root (python -m ...) so utils/ is importable

cv2.setNumThreads(0)                   # parallelism comes from the reader threads


VAL_NUM_SAMPLES = 10000      # sequences stored for --split val
TEST_NUM_SAMPLES = 1000
VAL_SEED = 42               # fixed seed: the same val subset on every build
TEST_SEED = 47


# ======================================================================
# filtering
# ======================================================================

def filter_results(results, seq_len=10):
    """
    DL3DV has no depth, overlap or occlusion. Sources were already chosen in preprocessing
    (frames within a window of the target, filtered by camera angle / distance) and carry
    "frame_gap", "angle" and "distance".

    Keep a sequence only if it has >= seq_len sources; if it has more, keep seq_len of them
    spread evenly over the frame order (same rule as preprocessing).
    """
    filtered = {}
    for key, data in results.items():
        sources = sorted(data["sources"], key=lambda s: s["frame_gap"])
        if len(sources) < seq_len:
            continue
        if len(sources) > seq_len:
            sources = sorted(sources, key=lambda s: s["angle"], reverse=True)[:seq_len]
            
        filtered[key] = {"target": data["target"], "sources": sources}

    print(f"Filtered results: kept {len(filtered):,} of {len(results):,} samples")
    return filtered


# ======================================================================
# files
# ======================================================================

def view_keys(data):
    """LMDB file keys one sequence needs (RGB only, DL3DV has no depth)."""
    return [f"rgb:{view['image']}" for view in [data["target"]] + list(data["sources"])]


def collect_files(seqs):
    """Unique (lmdb_key, path) pairs referenced by the sequences (RGB only)."""
    items = {}
    for data in seqs.values():
        for view in [data["target"]] + list(data["sources"]):
            items[f"rgb:{view['image']}"] = view["image"]
    return sorted(items.items())


def safe_read(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data if len(data) > 0 else None          # empty file counts as missing
    except Exception as e:
        print(f"\nWARNING: failed to read {path}: {e}")
        return None


def estimate_map_size(seqs, file_items, sample_files=1000, safety_factor=1.25):
    metadata = sum(len(pkl.dumps(d, protocol=pkl.HIGHEST_PROTOCOL)) + len(k) + 4 for k, d in seqs.items())

    step = max(1, len(file_items) // sample_files)               # spread the sample over all files
    sizes = []
    for _, path in tqdm(file_items[::step], desc="Estimating file size", leave=False):
        try:
            sizes.append(os.path.getsize(path))
        except OSError:
            pass
    files = (sum(sizes) / len(sizes)) * len(file_items) if sizes else 0

    total = (metadata + files + len(pkl.dumps(list(seqs.keys()))) + (1 << 20)) * safety_factor
    return int(total + (1 << 30))                                 # + 1 GB slack


# ======================================================================
# optional downscaling at pack time (your DUSt3R cropping.rescale_image_depthmap)
# ======================================================================

def collect_views(seqs):
    """Unique views (one per image): (rgb_key, rgb_path, K). K is needed to resize consistently."""
    views = {}
    for data in seqs.values():
        for v in [data["target"]] + list(data["sources"]):
            rk = f"rgb:{v['image']}"
            if rk not in views:
                views[rk] = (rk, v["image"], np.asarray(v["K"], np.float64))
    return [views[k] for k in sorted(views)]


def process_view(view, short_side=0, jpeg_quality=95):
    """
    Read one image and, if short_side > 0 and it is larger, resize it with
    cropping.rescale_image_depthmap(image, depth, K, (short_side, short_side), force=True):
    short side -> short_side, aspect kept, no crop, Lanczos, K via camera_matrix_of_crop.
    DL3DV has no depth: a tiny dummy map is passed so the call works with any cropping version.
    Images whose short side is already <= short_side are stored unchanged (no upscaling).
    Returns (rgb_key, bytes|None, K_new|None); K_new None = unchanged.
    """
    rk, path, K = view
    data = safe_read(path)
    if data is None or short_side <= 0:
        return rk, data, None

    image = Image.open(io.BytesIO(data)).convert("RGB")
    if min(image.size) <= short_side:
        return rk, data, None

    dummy_depth = np.zeros(image.size[::-1], np.uint8)
    out = cropping.rescale_image_depthmap(image, dummy_depth, K, (short_side, short_side), force=True)
    image, K_new = out[0], out[2]

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(jpeg_quality))
    return rk, buf.getvalue(), np.asarray(K_new, np.float32)


def rescale_sequence(data, new_K):
    """Copy of a sequence whose views' K match the resized images (views not resized keep their K)."""
    if not new_K:
        return data

    def view(v):
        K = new_K.get(f"rgb:{v['image']}")
        if K is None:
            return v
        v = dict(v)
        v["K"] = K
        return v

    out = dict(data)
    out["target"] = view(data["target"])
    out["sources"] = [view(s) for s in data["sources"]]
    return out


def pack_files(env, views, num_threads=64, commit_every=2000, short_side=0, jpeg_quality=95, new_K=None):
    """
    Read (and optionally resize) images in parallel, write from this thread.
    Returns (stored, set of missing keys); K of resized images go into `new_K`.
    """
    stored, missing, out_bytes = 0, set(), 0
    txn = env.begin(write=True)
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        it = pool.map(lambda v: process_view(v, short_side, jpeg_quality), views)
        for key, data, K_new in tqdm(it, total=len(views), desc="Packing RGB files"):
            if data is None:
                missing.add(key)
                continue
            txn.put(key.encode("utf-8"), data)
            out_bytes += len(data)
            if K_new is not None and new_K is not None:
                new_K[key] = K_new
            stored += 1
            if stored % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
    txn.commit()
    print(f"Stored files:  {stored:,}  ({out_bytes / 1024 ** 3:.2f} GB, {out_bytes / 1024 / max(stored, 1):.0f} KB/file)")
    print(f"Missing files: {len(missing):,}")
    return stored, missing


# ======================================================================
# build
# ======================================================================

def pack_lmdb(pkl_path, out_path, split, seq_len=10,
              num_threads=64, commit_every=2000, map_size_gb=None,
              resize_short_side=0, jpeg_quality=95):

    print("=" * 80)
    print(f"Building LMDB for split: {split}")
    print("=" * 80)
    print(f"Input pickle: {pkl_path}")
    print(f"Output LMDB:  {out_path}")
    if resize_short_side > 0:
        print(f"Resize:       short side -> {resize_short_side} px via cropping.rescale_image_depthmap "
              f"(Lanczos + JPEG q{jpeg_quality}, K updated)")

    if os.path.exists(out_path):
        raise FileExistsError(f"\nOutput already exists:\n{out_path}\n\nRemove it first to rebuild.")

    # build on node-local disk ($TMPDIR) if available, then move: LMDB writes are not safe on NFS
    build_dir = os.environ.get("TMPDIR")
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)
        work_path = os.path.join(build_dir, os.path.basename(out_path))
        for p in (work_path, work_path + "-lock"):
            if os.path.exists(p):
                os.remove(p)
        print(f"Build path:   {work_path}  (moved to the output when done)")
    else:
        work_path = out_path
        print("Build path:   output path directly ($TMPDIR not set)")

    # ---------------- load ----------------
    print("\nLoading preprocessed pickle...")
    with open(pkl_path, "rb") as f:
        results = pkl.load(f)
    print(f"Original sequences: {len(results):,}")

    # ---------------- filter ----------------
    print("\nFiltering sequences...")
    seqs = filter_results(results, seq_len=seq_len)
    del results
    if not seqs:
        raise RuntimeError("No sequences remain after filtering.")
    for key, data in seqs.items():
        if len(data["sources"]) != seq_len:
            raise RuntimeError(f"Sequence {key} has {len(data['sources'])} sources instead of {seq_len}.")
    print(f"All retained sequences have exactly {seq_len} sources.")

    # ---------------- which sequences ----------------
    if split == "val":
        order = sorted(seqs.keys())
        random.Random(VAL_SEED).shuffle(order)
        target_n = min(VAL_NUM_SAMPLES, len(order))
        print(f"\nValidation: randomly sampling {target_n:,} of {len(order):,} valid sequences (seed {VAL_SEED})")
    elif split == "test":
        order = sorted(seqs.keys())
        random.Random(TEST_SEED).shuffle(order)
        target_n = min(TEST_NUM_SAMPLES, len(order))
        print(f"\nTest: randomly sampling {target_n:,} of {len(order):,} valid sequences (seed {TEST_SEED})")
    else:
        order = list(seqs.keys())
        target_n = len(order)

    if map_size_gb is not None:
        map_size = int(map_size_gb * 1024 ** 3)
    else:
        print("\nEstimating LMDB map size...")
        est = {k: seqs[k] for k in order[:target_n]}
        map_size = estimate_map_size(est, collect_files(est))
    print(f"LMDB map size: {map_size / 1024 ** 3:.2f} GB")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    env = lmdb.open(work_path, map_size=map_size, subdir=False, meminit=False, lock=True, readahead=False)

    try:
        # Pack files, then write the complete sequences. For val, sequences dropped for a missing
        # file are replaced by the next ones in the shuffled order until target_n are stored.
        seq_keys, dropped = [], []
        packed, missing = set(), set()
        new_K = {}                                    # rgb key -> K of the resized image
        stored_files, pos = 0, 0

        while len(seq_keys) < target_n and pos < len(order):
            batch = order[pos: pos + (target_n - len(seq_keys))]
            pos += len(batch)
            batch_seqs = {k: seqs[k] for k in batch}

            # 1. files first (only ones not handled yet), so we know which are missing
            print(f"\nPacking RGB files for {len(batch):,} sequences...")
            views = [v for v in collect_views(batch_seqs)
                     if v[0] not in packed and v[0] not in missing]
            n_stored, new_missing = pack_files(env, views, num_threads=num_threads, commit_every=commit_every,
                                               short_side=resize_short_side, jpeg_quality=jpeg_quality,
                                               new_K=new_K)
            stored_files += n_stored
            missing |= new_missing
            packed |= {v[0] for v in views} - new_missing

            # 2. only complete sequences are written and indexed
            with env.begin(write=True) as txn:
                for key in tqdm(batch, desc="Writing sequence metadata"):
                    if missing and any(k in missing for k in view_keys(seqs[key])):
                        dropped.append(key)
                        continue
                    record = rescale_sequence(seqs[key], new_K)         # K matches the stored images
                    txn.put(f"seq:{key}".encode("utf-8"), pkl.dumps(record, protocol=pkl.HIGHEST_PROTOCOL))
                    seq_keys.append(key)

        if not seq_keys:
            raise RuntimeError("Every sequence references a missing file.")
        if len(seq_keys) < target_n:
            print(f"WARNING: only {len(seq_keys):,} complete sequences available (wanted {target_n:,})")

        rgb_count = sum(k.startswith("rgb:") for k in packed)

        info = {
            "format": "dl3dv_lmdb_v1",
            "split": split,
            "source_pkl": os.path.abspath(pkl_path),
            "num_sequences": len(seq_keys),
            "num_valid_sequences_before_sampling": len(order),
            "num_samples": target_n if split in ("val", "test") else None,
            "seed": {"val": VAL_SEED, "test": TEST_SEED}.get(split),
            "num_sequences_dropped_missing_files": len(dropped),
            "num_sources_per_sequence": seq_len,
            "num_unique_rgb": rgb_count,
            "num_files_stored": stored_files,
            "num_files_missing": len(missing),
            "with_depth": False,
            "resize_short_side": resize_short_side or None,     # None = original resolution
            "jpeg_quality": jpeg_quality if resize_short_side else None,
            "num_rgb_resized": len(new_K),
            # per-view fields: image, K, extrinsics (c2w, OpenCV); depth / mask are None
            # per-source fields: frame_gap, angle (deg), distance (normalized scene units)
            "source_fields": ["frame_gap", "angle", "distance"],
        }
        with env.begin(write=True) as txn:
            txn.put(b"__seq_keys__", pkl.dumps(seq_keys, protocol=pkl.HIGHEST_PROTOCOL))
            txn.put(b"__info__", pkl.dumps(info, protocol=pkl.HIGHEST_PROTOCOL))
        env.sync()
    finally:
        env.close()

    # ---------------- move to the output ----------------
    if work_path != out_path:
        print(f"\nMoving {work_path} -> {out_path} ...")
        shutil.move(work_path, out_path)
        if os.path.exists(work_path + "-lock"):
            os.remove(work_path + "-lock")

    # ---------------- summary ----------------
    print("\n" + "=" * 80)
    print("LMDB CREATION COMPLETE")
    print("=" * 80)
    print(f"Split:              {split}")
    print(f"Output:             {out_path}  ({os.path.getsize(out_path) / 1024 ** 3:.2f} GB)")
    print(f"Sequences:          {len(seq_keys):,}" + (f"  (random subset of {len(order):,})" if split in ("val", "test") else ""))
    print(f"Dropped (missing):  {len(dropped):,}" + (f"  e.g. {dropped[:3]}" if dropped else ""))
    print(f"Sources/sequence:   {seq_len}")
    print(f"Unique RGB:         {rgb_count:,}")
    print(f"Files stored:       {stored_files:,}")
    print(f"Files missing:      {len(missing):,}")
    if resize_short_side:
        print(f"Resized RGB:        {len(new_K):,}  (short side -> {resize_short_side} px, K rescaled)")
    print("Depth / overlap / occlusion: none (DL3DV); sources carry frame_gap, angle, distance")
    print("=" * 80)


def main():
    p = argparse.ArgumentParser(description="Convert a preprocessed DL3DV split to LMDB.")
    p.add_argument("--data_root", required=True, help="directory containing preprocessed_dl3dv_<subset>_<split>_set.pkl")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--subset", default="5K", help="DL3DV subset the pkl was built from")

    p.add_argument("--out", required=True, help="output LMDB path")
    p.add_argument("--seq_len", type=int, default=10, help="exactly this many sources per sequence")
    p.add_argument("--num_threads", type=int, default=64, help="threads used to read files")
    p.add_argument("--commit_every", type=int, default=2000, help="commit every N files")
    p.add_argument("--map_size_gb", type=float, default=None, help="LMDB map size in GB (default: estimated)")
    p.add_argument("--resize", type=int, default=0,
                   help="resize so the short side is this many px (e.g. 256); 0 = keep original size")
    p.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for resized RGB")
    args = p.parse_args()

    pkl_path = os.path.join(args.data_root, f"preprocessed_dl3dv_{args.subset}_{args.split}_set.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"\nCould not find preprocessed file:\n{pkl_path}")

    pack_lmdb(pkl_path=pkl_path, out_path=args.out, split=args.split, seq_len=args.seq_len,
              num_threads=args.num_threads, commit_every=args.commit_every,
              map_size_gb=args.map_size_gb,
              resize_short_side=args.resize, jpeg_quality=args.jpeg_quality)


if __name__ == "__main__":
    main()