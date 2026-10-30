import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")     # EXR depth re-encoding; before cv2 import
import io
import random
import shutil
import argparse
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor

import cv2
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

import utils.cropping as cropping      # run from the project root (python -m ...) so utils/ is importable

cv2.setNumThreads(0)      # parallelism comes from the reader threads; no nested OpenCV pools


VAL_NUM_SAMPLES = 10000      # sequences stored for --split val
TEST_NUM_SAMPLES = 1000
VAL_SEED = 42               # fixed seed: the same val subset on every build
TEST_SEED = 47


# ======================================================================
# filtering
# ======================================================================

def filter_results_by_overlap(results, min_overlap=0.10, seq_len=10):
    """
    Keep a sequence only if it has >= seq_len sources with overlap >= min_overlap;
    then keep the seq_len sources with the highest overlap.
    Occlusion is preserved in the source dicts but NOT used for filtering.
    """
    filtered = {}
    for key, data in results.items():
        valid = [s for s in data["sources"] if s["overlap"] >= min_overlap and s["overlap"] <= 0.95]
        if len(valid) < seq_len:
            continue
        valid = sorted(valid, key=lambda s: s["overlap"], reverse=True)[:seq_len]
        filtered[key] = {"target": data["target"], "sources": valid}

    print(f"Filtered results: kept {len(filtered):,} of {len(results):,} samples")
    return filtered


# ======================================================================
# files
# ======================================================================

def view_keys(data, with_depth):
    """LMDB file keys one sequence needs."""
    keys = []
    for view in [data["target"]] + list(data["sources"]):
        keys.append(f"rgb:{view['image']}")
        if with_depth and view.get("depth"):
            keys.append(f"depth:{view['depth']}")
    return keys


def collect_files(seqs, with_depth=True):
    """Unique (lmdb_key, path) pairs referenced by the sequences."""
    items = {}
    for data in seqs.values():
        for view in [data["target"]] + list(data["sources"]):
            items[f"rgb:{view['image']}"] = view["image"]
            if with_depth and view.get("depth"):
                items[f"depth:{view['depth']}"] = view["depth"]
    return sorted(items.items())


def safe_read(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data if len(data) > 0 else None          # empty file counts as missing
    except Exception as e:
        print(f"\nWARNING: failed to read {path}: {e}")
        return None


# ======================================================================
# optional downscaling at pack time (your DUSt3R cropping.rescale_image_depthmap)
# ======================================================================

def collect_views(seqs, with_depth=True):
    """
    Unique views (one per image) referenced by the sequences, with what resizing needs:
    (rgb_key, rgb_path, depth_key or None, depth_path or None, K).
    rgb and depth of a view are processed together so they get the same resize, and K comes along.
    """
    views = {}
    for data in seqs.values():
        for v in [data["target"]] + list(data["sources"]):
            rk = f"rgb:{v['image']}"
            if rk in views:
                continue
            has_depth = with_depth and bool(v.get("depth"))
            views[rk] = (rk, v["image"],
                         f"depth:{v['depth']}" if has_depth else None,
                         v["depth"] if has_depth else None,
                         np.asarray(v["K"], np.float64))
    return [views[k] for k in sorted(views)]


def _decode_depth(data, ext):
    """Depth bytes -> array in its stored dtype (.npy, or anything OpenCV reads: .png uint16, .exr ...)."""
    if ext == ".npy":
        return np.load(io.BytesIO(data), allow_pickle=False)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cannot decode depth '{ext}'; rebuild with --no_depth or add a decoder")
    return arr


def _encode_depth(arr, ext):
    """Back to the ORIGINAL format, so each dataset's depth decoder keeps working."""
    if ext == ".npy":
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()
    ok, enc = cv2.imencode(ext, arr)
    if not ok:
        raise RuntimeError(f"cannot re-encode depth as '{ext}'")
    return enc.tobytes()


def process_view(view, short_side=0, jpeg_quality=95):
    """
    Read one view's rgb (+ depth) and, if short_side > 0 and the image is larger, resize it with
    cropping.rescale_image_depthmap(image, depth, K, (short_side, short_side), force=True)
    -- the same call as dump_megadepth_crops.resize_short_side: short side -> short_side,
    aspect kept, no crop; image Lanczos, depth nearest, K via camera_matrix_of_crop.

    Images whose short side is already <= short_side are stored unchanged (no upscaling).
    Returns (rgb_key, rgb_bytes|None, depth_key|None, depth_bytes|None, K_new|None); K_new None = unchanged.
    """
    rk, rpath, dk, dpath, K = view
    rgb = safe_read(rpath)
    depth_raw = safe_read(dpath) if dk else None
    if rgb is None or short_side <= 0 or (dk and depth_raw is None):
        return rk, rgb, dk, depth_raw, None          # missing files: the sequence gets dropped later

    image = Image.open(io.BytesIO(rgb)).convert("RGB")
    if min(image.size) <= short_side:
        return rk, rgb, dk, depth_raw, None

    depth, ext = None, None
    if dk:
        ext = os.path.splitext(dpath)[1].lower()
        depth = _decode_depth(depth_raw, ext)
        if depth.shape[:2] != image.size[::-1]:
            raise RuntimeError(f"rgb {image.size[::-1]} / depth {depth.shape[:2]} size mismatch: {rpath}")

    out = cropping.rescale_image_depthmap(image, depth, K, (short_side, short_side), force=True)
    image, depth_new, K_new = out[0], out[1], out[2]

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(jpeg_quality))
    rgb_out = buf.getvalue()
    depth_out = _encode_depth(np.asarray(depth_new).astype(depth.dtype), ext) if dk else None
    return rk, rgb_out, dk, depth_out, np.asarray(K_new, np.float32)


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


def pack_views(env, views, num_threads=64, commit_every=2000, short_side=0, jpeg_quality=95, new_K=None):
    """
    Read (and optionally resize) views in parallel, write from this thread.
    Returns (files stored, set of missing keys, set of stored keys); resized K go into `new_K`.
    """
    stored, missing, done = 0, set(), set()
    out_bytes = 0
    txn = env.begin(write=True)
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        it = pool.map(lambda v: process_view(v, short_side, jpeg_quality), views)
        for rk, rgb, dk, depth, K_new in tqdm(it, total=len(views), desc="Packing views (rgb + depth)"):
            for key, data in ((rk, rgb), (dk, depth)):
                if key is None:
                    continue
                if data is None:
                    missing.add(key)
                    continue
                txn.put(key.encode("utf-8"), data)
                done.add(key)
                out_bytes += len(data)
                stored += 1
                if stored % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            if K_new is not None and new_K is not None and rgb is not None:
                new_K[rk] = K_new
    txn.commit()
    print(f"Stored files:  {stored:,}  ({out_bytes / 1024 ** 3:.2f} GB, "
          f"{out_bytes / 1024 / max(stored, 1):.0f} KB/file)")
    print(f"Missing files: {len(missing):,}")
    return stored, missing, done


# ======================================================================
# build
# ======================================================================

def pack_lmdb(pkl_path, out_path, split, min_overlap=0.10, seq_len=10,
              num_threads=64, commit_every=2000, map_size_gb=None, with_depth=True,
              resize_short_side=0, jpeg_quality=95):

    print("=" * 80)
    print(f"Building LMDB for split: {split}")
    print("=" * 80)
    print(f"Input pickle: {pkl_path}")
    print(f"Output LMDB:  {out_path}")
    if resize_short_side > 0:
        print(f"Resize:       short side -> {resize_short_side} px via cropping.rescale_image_depthmap "
              f"(rgb Lanczos + JPEG q{jpeg_quality}, depth nearest in its own format, K updated)")

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
    seqs = filter_results_by_overlap(results, min_overlap=min_overlap, seq_len=seq_len)
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
        map_size = estimate_map_size(est, collect_files(est, with_depth=with_depth))
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

            # 1. files first (only views not handled yet), so we know which are missing
            print(f"\nPacking RGB/depth files for {len(batch):,} sequences...")
            views = [v for v in collect_views(batch_seqs, with_depth=with_depth)
                     if v[0] not in packed and v[0] not in missing]
            n_stored, new_missing, new_done = pack_views(
                env, views, num_threads=num_threads, commit_every=commit_every,
                short_side=resize_short_side, jpeg_quality=jpeg_quality, new_K=new_K)
            stored_files += n_stored
            missing |= new_missing
            packed |= new_done

            # 2. only complete sequences are written and indexed
            with env.begin(write=True) as txn:
                for key in tqdm(batch, desc="Writing sequence metadata"):
                    if missing and any(k in missing for k in view_keys(seqs[key], with_depth)):
                        dropped.append(key)
                        continue
                    record = rescale_sequence(seqs[key], new_K)        # K matches the stored images
                    txn.put(f"seq:{key}".encode("utf-8"), pkl.dumps(record, protocol=pkl.HIGHEST_PROTOCOL))
                    seq_keys.append(key)

        if not seq_keys:
            raise RuntimeError("Every sequence references a missing file.")
        if len(seq_keys) < target_n:
            print(f"WARNING: only {len(seq_keys):,} complete sequences available (wanted {target_n:,})")

        rgb_count = sum(k.startswith("rgb:") for k in packed)
        depth_count = sum(k.startswith("depth:") for k in packed)

        info = {
            "format": "habitat_3d_lmdb_v1",
            "split": split,
            "source_pkl": os.path.abspath(pkl_path),
            "num_sequences": len(seq_keys),
            "num_valid_sequences_before_sampling": len(order),
            "val_num_samples": VAL_NUM_SAMPLES if split == "val" else None,
            "val_seed": VAL_SEED if split == "val" else None,
            "num_sequences_dropped_missing_files": len(dropped),
            "num_sources_per_sequence": seq_len,
            "num_unique_rgb": rgb_count,
            "num_unique_depth": depth_count,
            "num_files_stored": stored_files,
            "num_files_missing": len(missing),
            "min_overlap": min_overlap,
            "with_depth": with_depth,
            "resize_short_side": resize_short_side or None,     # None = original resolution
            "jpeg_quality": jpeg_quality if resize_short_side else None,
            "num_rgb_resized": len(new_K),
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
    print(f"Sequences:          {len(seq_keys):,}" + (f"  (random subset of {len(order):,})" if split == "val" else ""))
    print(f"Dropped (missing):  {len(dropped):,}" + (f"  e.g. {dropped[:3]}" if dropped else ""))
    print(f"Sources/sequence:   {seq_len}")
    print(f"Unique RGB:         {rgb_count:,}")
    if with_depth:
        print(f"Unique depth:       {depth_count:,}")
    print(f"Files stored:       {stored_files:,}")
    print(f"Files missing:      {len(missing):,}")
    if resize_short_side:
        print(f"Resized views:      {len(new_K):,}  (short side -> {resize_short_side} px, K rescaled)")
    print("\nFiltering:")
    print(f"  min overlap:      {min_overlap}")
    print("  occlusion:        NOT FILTERED")
    print("=" * 80)


def main():
    p = argparse.ArgumentParser(description="Convert a preprocessed Habitat 3D split to LMDB.")
    p.add_argument("--data_root", required=True, help="directory containing preprocessed_habitat_3d_<split>_set.pkl")
    p.add_argument("--split", required=True, choices=["train", "val", "test"], help="test reads the val pkl")
    p.add_argument("--dataset", required=True, default="habitat", help="test reads the val pkl")

    p.add_argument("--out", required=True, help="output LMDB path")
    p.add_argument("--min_overlap", type=float, default=0.10, help="minimum source-target overlap")
    p.add_argument("--seq_len", type=int, default=10, help="exactly this many sources per sequence")
    p.add_argument("--num_threads", type=int, default=64, help="threads used to read files")
    p.add_argument("--commit_every", type=int, default=2000, help="commit every N files")
    p.add_argument("--map_size_gb", type=float, default=500, help="LMDB map size in GB (default: estimated)")
    p.add_argument("--no_depth", action="store_true", help="do not store EXR depth files")
    p.add_argument("--resize", type=int, default=0,
                   help="downscale so the short side is this many px (e.g. 256); 0 = keep original size")
    p.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for resized RGB")
    args = p.parse_args()

    # the dataset has always used the val pkl for the test split
    # pkl_split = "val" if args.split == "test" else args.split
    pkl_path = os.path.join(args.data_root, f"preprocessed_{args.dataset}_3d_{args.split}_set.pkl")
    # preprocessed_dl3dv_5K_train_set
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"\nCould not find preprocessed file:\n{pkl_path}")

    pack_lmdb(pkl_path=pkl_path, out_path=args.out, split=args.split,
              min_overlap=args.min_overlap, seq_len=args.seq_len,
              num_threads=args.num_threads, commit_every=args.commit_every,
              map_size_gb=args.map_size_gb, with_depth=not args.no_depth,
              resize_short_side=args.resize, jpeg_quality=args.jpeg_quality)


if __name__ == "__main__":
    main()