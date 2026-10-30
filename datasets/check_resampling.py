"""
Stage-by-stage timing of every *_NVS dataset's get_data, without editing the dataset files.

For each dataset:
  1. cold  : N random samples, single process, per-stage timing
  2. warm  : the SAME samples again (if much faster -> the cost is I/O, not CPU)
  3. loader: real DataLoader on FRESH samples; throughput is measured on the second half of the
             batches only, so worker startup and the prefetch buffer don't inflate the numbers
Then a summary table across datasets.

Run from the project root:
    python -m datasets.time_all_loaders                         # all 8 training datasets
    python -m datasets.time_all_loaders --datasets dl3dv,megadepth --workers 16
    OMP_NUM_THREADS=1 python -m datasets.time_all_loaders       # mimic DataLoader workers in the main process
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")     # must be set before cv2 is imported

import argparse
import collections
import functools
import importlib
import time
import traceback
from contextlib import contextmanager

import numpy as np
from torch.utils.data import DataLoader, Dataset

print = functools.partial(print, flush=True)     # show progress immediately, even when piped to a file


CONFIG_DIR = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/configs"

# short name -> (module in datasets/, class name, config file)
DATASETS = {
    "hypersim":   ("hypersim_NVS",     "Hypersim_NVS",     "hypersim_nvs.yaml"),
    "taskonomy":  ("taskonomy_NVS",    "Taskonomy_NVS",    "taskonomy_nvs.yaml"),
    "replica":    ("replica_NVS",      "Replica_NVS",      "replica_nvs.yaml"),
    "megadepth":  ("megadepth_NVS",    "MegaDepth_NVS",    "megadepth_nvs.yaml"),
    "matterport": ("matterport_NVS",   "Matterport3D_NVS", "matterport3d_nvs.yaml"),
    "habitat":    ("habitat_hm3d_NVS", "Habitat_HM3D_NVS", "habitat_hm3d_nvs.yaml"),
    "dl3dv":      ("dl3dv_NVS",        "DL3DV_NVS",        "dl3dv_nvs.yaml"),
    "mapfree":    ("map_free_NVS",     "Mapfree_NVS",      "mapfree_nvs.yaml"),
}

# instance methods to time, if the dataset has them (label shown in the report)
METHODS = [
    ("get_data",                  "get_data (total)"),
    ("process_scene_sequence",    "  file loads + decode (sum)"),
    ("_get_sequence",             "    lmdb: sequence record"),
    ("_lmdb_get",                 "    lmdb: raw get"),
    ("_load_rgb",                 "    rgb load (+decode)"),
    ("_load_depth_and_mask",      "    depth load (+decode)"),
    ("load_image",                "    rgb load (+decode)"),
    ("load_exr_depth",            "    depth load (+decode)"),
    ("_crop_resize_if_necessary", "crop + resize"),
    ("train_image_aug",           "color aug + to-tensor"),
    ("image_transform",           "to-tensor"),
]

# module-level helpers to time (decoders etc.), if the module defines them
MODULE_FUNC_PREFIXES = ("_decode", "load_", "threshold_depth_map")


# ---------------------------------------------------------------------------
# timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = collections.defaultdict(float)
        self.n = collections.defaultdict(int)

    @contextmanager
    def __call__(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[key] += time.perf_counter() - t0
            self.n[key] += 1

    def wrap(self, key, fn):
        def wrapped(*a, **kw):
            with self(key):
                return fn(*a, **kw)
        wrapped.__wrapped__ = fn
        return wrapped

    def per_sample_ms(self, total_key="get_data (total)"):
        n = self.n.get(total_key, 0)
        return 1000 * self.t[total_key] / n if n else float("nan")

    def top_stage(self, total_key="get_data (total)"):
        ks = [k for k in self.t if k != total_key and not k.startswith(" ")]
        if not ks:
            return "-", 0.0
        k = max(ks, key=lambda k: self.t[k])
        return k.strip(), 100 * self.t[k] / max(self.t.get(total_key, 1e-9), 1e-9)

    def report(self, title, total_key="get_data (total)"):
        total = self.t.get(total_key, sum(self.t.values()))
        print(f"\n--- {title} ---")
        print(f"{'stage':38s} {'total s':>9s} {'calls':>7s} {'ms/call':>9s} {'% of total':>11s}")
        for k in sorted(self.t, key=lambda k: -self.t[k]):
            print(f"{k:38s} {self.t[k]:9.2f} {self.n[k]:7d} {1000 * self.t[k] / self.n[k]:9.1f} "
                  f"{100 * self.t[k] / max(total, 1e-9):10.1f}%")
        print(f"-> {self.per_sample_ms(total_key):.0f} ms per sample")


T = Timer()


def instrument(ds, module):
    for attr, label in METHODS:
        fn = getattr(ds, attr, None)
        if callable(fn):
            setattr(ds, attr, T.wrap(label, fn))
    # module globals the methods call by name (decoders, threshold_depth_map from dataset_util, ...)
    for name, obj in list(vars(module).items()):
        if (callable(obj) and not isinstance(obj, type) and name.startswith(MODULE_FUNC_PREFIXES)
                and not hasattr(obj, "__wrapped__")):
            setattr(module, name, T.wrap(f"    fn: {name}", obj))
    return ds


# ---------------------------------------------------------------------------
# DataLoader throughput
# ---------------------------------------------------------------------------

class _Samples(Dataset):
    def __init__(self, ds, idx, img_per_seq):
        self.ds, self.idx, self.k = ds, idx, img_per_seq

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds.get_data(seq_index=int(self.idx[i]), img_per_seq=self.k)


def _collate(batch):
    return batch          # no stacking: we only time loading


def time_loader(ds, idx, img_per_seq, num_workers, batch_size, n_batches):
    loader = DataLoader(_Samples(ds, idx, img_per_seq), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate,
                        prefetch_factor=2 if num_workers else None)
    it = iter(loader)
    t0 = time.perf_counter()
    stamps = []
    for _ in range(n_batches):
        next(it)
        stamps.append(time.perf_counter())
    del it, loader
    half = n_batches // 2
    steady_sps = (n_batches - half) * batch_size / max(stamps[-1] - stamps[half - 1], 1e-9)
    print(f"  num_workers={num_workers:3d}: first batch {stamps[0] - t0:6.2f}s | "
          f"steady {steady_sps:7.1f} samples/s  ({1000 * num_workers / max(steady_sps, 1e-9):.0f} ms/sample/worker)"
          if num_workers else
          f"  num_workers=  0: first batch {stamps[0] - t0:6.2f}s | steady {steady_sps:7.1f} samples/s")
    return steady_sps


# ---------------------------------------------------------------------------

def run_one(key, args):
    modname, clsname, cfg = DATASETS[key]
    print(f"\n{'=' * 90}\n{clsname}  ({cfg})\n{'=' * 90}")
    module = importlib.import_module(f"datasets.{modname}")
    cls = getattr(module, clsname)

    t0 = time.perf_counter()
    ds = cls(split=args.split, config_path=os.path.join(args.config_dir, cfg),
             max_views=args.max_views, n_target_views=args.n_target_views)
    print(f"init: {time.perf_counter() - t0:.1f}s, {len(ds):,} samples")

    img_per_seq = args.max_views - 1
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))
    idx = perm[:args.n_samples]
    instrument(ds, module)

    T.reset()
    for i in idx:
        ds.get_data(seq_index=int(i), img_per_seq=img_per_seq)
    T.report(f"COLD: {len(idx)} samples x {args.max_views} views, 1 process")
    cold_ms, (top, top_pct) = T.per_sample_ms(), T.top_stage()

    T.reset()
    for i in idx:
        ds.get_data(seq_index=int(i), img_per_seq=img_per_seq)
    T.report("WARM: same samples again (page cache)")
    warm_ms = T.per_sample_ms()

    sps = {}
    if args.workers:
        print(f"\n--- DataLoader: batch_size={args.batch_size}, {args.n_batches} batches, fresh samples ---")
        start = args.n_samples
        need = args.batch_size * args.n_batches
        for w in [int(x) for x in args.workers.split(",")]:
            if start + need > len(perm):
                print(f"  num_workers={w}: not enough samples left, skipped")
                continue
            sps[w] = time_loader(ds, perm[start:start + need], img_per_seq, w, args.batch_size, args.n_batches)
            start += need

    return dict(cold=cold_ms, warm=warm_ms, top=top, top_pct=top_pct, sps=sps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS), help=f"comma list of: {','.join(DATASETS)}")
    ap.add_argument("--config_dir", default=CONFIG_DIR)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_views", type=int, default=6, help="as in training (target + sources)")
    ap.add_argument("--n_target_views", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=20, help="samples for the cold/warm passes")
    ap.add_argument("--workers", default="16", help="comma list for the DataLoader test; '' to skip")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=80, help="throughput uses the second half only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_step", type=float, default=0.2, help="GPU s/it to compare against")
    ap.add_argument("--batch_per_gpu", type=int, default=64)
    args = ap.parse_args()

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; choose from {list(DATASETS)}")

    results = {}
    for k in keys:
        try:
            results[k] = run_one(k, args)
        except Exception:
            print(f"\n!! {k} failed:\n{traceback.format_exc()}")

    # ---------------- summary ----------------
    need = args.batch_per_gpu / args.gpu_step
    print(f"\n{'=' * 90}\nSUMMARY  (needed per rank: {args.batch_per_gpu} samples / {args.gpu_step}s = {need:.0f} samples/s)\n{'=' * 90}")
    ws = sorted({w for r in results.values() for w in r["sps"]})
    hdr = f"{'dataset':12s} {'cold ms/s':>10s} {'warm ms/s':>10s}  {'top stage':28s}" + \
          "".join(f" {'w=' + str(w) + ' sps':>11s}" for w in ws)
    print(hdr)
    for k, r in results.items():
        top = "{} ({:.0f}%)".format(r["top"][:22], r["top_pct"])
        row = "{:12s} {:10.0f} {:10.0f}  {:28s}".format(k, r["cold"], r["warm"], top)
        for w in ws:
            v = r["sps"].get(w)
            cell = "-" if v is None else "{:8.0f}{}".format(v, " ok" if v >= need else " !!")
            row += " {:>11s}".format(cell)
        print(row)
    print("\nms/s = ms per sample in one process.  !! = this dataset alone can't feed one GPU at --gpu_step.")
    print("A mixed batch costs the weighted mean over datasets, so one slow dataset slows every step.")