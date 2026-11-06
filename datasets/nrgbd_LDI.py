"""
NRGBD multi-view LDI dataset read from the LMDB written by preprocess_ldi_lmdb.py (--dataset nrgbd).

NRGBD is a ZERO-SHOT TEST benchmark: there is ONE LMDB (all scenes, only the frames selected for the overlap
matrices, every good frame an anchor), and this loader only serves split='test' from it -- a fixed subset of
num_test_keys (default 500) sequences, drawn once with seed 1 and dumped next to the LMDB, then always reloaded.
train / val raise an error so NRGBD can never end up in a training mix.

LMDB (in this order):
    lmdb_path=...                      constructor argument
    LMDB_PATHS.test in the config      e.g. LMDB_PATHS: {test: .../preprocessed_data/nrgbd_ldi_data.lmdb}
    <DUMP_ROOT>/LDIs/nrgbd_test_ldi_seq6.lmdb
    $LMDB_DIR set (the job copied the LMDBs to node-local disk): the same file name is read from $LMDB_DIR
The LMDB may have been built with any --split (for NRGBD every split holds every scene); only the dataset is checked.

Sample (lists of num_views = 6 views, like the other loaders):
    images      [S x (3, H, W)]   in [0, 1] (no augmentation)
    ldis        [S x (L, H, W)]   z-depth of layer k in metres (near -> far), 0 = no surface
    ldis_mask   [S x (L, H, W)]   1 where layer k exists
    depths      [S x (H, W)]      GT depth maps (metres), depth_masks [S x (H, W)]
    intrinsics  [S x (3, 3)]      K of the 224 x 224 views
    extrinsics  [S x (4, 4)]      world-to-camera (as stored)
    image_names [S x str], seq_name "<scene>/<anchor frame>", scene_name "<scene>"

Sequences: the stored ones, views in the stored order (view 0 = the anchor / target). Nothing is random in a sample.

Check samples (one point cloud + RGB image PER VIEW, in <out_dir>/<scene>__<anchor>/view<i>_<frame>.ply/.png):
    python -m datasets.nrgbd_ldi_lmdb --config_path configs/nrgbd_ldi_lmdb.yaml              # camera frame per view
    python -m datasets.nrgbd_ldi_lmdb --config_path configs/nrgbd_ldi_lmdb.yaml --frame world  # views overlay
Debugging (alignment of mesh layer 0 vs depth incl. pixel-shift test, layer order / gaps, cross-view reprojection,
hidden points outside the room; + |L0 - depth| error PNGs):
    python -m datasets.nrgbd_ldi_lmdb --config_path configs/nrgbd_ldi_lmdb.yaml --debug --num_samples 10 --no_ply
"""
import argparse
import io
import os
import os.path as osp
import pickle as pkl
import random

import lmdb
import numpy as np
import torch
import yaml
from fvcore.common.config import CfgNode
from PIL import Image

from .augmentation import PhotometricAug
from .base_dataset import BaseDataset

# ---------------------------------------------------------------------------
# LMDB handles: one read-only handle per (process, file). py-lmdb refuses to open the same file twice in one
# process, and a handle must not be used across fork(): a DataLoader worker drops the copy it inherited.
# ---------------------------------------------------------------------------
# Shared by the LDI LMDB loaders (an LMDB file can be open only once per process, whichever class opens it).
_LMDB_ENVS = lmdb.__dict__.setdefault("_ldi_open_envs", {})


def _open_lmdb(path):
    path = osp.realpath(path)
    key = (os.getpid(), path)
    if key not in _LMDB_ENVS:
        for k in [k for k in _LMDB_ENVS if k[1] == path and k[0] != os.getpid()]:
            _LMDB_ENVS.pop(k).close()
        _LMDB_ENVS[key] = lmdb.open(path, subdir=False, readonly=True, lock=False,
                                    readahead=False, meminit=False, max_readers=512)
    return _LMDB_ENVS[key]


def unpack_array(d):
    """Raw float16 bytes -> array."""
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])


def load_or_dump_keys(path, sequences, n, seed):
    """Fixed subset of n sequences: loaded from `path` if it exists, otherwise drawn (seeded) and dumped there.

    The file holds {"keys": ["<scene>/<anchor>", ...], "sequences": [(scene, [frame x num_views]), ...]}, so a
    loaded subset always has the same anchors AND the same views. Keys not in this LMDB are an error (the file
    belongs to another LMDB / preprocessing run: delete it to draw again).
    """
    if osp.isfile(path):
        with open(path, "rb") as f:
            saved = pkl.load(f)
        have = {f"{scene}/{frames[0]}" for scene, frames in sequences}
        missing = [k for k in saved["keys"] if k not in have]
        if missing:
            raise ValueError(f"{len(missing)} of {len(saved['keys'])} keys in {path} are not in this LMDB "
                             f"(e.g. {missing[0]}); delete the file to draw a new subset")
        print(f"loaded {len(saved['keys']):,} fixed keys from {path}")
        return [(scene, list(frames)) for scene, frames in saved["sequences"]]

    idx = sorted(random.Random(seed).sample(range(len(sequences)), min(n, len(sequences))))
    subset = [(sequences[i][0], list(sequences[i][1])) for i in idx]
    keys = [f"{scene}/{frames[0]}" for scene, frames in subset]
    try:
        os.makedirs(osp.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pkl.dump({"keys": keys, "sequences": subset}, f, protocol=pkl.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                                        # atomic: DDP ranks may race, same content
        print(f"dumped {len(keys):,} fixed keys to {path}")
    except OSError as e:                                             # read-only location: still deterministic
        print(f"could not write {path} ({e}); using the seeded subset without saving it")
    return subset


class NRGBD_LDI(BaseDataset):
    DATASET = "nrgbd"

    def __init__(self, split, config_path, lmdb_path=None, num_layers=None, overfit=False,
                 keys_dir=None, resample=None, num_test_keys=200, *args, **kwargs):
        if split != "test":
            raise ValueError(f"NRGBD is a zero-shot test benchmark: only split='test' is served (got '{split}')")
        if resample:                                                 # old configs: the option no longer exists
            raise ValueError("resample is no longer supported: the stored sequences are always used as they are")
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        opts = CfgNode(raw_cfg)
        self.opts = opts
        self.common_conf = opts.COMMON
        super().__init__(self.common_conf, *args, **kwargs)

        # ---------------- the single NRGBD LMDB ----------------
        if lmdb_path is None:
            paths = raw_cfg.get("LMDB_PATHS") or {}
            lmdb_path = paths.get("test") or osp.join(str(raw_cfg.get("DUMP_ROOT", "")), "LDIs",
                                                      f"{self.DATASET}_test_ldi_seq6.lmdb")
        lmdb_path = os.path.expandvars(lmdb_path)
        keys_dir = keys_dir or osp.dirname(osp.abspath(lmdb_path))     # key files: next to the original LMDB
        if os.environ.get("LMDB_DIR"):                               # copied to node-local disk by the job
            lmdb_path = osp.join(os.environ["LMDB_DIR"], osp.basename(lmdb_path))
        if not osp.isfile(lmdb_path):
            raise FileNotFoundError(f"{self.DATASET} LDI LMDB not found: {lmdb_path}\n"
                                    f"  config: {osp.abspath(config_path)}  LMDB_PATHS: {raw_cfg.get('LMDB_PATHS')}")
        self.lmdb_path = osp.realpath(lmdb_path)

        with _open_lmdb(lmdb_path).begin() as txn:
            meta = txn.get(b"__meta__")
        if meta is None:
            raise RuntimeError(f"{lmdb_path} has no __meta__: preprocessing did not finish")
        meta = pkl.loads(meta)
        if meta["dataset"] != self.DATASET:                          # any build split: NRGBD splits = every scene
            raise ValueError(f"{lmdb_path} holds {meta['dataset']} [{meta['split']}], not {self.DATASET}")

        self.meta = meta
        self.split = split
        self.overfit = overfit
        self.num_views = meta["num_views"]
        self.num_layers = min(num_layers or meta["num_layers"], meta["num_layers"])
        self.training = False
        self.image_aug = PhotometricAug()

        self.scene_frames = meta["scene_frames"]                     # {scene: sorted clean frames}
        self.sequences = meta["sequences"]                           # [(scene, [frame x num_views])]
        # Fixed subsets, dumped once next to the LMDB and loaded afterwards:
        #   overfit -> <lmdb stem>_overfitting_set_100_keys.pkl
        #   test    -> <lmdb stem>_test_<num_test_keys>_keys.pkl  (seed 1)
        stem = osp.splitext(osp.basename(lmdb_path))[0]
        if overfit:
            self.keys_path = osp.join(keys_dir, f"{stem}_overfitting_set_100_keys.pkl")
            self.sequences = load_or_dump_keys(self.keys_path, self.sequences, 100, seed=0)
        else:
            self.keys_path = osp.join(keys_dir, f"{stem}_test_{num_test_keys}_keys.pkl")
            self.sequences = load_or_dump_keys(self.keys_path, self.sequences, num_test_keys, seed=1)
        self.keys = [f"{scene}/{frames[0]}" for scene, frames in self.sequences]

        n_scenes = len({scene for scene, _ in self.sequences})
        print(f"NRGBD_LDI [test{' overfit' if overfit else ''}] {len(self.sequences):,} of {len(meta['sequences']):,} "
              f"sequences of {self.num_views} views from {n_scenes} scenes ({lmdb_path}) -- fixed keys {self.keys_path}")

    def __len__(self):
        return len(self.sequences)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_frame(self, txn, scene, stem):
        """One stored frame: image (PIL 224 x 224), depth (H, W), LDI (L, H, W), K (3, 3), world-to-camera (4, 4)."""
        buf = txn.get(f"frame:{scene}/{stem}".encode())
        if buf is None:
            raise KeyError(f"frame:{scene}/{stem} not in {self.lmdb_path}")
        rec = pkl.loads(buf)
        image = Image.open(io.BytesIO(rec["rgb"])).convert("RGB")
        depth = unpack_array(rec["depth"]).astype(np.float32)
        ldi = unpack_array(rec["ldi"]).astype(np.float32)[:self.num_layers]
        return image, depth, ldi, np.asarray(rec["K"], np.float32), np.asarray(rec["w2c"], np.float64)

    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=1.0, rng=None):
        if seq_index is None:
            raise ValueError("get_data needs seq_index (no random sequence pick)")
        scene, stems = self.sequences[seq_index]
        anchor = stems[0]                                            # stored order: view 0 = anchor / target

        with _open_lmdb(self.lmdb_path).begin() as txn:
            views = [self.read_frame(txn, scene, s) for s in stems]

        images = self.image_aug([v[0] for v in views], train=False)   # [(3, H, W) in [0, 1]], no augmentation
        return {
            "seq_name": f"{scene}/{anchor}",
            "scene_name": scene,
            "frame_num": len(views),
            "images": images,
            "ldis": [torch.from_numpy(np.ascontiguousarray(v[2])) for v in views],
            "ldis_mask": [torch.from_numpy((v[2] > 0).astype(np.float32)) for v in views],
            "depths": [v[1] for v in views],
            "depth_masks": [v[1] > 0 for v in views],
            "intrinsics": [v[3] for v in views],
            "extrinsics": [v[4].astype(np.float32) for v in views],
            "image_names": list(stems),
        }


# ======================================================================
# Check: one point cloud PER VIEW (GT depth in image colours, hidden LDI layers one colour each)
# ======================================================================
PALETTE = np.array([[0.9, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.3, 0.9], [0.9, 0.6, 0.0]])   # layer 1, 2, 3, ...


def view_points(image, depth, ldi, K, w2c, frame="camera"):
    """Points + colours of ONE view: layer 0 = GT depth in image colours, layer k >= 1 in PALETTE[k - 1].
    frame='camera': in the view's own camera frame (x right, y down, z forward); 'world': in the shared world frame."""
    image = image.permute(1, 2, 0).numpy() if torch.is_tensor(image) else np.asarray(image) / 255.0
    ldi = ldi.numpy() if torch.is_tensor(ldi) else np.asarray(ldi)
    depth = np.asarray(depth)
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(np.asarray(K, np.float64)).T
    c2w = np.linalg.inv(np.asarray(w2c, np.float64))
    pts, cols, counts = [], [], []
    for k, d in enumerate([depth] + list(ldi[1:])):
        m = d > 0
        p = rays[m] * d[m][:, None]
        if frame == "world":
            p = p @ c2w[:3, :3].T + c2w[:3, 3]
        pts.append(p)
        cols.append(image[m] if k == 0 else np.tile(PALETTE[(k - 1) % len(PALETTE)], (int(m.sum()), 1)))
        counts.append(int(m.sum()))
    return np.concatenate(pts), np.clip(np.concatenate(cols), 0, 1), counts


def save_view_plys(sample, out_dir, frame="camera"):
    """One .ply + the RGB image per view of a sample, in <out_dir>/<scene>__<anchor>/view<i>_<frame name>.*"""
    import open3d as o3d
    seq_dir = osp.join(out_dir, sample["seq_name"].replace("/", "__"))
    os.makedirs(seq_dir, exist_ok=True)
    for i, (image, depth, ldi, K, w2c, name) in enumerate(zip(sample["images"], sample["depths"], sample["ldis"],
                                                              sample["intrinsics"], sample["extrinsics"],
                                                              sample["image_names"])):
        pts, cols, counts = view_points(image, depth, ldi, K, w2c, frame)
        tag = f"view{i}{'_anchor' if i == 0 else ''}_{name}"
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud(osp.join(seq_dir, f"{tag}.ply"), pcd)
        rgb = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8) if torch.is_tensor(image) else np.asarray(image)
        Image.fromarray(rgb).save(osp.join(seq_dir, f"{tag}_rgb.png"))
        layers = ", ".join(f"L{k}: {n:,}" for k, n in enumerate(counts))
        print(f"  {tag}: {layers} points -> {seq_dir}")
    return seq_dir


# ======================================================================
# Debugging: alignment / layer / cross-view checks of one sample
# ======================================================================
def _pixel_rays(K, H, W):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(np.asarray(K, np.float64)).T   # (H, W, 3), z = 1


def _to_world(rays, d, m, c2w):
    return (rays[m] * d[m][:, None]) @ c2w[:3, :3].T + c2w[:3, 3]


def _pct(x):
    return f"{100 * x:.1f}%"


def debug_sample(sample, shift_px=1, tol=0.05, rel_tol=0.02, verbose=True):
    """
    Checks one sample and returns a dict of numbers (also printed):
      cameras      K sanity (principal point near the image centre, fx > 0), w2c a proper rotation
      alignment    mesh layer 0 (LDI L0) vs the GT depth map: median |err|, share within 5 cm / 2%,
                   and the same with the LDI shifted by +-shift_px pixels -- if a shift is better, the
                   intrinsics / pixel-centre convention is off
      layers       layers ordered near -> far, hidden-layer coverage, gap L1 - L0 (thin gaps = wall / slab
                   thickness), leading-zero holes (layer k set while layer k-1 is empty)
      cross-view   anchor (view 0) GT depth points projected into every other view and compared with that
                   view's GT depth where they land on its visible surface -> median error + inliers (pose /
                   intrinsics check that does not involve the mesh)
      room bounds  hidden points (layers >= 1, all views) outside the box of all visible points (+ 5 cm)
    """
    S = len(sample["images"])
    Ks = [np.asarray(k, np.float64) for k in sample["intrinsics"]]
    w2cs = [np.asarray(w, np.float64) for w in sample["extrinsics"]]
    depths = [np.asarray(d, np.float64) for d in sample["depths"]]
    ldis = [(l.numpy() if torch.is_tensor(l) else np.asarray(l)).astype(np.float64) for l in sample["ldis"]]
    H, W = depths[0].shape
    out = {"views": []}
    log = print if verbose else (lambda *a, **k: None)
    log(f"== debug {sample['seq_name']}  ({S} views, {H}x{W}, {ldis[0].shape[0]} layers)")

    vis_world, hid_world = [], []
    for i in range(S):
        K, w2c, gt, ldi = Ks[i], w2cs[i], depths[i], ldis[i]
        R = w2c[:3, :3]
        v = {"name": sample["image_names"][i]}
        # ---------- cameras ----------
        v["pp_offset_px"] = float(np.hypot(K[0, 2] - (W - 1) / 2, K[1, 2] - (H - 1) / 2))
        v["fx"], v["fy"] = float(K[0, 0]), float(K[1, 1])
        v["R_orth_err"] = float(np.abs(R @ R.T - np.eye(3)).max())
        v["R_det"] = float(np.linalg.det(R))
        # ---------- alignment: LDI L0 vs GT depth ----------
        def err_stats(l0):
            m = (gt > 0) & (l0 > 0)
            if not m.any():
                return float("nan"), float("nan")
            e = np.abs(l0[m] - gt[m])
            return float(np.median(e)), float((e < np.maximum(tol, rel_tol * gt[m])).mean())
        v["l0_med_err_m"], v["l0_within_tol"] = err_stats(ldi[0])
        shifts = {}
        for dy, dx in [(0, shift_px), (0, -shift_px), (shift_px, 0), (-shift_px, 0)]:
            shifts[(dy, dx)] = err_stats(np.roll(np.roll(ldi[0], dy, 0), dx, 1))[0]
        best = min(shifts, key=lambda k: shifts[k] if np.isfinite(shifts[k]) else np.inf)
        v["best_shift"] = best if shifts[best] < 0.8 * v["l0_med_err_m"] else (0, 0)
        v["gt_coverage"], v["l0_coverage"] = float((gt > 0).mean()), float((ldi[0] > 0).mean())
        # ---------- layers ----------
        L = ldi.shape[0]
        order_bad = 0
        holes = 0
        for k in range(1, L):
            both = (ldi[k] > 0) & (ldi[k - 1] > 0)
            order_bad += int((ldi[k][both] <= ldi[k - 1][both]).sum())
            holes += int(((ldi[k] > 0) & (ldi[k - 1] <= 0)).sum())
        v["order_violations"], v["layer_holes"] = order_bad, holes
        v["hidden_coverage"] = [float((ldi[k] > 0).mean()) for k in range(1, L)]
        m1 = (ldi[0] > 0) & (ldi[1] > 0) if L > 1 else np.zeros_like(gt, bool)
        gap = ldi[1][m1] - ldi[0][m1] if m1.any() else np.zeros(0)
        v["gap_L1_med_m"] = float(np.median(gap)) if len(gap) else float("nan")
        v["gap_L1_lt5cm"] = float((gap < 0.05).mean()) if len(gap) else float("nan")
        v["gap_L1_lt20cm"] = float((gap < 0.20).mean()) if len(gap) else float("nan")
        # ---------- world points for the cross-view / bounds checks ----------
        rays = _pixel_rays(K, H, W)
        c2w = np.linalg.inv(w2c)
        vis_world.append(_to_world(rays, gt, gt > 0, c2w))
        hid_world.append(np.concatenate([_to_world(rays, ldi[k], ldi[k] > 0, c2w) for k in range(1, L)])
                         if L > 1 else np.zeros((0, 3)))
        out["views"].append(v)
        log(f"  view{i} {v['name']:>10s} | K pp off {v['pp_offset_px']:.1f}px fx {v['fx']:.1f} | "
            f"R orth {v['R_orth_err']:.1e} det {v['R_det']:+.3f} | "
            f"L0 vs depth: med {100 * v['l0_med_err_m']:.2f}cm, {_pct(v['l0_within_tol'])} in tol, "
            f"best shift {v['best_shift']} | hidden cov {' / '.join(_pct(c) for c in v['hidden_coverage'])} | "
            f"L1-L0 gap med {100 * v['gap_L1_med_m']:.1f}cm (<5cm {_pct(v['gap_L1_lt5cm'])}, "
            f"<20cm {_pct(v['gap_L1_lt20cm'])}) | order viol {order_bad}, holes {holes}")

    # ---------- cross-view: anchor depth points -> other views ----------
    a = vis_world[0]
    if len(a) > 20000:
        a = a[np.random.default_rng(0).choice(len(a), 20000, replace=False)]
    out["cross_view"] = []
    for j in range(1, S):
        cam = a @ w2cs[j][:3, :3].T + w2cs[j][:3, 3]
        z = cam[:, 2]
        ok = z > 1e-3
        uv = cam[ok] @ Ks[j].T
        u, vv = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        ui, vi = np.round(u).astype(int), np.round(vv).astype(int)
        inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        zz, ui, vi = z[ok][inside], ui[inside], vi[inside]
        dj = depths[j][vi, ui]
        valid = dj > 0
        e = np.abs(zz[valid] - dj[valid])
        vis = e < np.maximum(tol, rel_tol * dj[valid])                # lands on view j's visible surface
        rec = {"view": j, "in_frame": float(inside.mean()) if len(inside) else 0.0,
               "visible_share": float(vis.mean()) if len(vis) else float("nan"),
               "med_err_visible_cm": float(100 * np.median(e[vis])) if vis.any() else float("nan")}
        out["cross_view"].append(rec)
        log(f"  cross-view 0 -> {j}: {_pct(rec['in_frame'])} of anchor points in frame, "
            f"{_pct(rec['visible_share'])} land on view {j}'s surface, median err there {rec['med_err_visible_cm']:.2f} cm")

    # ---------- room bounds ----------
    V = np.concatenate(vis_world)
    lo, hi = np.percentile(V, 0.5, axis=0) - 0.05, np.percentile(V, 99.5, axis=0) + 0.05
    Hd = np.concatenate(hid_world) if hid_world else np.zeros((0, 3))
    outside = ~np.all((Hd >= lo) & (Hd <= hi), axis=1) if len(Hd) else np.zeros(0, bool)
    out["bounds_lo"], out["bounds_hi"] = lo, hi
    out["hidden_outside_room"] = float(outside.mean()) if len(outside) else 0.0
    log(f"  room box lo {np.round(lo, 2)} hi {np.round(hi, 2)} | hidden points outside it: "
        f"{int(outside.sum()):,} of {len(Hd):,} ({_pct(out['hidden_outside_room'])})")

    # ---------- verdict ----------
    flags = []
    if any(v["pp_offset_px"] > 2 for v in out["views"]):
        flags.append("principal point far from the image centre (crop?)")
    if any(v["R_orth_err"] > 1e-3 or v["R_det"] < 0.99 for v in out["views"]):
        flags.append("extrinsics are not a proper rotation")
    if any(v["l0_within_tol"] < 0.9 for v in out["views"] if np.isfinite(v["l0_within_tol"])):
        flags.append("mesh layer 0 disagrees with the depth map (pose / mesh frame / K)")
    if any(v["best_shift"] != (0, 0) for v in out["views"]):
        flags.append("a pixel-shifted LDI matches the depth better (pixel-centre / intrinsics offset)")
    if any(v["order_violations"] for v in out["views"]):
        flags.append("layers not ordered near -> far")
    if any(r["med_err_visible_cm"] > 3 for r in out["cross_view"] if np.isfinite(r["med_err_visible_cm"])):
        flags.append("cross-view reprojection error > 3 cm (poses / intrinsics)")
    if out["hidden_outside_room"] > 0.01:
        flags.append("hidden points outside the room (modelling junk beyond walls / ceiling)")
    if any(np.isfinite(v["gap_L1_lt5cm"]) and v["gap_L1_lt5cm"] > 0.3 for v in out["views"]):
        flags.append("many hidden layers within 5 cm of the visible one (wall / slab thickness)")
    out["flags"] = flags
    log("  VERDICT: " + ("OK" if not flags else " | ".join(flags)))
    return out


def save_error_maps(sample, out_dir):
    """|LDI L0 - GT depth| per view as a PNG (black = 0, white >= 10 cm; red = only one of the two has depth)."""
    seq_dir = osp.join(out_dir, sample["seq_name"].replace("/", "__"))
    os.makedirs(seq_dir, exist_ok=True)
    for i, (gt, ldi, name) in enumerate(zip(sample["depths"], sample["ldis"], sample["image_names"])):
        gt = np.asarray(gt, np.float64)
        l0 = (ldi.numpy() if torch.is_tensor(ldi) else np.asarray(ldi))[0].astype(np.float64)
        err = np.clip(np.abs(l0 - gt) / 0.10, 0, 1)
        rgb = np.repeat((err * 255).astype(np.uint8)[..., None], 3, -1)
        one = (gt > 0) ^ (l0 > 0)
        rgb[one] = (255, 0, 0)
        rgb[(gt <= 0) & (l0 <= 0)] = (0, 0, 80)
        Image.fromarray(rgb).save(osp.join(seq_dir, f"view{i}_{name}_l0_err.png"))


def get_args():
    p = argparse.ArgumentParser(description="NRGBD LDI LMDB dataset check (test only): one point cloud per view")
    p.add_argument("--config_path", required=True)
    p.add_argument("--lmdb_path", default=None)
    p.add_argument("--num_test_keys", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--frame", choices=["camera", "world"], default="camera",
                   help="camera: each view in its own camera frame; world: shared frame (the views overlay in MeshLab)")
    p.add_argument("--out_dir", default="./ldi_lmdb_check_nrgbd")
    p.add_argument("--debug", action="store_true", help="print alignment / layer / cross-view / bounds checks")
    p.add_argument("--no_ply", action="store_true", help="skip the per-view .ply export")
    p.add_argument("--start", type=int, default=0, help="first sample index")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    ds = NRGBD_LDI("test", args.config_path, lmdb_path=args.lmdb_path, num_test_keys=args.num_test_keys)
    os.makedirs(args.out_dir, exist_ok=True)
    all_flags = []
    for i in range(args.start, min(args.start + args.num_samples, len(ds))):
        s = ds.get_data(seq_index=i)
        print(f"{s['seq_name']}: {len(s['images'])} views {s['image_names']}")
        if not args.no_ply:
            save_view_plys(s, args.out_dir, args.frame)
        if args.debug:
            r = debug_sample(s)
            save_error_maps(s, args.out_dir)
            all_flags += [(s["seq_name"], f) for f in r["flags"]]
    if args.debug:
        print(f"\n{len(all_flags)} flag(s) over {args.num_samples} samples")
        for name, f in all_flags:
            print(f"  {name}: {f}")