"""
ARKitScenes multi-view LDI dataset read from the LMDB written by preprocess_ldi_lmdb.py (--dataset arkitscenes).

Every frame in the LMDB is already clean (LDI layer 1 agrees with the depth map on >= 80% of the pixels) and
cropped to 224 x 224, so a sample only decodes: no listdir, no crop / resize, one LMDB file per split.

LMDB per split (in this order):
    lmdb_path=...                      constructor argument
    LMDB_PATHS.<split> in the config   e.g. LMDB_PATHS: {train: .../arkitscenes_train_ldi_seq6.lmdb, val: ..., test: ...}
    <DUMP_ROOT>/LDIs/arkitscenes_<split>_ldi_seq6.lmdb   (the preprocessing default)
    $LMDB_DIR set (the job copied the LMDBs to node-local disk): the same file name is read from $LMDB_DIR

Sample (lists of num_views = 6 views, like the other loaders):
    images      [S x (3, H, W)]   in [0, 1] (photometric augmentation when training)
    ldis        [S x (L, H, W)]   z-depth of layer k in metres (near -> far), 0 = no surface
    ldis_mask   [S x (L, H, W)]   1 where layer k exists
    depths      [S x (H, W)]      GT depth maps (metres), depth_masks [S x (H, W)]
    intrinsics  [S x (3, 3)]      K of the 224 x 224 views
    extrinsics  [S x (4, 4)]      world-to-camera relative to view 0 (view 0 = identity)
    image_names [S x str], seq_name "<scan>/<anchor frame>", scene_name "<scan>"
(batch["images"] is (B, S, 3, H, W) and batch["ldis"] (B, S, L, H, W) for finetune_ldi.py)

Sequences: the stored ones (one per anchor frame; val / test = the fixed subsets). Training redraws the views other
than the anchor every time from the anchor's scan (+-window frames), so it sees new combinations
(resample=False keeps the stored ones); val / test always use the stored sequences.

Check samples (writes a point cloud of the views of each sample, in view 0's frame):
    python -m datasets.arkitscenes_ldi_lmdb --split train --config_path configs/arkitscenes_ldi_lmdb.yaml
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
# Shared by the three LDI LMDB loaders (an LMDB file can be open only once per process, whichever class opens it).
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


class ARKitScenes_LDI(BaseDataset):
    DATASET = "arkitscenes"

    def __init__(self, split, config_path, lmdb_path=None, num_layers=None, overfit=False,
                keys_dir=None, *args, **kwargs):
        assert split in ("train", "val", "test"), split
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        opts = CfgNode(raw_cfg)
        self.opts = opts
        self.common_conf = opts.COMMON
        super().__init__(self.common_conf, *args, **kwargs)

        # ---------------- LMDB of this split ----------------
        if lmdb_path is None:
            paths = raw_cfg.get("LMDB_PATHS") or {}
            lmdb_path = paths.get(split) or osp.join(str(raw_cfg.get("DUMP_ROOT", "")), "LDIs",
                                                     f"{self.DATASET}_{split}_ldi_seq6.lmdb")
        lmdb_path = os.path.expandvars(lmdb_path)
        keys_dir = keys_dir or osp.dirname(osp.abspath(lmdb_path))     # key files: next to the original LMDB
        if os.environ.get("LMDB_DIR"):                               # copied to node-local disk by the job
            lmdb_path = osp.join(os.environ["LMDB_DIR"], osp.basename(lmdb_path))
        if not osp.isfile(lmdb_path):
            raise FileNotFoundError(f"{self.DATASET} LDI LMDB for split '{split}' not found: {lmdb_path}\n"
                                    f"  config: {osp.abspath(config_path)}  LMDB_PATHS: {raw_cfg.get('LMDB_PATHS')}")
        self.lmdb_path = osp.realpath(lmdb_path)

        with _open_lmdb(lmdb_path).begin() as txn:
            meta = txn.get(b"__meta__")
        if meta is None:
            raise RuntimeError(f"{lmdb_path} has no __meta__: preprocessing did not finish")
        meta = pkl.loads(meta)
        if meta["dataset"] != self.DATASET or meta["split"] != split:
            raise ValueError(f"{lmdb_path} holds {meta['dataset']} [{meta['split']}], not {self.DATASET} [{split}]")

        self.meta = meta
        self.split = split
        self.overfit = overfit
        self.num_views = meta["num_views"]
        self.window = meta["window"]
        self.num_layers = min(num_layers or meta["num_layers"], meta["num_layers"])
        self.training = split == "train" and not overfit
        self.resample = (split =="train")                            # default: all 6 stored views (no redraw)
        self.image_aug = PhotometricAug()

        self.scene_frames = meta["scene_frames"]                     # {scan: sorted clean frames}
        self.sequences = meta["sequences"]                           # [(scan, [frame x num_views])]
        # Fixed subsets, dumped once next to the LMDB and loaded afterwards:
        #   overfit (any split)  -> <lmdb stem>_overfit_<n_overfit>_keys.pkl
        #   val / test           -> <lmdb stem>_<split>_<n_eval>_keys.pkl
        stem = osp.splitext(osp.basename(lmdb_path))[0]
        if overfit:
            self.keys_path = osp.join(keys_dir, f"{stem}_overfit_100_overfit_keys.pkl")
            self.sequences = load_or_dump_keys(self.keys_path, self.sequences, 100, seed=0)
            self.resample = False                                    # always the same views
        elif split != "train":
            self.keys_path = osp.join(keys_dir, f"{stem}_{split}_2000_keys.pkl")
            self.sequences = load_or_dump_keys(self.keys_path, self.sequences, 2000, seed=1)
        else:
            self.keys_path = None
        self.keys = [f"{scene}/{frames[0]}" for scene, frames in self.sequences]

        print(f"ARKitScenes_LDI [{split}{' overfit' if overfit else ''}] {len(self.sequences):,} sequences of "
              f"{self.num_views} views from {len(self.scene_frames)} scans ({lmdb_path})"
              + (f" -- fixed keys {self.keys_path}" if self.keys_path else "")
              + (" -- views redrawn every time" if self.resample else ""))

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

    def redraw(self, scene, anchor):
        """Anchor + num_views - 1 random frames of its scan within +-window (whole scan if too few)."""
        pool = self.scene_frames[scene]
        i = pool.index(anchor)
        near = [j for j in range(max(0, i - self.window), min(len(pool), i + self.window + 1)) if j != i]
        if len(near) < self.num_views - 1:
            near = [j for j in range(len(pool)) if j != i]
        others = random.sample(near, self.num_views - 1)
        return [anchor] + [pool[j] for j in sorted(others)]

    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=1.0, rng=None):
        if seq_index is None:
            seq_index = random.randrange(len(self.sequences))
        scene, stems = self.sequences[seq_index]
        anchor = stems[0]
        if self.resample:
            stems = self.redraw(scene, anchor)
        if self.training:                                            # all 6 stored views, random order
            stems = random.sample(list(stems), len(stems))

        with _open_lmdb(self.lmdb_path).begin() as txn:
            views = [self.read_frame(txn, scene, s) for s in stems]

        ref_inv = np.linalg.inv(views[0][4])                         # poses relative to view 0
        images = self.image_aug([v[0] for v in views], train=self.training)   # [(3, H, W) in [0, 1]]
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
            "extrinsics": [(v[4] @ ref_inv).astype(np.float32) for v in views],
            "image_names": list(stems),
        }


# ======================================================================
# Check: point cloud of every view (GT depth in image colours, hidden LDI layers one colour each)
# ======================================================================
def save_views_ply(sample, ply_path):
    import open3d as o3d
    palette = np.array([[0.9, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.3, 0.9], [0.9, 0.6, 0.0]])
    pts, cols = [], []
    for image, depth, ldi, K, w2c in zip(sample["images"], sample["depths"], sample["ldis"],
                                         sample["intrinsics"], sample["extrinsics"]):
        image = image.permute(1, 2, 0).numpy() if torch.is_tensor(image) else np.asarray(image) / 255.0
        ldi = ldi.numpy()
        H, W = depth.shape
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(np.asarray(K, np.float64)).T
        c2w = np.linalg.inv(np.asarray(w2c, np.float64))
        for k, d in enumerate([depth] + list(ldi[1:])):
            m = d > 0
            pts.append((rays[m] * d[m][:, None]) @ c2w[:3, :3].T + c2w[:3, 3])
            cols.append(image[m] if k == 0 else np.tile(palette[(k - 1) % 4], (int(m.sum()), 1)))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(cols), 0, 1))
    o3d.io.write_point_cloud(ply_path, pcd)


def get_args():
    p = argparse.ArgumentParser(description="ARKitScenes LDI LMDB dataset check")
    p.add_argument("--split", default="train")
    p.add_argument("--config_path", required=True)
    p.add_argument("--lmdb_path", default=None)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--out_dir", default="./ldi_lmdb_check_arkitscenes")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    ds = ARKitScenes_LDI(args.split, args.config_path, lmdb_path=args.lmdb_path)
    ds.training = False                                              # raw colours
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(min(args.num_samples, len(ds))):
        s = ds.get_data(seq_index=i)
        path = osp.join(args.out_dir, s["seq_name"].replace("/", "__") + "_views.ply")
        save_views_ply(s, path)
        print(f"{s['seq_name']}: {len(s['images'])} views {s['image_names']}, ldis {tuple(s['ldis'][0].shape)} -> {path}")
        breakpoint()