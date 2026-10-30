"""
Pack one split of an LDI dataset (ScanNet_LDI / Matterport3D_LDI / ARKitScenes_LDI) into ONE LMDB file.

The dataset class does the work it already does (split, 80% good-frame filter, 224 x 224 crop), and every
good frame is stored ready to use, so the LMDB loader only decodes (no crop / resize, no listdir):

    key "frame:<scene>/<frame>"  -> pickle of
        rgb    JPEG bytes of the 224 x 224 view (quality 95)
        depth  zlib(float16 (224, 224)) metres, 0 = invalid
        ldi    zlib(float16 (L, 224, 224)) z-depth per layer, near -> far, 0 = no surface
        K      (3, 3) float32   intrinsics of the 224 x 224 view
        w2c    (4, 4) float32   world-to-camera (absolute; the loader makes them relative to view 0)
    key "__meta__"               -> pickle of
        dataset, split, img_size, num_layers, min_layer1_agreement, num_views, window
        scene_frames {scene: [frame, ...]}   sorted good frames per scene (the pool the views are drawn from)
        anchors      [(scene, frame), ...]   the samples (all good frames for train, the fixed subset for val / test)

Usage (one job per dataset / split; the file is written next to the dump unless --out is given):
    python build_ldi_lmdb.py --dataset "ScanNet_LDI(split='train', config_path='configs/scannet_ldi.yaml')"
        -> <dump_root>/LDIs/scannet_train_ldi.lmdb
Re-running skips frames already in the LMDB (an interrupted build resumes); --overwrite starts again.
"""
import argparse
import io
import os
import os.path as osp
import pickle as pkl
import zlib

import lmdb
import numpy as np
from tqdm import tqdm

from datasets.scannet_LDI import ScanNet_LDI                      # noqa: F401  (used through eval)
from datasets.matterport3D_LDI import Matterport3D_LDI             # noqa: F401
from datasets.arkitscenes_LDI import ARKitScenes_LDI               # noqa: F401


def pack_array(x, dtype=np.float16):
    x = np.ascontiguousarray(np.asarray(x, dtype))
    return {"shape": x.shape, "dtype": str(x.dtype), "data": zlib.compress(x.tobytes(), level=6)}


def unpack_array(d):
    return np.frombuffer(zlib.decompress(d["data"]), dtype=d["dtype"]).reshape(d["shape"])


def encode_jpeg(image, quality=95):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def frame_record(view):
    """load_view() output -> the stored record."""
    return pkl.dumps({
        "rgb": encode_jpeg(view["image"]),
        "depth": pack_array(np.where(view["depth_mask"], view["depth"], 0.0)),
        "ldi": pack_array(view["ldi"]),
        "K": np.asarray(view["K"], np.float32),
        "w2c": np.asarray(view["w2c"], np.float32),
    }, protocol=pkl.HIGHEST_PROTOCOL)


def build(ds, out_path, map_size_gb=200, commit_every=500, overwrite=False):
    if overwrite and osp.exists(out_path):
        os.remove(out_path)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    env = lmdb.open(out_path, subdir=False, map_size=int(map_size_gb * 1024 ** 3), readahead=False, meminit=False)

    frames = [(scene, stem) for scene, stems in ds.scene_pool.items() for stem in stems]
    with env.begin() as txn:
        todo = [f for f in frames if txn.get(f"frame:{f[0]}/{f[1]}".encode()) is None]
    print(f"{type(ds).__name__} [{ds.split}]: {len(frames):,} good frames, {len(todo):,} to write -> {out_path}")

    txn = env.begin(write=True)
    for i, (scene, stem) in enumerate(tqdm(todo, desc="writing")):
        txn.put(f"frame:{scene}/{stem}".encode(), frame_record(ds.load_view(scene, stem)))
        if (i + 1) % commit_every == 0:
            txn.commit()
            txn = env.begin(write=True)

    meta = {
        "dataset": ds.dataset, "split": ds.split, "img_size": ds.img_size, "num_layers": ds.num_layers,
        "num_views": ds.num_views, "window": ds.window,
        "scene_frames": {s: list(v) for s, v in ds.scene_pool.items()},
        "anchors": list(ds.frames),
    }
    txn.put(b"__meta__", pkl.dumps(meta, protocol=pkl.HIGHEST_PROTOCOL))   # written last: marks the LMDB complete
    txn.commit()
    env.sync()
    size_gb = osp.getsize(out_path) / 1024 ** 3
    print(f"done: {len(frames):,} frames, {len(meta['anchors']):,} samples, {len(meta['scene_frames'])} scenes, "
          f"{size_gb:.1f} GB ({1024 ** 2 * size_gb / max(1, len(frames)):.0f} KB / frame)")
    env.close()


def read_frame(txn, scene, stem):
    """The inverse of frame_record: (image uint8 (H, W, 3), depth (H, W), ldi (L, H, W), K, w2c)."""
    from PIL import Image
    rec = pkl.loads(txn.get(f"frame:{scene}/{stem}".encode()))
    image = np.array(Image.open(io.BytesIO(rec["rgb"])).convert("RGB"))
    return image, unpack_array(rec["depth"]).astype(np.float32), unpack_array(rec["ldi"]).astype(np.float32), \
        rec["K"], rec["w2c"]


def get_args():
    p = argparse.ArgumentParser(description="Pack an LDI dataset split into one LMDB")
    p.add_argument("--dataset", required=True,
                   help="dataset constructor, e.g. \"ScanNet_LDI(split='train', config_path='configs/scannet_ldi.yaml')\"")
    p.add_argument("--out", default=None, help="LMDB file (default: <dump_root>/LDIs/<dataset>_<split>_ldi.lmdb)")
    p.add_argument("--map_size_gb", type=float, default=200, help="LMDB map size (an upper bound, not allocated)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    ds = eval(args.dataset)
    out = args.out or osp.join(ds.ldi_root, f"{ds.dataset}_{ds.split}_ldi.lmdb")
    build(ds, out, map_size_gb=args.map_size_gb, overwrite=args.overwrite)