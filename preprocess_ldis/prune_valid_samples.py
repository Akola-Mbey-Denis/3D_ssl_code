"""
Keep only the sampled targets whose mesh LDI (.npz) exists, so the dataloader never hits a missing file.

Rewrites only the sample list written by arkitscenes_mesh_ldi.py, for each split:
    {root_dir}/arkitscenes_ldi_training_{split}_samples.json   {"keys": [...], ...}
The original is kept as *.json.orig (only the first time).

A key "<fold>/<video_id>__<image stem>" has the LDI "{video_id}__{image stem}_mesh_ldi.npz"
(arkitscenes_mesh_ldi.target_name; the key's second part is the target image stem).

Usage:
    python prune_ldi_sample_lists.py --splits train val            # --dry_run to only report
"""
import argparse
import json
import os
import os.path as osp
import shutil


def ldi_file_name(key):
    """"Training/41069025__41069025_28533.651" -> "41069025__41069025_28533.651_mesh_ldi.npz"."""
    scene_id, image_stem = key.split("__", 1)
    return f"{scene_id.split('/')[-1]}__{image_stem}_mesh_ldi.npz"


def prune_split(split, root_dir, save_dir, list_name, dry_run=False):
    json_path = osp.join(root_dir, list_name.format(split=split))
    ldi_dir = osp.join(save_dir, split, "supervision_data", "mesh_ldi")

    with open(json_path) as f:
        meta = json.load(f)
    existing = set(os.listdir(ldi_dir)) if osp.isdir(ldi_dir) else set()

    keep = [k for k in meta["keys"] if ldi_file_name(k) in existing]
    missing = [k for k in meta["keys"] if ldi_file_name(k) not in existing]
    print(f"[{split}] {len(keep)} of {len(meta['keys'])} keys have an LDI, {len(missing)} missing")
    for key in missing:
        print(f"    missing: {key}  ({ldi_file_name(key)})")
    if dry_run or not missing:
        return

    if not osp.exists(json_path + ".orig"):                          # keep the original once
        shutil.copy2(json_path, json_path + ".orig")
    meta = dict(meta, keys=keep, num_samples=len(keep), num_scenes=len({k.split("__")[0] for k in keep}))
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, json_path)                                       # atomic
    print(f"[{split}] rewrote {json_path} (original: {json_path}.orig)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--root_dir", default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/")
    p.add_argument("--save_dir", default="/scratch/dma9300/ldi_supervision_data/scannet_3d_256_max_depth_16_6_depth_layers")
    p.add_argument("--list_name", default="scannet_ldi_training_{split}_samples.json",
                   help="sample list in --root_dir, as written by arkitscenes_mesh_ldi.py")
    p.add_argument("--dry_run", action="store_true", help="only report, do not rewrite")
    args = p.parse_args()
    for split in args.splits:
        prune_split(split, args.root_dir, args.save_dir, args.list_name, dry_run=args.dry_run)