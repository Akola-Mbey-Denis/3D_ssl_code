#!/usr/bin/env python3
"""
Which ScanNet++ scenes have everything preprocess_scannetpp.py needs? Reads only (checks files / folders, opens nothing
big), prints a table and writes the usable scene lists to pass as --scene_list.

Needed per source:
    always : scans/mesh_aligned_0.05.ply
    iphone : iphone/colmap/{cameras,images}.txt, iphone/rgb/ with frames (extracted from the video)
    dslr   : dslr/colmap/{cameras,images}.txt,  dslr/resized_images/ with images
    optional (missing = no anonymisation masking): iphone/rgb_masks/, dslr/resized_anon_masks/

Usage:
    python check_scannetpp.py --scannetpp_dir /projects/rps/dff6142/fouheylab/jt4100/datasets/scannetpp --out_dir ./scannetpp_lists
    -> ./scannetpp_lists/scenes_iphone.txt, scenes_dslr.txt (complete scenes), check_scannetpp.csv (everything)
"""
import argparse
import csv
import os
import os.path as osp

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def count_images(folder):
    if not osp.isdir(folder):
        return 0
    with os.scandir(folder) as it:
        return sum(1 for e in it if e.is_file() and e.name.lower().endswith(IMAGE_EXTS))


def colmap_ok(folder):
    return all(osp.isfile(osp.join(folder, f)) and osp.getsize(osp.join(folder, f)) > 0
               for f in ("cameras.txt", "images.txt"))


def colmap_num_images(folder):
    """Registered images in images.txt (2 lines per image after the '#' header)."""
    if not colmap_ok(folder):
        return 0
    with open(osp.join(folder, "images.txt")) as f:
        lines = [line for line in f if not line.startswith("#")]
    return len(lines) // 2


def check_scene(scene_dir, mesh_name):
    r = {"mesh": osp.isfile(osp.join(scene_dir, "scans", mesh_name))}
    for src, rgb, masks in (("iphone", "iphone/rgb", "iphone/rgb_masks"),
                            ("dslr", "dslr/resized_images", "dslr/resized_anon_masks")):
        r[f"{src}_colmap"] = colmap_num_images(osp.join(scene_dir, src, "colmap"))
        r[f"{src}_images"] = count_images(osp.join(scene_dir, rgb))
        r[f"{src}_masks"] = count_images(osp.join(scene_dir, masks))
        r[f"{src}_video"] = src == "iphone" and osp.isfile(osp.join(scene_dir, "iphone", "rgb.mkv"))
        r[f"{src}_ok"] = r["mesh"] and r[f"{src}_colmap"] > 0 and r[f"{src}_images"] > 0
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scannetpp_dir", required=True, help="ScanNet++ root (with data/) or the folder of scenes itself")
    p.add_argument("--mesh_name", default="mesh_aligned_0.05.ply")
    p.add_argument("--min_images", type=int, default=6, help="a source needs at least this many frames to be usable")
    p.add_argument("--out_dir", default="./scannetpp_lists")
    args = p.parse_args()

    data = osp.join(args.scannetpp_dir, "data")
    if not osp.isdir(data):                                       # scenes directly in scannetpp_dir
        data = args.scannetpp_dir
    scenes = sorted(d for d in os.listdir(data) if osp.isdir(osp.join(data, d)) and not d.startswith("."))
    print(f"{len(scenes)} folders in {data}")
    if data == args.scannetpp_dir:
        print("NOTE: no data/ sub-folder -- give preprocess_scannetpp.py a --scannetpp_dir whose data/ holds the scenes "
              "(or adapt its data path)")

    rows = []
    for i, s in enumerate(scenes):
        r = check_scene(osp.join(data, s), args.mesh_name)
        for src in ("iphone", "dslr"):
            r[f"{src}_ok"] = r[f"{src}_ok"] and min(r[f"{src}_colmap"], r[f"{src}_images"]) >= args.min_images
        rows.append({"scene": s, **r})
        if (i + 1) % 50 == 0:
            print(f"  checked {i + 1}/{len(scenes)}")

    n = len(rows)
    def cnt(key):
        return sum(1 for r in rows if r[key])
    print(f"\n{'':40s}{'scenes':>8s}")
    print(f"{'mesh (' + args.mesh_name + ')':40s}{cnt('mesh'):>8d}")
    for src in ("iphone", "dslr"):
        print(f"{src + ' COLMAP':40s}{cnt(src + '_colmap'):>8d}")
        print(f"{src + ' images on disk':40s}{cnt(src + '_images'):>8d}")
        print(f"{src + ' masks':40s}{cnt(src + '_masks'):>8d}")
    print(f"{'iphone video only (rgb.mkv, no frames)':40s}"
          f"{sum(1 for r in rows if r['iphone_video'] and not r['iphone_images']):>8d}")
    print(f"{'USABLE --source iphone':40s}{cnt('iphone_ok'):>8d}")
    print(f"{'USABLE --source dslr':40s}{cnt('dslr_ok'):>8d}")
    print(f"{'usable with either':40s}{sum(1 for r in rows if r['iphone_ok'] or r['dslr_ok']):>8d}   of {n}")

    for src in ("iphone", "dslr"):
        ok = [r for r in rows if r[f"{src}_ok"]]
        if ok:
            frames = sorted(min(r[f"{src}_colmap"], r[f"{src}_images"]) for r in ok)
            print(f"{src}: frames per usable scene  min {frames[0]}  median {frames[len(frames) // 2]}  max {frames[-1]}"
                  f"  ({sum(f >= 1000 for f in frames)} scenes with >= 1000)")

    os.makedirs(args.out_dir, exist_ok=True)
    for src in ("iphone", "dslr"):
        with open(osp.join(args.out_dir, f"scenes_{src}.txt"), "w") as f:
            f.writelines(f"{r['scene']}\n" for r in rows if r[f"{src}_ok"])
    with open(osp.join(args.out_dir, "check_scannetpp.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out_dir}/scenes_iphone.txt, scenes_dslr.txt (usable scenes) and check_scannetpp.csv")


if __name__ == "__main__":
    main()