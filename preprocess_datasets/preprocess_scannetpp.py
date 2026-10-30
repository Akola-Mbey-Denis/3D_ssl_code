#!/usr/bin/env python3
"""
ScanNet++ -> low-res dump in the SAME layout as our ScanNet / Matterport dumps, so dump_ldis.py (LDIs) and
preprocess_ldi_lmdb.py (LMDB) run on it unchanged. Adapted from DUSt3R's preprocess_scannetpp.py, without the
precomputed pairs: up to --num_images frames per scene, uniformly over the capture (frame order).

Input (ScanNet++ release):
    <scannetpp_dir>/data/<scene>/dslr/colmap/{cameras,images}.txt, dslr/resized_images/*.JPG,
                                 dslr/resized_anon_masks/*.png
                                 iphone/colmap/{cameras,images}.txt, iphone/rgb/frame_*.jpg + iphone/rgb_masks/*.png,
                                 OR only iphone/rgb.mkv (+ iphone/rgb_mask.mkv): then the selected frames are decoded
                                 straight from the videos (frame_<i> = i-th video frame); nothing is extracted to disk
                                 scans/mesh_aligned_0.05.ply     laser-scan mesh, same frame as the COLMAP poses
Output, one folder per scene (written in the scene's own frame order):
    <output_dir>/<scene>/rgb/<frame>.jpg          undistorted, resized to --width (4:3 iPhone -> 640 x 480)
                         depth/<frame>.npy        depth rendered from the mesh (metres, float32, 0 = none / masked)
                         cameras.pkl              {"<frame>.jpg": {"K" (3,3) OpenCV at the saved size,
                                                                   "RT_c2w" (4,4) OpenCV cam2world, "depth": "<frame>.npy"}}
                         mesh.ply                 copy of scans/mesh_aligned_0.05.ply (for dump_ldis.py)
    cameras.pkl is written LAST: its presence = scene done (re-runs skip finished scenes).

Depth is rendered by ray casting the mesh (first hit, trimesh + embree), the same ray caster as dump_ldis.py, so no
OpenGL / EGL is needed and LDI layer 1 matches the depth maps exactly. Pixels covered by the anonymisation masks get
depth 0, as in DUSt3R.

Which frames:
    --source iphone (default): the iPhone video frames, thousands per scene -> 1000 uniform over time; consecutive
                               frames overlap, which suits the +-window sequences of preprocess_ldi_lmdb.py
    --source dslr            : the DSLR images (~a few hundred per scene, more accurately registered to the laser
                               scan); all of them if fewer than --num_images
Scenes: --scene_list (one per line, e.g. ScanNet++'s splits/nvs_sem_train.txt) or the sorted scenes in <dir>/data
(or <dir> itself if it has no data/). Scenes missing the mesh, the COLMAP files or the frames of --source are SKIPPED
(counted, reasons printed, listed in <output_dir>/skipped_scenes_<source>.txt); then the first --num_scenes COMPLETE
scenes are used, and --start_idx / --end_idx split that list across jobs (same list in every job).

Usage:
    python preprocess_scannetpp.py --scannetpp_dir /path/to/scannetpp --output_dir /path/to/low_res_scannetpp \
        --scene_list /path/to/scannetpp/splits/nvs_sem_train.txt --num_scenes 200 --num_images 1000
    python dump_ldis.py --dump_root /path/to/low_res_scannetpp --mesh_name mesh.ply --front_only
"""
import argparse
import os
import os.path as osp
import pickle as pkl
import re
import shutil

import cv2
import numpy as np
import PIL.Image as Image
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
except ImportError:
    from trimesh.ray.ray_triangle import RayMeshIntersector
    print("[WARN] pyembree not available: using trimesh's slow ray tracer")


# ======================================================================
# COLMAP (cameras.txt / images.txt only: the 3D points are not needed)
# ======================================================================
def pose_from_qwxyz_txyz(elems):
    """COLMAP world-to-camera (qw qx qy qz tx ty tz) -> OpenCV cam2world (4, 4)."""
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    w2c[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(w2c)


def load_colmap(sfm_dir):
    """-> {image name (without 'iphone/' / 'video/' prefix): {"camera": [model, W, H, params...], "c2w": (4, 4)}}"""
    cameras = {}
    with open(osp.join(sfm_dir, "cameras.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            cameras[int(c[0])] = [c[1]] + [float(x) for x in c[2:]]

    images = {}
    with open(osp.join(sfm_dir, "images.txt")) as f:
        lines = [line for line in f.read().splitlines() if not line.startswith("#")]
    for line in lines[0::2]:                          # every other line: the 2D points (not needed)
        e = line.split()
        if len(e) < 10:
            continue
        name = e[-1]
        for prefix in ("iphone/", "video/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        images[name] = {"camera": cameras[int(e[-2])], "c2w": pose_from_qwxyz_txyz(e[1:8])}
    return images


def uniform_subset(names, n):
    """n names spread uniformly over the (sorted) capture order; all of them if there are fewer."""
    names = sorted(names)
    if len(names) <= n:
        return names
    return [names[i] for i in np.round(np.linspace(0, len(names) - 1, n)).astype(int)]


# ======================================================================
# Undistort + resize
# ======================================================================
def undistort(camera, rgb, mask):
    """DUSt3R's undistort_images, in OpenCV intrinsics: -> K (3, 3) OpenCV, undistorted rgb, mask (255 = keep)."""
    model, width, height = camera[0], int(camera[1]), int(camera[2])
    fx, fy, cx, cy = camera[3:7]
    dist = np.array(camera[7:], dtype=np.float64)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K[:2, 2] -= 0.5                                   # COLMAP -> OpenCV pixel convention

    if model == "OPENCV_FISHEYE":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, dist, (width, height), np.eye(3), balance=0.0)
        new_K[0, 2], new_K[1, 2] = width / 2.0, height / 2.0            # principal point at the image centre
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    elif len(dist) == 0 or not np.any(dist):         # PINHOLE / no distortion
        return K, rgb, mask
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return new_K, rgb, mask


def resize(rgb, mask, K, width):
    """Resize to `width` (height keeps the aspect ratio); K follows (OpenCV pixel convention)."""
    H, W = rgb.shape[:2]
    s = width / W
    out_hw = (int(round(H * s)), width)
    rgb = np.array(Image.fromarray(rgb).resize((out_hw[1], out_hw[0]), Image.LANCZOS))
    mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    K = K.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
    return rgb, mask, K


# ======================================================================
# Depth from the mesh (first hit along each pixel ray, z-depth)
# ======================================================================
def render_depth(intersector, K, c2w, H, W, max_depth=20.0):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays_cam = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3) @ np.linalg.inv(K).T     # z = 1
    dirs = rays_cam @ c2w[:3, :3].T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape).copy()
    locations, index_ray, _ = intersector.intersects_location(origins, dirs / norms, multiple_hits=False)
    depth = np.zeros(H * W, np.float32)
    if len(locations):
        dist = np.linalg.norm(locations - origins[index_ray], axis=1)
        depth[index_ray] = dist / norms[index_ray, 0]                 # ray distance -> z-depth (rays_cam has z = 1)
    depth[(depth > max_depth) | ~np.isfinite(depth)] = 0.0
    return depth.reshape(H, W)


# ======================================================================
# Scenes
# ======================================================================
SOURCES = {
    #          colmap dir          rgb dir                  mask dir                   mask name from image name
    "iphone": ("iphone/colmap", "iphone/rgb", "iphone/rgb_masks", lambda n: osp.splitext(n)[0] + ".png"),
    "dslr": ("dslr/colmap", "dslr/resized_images", "dslr/resized_anon_masks", lambda n: osp.splitext(n)[0] + ".png"),
}


def data_root(scannetpp_dir):
    """Folder holding the scene folders: <dir>/data (ScanNet++ release layout) or <dir> itself."""
    data = osp.join(scannetpp_dir, "data")
    return data if osp.isdir(data) else scannetpp_dir


def _has_images(folder):
    if not osp.isdir(folder):
        return False
    with os.scandir(folder) as it:
        return any(e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png")) for e in it)


def missing_data(scene_dir, source, mesh_name):
    """What this scene lacks for `source` ([] = complete): the mesh, COLMAP cameras.txt / images.txt and the frames
    (for iPhone: frames extracted from the video into iphone/rgb/). Masks are optional (no masking without them)."""
    colmap_dir, rgb_dir, _, _ = SOURCES[source]
    missing = []
    if not osp.isfile(osp.join(scene_dir, "scans", mesh_name)):
        missing.append(f"scans/{mesh_name}")
    for f in ("cameras.txt", "images.txt"):
        path = osp.join(scene_dir, colmap_dir, f)
        if not (osp.isfile(path) and osp.getsize(path) > 0):
            missing.append(f"{colmap_dir}/{f}")
    has_video = source == "iphone" and osp.isfile(osp.join(scene_dir, "iphone", "rgb.mkv"))
    if not (_has_images(osp.join(scene_dir, rgb_dir)) or has_video):    # iPhone: frames or the video they come from
        missing.append(f"{rgb_dir}/ frames" + (" or iphone/rgb.mkv" if source == "iphone" else ""))
    return missing


# ======================================================================
# Frames: image files, or (iPhone) decoded from the video
# ======================================================================
def frame_number(name):
    """frame_000123.jpg -> 123 (= index of the frame in iphone/rgb.mkv)."""
    return int(re.findall(r"\d+", osp.basename(name))[-1])


def video_num_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path} (needs an FFmpeg-enabled OpenCV build)")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def frames_from_video(video, mask_video, names):
    """Yield (name, rgb (H, W, 3) uint8, mask (H, W) uint8 or None) for `names` (sorted by frame number), decoding
    the video once from the start: frame_<i> = the i-th frame of rgb.mkv (frames numbered from 0, as the ScanNet++
    toolkit extracts them). Frames in between are only grabbed, and each selected frame is handed on right away, so
    memory stays at one frame whatever --num_images is."""
    cap = cv2.VideoCapture(video)
    mcap = cv2.VideoCapture(mask_video) if mask_video and osp.exists(mask_video) else None
    current = -1                                                      # index of the last frame read
    for name in names:
        target = frame_number(name)
        ok = True
        while ok and current < target - 1:                            # skip to the frame before the target
            ok = cap.grab()
            if mcap is not None:
                mcap.grab()
            current += 1
        ok, bgr = cap.read()
        current += 1
        mask = None
        if mcap is not None:
            mok, mframe = mcap.read()
            if mok:
                gray = mframe if mframe.ndim == 2 else cv2.cvtColor(mframe, cv2.COLOR_BGR2GRAY)
                mask = np.where(gray > 127, 255, 0).astype(np.uint8)  # undo compression noise: 255 = keep
        if not ok:
            print(f"[WARN] {video}: could not decode frame {target}; stopping this scene's video here")
            break
        yield name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask
    cap.release()
    if mcap is not None:
        mcap.release()


def frames_from_files(rgb_dir, mask_dir, mask_name, names):
    for name in names:
        rgb = np.array(Image.open(osp.join(rgb_dir, name)).convert("RGB"))
        mpath = osp.join(mask_dir, mask_name(name))
        yield name, rgb, (np.array(Image.open(mpath).convert("L")) if osp.exists(mpath) else None)


def process_scene(scene, args):
    data_dir = osp.join(data_root(args.scannetpp_dir), scene)
    out_dir = osp.join(args.output_dir, scene)
    if osp.exists(osp.join(out_dir, "cameras.pkl")):
        return "already done"
    colmap_dir, rgb_dir, mask_dir, mask_name = SOURCES[args.source]
    mesh_path = osp.join(data_dir, "scans", args.mesh_name)

    images = load_colmap(osp.join(data_dir, colmap_dir))
    video = osp.join(data_dir, "iphone", "rgb.mkv")
    use_video = args.source == "iphone" and not _has_images(osp.join(data_dir, rgb_dir)) and osp.isfile(video)
    if use_video:                                                     # registered frames inside the video
        n_video = video_num_frames(video)
        available = [n for n in images if frame_number(n) < n_video]
        where = f"rgb.mkv ({n_video} frames)"
    else:
        available = [n for n in images if osp.exists(osp.join(data_dir, rgb_dir, n))]
        where = rgb_dir
    if len(available) < len(images):
        print(f"[WARN] {scene}: {len(images) - len(available)} COLMAP images not found in {where}")
    if len(available) < args.min_images:
        return f"skipped: only {len(available)} registered frames available (< --min_images {args.min_images})"
    selected = uniform_subset(available, args.num_images)
    if use_video:
        selected = sorted(selected, key=frame_number)                 # decoding order
        frames = frames_from_video(video, osp.join(data_dir, "iphone", "rgb_mask.mkv"), selected)
    else:
        frames = frames_from_files(osp.join(data_dir, rgb_dir), osp.join(data_dir, mask_dir), mask_name, selected)

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    intersector = RayMeshIntersector(mesh)
    os.makedirs(osp.join(out_dir, "rgb"), exist_ok=True)
    os.makedirs(osp.join(out_dir, "depth"), exist_ok=True)

    cameras, coverage = {}, []
    for name, rgb, mask in tqdm(frames, total=len(selected), desc=scene, leave=False):
        info = images[name]
        if mask is None:                                              # no anonymisation mask: keep every pixel
            mask = np.full(rgb.shape[:2], 255, np.uint8)
        if (rgb.shape[1], rgb.shape[0]) != (int(info["camera"][1]), int(info["camera"][2])):
            raise ValueError(f"{scene}/{name}: frame {rgb.shape[1]}x{rgb.shape[0]} but COLMAP camera "
                             f"{int(info['camera'][1])}x{int(info['camera'][2])}")

        K, rgb, mask = undistort(info["camera"], rgb, mask)
        rgb, mask, K = resize(rgb, mask, K, args.width)
        H, W = rgb.shape[:2]
        depth = render_depth(intersector, K, info["c2w"], H, W, max_depth=args.max_depth)
        depth[mask < 255] = 0.0                                        # anonymised pixels: no depth (as DUSt3R)

        stem = osp.splitext(osp.basename(name))[0]
        Image.fromarray(rgb).save(osp.join(out_dir, "rgb", f"{stem}.jpg"), quality=95)
        np.save(osp.join(out_dir, "depth", f"{stem}.npy"), depth)
        cameras[f"{stem}.jpg"] = {"K": K.astype(np.float32), "RT_c2w": info["c2w"].astype(np.float32),
                                  "depth": f"{stem}.npy"}
        coverage.append(float((depth > 0).mean()))

    mesh_out = osp.join(out_dir, "mesh.ply")                           # dump_ldis.py reads <scene>/mesh.ply
    if not osp.exists(mesh_out):                                       # a copy: the dump does not depend on the source
        shutil.copyfile(mesh_path, mesh_out + ".tmp")
        os.replace(mesh_out + ".tmp", mesh_out)
    tmp = osp.join(out_dir, "cameras.pkl.tmp")
    with open(tmp, "wb") as f:
        pkl.dump(cameras, f)
    os.replace(tmp, osp.join(out_dir, "cameras.pkl"))                 # last: marks the scene done
    return (f"{len(cameras)} frames ({len(available)} available, {args.source}"
            f"{', decoded from rgb.mkv' if use_video else ''}) | {W}x{H} | "
            f"pixels with depth: {np.mean(coverage):.0%}")


def get_scenes(args):
    """The complete scenes (see missing_data) of --scene_list or of the data folder, then the first --num_scenes of
    them, then [start_idx, end_idx). Incomplete scenes are skipped and reported (counts + the first few reasons)."""
    root = data_root(args.scannetpp_dir)
    if args.scene_list:
        with open(args.scene_list) as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        scenes = sorted(d for d in os.listdir(root) if osp.isdir(osp.join(root, d)) and not d.startswith("."))

    complete, skipped = [], {}
    for scene in scenes:
        missing = missing_data(osp.join(root, scene), args.source, args.mesh_name) \
            if osp.isdir(osp.join(root, scene)) else ["the scene folder"]
        if missing:
            skipped[scene] = missing
        else:
            complete.append(scene)
    print(f"{len(scenes)} scenes in {root}: {len(complete)} complete for --source {args.source}, "
          f"{len(skipped)} skipped")
    if skipped:
        reasons = {}
        for missing in skipped.values():
            for m in missing:
                reasons[m.split(" (")[0]] = reasons.get(m.split(" (")[0], 0) + 1
        for m, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {n:4d} scenes without {m}")
        for scene, missing in list(skipped.items())[:5]:
            print(f"    e.g. {scene}: missing {', '.join(missing)}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(osp.join(args.output_dir, f"skipped_scenes_{args.source}.txt"), "w") as f:
            f.writelines(f"{s}\t{'; '.join(m)}\n" for s, m in skipped.items())

    scenes = complete[:args.num_scenes] if args.num_scenes else complete
    if args.num_scenes and len(complete) < args.num_scenes:
        print(f"[WARN] only {len(complete)} complete scenes (< --num_scenes {args.num_scenes})")
    end = len(scenes) if args.end_idx is None else min(args.end_idx, len(scenes))
    return scenes[args.start_idx:end]


def get_parser():
    p = argparse.ArgumentParser(description="ScanNet++ -> low-res dump (rgb / depth / cameras.pkl / mesh) for the LDI pipeline")
    p.add_argument("--scannetpp_dir", default="/projects/rps/dff6142/fouheylab/jt4100/datasets/scannetpp/", help="ScanNet++ root (contains data/ and splits/)")
    p.add_argument("--output_dir", default="/scratch/dma9300/scannetpp_preprocessed/")
    p.add_argument("--scene_list", default=None, help="one scene per line (e.g. splits/nvs_sem_train.txt)")
    p.add_argument("--num_scenes", type=int, default=200, help="first N COMPLETE scenes of the list (0 = all)")
    p.add_argument("--start_idx", type=int, default=0, help="first scene of the (truncated) list, for job arrays")
    p.add_argument("--end_idx", type=int, default=None, help="end scene, exclusive")
    p.add_argument("--source", choices=list(SOURCES), default="iphone")
    p.add_argument("--num_images", type=int, default=2000, help="frames per scene, uniform over the capture")
    p.add_argument("--min_images", type=int, default=6, help="skip scenes with fewer registered frames on disk")
    p.add_argument("--width", type=int, default=640, help="saved image width (iPhone 4:3 -> 640 x 480)")
    p.add_argument("--mesh_name", default="mesh_aligned_0.05.ply")
    p.add_argument("--max_depth", type=float, default=20.0, help="DUSt3R / ScanNet++ render zfar")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    scenes = get_scenes(args)
    print(f"processing {len(scenes)} scenes -> {args.output_dir} ({args.source}, up to {args.num_images} frames each)")
    for scene in tqdm(scenes, desc="scenes"):
        tqdm.write(f"{scene}: {process_scene(scene, args)}")