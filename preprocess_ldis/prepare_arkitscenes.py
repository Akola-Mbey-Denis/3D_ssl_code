"""
ARKitScenes raw captures -> per-frame RGB, intrinsics, pose and mesh-rendered depth.

Input per capture (download_arkits.py raw ... --raw_dataset_assets vga_wide vga_wide_intrinsics
lowres_wide.traj mesh):
    {raw_dir}/{fold}/{video_id}/vga_wide/{video_id}_{ts}.png          640x480 RGB, 30 fps
    {raw_dir}/{fold}/{video_id}/vga_wide_intrinsics/{video_id}_{ts}.pincam   "w h fx fy cx cy"
    {raw_dir}/{fold}/{video_id}/lowres_wide.traj                       60 fps poses
    {raw_dir}/{fold}/{video_id}/{video_id}_3dod_mesh.ply               ARKit mesh

lowres_wide.traj line: "ts  rx ry rz  tx ty tz" = world->camera rotation (axis-angle) and
translation, as read by ARKitScenes' TrajStringToMatrix; we store its inverse (camera->world).

Each vga_wide frame gets a pose interpolated at its timestamp, as in DUSt3R's ARKitScenes
preprocessing (linear for the position, spherical for the rotation; DUSt3R uses squad); frames
outside the trajectory or whose nearest trajectory sample is further than --max_dt seconds (a
tracking gap) are skipped. Up to --max_frames frames are kept, evenly spaced. Depth = first mesh hit
per pixel (camera-frame z, metres, 0 = no hit), rendered with pyrender as in DUSt3R's ScanNet++
preprocessing (OpenGL camera = c2w @ OPENGL_TO_OPENCV, DEPTH_ONLY, znear/zfar); --renderer embree
ray-casts instead. Both use the LDI scripts' pixel convention (pixel (u, v) <-> K^-1 [u, v, 1]).

Each capture is made upright as in DUSt3R (find_scene_orientation: sky UP/DOWN/LEFT/RIGHT from the
average device up/right vectors vs world +z); the image is rotated and K, c2w and the image size are
changed to match, and depth is rendered with the upright camera. --no_upright keeps the original.

Output, a separate preprocessed tree (the raw download is only read):
    {out_dir}/{fold}/{video_id}/rgb/{video_id}_{ts}.jpg      RGB (upright unless --no_upright)
    {out_dir}/{fold}/{video_id}/depth/{video_id}_{ts}.npy    float32 z-depth (mesh-rendered), same size
    {out_dir}/{fold}/{video_id}/cameras.npz   frame_ids, K (N,3,3), c2w (N,4,4), sky_direction
    {out_dir}/{fold}/{video_id}/mesh.ply      copy of {video_id}_3dod_mesh.ply (for the LDI script)
    {out_dir}/{fold}/{video_id}/check.png     RGB / depth / overlay of 3 frames (--check)
"""

import argparse
import glob
import os
import os.path as osp
import shutil

import numpy as np
import pandas as pd
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from trimesh.ray.ray_pyembree import RayMeshIntersector


def read_traj(path):
    """
    -> timestamps (M,), c2w (M, 4, 4). Same as DUSt3R's read_traj: axis-angle + translation is
    world->camera (cv2.Rodrigues == Rotation.from_rotvec), inverted to camera->world; timestamps
    rounded to 3 decimals like the frame file names.
    """
    data = np.loadtxt(path, ndmin=2)
    ts = np.round(data[:, 0], 3)
    w2c = np.tile(np.eye(4), (len(data), 1, 1))
    w2c[:, :3, :3] = Rotation.from_rotvec(data[:, 1:4]).as_matrix()
    w2c[:, :3, 3] = data[:, 4:7]
    return ts, np.linalg.inv(w2c)


def read_pincam(path):
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, int(w), int(h)


class PoseInterpolator:
    """Camera->world pose at any time: linear position, slerp rotation (DUSt3R: linear + squad)."""

    def __init__(self, ts, c2w):
        ts, idx = np.unique(ts, return_index=True)                       # sorted, no duplicate stamps
        self.ts, self.c2w = ts, c2w[idx]
        self.slerp = Slerp(ts, Rotation.from_matrix(self.c2w[:, :3, :3]))

    def gap(self, t):
        """Time to the nearest trajectory sample (inf outside the trajectory)."""
        if t < self.ts[0] or t > self.ts[-1]:
            return np.inf
        j = np.searchsorted(self.ts, t)
        return min(abs(self.ts[min(j, len(self.ts) - 1)] - t), abs(self.ts[max(j - 1, 0)] - t))

    def __call__(self, t):
        pose = np.eye(4)
        pose[:3, :3] = self.slerp([t]).as_matrix()[0]
        pose[:3, 3] = [np.interp(t, self.ts, self.c2w[:, k, 3]) for k in range(3)]
        return pose


def find_pincam(cap_dir, vid, ts_str):
    """Intrinsics file of a frame; like DUSt3R, also accept a +-1 ms timestamp mismatch."""
    t = float(ts_str)
    for s in (ts_str, f"{t - 0.001:.3f}", f"{t + 0.001:.3f}"):
        path = osp.join(cap_dir, "vga_wide_intrinsics", f"{vid}_{s}.pincam")
        if osp.exists(path):
            return path
    return None


def frame_ts(path):
    """{video_id}_{ts}.png -> ts (float) and the ts string."""
    s = osp.splitext(osp.basename(path))[0].split("_", 1)[1]
    return float(s), s


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class PyrenderDepth:
    """
    Depth rendering as in DUSt3R's preprocess_scannetpp.py: one pyrender scene per mesh, an
    IntrinsicsCamera added at c2w @ OPENGL_TO_OPENCV per frame, DEPTH_ONLY render (z-depth, 0 = miss).

    Differences from that script, both so the depth matches the ray-cast LDIs exactly:
    - pyrender samples pixel centres, so cx, cy are shifted by +0.5 (pixel (u, v) <-> K^-1 [u, v, 1],
      the convention of the LDI scripts and of the ARKit .pincam intrinsics);
    - the material is double-sided, so back faces are not culled (no seeing through a wall whose
      triangles face away from the camera).
    """

    def __init__(self, mesh, znear=0.05, zfar=20.0):
        import pyrender
        self.pyrender = pyrender
        material = pyrender.MetallicRoughnessMaterial(doubleSided=True)
        self.scene = pyrender.Scene()
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
        self.renderer = pyrender.OffscreenRenderer(1, 1)
        self.znear, self.zfar = znear, zfar

    def __call__(self, K, c2w, H, W, max_depth=16.0):
        pr = self.pyrender
        self.renderer.viewport_width, self.renderer.viewport_height = W, H
        cam = pr.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5,
                                  znear=self.znear, zfar=self.zfar)
        node = self.scene.add(cam, pose=c2w @ OPENGL_TO_OPENCV)
        depth = self.renderer.render(self.scene, flags=pr.RenderFlags.DEPTH_ONLY)
        self.scene.remove_node(node)                                          # dont forget to remove camera
        depth = depth.astype(np.float32)
        depth[depth > max_depth] = 0.0
        # breakpoint()
        return depth

    def close(self):
        self.renderer.delete()


class EmbreeDepth:
    def __init__(self, mesh):
        self.intersector = RayMeshIntersector(mesh)

    def __call__(self, K, c2w, H, W, max_depth=16.0):
        return render_depth(self.intersector, K, c2w, H, W, max_depth)

    def close(self):
        pass


def render_depth(intersector, K, c2w, H, W, max_depth=16.0):
    """Ray-cast z-depth of the first mesh hit per pixel (0 = no hit), integer pixel coordinates."""
    v, u = np.mgrid[0:H, 0:W]
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).astype(np.float64)
    rays_cam = pix @ np.linalg.inv(K).T                                   # z = 1
    dirs = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
    dirs_w = dirs @ c2w[:3, :3].T
    origins = np.broadcast_to(c2w[:3, 3], dirs_w.shape).copy()
    loc, idx_ray, _ = intersector.intersects_location(origins, dirs_w, multiple_hits=False)
    depth = np.zeros(H * W, np.float32)
    if len(loc):
        z = (loc - origins[idx_ray]) @ c2w[:3, 2]                         # camera-frame z
        ok = (z > 0) & (z <= max_depth)
        depth[idx_ray[ok]] = z[ok]
    return depth.reshape(H, W)


def save_check(out_path, rgbs, depths, max_depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(rgbs)
    fig, axs = plt.subplots(n, 3, figsize=(12, 3.2 * n), squeeze=False)
    for i, (rgb_path, d) in enumerate(zip(rgbs, depths)):
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        axs[i, 0].imshow(rgb)
        axs[i, 1].imshow(np.ma.masked_where(d <= 0, d), cmap="Spectral_r", vmin=0, vmax=max_depth)
        # depth edges over the image: misaligned edges = wrong pose convention / intrinsics
        gy, gx = np.gradient(d)
        edges = (np.hypot(gx, gy) > 0.1) & (d > 0)
        over = rgb.copy()
        over[edges] = [255, 0, 0]
        axs[i, 2].imshow(over)
        for a, t in zip(axs[i], ["RGB", "mesh depth", "depth edges on RGB"]):
            a.set_title(t)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------
# Upright orientation (DUSt3R's find_scene_orientation, adapted)
# ----------------------------------------------------------------------
# PIL transpose that makes the image upright, and the pixel map it applies to integer pixel
# coordinates of a W x H image: (u, v) -> M @ (u, v) + t
_ROT = {
    "UP":    (None,                   np.array([[1, 0], [0, 1]]),   lambda W, H: np.array([0, 0])),
    "RIGHT": (Image.Transpose.ROTATE_90,  np.array([[0, 1], [-1, 0]]),  lambda W, H: np.array([0, W - 1])),
    "LEFT":  (Image.Transpose.ROTATE_270, np.array([[0, -1], [1, 0]]),  lambda W, H: np.array([H - 1, 0])),
    "DOWN":  (Image.Transpose.ROTATE_180, np.array([[-1, 0], [0, -1]]), lambda W, H: np.array([W - 1, H - 1])),
}


def find_scene_orientation(traj_c2w):
    """
    Sky direction of a capture, as in DUSt3R's preprocess_arkitscenes.find_scene_orientation:
    average the device up (-y) and right (+x) vectors over the trajectory and compare them with
    world up (+z). Returns "UP", "DOWN", "LEFT" or "RIGHT".
    """
    up = (traj_c2w[:, :3, :3] @ np.array([0.0, -1.0, 0.0])).mean(0)
    right = (traj_c2w[:, :3, :3] @ np.array([1.0, 0.0, 0.0])).mean(0)
    up_world = np.array([0.0, 0.0, 1.0])
    ang_up = np.degrees(np.arccos(np.clip(up @ up_world / max(np.linalg.norm(up), 1e-8), -1, 1)))
    ang_right = np.degrees(np.arccos(np.clip(right @ up_world / max(np.linalg.norm(right), 1e-8), -1, 1)))
    if abs(ang_up - 90.0) < abs(ang_right - 90.0):                      # device up is horizontal: rolled 90
        return "LEFT" if ang_right > 90.0 else "RIGHT"
    return "DOWN" if ang_up > 90.0 else "UP"


def upright_camera(K, c2w, W, H, sky):
    """
    K, c2w and size of the camera whose image is the upright-rotated image.
    Derived from the pixel map (u', v') = M (u, v) + t: the camera turns about its optical axis
    by M (x' = M x in normalised coordinates), so K' = diag(M diag(fx, fy) M^T), c' = M c + t
    (exact for any principal point; DUSt3R only swaps cx and cy).
    """
    if sky == "UP":
        return K, c2w, W, H
    _, M, t = _ROT[sky]
    f = M @ np.diag([K[0, 0], K[1, 1]]) @ M.T
    c = M @ K[:2, 2] + t(W, H)
    K2 = np.array([[abs(f[0, 0]), 0, c[0]], [0, abs(f[1, 1]), c[1]], [0, 0, 1.0]])
    R = np.eye(4)
    R[:2, :2] = M                                                          # p_new = R p_old (camera frame)
    c2w2 = c2w @ np.linalg.inv(R)
    W2, H2 = (H, W) if sky in ("LEFT", "RIGHT") else (W, H)
    return K2, c2w2, W2, H2


def process_capture(cap_dir, out_dir, max_frames=1000, max_dt=0.05, max_depth=16.0, check=False,
                    overwrite=False, renderer="pyrender", upright=True):
    vid = osp.basename(cap_dir.rstrip("/"))
    fold = osp.basename(osp.dirname(cap_dir.rstrip("/")))                  # Training / Validation
    dst = osp.join(out_dir, fold, vid)
    if osp.exists(osp.join(dst, "cameras.npz")) and not overwrite:
        return "exists"

    mesh_path = osp.join(cap_dir, f"{vid}_3dod_mesh.ply")
    traj_path = osp.join(cap_dir, "lowres_wide.traj")
    frames = sorted(glob.glob(osp.join(cap_dir, "vga_wide", "*.png")), key=lambda p: frame_ts(p)[0])
    if not (osp.exists(mesh_path) and osp.exists(traj_path) and frames):
        return "missing assets"

    traj_ts, traj_c2w = read_traj(traj_path)
    poses = PoseInterpolator(traj_ts, traj_c2w)

    # keep frames inside the trajectory (no tracking gap) that have intrinsics; pose interpolated at t
    matched = []
    for f in frames:
        t, ts_str = frame_ts(f)
        pincam = find_pincam(cap_dir, vid, ts_str)
        if poses.gap(t) <= max_dt and pincam is not None:
            matched.append((f, ts_str, t, pincam))
    if not matched:
        return "no frames matched a pose"
    if 0 < max_frames < len(matched):                                          # evenly spaced subset
        keep = np.linspace(0, len(matched) - 1, max_frames).round().astype(int)
        matched = [matched[i] for i in np.unique(keep)]

    sky = find_scene_orientation(traj_c2w) if upright else "UP"

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    render = PyrenderDepth(mesh) if renderer == "pyrender" else EmbreeDepth(mesh)

    os.makedirs(osp.join(dst, "depth"), exist_ok=True)
    os.makedirs(osp.join(dst, "rgb"), exist_ok=True)
    Ks, c2ws, ids, check_rgb, check_depth = [], [], [], [], []
    check_idx = set(np.linspace(0, len(matched) - 1, 3).round().astype(int).tolist()) if check else set()
    for i, (f, ts_str, t, pincam) in enumerate(tqdm(matched, desc=vid, leave=False)):
        K, w, h = read_pincam(pincam)
        c2w = poses(t)                                                     # interpolated at the frame time
        # rotate the image upright (no-op for UP / --no_upright) and render depth with that camera
        K, c2w, w, h = upright_camera(K, c2w, w, h, sky)
        img = Image.open(f).convert("RGB")
        if _ROT[sky][0] is not None:
            img = img.transpose(_ROT[sky][0])
        assert img.size == (w, h), (img.size, (w, h))
        f = osp.join(dst, "rgb", f"{vid}_{ts_str}.jpg")
        img.save(f, quality=95)
        depth = render(K, c2w, h, w, max_depth)
        np.save(osp.join(dst, "depth", f"{vid}_{ts_str}.npy"), depth)
        Ks.append(K)
        c2ws.append(c2w)
        ids.append(ts_str)
        if i in check_idx:
            check_rgb.append(f)
            check_depth.append(depth)

    render.close()
    np.savez(osp.join(dst, "cameras.npz"), frame_ids=np.array(ids), K=np.stack(Ks), c2w=np.stack(c2ws),
             sky_direction=np.array(sky), upright=np.array(upright), fold=np.array(fold))
    shutil.copy2(mesh_path, osp.join(dst, "mesh.ply"))
    # breakpoint()
    if check:
        save_check(f"./check_{vid}.png", check_rgb, check_depth, max_depth)
    # breakpoint()
    return f"{len(ids)} frames, sky {sky}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="/scratch/dma9300/ARKitScenes/raw",
                   help="folder with Training/ and Validation/ capture folders")
    p.add_argument("--out_dir", default="/scratch/dma9300/preprocessed_arkitscenes")
    p.add_argument("--video_id_csv", default=None, help="optional: only these captures")
    p.add_argument("--max_frames", type=int, default=1000, help="-1 = render every frame with a pose")
    p.add_argument("--max_dt", type=float, default=0.05,
                   help="skip frames whose nearest trajectory sample is further than this (s): tracking gaps")
    p.add_argument("--max_depth", type=float, default=16.0)
    p.add_argument("--check", action="store_true", help="save check.png (depth edges over RGB)")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--renderer", choices=["pyrender", "embree"], default="pyrender")
    p.add_argument("--no_upright", action="store_true",
                   help="keep the original (possibly sideways) orientation; default rotates every capture "
                        "upright (DUSt3R's find_scene_orientation)")
    p.add_argument("--pyopengl-platform", type=str, default="egl",
                   help="PYOPENGL_PLATFORM for headless pyrender (egl on GPU nodes, osmesa on CPU nodes)")
    args = p.parse_args()
    if args.renderer == "pyrender" and args.pyopengl_platform.strip():
        os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform      # must be set before importing pyrender

    caps = sorted(glob.glob(osp.join(args.raw_dir, "*", "*")))
    caps = [c for c in caps if osp.isdir(osp.join(c, "vga_wide"))]
    if args.video_id_csv:
        keep = set(pd.read_csv(args.video_id_csv, dtype=str)["video_id"])
        caps = [c for c in caps if osp.basename(c) in keep]
    caps = caps[args.start_idx:(None if args.end_idx == -1 else args.end_idx + 1)]
    for c in caps:
        print(osp.basename(c), process_capture(c, args.out_dir, args.max_frames, args.max_dt,
                                               args.max_depth, args.check, args.overwrite, args.renderer,
                                               not args.no_upright))


if __name__ == "__main__":
    main()