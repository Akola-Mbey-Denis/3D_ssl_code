"""
Matterport3D mesh LDIs for EVERY frame of the low-res Matterport dump, at 224 x 224 (same output as dump_ldis.py).

Input (the low-res dump, same layout as the ScanNet dump):
    <dump_root>/<house>/rgb/<frame>.jpg
                        depth/<frame>.npy          metres, 0 = invalid
                        cameras.pkl                {rgb file name: {"K", "RT_c2w" (OpenCV cam2world), "depth"}}
Meshes, as in the Matterport mesh-LDI script / the MatterportDataLDI dataloader's load_mesh():
    house Poisson mesh  {matterport_dir}/poisson_meshes/{house}/poisson_meshes/{house}.ply   (matterport_parse.preload_meshes)
                        or {poisson_mesh_dir}/{house}.ply                                      (preload_meshes_poisson_from_dir)
    per-view face ids   {faces_cachedir}/{house}/{frame}.mat                                  (matterport_parse.load_face_metadata)
    Each view is ray-cast against the faces it sees only (mesh["faces"][fids] -> delete_unused_verts), so rays
    do not run through doorways into other rooms; views without face ids are skipped (--full_house: whole house).

Every frame is cropped / resized to 224 x 224 exactly like the dataloaders (crop around the principal point,
then resize, aug_crop=-1) -> K at 224, then one ray per pixel centre, all mesh hits in [min_depth, max_depth],
back faces dropped from the hidden layers (as the Matterport script), consecutive layers >= min_sep apart,
first num_layers kept.

Output, one file per frame (same name as the frame):
    <out_root>/LDIs/<house>/<frame>.npz     (out_root defaults to dump_root)
        ldi  (224, 224, 4) float16   layer k = k-th surface along the pixel's ray, near -> far, 0 = no surface
                                     z-depth in metres by default (--ldi_type ray_distance for ray distances)
    A frame whose .npz already exists is skipped (re-runs only compute the missing frames).

Train the LDI probes with aug_crop=-1 (no random scale jitter), so the image crop and the LDI line up.

Usage:
    python dump_ldis_matterport.py --dump_root <low_res_matterport_dump> --start_idx 0 --end_idx 20   # houses 0..19
"""

import argparse
import os
import os.path as osp
import pickle as pkl

import numpy as np
import PIL
import trimesh
from PIL import Image
from tqdm import tqdm
from trimesh.ray.ray_pyembree import RayMeshIntersector

import utils.cropping as cropping
import utils.matterport_parse2 as matterport_parse


# ======================================================================
# Loading
# ======================================================================
def load_image(file_path):
    return np.array(Image.open(file_path).convert("RGB"))


def load_depth(depth_path):
    """Depth in metres (.npy), non-finite -> 0."""
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def rotate_inverse(se3):
    assert se3.shape == (4, 4), f"Expected (4, 4), got {se3.shape}"
    R, T = se3[:3, :3], se3[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T
    return out


def crop_resize(image, depthmap, intrinsics, resolution, aug_crop=-1, mask=None, rng=None):
    """The dataloaders' _crop_resize_if_necessary (crop around the principal point, then resize)."""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, (l, t, r, b), mask=mask)

    W, H = image.size
    assert resolution[0] >= resolution[1]
    if H > 1.1 * W:
        resolution = resolution[::-1]
    elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
        if rng.integers(2):
            resolution = resolution[::-1]

    target_resolution = np.array(resolution)
    if aug_crop > 1:
        target_resolution += rng.integers(0, aug_crop)
    image, depthmap, intrinsics, mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, mask=mask)

    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask=mask)
    return image, depthmap, intrinsics2, mask


# ======================================================================
# Ray generation + mesh intersection (from the ScanNet mesh-LDI script)
# ======================================================================
def generate_rays(K, RT_w2c, H, W):
    """One ray per pixel: origins (H*W, 3), unit dirs (H*W, 3) in world frame."""
    RT_c2w = rotate_inverse(RT_w2c)
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).T
    dirs_cam = (np.linalg.inv(K) @ pixels).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    R, T = RT_c2w[:3, :3], RT_c2w[:3, 3]
    ray_dirs = dirs_cam @ R.T
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_origins = np.broadcast_to(T, ray_dirs.shape).copy()
    return ray_origins, ray_dirs


def generate_layered_depths(mesh, ray_origins, ray_dirs, H, W, max_layers=4, min_depth=0.001, max_depth=16.0,
                            front_only=False, min_sep=16.0 / 256, intersector=None):
    """
    LDI by intersecting one ray per pixel with the mesh (trimesh + embree, all hits per ray).

    Hits outside [min_depth, max_depth] are dropped and the rest are sorted by distance.
    With front_only, hidden layers keep only front-facing hits (surfaces the ray enters);
    the first hit is always kept, so layer 1 is the visible surface.
    Consecutive kept layers are at least min_sep apart along the ray; the nearer hit wins.

    Returns
        dist   (H, W, L) float32  ray distance, 0 = none
        facing (H, W, L) int8     +1 front face, -1 back face, 0 none
    """
    L = max_layers
    n = H * W
    dist = np.zeros((n, L), np.float32)
    facing = np.zeros((n, L), np.int8)

    if intersector is None:
        intersector = RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=True)

    if len(locations) > 0:
        depths = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        valid = (depths >= min_depth) & (depths <= max_depth)
        depths, index_ray, index_tri = depths[valid], index_ray[valid], index_tri[valid]

        # sort hits by ray, then by distance
        order = np.lexsort((depths, index_ray))
        depths, index_ray, index_tri = depths[order], index_ray[order], index_tri[order]

        is_front = (mesh.face_normals[index_tri] * ray_dirs[index_ray]).sum(-1) < 0

        if front_only:
            is_first = np.r_[True, index_ray[1:] != index_ray[:-1]]
            keep_hit = is_first | is_front
            depths, index_ray, is_front = depths[keep_hit], index_ray[keep_hit], is_front[keep_hit]

        # minimum separation between consecutive kept layers (measured to the previous KEPT hit)
        if min_sep > 0 and len(depths) > 0:
            first = np.searchsorted(index_ray, index_ray, side="left")
            rank = np.arange(len(index_ray)) - first
            last_kept = np.full(n, -np.inf)
            keep_sep = np.zeros(len(depths), bool)
            for j in range(rank.max() + 1):
                idx = np.nonzero(rank == j)[0]
                ok = depths[idx] - last_kept[index_ray[idx]] >= min_sep
                keep_sep[idx[ok]] = True
                last_kept[index_ray[idx[ok]]] = depths[idx[ok]]
            depths, index_ray, is_front = depths[keep_sep], index_ray[keep_sep], is_front[keep_sep]

        # layer k of a ray = its k-th remaining hit
        first = np.searchsorted(index_ray, index_ray, side="left")
        layer = np.arange(len(index_ray)) - first
        keep = layer < L
        r, k = index_ray[keep], layer[keep]
        dist[r, k] = depths[keep]
        facing[r, k] = np.where(is_front[keep], 1, -1)

    return dist.reshape(H, W, L), facing.reshape(H, W, L)


def ray_distance_to_depth(dist, K):
    """Ray distances (H, W, L) -> camera-frame z-depths; 0 (no hit) stays 0."""
    H, W = dist.shape[:2]
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    return (dist / np.linalg.norm(rays, axis=-1)[..., None]).astype(np.float32)


def layer1_agreement(ldi_depth, target_depth, tol=0.05, rel_tol=0.02):
    """How well mesh layer 1 matches the depth map (both z-depths)."""
    m = (ldi_depth[..., 0] > 0) & (target_depth > 0)
    if not m.any():
        return {"pixels": 0}
    gt = target_depth[m]
    err = np.abs(ldi_depth[..., 0][m] - gt)
    thr = np.maximum(tol, rel_tol * gt)            # 5 cm, or 2% of the GT depth if larger
    return {
        "pixels": int(m.sum()),
        "coverage": float(m.mean()),
        "median_abs_err_m": float(np.median(err)),
        "frac_within_tol": float((err < thr).mean()),
    }


# ======================================================================
# Scenes
# ======================================================================
def list_scenes(dump_root):
    """Scene folders whose dump finished (cameras.pkl present), sorted."""
    return [s for s in sorted(os.listdir(dump_root))
            if s != "LDIs" and osp.exists(osp.join(dump_root, s, "cameras.pkl"))]


class MatterportMeshes:
    """
    Loads each house mesh once and returns the per-view trimesh, as the MatterportDataLDI dataloader's load_mesh():
      mesh  : matterport_parse.preload_meshes (or preload_meshes_poisson_from_dir)
      fids  : matterport_parse.load_face_metadata
      subset: mesh["faces"][fids] -> matterport_parse.delete_unused_verts -> trimesh.Trimesh
    A view without face ids gets None (skipped). With face_subset=False the full house mesh is used for every view.
    """

    def __init__(self, matterport_dir, faces_cachedir=None, face_subset=True, poisson_mesh_dir=None):
        self.matterport_dir = matterport_dir
        self.poisson_mesh_dir = poisson_mesh_dir
        self.faces_cachedir = faces_cachedir
        self.face_subset = face_subset
        if face_subset and faces_cachedir is None:
            raise ValueError("face_subset needs faces_cachedir")
        self._house_id, self._mesh, self._full = None, None, None

    def mesh_path(self, house_id):
        if self.poisson_mesh_dir is not None:
            return osp.join(self.poisson_mesh_dir, f"{house_id}.ply")
        return osp.join(self.matterport_dir, "poisson_meshes", house_id, "poisson_meshes", f"{house_id}.ply")

    def has_mesh(self, house_id):
        return osp.exists(self.mesh_path(house_id))

    def _load_house(self, house_id):
        if house_id == self._house_id:
            return
        if self.poisson_mesh_dir is not None:
            houses2mesh = matterport_parse.preload_meshes_poisson_from_dir(house_id, self.poisson_mesh_dir)
        else:
            houses2mesh = matterport_parse.preload_meshes(house_id, self.matterport_dir)
        self._mesh, self._full, self._house_id = houses2mesh[house_id], None, house_id

    def load(self, house_id, image_name):
        """(trimesh.Trimesh, RayMeshIntersector) for one view, or (None, None) if the view has no face ids."""
        self._load_house(house_id)
        if not self.face_subset:
            if self._full is None:
                mesh = trimesh.Trimesh(vertices=self._mesh["vertices"], faces=self._mesh["faces"], process=False)
                self._full = (mesh, RayMeshIntersector(mesh))          # one BVH for the whole house
            return self._full
        if not osp.exists(osp.join(self.faces_cachedir, house_id, f"{image_name}.mat")):
            return None, None
        metadata = {"house_id": house_id, "image_names": f"{image_name}.jpg"}
        fids = matterport_parse.load_face_metadata(metadata=metadata, face_cachedir=self.faces_cachedir)[0]
        if fids is None:
            return None, None
        mesh = {"faces": self._mesh["faces"] * 1, "vertices": self._mesh["vertices"] * 1}
        mesh["faces"] = mesh["faces"][fids]
        mesh = matterport_parse.delete_unused_verts(mesh)
        if len(mesh["faces"]) == 0:
            return None, None
        mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
        return mesh, RayMeshIntersector(mesh)


def plot_gt_depth_and_ldi(gt_depth, ldi_depth, save_path="ldis_gt_depthmap.png", vmin=0.0, vmax=16.0,
                          cmap="Spectral_r", title=None):
    """
    GT depth and every LDI layer side by side, on one shared color scale.

    gt_depth  (H, W)    z-depth, 0 = invalid
    ldi_depth (H, W, L) z-depth per layer, 0 = no hit
    Invalid pixels / missing hits are left blank.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = ldi_depth.shape[-1]
    panels = [("GT depth", gt_depth)] + [(f"LDI layer {k + 1}", ldi_depth[..., k]) for k in range(L)]

    fig, axs = plt.subplots(1, L + 1, figsize=(3.5 * (L + 1), 3.8), constrained_layout=True)
    for ax, (name, d) in zip(axs, panels):
        valid = np.isfinite(d) & (d > 0)
        im = ax.imshow(np.ma.masked_where(~valid, d), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} ({valid.mean():.0%})")
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label="depth (m)")
    if title is not None:
        fig.suptitle(title)

    os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def process_scene(scene, args, meshes):
    scene_dir = osp.join(args.dump_root, scene)
    out_dir = osp.join(args.out_root, "LDIs", scene)
    if not meshes.has_mesh(scene):
        print(f"[MESH] {scene}: no mesh at {meshes.mesh_path(scene)}, skipping the house")
        return "no_mesh"

    with open(osp.join(scene_dir, "cameras.pkl"), "rb") as f:
        cameras = pkl.load(f)
    rgb_dir = osp.join(scene_dir, "rgb")
    frames = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not args.overwrite:                                           # only the frames without an LDI yet
        frames = [f for f in frames if not osp.exists(osp.join(out_dir, f"{osp.splitext(f)[0]}.npz"))]
    if not frames:
        return "all frames already done"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed=42)                             # unused for square 224 x 224 crops
    n_written, agree, hidden, n_layers = 0, [], [], []
    for frame in tqdm(frames, desc=scene, leave=False):
        stem = osp.splitext(frame)[0]
        out_path = osp.join(out_dir, f"{stem}.npz")
        cam = cameras.get(frame)
        if cam is None:
            print(f"[SKIP] {scene}/{frame}: not in cameras.pkl")
            continue
        c2w = np.asarray(cam["RT_c2w"], np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        if not np.isfinite(c2w).all():
            print(f"[SKIP] {scene}/{frame}: invalid pose")
            continue

        image = load_image(osp.join(rgb_dir, frame))
        depth = load_depth(osp.join(scene_dir, "depth", cam.get("depth", f"{stem}.npy")))
        K = np.asarray(cam["K"], np.float64)
        _, depth, K, _ = crop_resize(image, depth, K, (args.resolution, args.resolution),
                                     aug_crop=-1, mask=(depth > 0).astype(np.int32), rng=rng)
        depth = np.asarray(depth, np.float32)
        H, W = depth.shape

        mesh, intersector = meshes.load(scene, stem)                 # this view's faces only
        if mesh is None:
            print(f"[SKIP] {scene}/{frame}: no face ids")
            continue

        ray_origins, ray_dirs = generate_rays(K=K, RT_w2c=rotate_inverse(c2w), H=H, W=W)
        ldi_dist, facing = generate_layered_depths(
            mesh, ray_origins, ray_dirs, H, W, max_layers=args.num_layers, min_depth=args.min_depth,
            max_depth=args.max_depth, front_only=not args.keep_back_faces, min_sep=args.min_sep, intersector=intersector)
        ldi_depth = ray_distance_to_depth(ldi_dist, K)

         
        # plot_gt_depth_and_ldi(gt_depth = depth, ldi_depth = ldi_depth, save_path=f"./ldis_gt_depthmap-matterport_{stem}.png", vmin=0.0, vmax=16.0, cmap="Spectral_r", title=None)
        # breakpoint()

        ldi = ldi_depth if args.ldi_type == "z_depth" else ldi_dist
        ldi = np.where(np.isfinite(ldi) & (ldi > 0), ldi, 0.0).astype(np.float16)   # invalid / no surface = 0
        tmp = out_path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, ldi=ldi)
        os.replace(tmp, out_path)                                     # no half-written files
        n_written += 1

        target = depth.copy()
        target[target > args.max_depth] = 0.0
        stats = layer1_agreement(ldi_depth, target)
        has1 = facing[..., 0] != 0
        stats["frac_layer1_front_facing"] = float((facing[..., 0][has1] == 1).mean()) if has1.any() else 0.0
        stats["layers_per_pixel_mean"] = float((ldi_dist > 0).sum(-1).mean())
        stats["frac_pixels_with_hidden_layer"] = float((ldi_dist[..., 1] > 0).mean())
        agree.append(stats.get("frac_within_tol", np.nan))
        hidden.append(stats["frac_pixels_with_hidden_layer"])
        n_layers.append(stats["layers_per_pixel_mean"])
        if stats.get("frac_within_tol", 1.0) < 0.5:
            print(f"[WARN] {scene}/{stem}: only {stats['frac_within_tol']:.0%} of layer 1 agrees with the depth "
                  f"map -- check pose convention / depth units / mesh frame")
        if stats["frac_layer1_front_facing"] < 0.5 and has1.any():
            print(f"[WARN] {scene}/{stem}: layer 1 is mostly back-facing -- inverted face winding?")

    if n_written == 0:
        return "0 written"
    return (f"{n_written} written | layer 1 within tol of depth: {np.nanmean(agree):.1%} | "
            f"pixels with a hidden layer: {np.mean(hidden):.1%} | layers/pixel: {np.mean(n_layers):.2f}")


def get_args():
    p = argparse.ArgumentParser(description="Matterport3D mesh LDIs (224 x 224, 4 layers, float16) for every frame of the low-res dump")
    p.add_argument("--dump_root", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_matterport3d/",
                   help="low-res dump: <dump_root>/<house>/{rgb, depth, cameras.pkl}")
    p.add_argument("--out_root", type=str, default=None, help="LDIs go to <out_root>/LDIs/<scene>/ (default: dump_root)")
    p.add_argument("--matterport_dir", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/matterport3D/scans/",
                   help="Matterport3D root: meshes at {root}/poisson_meshes/{house}/poisson_meshes/{house}.ply")
    p.add_argument("--poisson_mesh_dir", type=str, default=None,
                   help="optional flat folder of {house}.ply Poisson meshes; overrides the root layout")
    p.add_argument("--faces_cachedir", type=str,
                   default="/projects/rps/dff6142/fouheylab/dma9300/occ_project/preprocessed_training_data/mp3d_data_zmx8",
                   help="per-view face ids: {faces_cachedir}/{house}/{frame}.mat")
    p.add_argument("--full_house", action="store_true",
                   help="ray-cast the whole house mesh for every view (default: the view's face subset)")
    p.add_argument("--keep_back_faces", action="store_true",
                   help="keep back-facing hits in the hidden layers (default: dropped, as the Matterport script)")
    p.add_argument("--start_idx", type=int, default=0, help="first scene (index in the sorted scene list)")
    p.add_argument("--end_idx", type=int, default=None, help="end scene, exclusive (default: all)")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ldi_type", choices=["z_depth", "ray_distance"], default="z_depth",
                   help="z_depth: same quantity as the depth maps (default); ray_distance: as the Matterport LDI script")
    p.add_argument("--min_depth", type=float, default=0.10)
    p.add_argument("--max_depth", type=float, default=16.0)
    p.add_argument("--min_sep", type=float, default=16.0 / 256.0,
                   help="minimum ray-distance separation between consecutive layers (m); 0 = off")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.out_root = args.out_root or args.dump_root
    meshes = MatterportMeshes(matterport_dir=args.matterport_dir, faces_cachedir=args.faces_cachedir,
                              face_subset=not args.full_house, poisson_mesh_dir=args.poisson_mesh_dir)
    scenes = list_scenes(args.dump_root)
    end = len(scenes) if args.end_idx is None else min(args.end_idx, len(scenes))
    todo = scenes[args.start_idx:end+1]
    print(f"{len(scenes)} houses in {args.dump_root}; processing [{args.start_idx}, {end}) = {len(todo)} houses "
          f"-> {osp.join(args.out_root, 'LDIs')}")

    for scene in tqdm(todo, desc="houses"):
        status = process_scene(scene, args, meshes)
        tqdm.write(f"{scene}: {status}")