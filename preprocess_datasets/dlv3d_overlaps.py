import os
import json
import argparse
import pickle as pkl
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Every view is center-cropped to IMAGE_SIZE x IMAGE_SIZE (short side resized to IMAGE_SIZE,
# like the 518 crops with aug_crop=-1) and EVERY pixel of that grid is sampled.
IMAGE_SIZE = 480
N_DEPTHS = 128        # depth samples per pixel ray, uniform (linspace) in [min_depth, max_depth]


def rotation_inverse(RT_c2w: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 camera-to-world matrix (rigid transform) using NumPy.

    Args:
        RT_c2w (np.ndarray): 4x4 camera-to-world extrinsic matrix [R | t; 0 0 0 1]

    Returns:
        np.ndarray: 4x4 world-to-camera matrix.
    """
    assert RT_c2w.shape == (4, 4), "RT_c2w must be 4x4"

    R = RT_c2w[:3, :3]  # Rotation part (3x3)
    t = RT_c2w[:3, 3]   # Translation part (3,)

    R_inv = R.T          # Inverse rotation
    t_inv = -R_inv @ t   # Adjusted translation

    RT_w2c = np.eye(4, dtype=RT_c2w.dtype)
    RT_w2c[:3, :3] = R_inv
    RT_w2c[:3, 3] = t_inv

    return RT_w2c


def points_inside_image(uv, image_size):
        """
        Check which projected points fall inside an image.

        Args:
            uv:
                Projected image coordinates, shape (N, 2).
                uv[:, 0] = x / column
                uv[:, 1] = y / row

            image_size:
                (H, W)

        Returns:
            inside:
                Boolean tensor of shape (N,).
        """
        H, W = image_size[:2]

        # Make sure uv is a tensor
        if not torch.is_tensor(uv):
            uv = torch.as_tensor(uv)

        x = uv[:, 0]
        y = uv[:, 1]

        inside = (
            torch.isfinite(x)
            & torch.isfinite(y)
            & (x >= 0)
            & (x < W)
            & (y >= 0)
            & (y < H)
        )

        return inside


def project_points(K: torch.Tensor, RT_c2w: np.ndarray, world_points: torch.Tensor, image_size: tuple):
    """
    Project 3D world points into camera image space (torch, so it runs on GPU if available).

    Args:
        K (torch.Tensor): (3, 3) intrinsic matrix.
        RT_c2w (np.ndarray): (4, 4) camera-to-world extrinsic matrix.
        world_points (torch.Tensor): (N, 3) points in world coordinates.
        image_size (tuple): (H, W) image dimensions.

    Returns:
        uv (torch.Tensor): (N, 2) projected pixel coordinates (float).
        z (torch.Tensor):  (N,) depth values in camera space.
    """
    # Invert RT to get world → camera
    RT_w2c = torch.from_numpy(rotation_inverse(RT_c2w)).to(world_points)  # (4, 4)

    # Transform world → camera
    cam_points = world_points @ RT_w2c[:3, :3].T + RT_w2c[:3, 3]  # (N, 3)

    # Depth in camera coordinates
    z = cam_points[:, 2]  # (N,)

    # Project into pixel space
    proj = cam_points @ K.to(world_points).T  # (N, 3)

    uv = proj[:, :2] / (proj[:, 2:3] + 1e-8)  # (N, 2)

    return uv, z


def sample_frustum_points(K: np.ndarray, RT_c2w: np.ndarray, image_size: tuple,
                          min_depth: float, max_depth: float, n_depths: int = N_DEPTHS,
                          device: str = "cpu") -> torch.Tensor:
    """
    Sample 3D world points filling a camera frustum: every pixel of the (H, W) grid,
    back-projected at n_depths depths in [min_depth, max_depth] (linspace).
    Same back-projection as project_to_world_points, with sampled depths instead of a depth map.

    Returns:
        torch.Tensor: (H * W * n_depths, 3) world points.
    """
    H, W = image_size

    # Full pixel grid (u = x, v = y), as in project_to_world_points
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pixel_coords = np.stack([u.flatten(), v.flatten(), np.ones(H * W)], axis=0)  # (3, HW)

    # Rays in camera coordinates (z = 1)
    rays = np.linalg.inv(K) @ pixel_coords  # (3, HW)

    # Depth samples, uniform in depth
    depths = np.linspace(min_depth, max_depth, n_depths)  # (D,)

    rays = torch.from_numpy(rays).float().to(device)
    depths = torch.from_numpy(depths).float().to(device)

    # Camera-space points (HW * D, 3)
    cam_points = (rays[:, :, None] * depths[None, None, :]).reshape(3, -1).T

    # Transform to world coordinates
    RT = torch.from_numpy(RT_c2w).float().to(device)
    world_points = cam_points @ RT[:3, :3].T + RT[:3, 3]

    return world_points


def compute_frustum_overlap(
        K1,
        K2,
        Rt1_c2w,
        Rt2_c2w,
        points1,
        points2,
        image_size,
        min_depth,
        max_depth,
    ):
        """
        Directional frustum overlap without depth maps.

        Convention (as in the depth-based version):
            View 1 = reference / target view.
            View 2 = candidate source view.

        overlap1:
            Fraction of view 2's frustum samples that land inside view 1's frustum
            (inside the image and min_depth <= z <= max_depth in view 1).
            -> camera_overlap[view1, view2]

        overlap2:
            Fraction of view 1's frustum samples that land inside view 2's frustum.
            -> camera_overlap[view2, view1]

        No occlusion: without depth there is no first-hit surface to test against.
        """

        # View 2 samples -> view 1  (overlap1, view 1 = target)
        uv21, z21 = project_points(K=K1, RT_c2w=Rt1_c2w, world_points=points2, image_size=image_size)
        inside1 = points_inside_image(uv21, image_size=image_size)
        valid21 = inside1 & (z21 >= min_depth) & (z21 <= max_depth)

        # View 1 samples -> view 2  (overlap2, view 2 = target)
        uv12, z12 = project_points(K=K2, RT_c2w=Rt2_c2w, world_points=points1, image_size=image_size)
        inside2 = points_inside_image(uv12, image_size=image_size)
        valid12 = inside2 & (z12 >= min_depth) & (z12 <= max_depth)

        return {
            "overlap1": valid21.float().mean().item(),
            "overlap2": valid12.float().mean().item(),
        }


def rgb_to_transforms_path(rgb_path: str) -> str:
    """<scene>/images_8/frame_00001.png -> <scene>/transforms.json"""
    return os.path.join(os.path.dirname(os.path.dirname(rgb_path)), "transforms.json")


def load_dl3dv_cameras(list_rgbs: List[str], image_size: int = IMAGE_SIZE):
    """
    Read DL3DV transforms.json (nerfstudio / OpenGL RT_c2w) and return, for every image in list_rgbs
    (same order; images without a pose in the json are dropped):
        rgb paths, K (for the image_size x image_size center crop), RT_c2w (OpenCV).

    Scale: COLMAP poses have an arbitrary scale per scene, so camera centers are
    re-centered and scaled so that their mean distance to the centroid is 1.
    min_depth / max_depth are expressed in this unit.
    """
    with open(rgb_to_transforms_path(list_rgbs[0]), "r") as f:
        meta = json.load(f)

    # json frames by image file name (json paths say images/..., files are in images_8/)
    frame_by_name = {os.path.basename(fr["file_path"]): fr for fr in meta["frames"]}

    # Real resolution on disk (intrinsics in the json may be for a different resolution)
    W_img, H_img = Image.open(list_rgbs[0]).size

    names, Ks, RTs = [], [], []
    for rgb_path in list_rgbs:
        fr = frame_by_name.get(os.path.basename(rgb_path))
        if fr is None:
            continue
        w, h = fr.get("w", meta["w"]), fr.get("h", meta["h"])
        fx, fy = fr.get("fl_x", meta["fl_x"]), fr.get("fl_y", meta["fl_y"])
        cx, cy = fr.get("cx", meta["cx"]), fr.get("cy", meta["cy"])

        # json resolution -> image on disk
        sx, sy = W_img / w, H_img / h
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy

        # resize short side to image_size, then center crop image_size x image_size
        s = image_size / min(W_img, H_img)
        W_new, H_new = round(W_img * s), round(H_img * s)
        left, top = (W_new - image_size) // 2, (H_new - image_size) // 2
        K = np.array([[fx * s, 0, cx * s - left],
                      [0, fy * s, cy * s - top],
                      [0, 0, 1]], dtype=np.float32)

        # OpenGL (x right, y up, z back) -> OpenCV (x right, y down, z forward)
        RT_c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        RT_c2w[2, :] *= -1
        RT_c2w = RT_c2w[np.array([1, 0, 2, 3]), :]
        RT_c2w[0:3, 1:3] *= -1
        names.append(rgb_path)
        Ks.append(K)
        RTs.append(RT_c2w)

    # Normalize scene scale
    RTs = np.stack(RTs)
    centers = RTs[:, :3, 3]
    centroid = centers.mean(0)
    scale = np.linalg.norm(centers - centroid, axis=1).mean()
    if scale < 1e-8:
        scale = 1.0
    RTs[:, :3, 3] = (centers - centroid) / scale

    return names, Ks, list(RTs)


def compute_scene_camera_overlaps(
    scene_files: Dict[str, str],
    overlap_basedir: str,
    min_depth: float,
    max_depth: float,
    start_idx: int = 0,
    end_idx: int = None,
):
    """
    Frustum-based camera overlaps for a subset of DL3DV scenes.

    Args:
        scene_files: dict mapping scene_name -> list of image paths (images_8/*)
        start_idx: start index of scenes to process
        end_idx: end index (exclusive) of scenes to process
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scene_names = list(scene_files.keys())
    if end_idx is None or end_idx > len(scene_names):
        end_idx = len(scene_names)

    image_size = (IMAGE_SIZE, IMAGE_SIZE)

    for scene_name in scene_names[start_idx:end_idx]:
        list_rgbs = scene_files[scene_name]

        overlaps_dir = os.path.join(overlap_basedir, scene_name)
        overlap_file = os.path.join(overlaps_dir, "camera_overlap.npy")

        # Skip already processed scenes
        if os.path.exists(overlap_file):
            print("Already processed. Skipping.")
            continue

        try:
            names, Ks, RTs = load_dl3dv_cameras(list_rgbs)
        except Exception as e:
            print(f"Skipping {scene_name}: {e}")
            continue

        

        os.makedirs(overlaps_dir, exist_ok=True)
        N = len(names)

        Ks_t = [torch.from_numpy(K).to(device) for K in Ks]

        camera_overlap = np.full((N, N), -1.0, dtype=np.float32)
        np.fill_diagonal(camera_overlap, 1.0)

        for i in tqdm(range(N), desc=f"Computing overlaps {scene_name}"):
            # Frustum samples: 480*480*256 points (~0.7 GB) per view,
            # so only two views' samples are kept in memory at a time
            points_i = sample_frustum_points(Ks[i], RTs[i], image_size, min_depth, max_depth, device=device)

            for j in range(i + 1, N):
                points_j = sample_frustum_points(Ks[j], RTs[j], image_size, min_depth, max_depth, device=device)

                overlap_dict = compute_frustum_overlap(
                    K1=Ks_t[i],
                    K2=Ks_t[j],
                    Rt1_c2w=RTs[i],
                    Rt2_c2w=RTs[j],
                    points1=points_i,
                    points2=points_j,
                    image_size=image_size,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                breakpoint()

                # Directional entries: camera_overlap[target, src]
                camera_overlap[i, j] = overlap_dict["overlap1"]   # target i, source j
                camera_overlap[j, i] = overlap_dict["overlap2"]   # target j, source i
                del points_j

            del points_i

        with open(os.path.join(overlaps_dir, "frame_names.pkl"), "wb") as f:
            pkl.dump(names, f)
        np.save(overlap_file, camera_overlap)   # saved last = scene done


def collect_scene_files(root_dir: str, image_dir: str = "images_8") -> Dict[str, List[str]]:
    """
    DL3DV layout: <root_dir>/<scene_hash>/{transforms.json, images_8/*.png}
    (root_dir = e.g. /scratch/dma9300/DL3DV-480P/5K)

    Returns: scene_name ("<scene_hash>") -> sorted list of all image paths of the scene
    """
    scene_files = {}
    for scene in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene)
        rgb_dir = os.path.join(scene_dir, image_dir)
        if not (os.path.isdir(rgb_dir) and os.path.exists(os.path.join(scene_dir, "transforms.json"))):
            continue
        files = sorted(
            os.path.join(rgb_dir, f)
            for f in os.listdir(rgb_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if len(files) > 0:
            scene_files[scene] = files
    return scene_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of scenes to process")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of scenes to process (exclusive)")
    parser.add_argument("--root_dir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dl3dv_datasets/5K/", help="Subset folder containing the scene hash folders")
    parser.add_argument("--scene_file", type=str, default=None, help="Pickle file containing scene files (default: dl3dv_scene_files_<subset>.pkl)")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Near plane (min depth along the rays)")
    parser.add_argument("--max_depth", type=float, default=8.0, help="Far plane (max depth along the rays)")
    args = parser.parse_args()

    root_dir = args.root_dir
    subset = os.path.basename(os.path.normpath(root_dir))   # e.g. 5K
    if args.scene_file is None:
        args.scene_file = f"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/dl3dv_scene_files_{subset}.pkl"
    overlap_basedir = os.path.join("/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/dl3dv_overlaps/", subset)

    if not os.path.exists(args.scene_file):
        all_scene_files = collect_scene_files(root_dir=root_dir)

        with open(f"{args.scene_file}", "wb") as f:
            pkl.dump(all_scene_files, f)

    with open(args.scene_file, "rb") as f:
        all_scene_files = pkl.load(f)

    compute_scene_camera_overlaps(all_scene_files, overlap_basedir=overlap_basedir,
                                  min_depth=args.min_depth, max_depth=args.max_depth,
                                  start_idx=args.start_idx, end_idx=args.end_idx)