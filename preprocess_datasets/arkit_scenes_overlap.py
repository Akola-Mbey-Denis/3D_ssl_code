"""
ARKitScenes overlaps, same as the ScanNet / Hypersim / MegaDepth overlap scripts, on the frames written by
prepare_arkitscenes.py.

  <dump_root>/<fold>/<video_id>/rgb/<video_id>_<ts>.jpg, depth/<video_id>_<ts>.npy (meters, 0 = invalid),
  cameras.npz (frame_ids, K (N,3,3), c2w (N,4,4) OpenCV c2w; upright)
  A scene is "<fold>/<video_id>" (fold = Training / Validation).

Every frame is cropped/resized to 518 x 518 (_crop_resize_if_necessary, centered on the principal point,
deterministic) and compute_frustum_overlap is run on every pair.

  <overlap_basedir>/<scene>/frame_names.pkl      row / col i of the matrices <-> frame_names[i] (rgb file name)
                           camera_occlusion.npy  camera_occlusion[target, src]
                           camera_overlap.npy    camera_overlap[target, src], written last (its presence = scene done)

Usage:
  python arkitscenes_overlaps.py --start_idx 0 --end_idx 9      # scenes 0..9 (end inclusive)
"""
import os
import argparse
import pickle as pkl
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from tqdm import tqdm
import open3d as o3d

import utils.cropping as cropping

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

def project_points(K: np.ndarray, RT_c2w: np.ndarray, world_points: np.ndarray, image_size: tuple):
    """
    Project 3D world points into camera image space using NumPy.

    Args:
        K (np.ndarray): (3, 3) intrinsic matrix.
        RT_c2w (np.ndarray): (4, 4) camera-to-world extrinsic matrix.
        world_points (np.ndarray): (N, 3) points in world coordinates.
        image_size (tuple): (H, W) image dimensions.

    Returns:
        uv (np.ndarray): (N, 2) projected pixel coordinates (float).
        z (np.ndarray):  (N,) depth values in camera space.
    """
    H, W = image_size

    # Invert RT to get world → camera
    RT_w2c = rotation_inverse(RT_c2w)  # (4, 4)

    N = world_points.shape[0]

    # Convert world points to homogeneous coordinates
    world_h = np.hstack([world_points, np.ones((N, 1), dtype=world_points.dtype)])  # (N, 4)

    # Transform world → camera
    cam_points = (RT_w2c @ world_h.T).T[:, :3]  # (N, 3)

    # Depth in camera coordinates
    z = cam_points[:, 2] # (N,)

    # Project into pixel space
    proj = (K @ cam_points.T).T #(N, 3)

    uv = proj[:, :2] / (proj[:, 2:3] + 1e-8) # (N, 2)

    return uv, z

def project_to_world_points(depth: np.ndarray, K: np.ndarray, RT_c2w: np.ndarray,depth_mask: np.ndarray,) -> np.ndarray:
    """
    Project a depth map into 3D world coordinates using intrinsics and 
    a camera-to-world transformation matrix.

    Args:
        depth (np.ndarray): Depth map of shape (H, W), float.
        K (np.ndarray): Camera intrinsic matrix of shape (3, 3), float.
        RT_c2w (np.ndarray): Extrinsic matrix of shape (4, 4), mapping camera → world, float.
        depth_mask (np.ndarray): Boolean mask of valid depth values, shape (H, W).

    Returns:
        np.ndarray: 3D world points of shape (N, 3), where N is the number of valid depth pixels.
    """
    H, W = depth.shape

    # Generate pixel coordinates (u = x, v = y)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape (H, W)
    
    # Flatten pixel grid and depth
    u_flat, v_flat = u.flatten(), v.flatten()
    depth_flat = depth.flatten()

    # Apply valid depth mask
    mask_flat = depth_mask.flatten()
    u_valid, v_valid, depth_valid = u_flat[mask_flat], v_flat[mask_flat], depth_flat[mask_flat]

    # Build homogeneous pixel coordinates (3, N)
    pixel_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)  # (3, N)

    # Backproject to camera coordinates
    K_inv = np.linalg.inv(K)
    cam_points = K_inv @ pixel_coords  # (3, N)

    # Scale rays by depth → camera space points (N, 3)
    cam_points = (cam_points * depth_valid[np.newaxis, :]).T

    # Convert to homogeneous coordinates (N, 4)
    cam_points_h = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])

    # Transform to world coordinates
    world_points_h = (RT_c2w @ cam_points_h.T).T  # (N, 4)
    world_points = world_points_h[:, :3]  # (N, 3)

    return world_points.astype(np.float32)

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

def pixel_to_ndc(uv, image_size):
        """
        Convert pixel coordinates to NDC coordinates expected by
        grid_sample(..., align_corners=True).

        Args:
            uv: (N, 2), pixel coordinates (u, v)
            image_size: (H, W)

        Returns:
            uv_ndc: (N, 2), coordinates in [-1, 1]
        """
        H, W = image_size

        u = uv[:, 0]
        v = uv[:, 1]

        u_ndc = 2.0 * u / (W - 1) - 1.0
        v_ndc = 2.0 * v / (H - 1) - 1.0

        return np.stack([u_ndc, v_ndc], axis=1)

def interpolate_depth(depth, uv):
        """
        Interpolate depth values at projected image coordinates.

        Args:
            depth:
                (H, W) depth map.

            uv:
                (N, 2) coordinates in NDC:
                [-1, 1] x [-1, 1]
                where uv[:, 0] = x and uv[:, 1] = y.

        Returns:
            sampled_depth:
                (N,) depth values.
        """
        uv  = pixel_to_ndc(uv,depth.shape)

        if not torch.is_tensor(depth):
            depth = torch.from_numpy(depth)

        if not torch.is_tensor(uv):
            uv = torch.from_numpy(uv)

        depth = depth.float()
        uv = uv.float()

        device = depth.device
        uv = uv.to(device)

        H, W = depth.shape

        # grid_sample expects:
        #   (N, H_out, W_out, 2)
        #
        # We want one sample for each uv point:
        #   (1, N, 1, 2)
        grid = uv.view(1, -1, 1, 2)

        depth_input = depth.unsqueeze(0).unsqueeze(0)
        # (1, 1, H, W)

        sampled = F.grid_sample(
            depth_input,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # (1, 1, N, 1) -> (N,)
        sampled_depth = sampled[0, 0, :, 0]

        return sampled_depth

def compute_frustum_overlap(
        K1,
        K2,
        Rt1_c2w,
        depth1,
        Rt2_c2w,
        depth2,
        mask1,
        mask2,
        eps=5e-2,
    ):
        """
        Compute pairwise depth-based overlap and directional occlusion.

        Convention:
            View 1 = reference / target view.
            View 2 = candidate source view.

        overlap1:
            Fraction of valid surfaces in view 2 that project into a valid
            observed-depth region of view 1.

        occlusion1:
            Fraction of overlapping surfaces from view 2 that are behind
            view 1's first-hit depth.

            This answers:
                "How much of what view 2 sees is hidden from view 1?"

            High occlusion1 means view 2 contains more surfaces that are
            hidden from target view 1.

        overlap2:
            Fraction of valid surfaces in view 1 that project into a valid
            observed-depth region of view 2.

        occlusion2:
            Fraction of overlapping surfaces from view 1 that are behind
            view 2's first-hit depth.

        Returns:
            dict:
                overlap1
                occlusion1
                overlap2
                occlusion2
        """

        # =========================================================
        # 1. Valid depth pixels
        # =========================================================

        depth1 = np.asarray(depth1)
        depth2 = np.asarray(depth2)
        mask1 = np.asarray(mask1).astype(bool)
        mask2 = np.asarray(mask2).astype(bool)

        mask1 = (
            mask1
            & np.isfinite(depth1)
            & (depth1 > 0.01)
        )

        mask2 = (
            mask2
            & np.isfinite(depth2)
            & (depth2 > 0.01)
        )

        num_surface1 = np.count_nonzero(mask1)
        num_surface2 = np.count_nonzero(mask2)

        if num_surface1 == 0 or num_surface2 == 0:
            return {
                "overlap1": 0.0,
                "occlusion1": 0.0,
                "overlap2": 0.0,
                "occlusion2": 0.0,
            }

        # =========================================================
        # 2. Convert first-hit depth maps to world-space surfaces
        # =========================================================

        world_points1 = project_to_world_points(
            depth=depth1,
            K=K1,
            RT_c2w=Rt1_c2w,
            depth_mask=mask1,
        )

        world_points2 = project_to_world_points(
            depth=depth2,
            K=K2,
            RT_c2w=Rt2_c2w,
            depth_mask=mask2,
        )

        # Make sure downstream operations are NumPy.
        world_points1 = np.asarray(world_points1)
        world_points2 = np.asarray(world_points2)

        # =========================================================
        # 3. View 1 surfaces -> View 2
        #
        # Used for:
        #   overlap2
        #   occlusion2
        # =========================================================

        uv12, z12 = project_points(
            K=K2,
            RT_c2w=Rt2_c2w,
            world_points=world_points1,
            image_size=depth2.shape,
        )

        uv12 = np.asarray(uv12)
        z12 = np.asarray(z12).reshape(-1)

        inside2 = np.asarray(points_inside_image(uv12, image_size=depth2.shape)).astype(bool)

        # breakpoint()

        target_depth2 = np.asarray(interpolate_depth(depth2,uv12)).reshape(-1)

        target_mask2 = (
            np.asarray(
                interpolate_depth(
                    mask2.astype(np.float32),
                    uv12,
                )
            ).reshape(-1)
            > 0.5
        )

        valid12 = (
            inside2
            & target_mask2
            & np.isfinite(z12)
            & (z12 > 0.01)
            & np.isfinite(target_depth2)
            & (target_depth2 > 0.01)
        )

        # ---------------------------------------------------------
        # View-1 surfaces that are behind view-2's first surface
        # ---------------------------------------------------------



        occluded12 = (
            valid12
            & (z12 > target_depth2 + eps)
        )

        # =========================================================
        # 4. View 2 surfaces -> View 1
        #
        # Used for:
        #   overlap1
        #   occlusion1
        #
        # THIS IS THE IMPORTANT DIRECTION WHEN VIEW 1 IS TARGET.
        # =========================================================

        uv21, z21 = project_points(
            K=K1,
            RT_c2w=Rt1_c2w,
            world_points=world_points2,
            image_size=depth1.shape,
        )

        uv21 = np.asarray(uv21)
        z21 = np.asarray(z21).reshape(-1)

        inside1 = np.asarray(
            points_inside_image(
                uv21,
                image_size=depth1.shape,
            )
        ).astype(bool)

        target_depth1 = np.asarray(interpolate_depth(depth1, uv21)).reshape(-1)

        target_mask1 = (
            np.asarray(
                interpolate_depth(
                    mask1.astype(np.float32),
                    uv21,
                )
            ).reshape(-1)
            > 0.5
        )

        valid21 = (
            inside1
            & target_mask1
            & np.isfinite(z21)
            & (z21 > 0.01)
            & np.isfinite(target_depth1)
            & (target_depth1 > 0.01)
        )


        # ---------------------------------------------------------
        # View-2 surfaces that are behind view-1's first surface
        # ---------------------------------------------------------


        occluded21 = (
            valid21
            & (z21 > target_depth1 + eps)
        )

        # =========================================================
        # 5. Directional overlap
        # =========================================================

        num_overlap12 = np.count_nonzero(valid12)
        num_overlap21 = np.count_nonzero(valid21)

        # ---------------------------------------------------------
        # View 1 = TARGET
        #
        # How much of view 2 is testable from view 1?
        # ---------------------------------------------------------

        overlap1 = (
            num_overlap21 / num_surface2
            if num_surface2 > 0
            else 0.0
        )

        # ---------------------------------------------------------
        # View 2 = TARGET
        #
        # How much of view 1 is testable from view 2?
        # ---------------------------------------------------------

        overlap2 = (
            num_overlap12 / num_surface1
            if num_surface1 > 0
            else 0.0
        )

        # =========================================================
        # 6. Directional occlusion
        # =========================================================

        # ---------------------------------------------------------
        # TARGET = VIEW 1
        #
        # Of the view-2 surfaces overlapping view 1,
        # what fraction is hidden behind view 1?
        # ---------------------------------------------------------

        occlusion1 = (
            np.count_nonzero(occluded21) / num_overlap21
            if num_overlap21 > 0
            else 0.0
        )

        # ---------------------------------------------------------
        # TARGET = VIEW 2
        #
        # Of the view-1 surfaces overlapping view 2,
        # what fraction is hidden behind view 2?
        # ---------------------------------------------------------

        occlusion2 = (
            np.count_nonzero(occluded12) / num_overlap12
            if num_overlap12 > 0
            else 0.0
        )

        return {
            "overlap1": float(overlap1),
            "occlusion1": float(occlusion1),
            "overlap2": float(overlap2),
            "occlusion2": float(occlusion2),
        }

def load_image(file_path):

    """
    Load a JPEG image and return as a NumPy array (H, W, 3) in RGB order.
    """

    img = Image.open(file_path).convert("RGB")


    return np.array(img)

def _crop_resize_if_necessary(image, depthmap, intrinsics, resolution, aug_crop = 16,  mask = None, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        # breakpoint()
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        # breakpoint()
        # assert min_margin_x > W/5, f'Bad principal point in view={info}'
        # assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if aug_crop > 1:
            target_resolution += rng.integers(0, aug_crop)
        image, depthmap, intrinsics, mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, mask = mask)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, mask  = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)
        
        # breakpoint()
        return image, depthmap, intrinsics2, mask

def rgbd_to_pointcloud(rgb, depth, K, RT_c2w, mask=None,  save_path="arkitscenes_point_cloud.ply"):
    """
    Project RGB-D image to colored world point cloud.
    
    Args:
        rgb: (H, W, 3) RGB image, uint8 [0-255] or float [0-1]
        depth: (H, W) depth map
        K: (3, 3) camera intrinsic matrix
        RT_c2w: (4, 4) cam2world matrix
        mask: (H, W) optional binary mask, True/1 = valid
        save_path: optional path to save PLY file
    
    Returns:
        points: (N, 4) world coordinates (homogeneous)
        colors: (N, 3) RGB colors [0-1]
    """
    H, W = depth.shape
    
    # Create pixel grid (homogeneous)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones((H, W))
    pixels = np.stack([u, v, ones], axis=-1).reshape(-1, 3)  # (N, 3)
    z = depth.flatten() 
    
    # Filter valid depth
    valid = z > 0
    
    # Apply mask if provided
    if mask is not None:
        valid = valid & mask.flatten().astype(bool)
    
    pixels = pixels[valid]
    z = z[valid]
    
    # Unproject: p_cam = z * K_inv @ [u, v, 1]
    K_inv = np.linalg.inv(K)
    pts_cam = (K_inv @ pixels.T).T * z[:, None]  # (N, 3)
    
    # Homogeneous camera coordinates
    pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])  # (N, 4)
    
    # Transform to world
    pts_world = (RT_c2w @ pts_cam_h.T).T  # (N, 4)
    
    # Get colors
    colors = rgb.reshape(-1, 3)[valid]
    if colors.max() > 1:
        colors = colors / 255.0
    
    # Save with Open3D
    if save_path:
        print(f"Saving colored point cloud - file {save_path}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_path, pcd)


def collect_scene_files(root_dir: str, folds=("Training", "Validation")) -> Dict[str, List[str]]:
    """
    Preprocessed ARKitScenes captures: "<fold>/<video_id>" -> rgb file paths of the capture, sorted by
    timestamp. Only captures whose preprocessing finished (cameras.npz present) are listed.

    Args:
        root_dir (str): output folder of prepare_arkitscenes.py (<fold>/<video_id>/...).

    Returns:
        Dict[str, List[str]]: "<fold>/<video_id>" -> [rgb file paths, sorted by timestamp]
    """
    scene_files = {}
    for fold in folds:
        fold_dir = os.path.join(root_dir, fold)
        if not os.path.isdir(fold_dir):
            continue
        for video_id in sorted(os.listdir(fold_dir)):
            scene_dir = os.path.join(fold_dir, video_id)
            if not os.path.exists(os.path.join(scene_dir, "cameras.npz")):
                continue
            rgb_dir = os.path.join(scene_dir, "rgb")
            names = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
            names.sort(key=lambda f: float(os.path.splitext(f)[0].split("_", 1)[1]))   # <video_id>_<ts>.jpg
            scene_files[f"{fold}/{video_id}"] = [os.path.join(rgb_dir, f) for f in names]
    return scene_files


def load_dumped_scene(scene_dir, rgb_paths, resolution=(518, 518)):
    """
    Read one capture written by prepare_arkitscenes.py, in the order of `rgb_paths`
    (the scene's list from collect_scene_files), and crop/resize every frame to `resolution`
    (default 518 x 518) with _crop_resize_if_necessary: centered on the principal point,
    K updated to the crop. RT_c2w is unchanged by cropping.

    Returns:
        names:      [rgb file names]                      -> row/col i of the overlap matrices
        depths:     [(518, 518) float32, meters, 0 = invalid]
        masks:      [(518, 518) bool]                     valid depth after the crop
        cam_params: [(K (3,3), RT_c2w (4,4))]             K at the cropped resolution
    """
    cameras = np.load(os.path.join(scene_dir, "cameras.npz"))
    video_id = os.path.basename(os.path.normpath(scene_dir))
    frame_index = {str(fid): i for i, fid in enumerate(cameras["frame_ids"])}      # <ts> -> row of K / c2w

    rng = np.random.default_rng(seed=42)
    names = [os.path.basename(p) for p in rgb_paths]
    depths, masks, cam_params = [], [], []
    for rgb_path, rgb_name in zip(rgb_paths, names):
        stem = os.path.splitext(rgb_name)[0]                                          # <video_id>_<ts>
        i = frame_index[stem[len(video_id) + 1:]]
        image = load_image(rgb_path)
        depth = np.load(os.path.join(scene_dir, "depth", stem + ".npy")).astype(np.float32)
        K = np.asarray(cameras["K"][i], np.float32)
        RT_c2w = np.asarray(cameras["c2w"][i], np.float32)
        mask = (depth > 0).astype(np.int32)

        # aug_crop=-1: deterministic crop (no random size jitter) for overlap computation
        image_new, depth_new, K_new, mask_new = _crop_resize_if_necessary(
            image=image, depthmap=depth, intrinsics=K, resolution=resolution,
            aug_crop=-1, mask=mask, rng=rng, info=rgb_name)

        depth_new = np.asarray(depth_new, np.float32)
        depths.append(depth_new)
        masks.append((np.asarray(mask_new) > 0) & (depth_new > 0))
        cam_params.append((np.asarray(K_new, np.float32), RT_c2w))

        # rgbd_to_pointcloud(rgb = np.asarray(image_new), depth = depth_new, K = K_new, RT_c2w = RT_c2w, mask = mask_new,  save_path="arkitscenes_point_cloud.ply")
        # breakpoint()


    return names, depths, masks, cam_params


def compute_scene_camera_overlaps(
    scene_files: Dict[str, List[str]],
    dump_root: str,
    overlap_basedir: str,
    start_idx: int = 0,
    end_idx: int = None
):
    """
    Compute camera overlaps and covisibility for a subset of preprocessed ARKitScenes captures.
    Images, depths and cameras come from prepare_arkitscenes.py (upright, depth in meters,
    K and RT_c2w already processed), so no camera preprocessing happens here.

    Args:
        scene_files: "<fold>/<video_id>" -> rgb file paths (collect_scene_files on the preprocessed data)
        dump_root: folder with <fold>/<video_id>/{rgb, depth, cameras.npz}
        overlap_basedir: output folder, one sub-folder per scene (<fold>/<video_id>)
        start_idx: start index of scenes to process
        end_idx: end index (inclusive) of scenes to process
    """
    scene_names = list(scene_files.keys())
    if end_idx is None or end_idx >= len(scene_names):
        end_idx = len(scene_names) - 1

    # Loop over only the specified range (end inclusive)
    for scene_name in scene_names[start_idx:end_idx + 1]:

        scene_dir = os.path.join(dump_root, scene_name)
        overlaps_dir = os.path.join(overlap_basedir, scene_name)
        file_path = os.path.join(overlaps_dir, "camera_overlap.npy")

        if os.path.exists(file_path):
            print(f"{scene_name}: already processed!")
            continue
        if not os.path.exists(os.path.join(scene_dir, "cameras.npz")):
            print(f"{scene_name}: not preprocessed yet, skipping")
            continue

        os.makedirs(overlaps_dir, exist_ok=True)

        # every dumped frame is already valid: all frames take part, cropped to 518 x 518
        names, depths, masks, cam_params = load_dumped_scene(scene_dir, scene_files[scene_name],
                                                             resolution=(518, 518))
        N = len(names)

        # Initialize matrices
        camera_overlap = np.full((N, N), -1.0, dtype=np.float32)
        camera_occlusion = np.full((N, N), -1.0, dtype=np.float32)
        np.fill_diagonal(camera_overlap, 1.0)
        np.fill_diagonal(camera_occlusion, 0.0)

        for target_idx in tqdm(range(N), desc=f"Computing overlaps {scene_name}"):
            depth_i = depths[target_idx]
            K_i, RT_c2w_i = cam_params[target_idx]
            mask_i = masks[target_idx]
            
            for src_idx in range(target_idx + 1, N):
                depth_j = depths[src_idx]
                K_j, RT_c2w_j = cam_params[src_idx]
                mask_j = masks[src_idx]

                overlap_dict = compute_frustum_overlap(
                    K1=K_i,
                    K2=K_j,
                    mask1 = mask_i,
                    mask2 = mask_j,
                    Rt1_c2w=RT_c2w_i,
                    depth1=depth_i,
                    Rt2_c2w=RT_c2w_j,
                    depth2=depth_j,
                )

                # breakpoint()

               # Fill symmetric entries
                camera_overlap[target_idx, src_idx] = overlap_dict["overlap1"]
                camera_overlap[src_idx, target_idx] = overlap_dict["overlap2"]
                camera_occlusion[target_idx, src_idx] =  overlap_dict["occlusion1"]
                camera_occlusion[src_idx, target_idx] =  overlap_dict["occlusion2"]

        # row/col i of both matrices <-> names[i] (rgb file name <video_id>_<ts>.jpg)
        # with open(os.path.join(overlaps_dir, "frame_names.pkl"), "wb") as f:
        #     pkl.dump(names, f)
        np.save(os.path.join(overlaps_dir, "camera_occlusion.npy"), camera_occlusion)
        np.save(file_path, camera_overlap)            # written last: marks the scene as done



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of scenes to process")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of scenes to process (inclusive)")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/arkitscenes_overlap_scene_files.pkl", help="Pickle containing the scene files of the preprocessed dataset")
    parser.add_argument("--dump_root", type=str, default="/scratch/dma9300/preprocessed_arkitscenes", help="Output folder of prepare_arkitscenes.py (the preprocessed dataset)")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/arkitscenes_overlaps/", help="Where to save the overlap matrices")
    args = parser.parse_args()

    # scene dictionary from the PREPROCESSED dataset: "<fold>/<video_id>" -> its rgb file paths
    if not os.path.exists(args.scene_file):

        all_scene_files = collect_scene_files(root_dir = args.dump_root)

        with open(f"{args.scene_file}", "wb") as f:
            pkl.dump(all_scene_files, f)


    with open(args.scene_file, "rb") as f:
        all_scene_files = pkl.load(f)


    compute_scene_camera_overlaps(
        all_scene_files,
        dump_root=args.dump_root,
        overlap_basedir=args.overlap_basedir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )