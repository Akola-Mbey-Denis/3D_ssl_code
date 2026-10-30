import os
import numpy as np
# import habitat_sim
import time
import argparse
import json
from sklearn.neighbors import NearestNeighbors
import cv2
import torch
import PIL 
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # it works well.
from .paths import list_scenes_available
import torch.nn.functional as F


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




def compute_points_visibility(
    uv: np.ndarray,
    z: np.ndarray,
    depthmap: np.ndarray,
    depth_mask: np.ndarray,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Compute visibility mask of projected points in a target depth map using NumPy.

    Args:
        uv (np.ndarray): Projected pixel coordinates (N, 2), float.
        z (np.ndarray): Projected depth values (N,), float.
        depthmap (np.ndarray): Target depth map (H, W), float.

    Returns:
        np.ndarray: Boolean mask of shape (N,) indicating whether each projected
                    point is consistent with the target depth map.
    """
    H, W = depthmap.shape
    N = uv.shape[0]

    # Initialize output visibility mask
    visibility_mask = np.zeros(N, dtype=bool)

    # Split uv into separate arrays
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)

    # Bounds check (points must fall inside the depth map)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.01)
    
    # Filter valid indices
    u_valid, v_valid, z_valid = u[in_bounds], v[in_bounds], z[in_bounds]

    depth_values = depthmap[v_valid, u_valid]
    depth_mask_values = depth_mask[v_valid, u_valid]

    visible =  (z_valid <= depth_values + eps ) & depth_mask_values

    # Mark visible points in the full mask
    visibility_mask[in_bounds] = visible

    return visibility_mask




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

def compute_frustum_overlap(
        K,
        Rt1_c2w,
        depth1,
        Rt2_c2w,
        depth2,
        mask1,
        mask2,
        eps=1e-3,
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
            K=K,
            RT_c2w=Rt1_c2w,
            depth_mask=mask1,
        )

        world_points2 = project_to_world_points(
            depth=depth2,
            K=K,
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
            K=K,
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
            K=K,
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



def load_exr_depth(depth_path: str) -> np.ndarray:

    """
        Load a depth map from a .exr file using OpenCV.

        Args:
            depth_path (str): Path to the .exr depth file.

        Returns:
            np.ndarray: Depth map of shape (H, W), dtype float32.
    """
        
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    
    return depth.astype(np.float32)


def load_camera_params(file_path):
    """
    Load camera parameters (extrinsics and intrinsics) from a JSON file.

    Returns:
        K (np.ndarray): 3x3 intrinsic matrix
        RT_c2w (np.ndarray): 4x4 extrinsic matrix (camera to world)
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    RT_c2w = np.array(data["extrinsics"], dtype=np.float32)
    K = np.array(data["intrinsics"], dtype=np.float32)

    return K, RT_c2w

def has_sufficient_valid_depth(depth: np.ndarray, threshold: float = 0.10, min_fraction: float = 0.6) -> bool:
    """
    Check if a depth map has at least `min_fraction` of readings above `threshold`.

    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        threshold (float): Minimum depth value to be considered valid.
        min_fraction (float): Minimum fraction of valid depth values required.

    Returns:
        bool: True if fraction of valid depths >= min_fraction, else False.
    """
    num_valid = np.sum(depth > threshold)
    total_pixels = depth.size
    fraction_valid = num_valid / total_pixels

    return fraction_valid >= min_fraction



def is_valid_image(image_path, valid_pixels_frac=0.05):
    """
    Check if the fraction of black pixels in an RGB image is < valid_pixels_frac.

    Args:
        image_path (str): Path to the RGB image.
        valid_pixels_frac (float): Maximum allowed fraction of black pixels.

    Returns:
        bool: True if black pixel fraction < valid_pixels_frac, else False.
    """

    # Load image as RGB
    image = np.array(Image.open(image_path).convert("RGB"))

    # Ensure it's RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be an RGB image of shape (H, W, 3).")

    black_mask = np.all(image < 6, axis=-1)
    black_pixels = np.sum(black_mask)

    total_pixels = image.shape[0] * image.shape[1]
    fraction_black = black_pixels / total_pixels

    return fraction_black < valid_pixels_frac


if __name__ == "__main__":
    os.umask(2)

    parser = argparse.ArgumentParser(
        description="Compute pairwise camera overlap and occlusion for Habitat scenes."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/dma9300/preprocessed_habitat_matterport3d/",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of scenes to process.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="End index (exclusive) of scenes to process. Use -1 for all scenes.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Find scenes and determine scene range
    # ------------------------------------------------------------------
    scenes_data = list_scenes_available(base_output_dir=args.output_dir)

    num_scenes = len(scenes_data)
    start_idx = args.start_idx
    end_idx = num_scenes if args.end_idx == -1 else min(args.end_idx, num_scenes)

    if not (0 <= start_idx < end_idx <= num_scenes):
        raise ValueError(
            f"Invalid scene range: start_idx={start_idx}, "
            f"end_idx={end_idx}, num_scenes={num_scenes}"
        )

    print(
        f"Processing scenes {start_idx} to {end_idx - 1} "
        f"out of {num_scenes} total"
    )

    overlap_basedir = (
        "/projects/rps/dff6142/fouheylab/shared_datasets/"
        "dma9300/dataset_overlaps/habitat_overlaps"
    )

    # ------------------------------------------------------------------
    # Process scenes
    # ------------------------------------------------------------------
    for scene_data in tqdm(
        scenes_data[start_idx:end_idx],
        desc="Scenes",
    ):
        scene_dir = scene_data.output_dir
        scene_name = os.path.basename(os.path.dirname(scene_dir))

        print(f"\nProcessing scene: {scene_name}")

        # Paths
        rgb_dir = os.path.join(scene_dir, "rgbs")
        depth_dir = os.path.join(scene_dir, "depths")
        camera_dir = os.path.join(scene_dir, "cameras")

        overlaps_dir = os.path.join(overlap_basedir, scene_name)
        overlap_file = os.path.join(overlaps_dir, "camera_overlap.npy")
        occlusion_file = os.path.join(overlaps_dir, "camera_occlusion.npy")

        # Skip already processed scenes
        if os.path.exists(overlap_file) and os.path.exists(occlusion_file):
            print("Already processed. Skipping.")
            continue

        os.makedirs(overlaps_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Determine number of images
        # ------------------------------------------------------------------
        rgb_images = sorted(
            f for f in os.listdir(rgb_dir) if f.endswith(".jpeg")
        )
        num_images = len(rgb_images)

        # ------------------------------------------------------------------
        # Preload valid images
        #
        # valid_indices[k] gives the ORIGINAL image index.
        # depths[k], depth_masks[k], and cam_params[k] correspond to
        # that same valid image.
        # ------------------------------------------------------------------
        valid_indices = []
        depths = []
        depth_masks = []
        cam_params = []

        for image_idx in tqdm(
            range(num_images),
            desc="Preloading",
            leave=False,
        ):
            depth_path = os.path.join(
                depth_dir,
                f"{image_idx:05d}_depth.exr",
            )
            image_path = os.path.join(
                rgb_dir,
                f"{image_idx:05d}.jpeg",
            )
            camera_path = os.path.join(
                camera_dir,
                f"{image_idx:05d}_cam.json",
            )

            depth = load_exr_depth(depth_path)

            if not has_sufficient_valid_depth(
                depth,
                threshold=0.10,
                min_fraction=0.6,
            ):
                continue

            if not is_valid_image(image_path):
                continue

            valid_indices.append(image_idx)
            depths.append(depth)
            depth_masks.append(depth > 0)
            cam_params.append(load_camera_params(camera_path))

        num_valid = len(valid_indices)

        print(
            f"Valid images: {num_valid}/{num_images}"
        )

        # ------------------------------------------------------------------
        # Initialize overlap / occlusion matrices.
        #
        # These matrices use ORIGINAL image indices:
        #   camera_overlap[image_idx_i, image_idx_j]
        # ------------------------------------------------------------------
        camera_overlap = np.full(
            (num_images, num_images),
            -1.0,
            dtype=np.float32,
        )
        camera_occlusion = np.full(
            (num_images, num_images),
            -1.0,
            dtype=np.float32,
        )

        # Self-overlap
        for image_idx in valid_indices:
            camera_overlap[image_idx, image_idx] = 1.0
            camera_occlusion[image_idx, image_idx] = 0.0

        # ------------------------------------------------------------------
        # Compute pairwise overlap / occlusion
        # ------------------------------------------------------------------
        for valid_pos_i in tqdm(
            range(num_valid),
            desc="Computing overlaps",
            leave=False,
        ):
            # valid_pos_i indexes the compact preloaded lists
            image_idx_i = valid_indices[valid_pos_i]

            depth_i = depths[valid_pos_i]
            mask_i = depth_masks[valid_pos_i]
            _, RT_c2w_i = cam_params[valid_pos_i]

            for valid_pos_j in range(
                valid_pos_i + 1,
                num_valid,
            ):
                # valid_pos_j indexes the compact preloaded lists
                image_idx_j = valid_indices[valid_pos_j]

                depth_j = depths[valid_pos_j]
                mask_j = depth_masks[valid_pos_j]
                K_j, RT_c2w_j = cam_params[valid_pos_j]

                overlap = compute_frustum_overlap(
                    K=K_j,
                    Rt1_c2w=RT_c2w_i,
                    depth1=depth_i,
                    Rt2_c2w=RT_c2w_j,
                    depth2=depth_j,
                    mask1=mask_i,
                    mask2=mask_j,
                )

                # Store using ORIGINAL image indices
                camera_overlap[image_idx_i, image_idx_j] = (
                    overlap["overlap1"]
                )
                camera_overlap[image_idx_j, image_idx_i] = (
                    overlap["overlap2"]
                )

                camera_occlusion[image_idx_i, image_idx_j] = (
                    overlap["occlusion1"]
                )
                camera_occlusion[image_idx_j, image_idx_i] = (
                    overlap["occlusion2"]
                )

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------
        np.save(overlap_file, camera_overlap)
        np.save(occlusion_file, camera_occlusion)

        print(f"Saved overlap:   {overlap_file}")
        print(f"Saved occlusion: {occlusion_file}")

        # Explicitly release large arrays before processing next scene
        del depths
        del depth_masks
        del cam_params