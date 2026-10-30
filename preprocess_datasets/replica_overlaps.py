import os
import numpy as np
import time
import argparse
import json
import cv2
import torch
import PIL 
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # it works well.
# from paths import list_scenes_available
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import imageio.v2 as imageio
from typing import Dict, List
from collections import defaultdict
from typing import Dict, List
import open3d as o3d
import torch.nn.functional as F



def view_camera_parameters(point_info):
    """
    Returns: (RT_cam2world, K) both in OpenGL coordinate system:
        +X right, +Y up, -Z forward.
    https://github.com/nileshkulkarni/mv_drdf/blob/master/rgbd_drdf/data_utils/base_rgbd.py#L215

    """

    # ---- Extract metadata ----
    location = point_info["camera_location"]      # (Tx, Tz, Ty) from dataset
    rotation = point_info["camera_rotation_final"]      # Blender XYZ euler
    W = point_info["resolution"]
    H = point_info["resolution"]
    fov = point_info["field_of_view_rads"]


    # Build rotation matrix from Euler angles in 'sxyz' order
    R_euler = R.from_euler('xyz', rotation).as_matrix()  # shape (3,3)

    # Multiply by Rx(pi) to flip the camera coordinate frame
    Rx = R.from_euler('x', np.pi).as_matrix()
    R_cam2world = R_euler @ Rx

    # Build 4x4 cam2world matrix
    RT_4x4 = np.eye(4, dtype=np.float32)
    RT_4x4[:3, :3] = R_cam2world
    RT_4x4[:3, 3] = location

    K = get_intrinsics_from_fov(fov = fov, W = W, H = H)

    return RT_4x4, K



def get_intrinsics_from_fov(fov, W, H, degrees=False):
    """
    Recover camera intrinsics K from a single field of view (Taskonomy-style).
    Taskonomy provides horizontal FOV.

    Args:
        fov: horizontal FOV (radians by default, or degrees if degrees=True)
        W: image width in pixels
        H: image height in pixels
        degrees: set True if fov is provided in degrees

    Returns:
        K: (3,3) intrinsic matrix
    """
    if degrees:
        fov = np.radians(fov)

    # Horizontal focal length
    fx = W / (2 * np.tan(fov / 2))

    # Convert HFOV -> VFOV
    fov_y = 2 * np.arctan(np.tan(fov / 2) * (H / W))

    # Vertical focal length
    fy = H / (2 * np.tan(fov_y / 2))

    # Principal point (center pixel)
    cx = W / 2
    cy = H / 2

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    return K


def compute_opengl_projection_matrix(fov, znear, zfar, aspect_ratio=1.0, degrees=False):
    """
    Compute OpenGL-style perspective projection matrix (KNDC).

    Args:
        fov: vertical field of view (radians by default, or degrees if `degrees=True`)
        znear: near clipping plane
        zfar: far clipping plane
        aspect_ratio: width / height (default 1.0 for square pixels)
        degrees: if True, `fov` is in degrees

    Returns:
        K: (4, 4) OpenGL projection matrix
    """
    if degrees:
        fov = np.radians(fov)

    tan_half_fov = np.tan(fov / 2)

    max_y = tan_half_fov * znear
    min_y = -max_y
    max_x = max_y * aspect_ratio
    min_x = -max_x

    # OpenGL uses right-handed system, z in [-1, 1]
    K = np.zeros((4, 4), dtype=np.float32)
    K[0, 0] = 2 * znear / (max_x - min_x)
    K[1, 1] = 2 * znear / (max_y - min_y)
    K[0, 2] = (max_x + min_x) / (max_x - min_x)  # 0 for symmetric
    K[1, 2] = (max_y + min_y) / (max_y - min_y)  # 0 for symmetric
    K[2, 2] = -(zfar + znear) / (zfar - znear)   # OpenGL: z mapped to [-1,1]
    K[2, 3] = -2 * zfar * znear / (zfar - znear)
    K[3, 2] = -1.0

    return K



def opengl_kndc_to_opencv_K(K_ndc, image_size):
    """
    Convert an OpenGL KNDC (NDC) projection matrix to an OpenCV-style intrinsic matrix.

    Args:
        K_ndc: (4,4) OpenGL projection matrix
        image_size: (H, W) tuple
    
    Returns:
        K: (3,3) OpenCV intrinsic matrix
    """
    H, W = image_size
    s = min(H, W)

    # Extract normalized focal lengths and principal point offsets
    fx_ndc = K_ndc[0, 0]
    fy_ndc = K_ndc[1, 1]
    px_ndc = K_ndc[0, 2]
    py_ndc = K_ndc[1, 2]

    # Convert NDC units to pixels
    fx = fx_ndc * s / 2.0
    fy = fy_ndc * s / 2.0
    cx = W / 2.0 - px_ndc * s / 2.0
    cy = H / 2.0 - py_ndc * s / 2.0

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)

    return K






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



def rgbd_to_pointcloud(rgb, depth, K, RT_c2w, mask=None,  save_path="replica_point_cloud.ply"):
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





def interpolate_depth(depth_map, coords):
    """
    Interpolate depth values at specified coordinates.
    
    Args:
        depth_map: (H, W) depth tensor
        coords: (N, 2) normalized coordinates in [-1, 1]
                where coords[:, 0] is x and coords[:, 1] is y
    
    Returns:
        depths: (N,) interpolated depth values
    """
    # Add batch and channel dimensions for grid_sample
    depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    coords_grid = coords.unsqueeze(0).unsqueeze(0)   # (1, 1, N, 2)
    
    # Interpolate
    sampled_depth = F.grid_sample(
        depth_map, 
        coords_grid,
        mode='bilinear',
        padding_mode='zeroes',
        align_corners=True
    )

    ones = torch.ones_like(depth_map)
    in_depth = F.grid_sample(
        ones,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)

    in_depth = in_depth.permute(1, 2, 0)
    sampled_depth[in_depth < 0.99] = 0.00
    
    depths = sampled.squeeze()
    
    return depths


def load_image(file_path):

    """
    Load a JPEG image and return as a NumPy array (H, W, 3) in RGB order.
    """

    img = Image.open(file_path).convert("RGB")

    return np.array(img)


# def load_depth(path):
#     """
#     Load quantized raycast depth and return metric depth in meters.
#     Depth is expected to be stored as integers with 1/512 m resolution.

#     Args:
#         path (str): Path to the depth file (e.g., PNG, uint16).

#     Returns:
#         depth_m (np.ndarray, float32): Depth map in meters.
#     """
#     # Read image (uint16 expected)
#     depth_int = imageio.imread(path)

#     breakpoint()

#     # Convert to float (metric depth)
#     depth_m = depth_int.astype(np.float32) / 512.0

#     return depth_m

def load_depth_and_mask(path):
    """
        Load z-buffer depth (uint16) and return:
        - metric depth in meters
        - valid mask (True = valid depth)

        Depth is expected to be stored as uint16 integers
        with increments of 1/512 m. Invalid pixels typically
        have value 65535 (2**16 - 1).

        Args:
            path (str): Path to the depth file (PNG, uint16).

        Returns:
            depth_m (np.ndarray, float32): Depth map in meters.
            mask (np.ndarray, bool): Valid depth mask.
    """

    # Load uint16 depth
    depth_int = imageio.imread(path).astype(np.uint16)

    # Invalid depth value
    invalid_val = (2**16 - 1)

    # Valid mask
    mask = depth_int < invalid_val

    # Convert to metric depth (meters)
    depth_m = depth_int.astype(np.float32) / 512.0

    # replace invalid with 0 
    depth_m[~mask] = 0.00

    return depth_m, mask


def load_valid_mask(mask_path: str) -> np.ndarray:
    """
    Load a 1-channel 8-bit PNG as a boolean mask.
    0 -> False (missing geometry)
    255 -> True (valid geometry)
    
    Args:
        png_path: Path to the PNG file.
    
    Returns:
        mask: np.ndarray of shape (H, W), dtype=bool
    """
    img = imageio.imread(mask_path)  # shape (H, W), dtype=uint8
    mask = img.astype(bool)      # 0->False, 255->True
    return mask




def load_camera_params(file_path):
    """
    Load camera parameters (extrinsics and intrinsics) from a JSON file.

    Returns:
        K (np.ndarray): 3x3 intrinsic matrix
        RT_c2w (np.ndarray): 4x4 extrinsic matrix (camera to world)
    """
    with open(file_path, "r") as f:
        data = json.load(f)


    return data

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


def rgb_to_depth_path(rgb_path: str) -> str:
    """
    Convert an RGB file path into a depth_z_buffer depth file path by replacing:
    - the first 'rgb' directory with 'depth_euclidean'
    - the 'rgb' in the filename with 'depth_euclidean'
    """
    # Replace the first 'rgb' directory
    depth_path = rgb_path.replace("/rgb/", "/depth_zbuffer/", 1)

    # Replace 'rgb' in the filename
    dir_name, filename = os.path.split(depth_path)
    filename = filename.replace("rgb", "depth_zbuffer")
    
    return os.path.join(dir_name, filename)


def rgb_to_mask_path(rgb_path: str) -> str:
    """
    Convert an RGB file path into a Euclidean depth file path by replacing:
    - the first 'rgb' directory with 'depth_euclidean'
    - the 'rgb' in the filename with 'depth_euclidean'
    """
    # point_9_view_9_domain_mask_valid.png
    # Replace the first 'rgb' directory
    depth_path = rgb_path.replace("/rgb/", "/mask_valid/", 1)

    # Replace 'rgb' in the filename
    dir_name, filename = os.path.split(depth_path)
    filename = filename.replace("rgb", "mask_valid")
    
    return os.path.join(dir_name, filename)



def rgb_to_pointinfo_path(rgb_path: str) -> str:
    """
    Convert an RGB file path into a point_info file path by replacing:
    - the first 'rgb' directory with 'point_info'
    - the 'rgb' in the filename with 'point_info'
    - the file extension to .json

    """
    # Replace the first 'rgb' directory
    pointinfo_path = rgb_path.replace("/rgb/", "/point_info/", 1)

    # Replace 'rgb' in the filename
    dir_name, filename = os.path.split(pointinfo_path)
    filename = filename.replace("rgb", "fixatedpose")

    # Change extension to .json
    filename = os.path.splitext(filename)[0] + ".json"
    
    return os.path.join(dir_name, filename)


def load_png_image(image_path: str) -> np.ndarray:
    """
    Load a PNG image as a NumPy array.

    Args:
        image_path (str): Path to the PNG image.

    Returns:
        np.ndarray: Image array (H, W) for grayscale or (H, W, 3) for RGB.
    """
    img = iio.imread(image_path)
    

    return img


def pad_to_size_with_intrinsics(img, depth, mask, K, new_H, new_W):
    """
    Pad image, depth, and mask to (new_H, new_W) via replicate padding,
    and adjust intrinsics accordingly.

    Args:
        img   (np.ndarray): HxWx3 image
        depth (np.ndarray): HxW depth map
        mask  (np.ndarray): HxW mask
        K     (np.ndarray): 3x3 intrinsic matrix
        new_H (int): target height
        new_W (int): target width

    Returns:
        img_padded, depth_padded, mask_padded, K_new
    """

    # ----------------------------
    # Get input resolution
    # ----------------------------
    H, W = depth.shape  # depth is HxW, avoids ambiguity
 
    assert img.shape[0] == H and img.shape[1] == W, "Image and depth shape mismatch"
    assert mask.shape == (H, W), "Mask shape mismatch"

    # ----------------------------
    # Compute padding (symmetric)
    # ----------------------------
    pad_h = (new_H - H)
    pad_w = (new_W - W)

    assert pad_h >= 0 and pad_w >= 0, "New size must be >= input size"

    # split evenly on both sides
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # ----------------------------
    # Replicate padding
    # ----------------------------
    def pad_replicate(x):
        if x.ndim == 2:  # depth or mask
            return np.pad(
                x,
                ((top, bottom), (left, right)),
                mode='edge'
            )
        else:  # image HxWx3
            return np.pad(
                x,
                ((top, bottom), (left, right), (0, 0)),
                mode='edge'
            )

    img_padded = pad_replicate(img)
    depth_padded = pad_replicate(depth)
    mask_padded = pad_replicate(mask)

    # ----------------------------
    # Adjust intrinsics
    # ----------------------------
    K_new = K.copy().astype(np.float32)
    K_new[0, 2] += left   # shift cx
    K_new[1, 2] += top    # shift cy
    # fx, fy unchanged (padding does not scale)

    return img_padded, depth_padded, mask_padded, K_new



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
        K1,
        K2,
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



def compute_scene_camera_overlaps(
    scene_files: Dict[str, Dict[str, List[str]]],
    start_idx: int = 0,
    end_idx: int = None,
):
    """
    Compute camera overlaps and covisibility for a subset of scenes.

    Args:
        scene_files: dict mapping scene_name -> {'rgb': [...], ...}
        start_idx: start index of scenes to process
        end_idx: end index (exclusive) of scenes to process
    """
    scene_names = list(scene_files.keys())
    if end_idx is None or end_idx > len(scene_names):
        end_idx = len(scene_names)

    # Loop over only the specified range
    for scene_name in scene_names[start_idx:end_idx]:
        list_rgbs = scene_files[scene_name]
        N = len(list_rgbs)
        depths = []
        images = []
        cam_params = []
        valid_indices = []
        depth_masks = []

        
        overlap_basedir = os.path.join("/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/replica_overlaps/")


        overlaps_dir = os.path.join(overlap_basedir, scene_name)

        overlap_file = os.path.join(overlaps_dir, "camera_overlap.npy")
        occlusion_file = os.path.join(overlaps_dir, "occlusion_matrix.npy")

        # Skip already processed scenes
        if os.path.exists(overlap_file) and os.path.exists(occlusion_file):
            print("Already processed. Skipping.")
            continue

        os.makedirs(overlaps_dir, exist_ok=True)
        
      
        for i in tqdm(range(N), desc=f"Preloading {scene_name}"):

            depth_path = rgb_to_depth_path(list_rgbs[i])

            depth, mask_depth  = load_depth_and_mask(depth_path)

            rgb_image = load_image(list_rgbs[i])

            cameras_dict = load_camera_params(rgb_to_pointinfo_path(list_rgbs[i]))

            RT_c2w, K = view_camera_parameters(cameras_dict) # cam2world

            mask_path = rgb_to_mask_path(list_rgbs[i])

            if os.path.exists(mask_path):
                mask = load_valid_mask(mask_path)
                mask = mask & (mask_depth)
            else:
                mask = mask_depth

            rgb_image_new, depth_new, mask_new, K_new =  pad_to_size_with_intrinsics(img = rgb_image, depth = depth, mask = mask.astype(np.int32), K = K, new_H = 518, new_W = 518)
            # rgbd_to_pointcloud(rgb = rgb_image_new, depth = depth_new, K = K_new, RT_c2w = RT_c2w,  save_path=f"replica_point_cloud_{i}.ply")
            # plt.imshow(rgb_image)
            # plt.savefig(f"replica_image_{i}.png")
            # plt.close()
            # breakpoint()

            if has_sufficient_valid_depth(depth_new, threshold = 0.10, min_fraction = 0.6):
                valid_indices.append(i)
                depths.append(depth_new)
                # images.append(rgb_image_new)
                depth_mask =  (mask_new > 0).astype(np.bool_)
                depth_masks.append(depth_mask)
                cam_params.append((K_new, RT_c2w))

        # Initialize matrices
        camera_overlap = np.full((N, N), -1.0, dtype=np.float32)
        occlusion_matrix = np.full((N, N), -1.0, dtype=np.float32)
        # frustum_iou = np.full((N, N), -1.0, dtype=np.float32)
        # covisibility = np.full((N, N), -1.0, dtype=np.float32)


        for index in valid_indices:
            camera_overlap[index, index] = 1.0
            occlusion_matrix[index, index] = 1.0
            # frustum_iou[index, index] = 1.0
            # covisibility[index, index] = 1.0


        for idx_i, target_idx in enumerate(tqdm(valid_indices, desc=f"Computing overlaps {scene_name}")):
            depth_i = depths[idx_i]  # corrected indexing
            K_i, RT_c2w_i = cam_params[idx_i]
            mask_i = depth_masks[idx_i]
            mask_i = mask_i 

        
            for idx_j in range(idx_i + 1, len(valid_indices)):
                src_idx = valid_indices[idx_j]
                depth_j = depths[idx_j]
                K_j, RT_c2w_j = cam_params[idx_j]
                mask_j = depth_masks[idx_j]

                mask_j = mask_j 

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

                # Fill symmetric entries
                camera_overlap[target_idx, src_idx] = overlap_dict["overlap1"]
                camera_overlap[src_idx, target_idx] = overlap_dict["overlap2"]
                occlusion_matrix[target_idx, src_idx] = overlap_dict["occlusion1"]
                occlusion_matrix[src_idx, target_idx] = overlap_dict["occlusion2"]
            # breakpoint()   
        np.save(overlap_file, camera_overlap)
        np.save(occlusion_file, occlusion_matrix)



def collect_scene_files(root_dir: str, max_images: int = 1000) -> Dict[str, List[str]]:
    """
    Collect file paths for each unique scene, merging sub-scene directories.
    Each scene will have at most `max_images` files.

    Args:
        root_dir (str): Root directory containing subdirectories like 'ai_055_009-cam_00'.
        max_images (int): Maximum number of images per scene.

    Returns:
        Dict[str, List[str]]: scene_name -> list of file paths
    """
    scene_files = defaultdict(list)

    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Extract unique scene name (everything before first '-')
        scene_name = subdir.split('-')[0]

        # breakpoint()

        # Get all files in the subdir, sorted
        files_in_subdir = sorted([
            os.path.join(subdir_path, f)
            for f in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, f))
        ])

        # breakpoint()

        # Subsample if needed
        if len(files_in_subdir) > max_images:
            indices = np.linspace(0, len(files_in_subdir) - 1, max_images, dtype=int)
            files_in_subdir = [files_in_subdir[i] for i in indices]

        # Append to the scene list
        scene_files[scene_name].extend(files_in_subdir)

    return scene_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of scenes to process")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of scenes to process (exclusive)")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/replica_scene_files_1000.pkl", help="Pickle file containing scene files")
    args = parser.parse_args()

    rgb_dir = "/scratch/dma9300/replica/rgb/replica"

    if not os.path.exists(args.scene_file):
        all_scene_files = collect_scene_files(root_dir = rgb_dir)

        with open(f"{args.scene_file}", "wb") as f:
            pkl.dump(all_scene_files, f)

    with open(args.scene_file, "rb") as f:
        all_scene_files = pkl.load(f)

    compute_scene_camera_overlaps(all_scene_files, start_idx=args.start_idx, end_idx=args.end_idx)


