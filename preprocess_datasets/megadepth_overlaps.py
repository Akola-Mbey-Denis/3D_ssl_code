# import os
# import numpy as np
# import time
# import argparse
# import json
# import cv2
# import torch
# import PIL 
# from PIL import Image
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # it works well.
# import pickle as pkl
# from scipy.spatial.transform import Rotation as R
# import imageio.v2 as imageio
# from typing import Dict, List
# from collections import defaultdict
# import numpy as np
# import open3d as o3d
# import h5py
# from pathlib import Path
# import utils.cropping as cropping
# import torch.nn.functional as F


# def compute_opengl_projection_matrix(
#         fov,           # Taskonomy horizontal FOV
#         znear, zfar,   # clipping planes
#         width_pixels,
#         height_pixels,
#         degrees=False
#     ):
#     """
#     Compute OpenGL projection matrix for Taskonomy cameras.
#     Taskonomy provides ONLY the horizontal FOV (fov).

#     Args:
#         fov            : horizontal field of view (radians or degrees)
#         znear, zfar    : near/far clipping planes
#         width_pixels   : image width
#         height_pixels  : image height
#         degrees        : whether fov is in degrees

#     Returns:
#         4×4 OpenGL projection matrix
#     """

#     if degrees:
#         fov = np.radians(fov)

#     # Horizontal FOV from dataset
#     fovx = fov

#     # Derive vertical FOV using aspect ratio
#     aspect = width_pixels / height_pixels
#     fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) / aspect)

#     # Near plane extents
#     f_h = np.tan(fovy / 2.0) * znear  # top/bottom
#     f_w = np.tan(fovx / 2.0) * znear  # left/right

#     left   = -f_w
#     right  =  f_w
#     bottom = -f_h
#     top    =  f_h

#     # OpenGL projection matrix
#     K = np.zeros((4, 4), dtype=np.float32)
#     K[0, 0] = 2.0 * znear / (right - left)
#     K[1, 1] = 2.0 * znear / (top - bottom)
#     K[0, 2] = (right + left) / (right - left)
#     K[1, 2] = (top + bottom) / (top - bottom)
#     K[2, 2] = -(zfar + znear) / (zfar - znear)
#     K[2, 3] = -2.0 * zfar * znear / (zfar - znear)
#     K[3, 2] = -1.0

#     return K


# def opengl_kndc_to_opencv_K(K_ndc, W, H):
#     """
#     Convert an OpenGL KNDC (NDC) projection matrix to an OpenCV-style intrinsic matrix.

#     Args:
#         K_ndc: (4,4) OpenGL projection matrix (z in [-1,1])
#         W: image width in pixels
#         H: image height in pixels

#     Returns:
#         K: (3,3) OpenCV intrinsic matrix
#     """
#     # Horizontal and vertical focal lengths
#     fx = K_ndc[0, 0] * W / 2.0
#     fy = K_ndc[1, 1] * H / 2.0

#     # Principal point
#     cx = (1.0 - K_ndc[0, 2]) * W / 2.0
#     cy = (1.0 + K_ndc[1, 2]) * H / 2.0

#     K = np.array([
#         [fx, 0.0, cx],
#         [0.0, fy, cy],
#         [0.0, 0.0, 1.0]
#     ], dtype=np.float32)

#     return K





# def view_camera_parameters(point_info):
#     """
#     Returns: (RT_cam2world, K) both in OpenGL coordinate system:
#         +X right, +Y up, -Z forward.
#     https://github.com/nileshkulkarni/mv_drdf/blob/master/rgbd_drdf/data_utils/base_rgbd.py#L215

#     """

#     # ---- Extract metadata ----
#     location = point_info["camera_location"]      # (Tx, Tz, Ty) from dataset
#     rotation = point_info["camera_rotation_final"]      # Blender XYZ euler
#     W = point_info["resolution"]
#     H = point_info["resolution"]
#     fov = point_info["field_of_view_rads"]


#     # Build rotation matrix from Euler angles in 'sxyz' order
#     R_euler = R.from_euler('xyz', rotation).as_matrix()  # shape (3,3)

#     # Multiply by Rx(pi) to flip the camera coordinate frame
#     Rx = R.from_euler('x', np.pi).as_matrix()
#     R_cam2world = R_euler @ Rx

#     # Build 4x4 cam2world matrix
#     RT_4x4 = np.eye(4, dtype=np.float32)
#     RT_4x4[:3, :3] = R_cam2world
#     RT_4x4[:3, 3] = location

#     K = get_intrinsics_from_fov(fov = fov, W = W, H = H)

#     return RT_4x4, K




# def get_intrinsics_from_fov(fov, W, H, degrees=False):
#     """
#     Recover camera intrinsics K from a single field of view (Taskonomy-style).
#     Taskonomy provides horizontal FOV.

#     Args:
#         fov: horizontal FOV (radians by default, or degrees if degrees=True)
#         W: image width in pixels
#         H: image height in pixels
#         degrees: set True if fov is provided in degrees

#     Returns:
#         K: (3,3) intrinsic matrix
#     """
#     if degrees:
#         fov = np.radians(fov)

#     # Horizontal focal length
#     fx = W / (2 * np.tan(fov / 2))

#     # Convert HFOV -> VFOV
#     fov_y = 2 * np.arctan(np.tan(fov / 2) * (H / W))

#     # Vertical focal length
#     fy = H / (2 * np.tan(fov_y / 2))

#     # Principal point (center pixel)
#     cx = W / 2
#     cy = H / 2

#     K = np.array([
#         [fx, 0,  cx],
#         [0,  fy, cy],
#         [0,  0,  1]
#     ], dtype=np.float32)

#     return K




# def rotation_inverse(RT_c2w: np.ndarray) -> np.ndarray:
#     """
#     Compute the inverse of a 4x4 camera-to-world matrix (rigid transform) using NumPy.

#     Args:
#         RT_c2w (np.ndarray): 4x4 camera-to-world extrinsic matrix [R | t; 0 0 0 1]

#     Returns:
#         np.ndarray: 4x4 world-to-camera matrix.
#     """
#     assert RT_c2w.shape == (4, 4), "RT_c2w must be 4x4"

#     R = RT_c2w[:3, :3]  # Rotation part (3x3)
#     t = RT_c2w[:3, 3]   # Translation part (3,)

#     R_inv = R.T          # Inverse rotation
#     t_inv = -R_inv @ t   # Adjusted translation

#     RT_w2c = np.eye(4, dtype=RT_c2w.dtype)
#     RT_w2c[:3, :3] = R_inv
#     RT_w2c[:3, 3] = t_inv

#     return RT_w2c




# def project_points(K: np.ndarray, RT_c2w: np.ndarray, world_points: np.ndarray, image_size: tuple):
#     """
#     Project 3D world points into camera image space using NumPy.

#     Args:
#         K (np.ndarray): (3, 3) intrinsic matrix.
#         RT_c2w (np.ndarray): (4, 4) camera-to-world extrinsic matrix.
#         world_points (np.ndarray): (N, 3) points in world coordinates.
#         image_size (tuple): (H, W) image dimensions.

#     Returns:
#         uv (np.ndarray): (N, 2) projected pixel coordinates (float).
#         z (np.ndarray):  (N,) depth values in camera space.
#     """
#     H, W = image_size

#     # Invert RT to get world → camera
#     RT_w2c = rotation_inverse(RT_c2w)  # (4, 4)

#     N = world_points.shape[0]

#     # Convert world points to homogeneous coordinates
#     world_h = np.hstack([world_points, np.ones((N, 1), dtype=world_points.dtype)])  # (N, 4)

#     # Transform world → camera
#     cam_points = (RT_w2c @ world_h.T).T[:, :3]  # (N, 3)

#     # Depth in camera coordinates
#     z = cam_points[:, 2] # (N,)


#     # Project into pixel space
#     proj = (K @ cam_points.T).T #(N, 3)

#     uv = proj[:, :2] / (proj[:, 2:3] + 1e-8) # (N, 2)


#     return uv, z



# def compute_points_visibility(
#     uv: np.ndarray,
#     z: np.ndarray,
#     depthmap: np.ndarray,
#     depth_mask: np.ndarray,
#     eps: float = 1e-3
# ) -> np.ndarray:
#     """
#     Compute visibility mask of projected points in a target depth map using NumPy.

#     Args:
#         uv (np.ndarray): Projected pixel coordinates (N, 2), float.
#         z (np.ndarray): Projected depth values (N,), float.
#         depthmap (np.ndarray): Target depth map (H, W), float.

#     Returns:
#         np.ndarray: Boolean mask of shape (N,) indicating whether each projected
#                     point is consistent with the target depth map.
#     """
#     H, W = depthmap.shape
#     N = uv.shape[0]

#     # Initialize output visibility mask
#     visibility_mask = np.zeros(N, dtype=bool)

#     # Split uv into separate arrays
#     u = np.round(uv[:, 0]).astype(np.int64)
#     v = np.round(uv[:, 1]).astype(np.int64)

#     # Bounds check (points must fall inside the depth map)
#     in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.01)
    
#     # Filter valid indices
#     u_valid, v_valid, z_valid = u[in_bounds], v[in_bounds], z[in_bounds]

#     depth_values = depthmap[v_valid, u_valid]
#     depth_mask_values = depth_mask[v_valid, u_valid]

#     visible =  (z_valid <= depth_values + eps ) & depth_mask_values

#     # Mark visible points in the full mask
#     visibility_mask[in_bounds] = visible

#     return visibility_mask


# def load_valid_mask(png_path: str) -> np.ndarray:
#     """
#     Load a 1-channel 8-bit PNG as a boolean mask.
#     0 -> False (missing geometry)
#     255 -> True (valid geometry)
    
#     Args:
#         png_path: Path to the PNG file.
    
#     Returns:
#         mask: np.ndarray of shape (H, W), dtype=bool
#     """
#     mask = imageio.imread(png_path)  # shape (H, W), dtype=uint8
#     mask = mask.astype(bool)      # 0->False, 255->True
#     return mask




# def project_to_world_points(depth: np.ndarray, K: np.ndarray, RT_c2w: np.ndarray,depth_mask: np.ndarray,) -> np.ndarray:
#     """
#     Project a depth map into 3D world coordinates using intrinsics and 
#     a camera-to-world transformation matrix.

#     Args:
#         depth (np.ndarray): Depth map of shape (H, W), float.
#         K (np.ndarray): Camera intrinsic matrix of shape (3, 3), float.
#         RT_c2w (np.ndarray): Extrinsic matrix of shape (4, 4), mapping camera → world, float.
#         depth_mask (np.ndarray): Boolean mask of valid depth values, shape (H, W).

#     Returns:
#         np.ndarray: 3D world points of shape (N, 3), where N is the number of valid depth pixels.
#     """
#     H, W = depth.shape

#     # Generate pixel coordinates (u = x, v = y)
#     u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape (H, W)
    
#     # Flatten pixel grid and depth
#     u_flat, v_flat = u.flatten(), v.flatten()
#     depth_flat = depth.flatten()

#     # Apply valid depth mask
#     mask_flat = depth_mask.flatten()
#     u_valid, v_valid, depth_valid = u_flat[mask_flat], v_flat[mask_flat], depth_flat[mask_flat]

#     # Build homogeneous pixel coordinates (3, N)
#     pixel_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)  # (3, N)

#     # Backproject to camera coordinates
#     K_inv = np.linalg.inv(K)
#     cam_points = K_inv @ pixel_coords  # (3, N)

#     # Scale rays by depth → camera space points (N, 3)
#     cam_points = (cam_points * depth_valid[np.newaxis, :]).T

#     # Convert to homogeneous coordinates (N, 4)
#     cam_points_h = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])

#     # Transform to world coordinates
#     world_points_h = (RT_c2w @ cam_points_h.T).T  # (N, 4)
#     world_points = world_points_h[:, :3]  # (N, 3)

#     return world_points.astype(np.float32)



# def compute_frustum_overlap(
#         K1,
#         K2,
#         Rt1_c2w,
#         depth1,
#         Rt2_c2w,
#         depth2,
#         mask1,
#         mask2,
#         eps=1e-3,
#     ):
#         """
#         Compute pairwise depth-based overlap and directional occlusion.

#         Convention:
#             View 1 = reference / target view.
#             View 2 = candidate source view.

#         overlap1:
#             Fraction of valid surfaces in view 2 that project into a valid
#             observed-depth region of view 1.

#         occlusion1:
#             Fraction of overlapping surfaces from view 2 that are behind
#             view 1's first-hit depth.

#             This answers:
#                 "How much of what view 2 sees is hidden from view 1?"

#             High occlusion1 means view 2 contains more surfaces that are
#             hidden from target view 1.

#         overlap2:
#             Fraction of valid surfaces in view 1 that project into a valid
#             observed-depth region of view 2.

#         occlusion2:
#             Fraction of overlapping surfaces from view 1 that are behind
#             view 2's first-hit depth.

#         Returns:
#             dict:
#                 overlap1
#                 occlusion1
#                 overlap2
#                 occlusion2
#         """

#         # =========================================================
#         # 1. Valid depth pixels
#         # =========================================================

#         depth1 = np.asarray(depth1)
#         depth2 = np.asarray(depth2)
#         mask1 = np.asarray(mask1).astype(bool)
#         mask2 = np.asarray(mask2).astype(bool)

#         mask1 = (
#             mask1
#             & np.isfinite(depth1)
#             & (depth1 > 0.01)
#         )

#         mask2 = (
#             mask2
#             & np.isfinite(depth2)
#             & (depth2 > 0.01)
#         )

#         num_surface1 = np.count_nonzero(mask1)
#         num_surface2 = np.count_nonzero(mask2)

#         if num_surface1 == 0 or num_surface2 == 0:
#             return {
#                 "overlap1": 0.0,
#                 "occlusion1": 0.0,
#                 "overlap2": 0.0,
#                 "occlusion2": 0.0,
#             }

#         # =========================================================
#         # 2. Convert first-hit depth maps to world-space surfaces
#         # =========================================================

#         world_points1 = project_to_world_points(
#             depth=depth1,
#             K=K1,
#             RT_c2w=Rt1_c2w,
#             depth_mask=mask1,
#         )

#         world_points2 = project_to_world_points(
#             depth=depth2,
#             K=K2,
#             RT_c2w=Rt2_c2w,
#             depth_mask=mask2,
#         )

#         # Make sure downstream operations are NumPy.
#         world_points1 = np.asarray(world_points1)
#         world_points2 = np.asarray(world_points2)

#         # =========================================================
#         # 3. View 1 surfaces -> View 2
#         #
#         # Used for:
#         #   overlap2
#         #   occlusion2
#         # =========================================================

#         uv12, z12 = project_points(
#             K=K2,
#             RT_c2w=Rt2_c2w,
#             world_points=world_points1,
#             image_size=depth2.shape,
#         )

#         uv12 = np.asarray(uv12)
#         z12 = np.asarray(z12).reshape(-1)

#         inside2 = np.asarray(points_inside_image(uv12, image_size=depth2.shape)).astype(bool)

#         # breakpoint()

#         target_depth2 = np.asarray(interpolate_depth(depth2,uv12)).reshape(-1)

#         target_mask2 = (
#             np.asarray(
#                 interpolate_depth(
#                     mask2.astype(np.float32),
#                     uv12,
#                 )
#             ).reshape(-1)
#             > 0.5
#         )

#         valid12 = (
#             inside2
#             & target_mask2
#             & np.isfinite(z12)
#             & (z12 > 0.01)
#             & np.isfinite(target_depth2)
#             & (target_depth2 > 0.01)
#         )

#         # ---------------------------------------------------------
#         # View-1 surfaces that are behind view-2's first surface
#         # ---------------------------------------------------------



#         occluded12 = (
#             valid12
#             & (z12 > target_depth2 + eps)
#         )

#         # =========================================================
#         # 4. View 2 surfaces -> View 1
#         #
#         # Used for:
#         #   overlap1
#         #   occlusion1
#         #
#         # THIS IS THE IMPORTANT DIRECTION WHEN VIEW 1 IS TARGET.
#         # =========================================================

#         uv21, z21 = project_points(
#             K=K1,
#             RT_c2w=Rt1_c2w,
#             world_points=world_points2,
#             image_size=depth1.shape,
#         )

#         uv21 = np.asarray(uv21)
#         z21 = np.asarray(z21).reshape(-1)

#         inside1 = np.asarray(
#             points_inside_image(
#                 uv21,
#                 image_size=depth1.shape,
#             )
#         ).astype(bool)

#         target_depth1 = np.asarray(interpolate_depth(depth1, uv21)).reshape(-1)

#         target_mask1 = (
#             np.asarray(
#                 interpolate_depth(
#                     mask1.astype(np.float32),
#                     uv21,
#                 )
#             ).reshape(-1)
#             > 0.5
#         )

#         valid21 = (
#             inside1
#             & target_mask1
#             & np.isfinite(z21)
#             & (z21 > 0.01)
#             & np.isfinite(target_depth1)
#             & (target_depth1 > 0.01)
#         )


#         # ---------------------------------------------------------
#         # View-2 surfaces that are behind view-1's first surface
#         # ---------------------------------------------------------


#         occluded21 = (
#             valid21
#             & (z21 > target_depth1 + eps)
#         )

#         # =========================================================
#         # 5. Directional overlap
#         # =========================================================

#         num_overlap12 = np.count_nonzero(valid12)
#         num_overlap21 = np.count_nonzero(valid21)

#         # ---------------------------------------------------------
#         # View 1 = TARGET
#         #
#         # How much of view 2 is testable from view 1?
#         # ---------------------------------------------------------

#         overlap1 = (
#             num_overlap21 / num_surface2
#             if num_surface2 > 0
#             else 0.0
#         )

#         # ---------------------------------------------------------
#         # View 2 = TARGET
#         #
#         # How much of view 1 is testable from view 2?
#         # ---------------------------------------------------------

#         overlap2 = (
#             num_overlap12 / num_surface1
#             if num_surface1 > 0
#             else 0.0
#         )

#         # =========================================================
#         # 6. Directional occlusion
#         # =========================================================

#         # ---------------------------------------------------------
#         # TARGET = VIEW 1
#         #
#         # Of the view-2 surfaces overlapping view 1,
#         # what fraction is hidden behind view 1?
#         # ---------------------------------------------------------

#         occlusion1 = (
#             np.count_nonzero(occluded21) / num_overlap21
#             if num_overlap21 > 0
#             else 0.0
#         )

#         # ---------------------------------------------------------
#         # TARGET = VIEW 2
#         #
#         # Of the view-1 surfaces overlapping view 2,
#         # what fraction is hidden behind view 2?
#         # ---------------------------------------------------------

#         occlusion2 = (
#             np.count_nonzero(occluded12) / num_overlap12
#             if num_overlap12 > 0
#             else 0.0
#         )

#         return {
#             "overlap1": float(overlap1),
#             "occlusion1": float(occlusion1),
#             "overlap2": float(overlap2),
#             "occlusion2": float(occlusion2),
#         }




# def load_image(file_path):

#     """
#     Load a JPEG image and return as a NumPy array (H, W, 3) in RGB order.
#     """

#     img = Image.open(file_path).convert("RGB")

#     return np.array(img)



# def load_depth(depth_path):
#     """
#     Loads a depth map from:
#         root/depths/<tag_without_ext>.h5

#     Parameters
#     ----------
#     root : str
#         Dataset root directory

#     Returns
#     -------
#     depthmap : np.ndarray
#         Depth map as float32 array (H, W)
#     """
#     with h5py.File(depth_path, "r") as hd5:
#         depthmap = np.asarray(hd5["depth"])

#     mask = depthmap >0.00

#     return depthmap, mask





# def load_camera_params(file_path):
#     """
#     Load camera parameters (extrinsics and intrinsics) from a JSON file.

#     Returns:
#         K (np.ndarray): 3x3 intrinsic matrix
#         RT_c2w (np.ndarray): 4x4 extrinsic matrix (camera to world)
#     """
#     with open(file_path, "r") as f:
#         data = json.load(f)


#     return data

# def has_sufficient_valid_depth(depth: np.ndarray, threshold: float = 0.10, min_fraction: float = 0.6) -> bool:
#     """
#     Check if a depth map has at least `min_fraction` of readings above `threshold`.

#     Args:
#         depth (np.ndarray): Depth map of shape (H, W).
#         threshold (float): Minimum depth value to be considered valid.
#         min_fraction (float): Minimum fraction of valid depth values required.

#     Returns:
#         bool: True if fraction of valid depths >= min_fraction, else False.
#     """
#     num_valid = np.sum(depth > threshold)
#     total_pixels = depth.size
#     fraction_valid = num_valid / total_pixels

#     return fraction_valid >= min_fraction


# def rgb_to_depth_path(rgb_path: str) -> str:
#     """
#     Convert an RGB file path into a Euclidean depth file path by replacing:
#     - the first 'rgb' directory with 'depth_euclidean'
#     - the 'rgb' in the filename with 'depth_euclidean'
#     """
#     # Replace the first 'rgb' directory
#     depth_path = rgb_path.replace("/rgb/", "/depth_zbuffer/", 1)

#     # Replace 'rgb' in the filename
#     dir_name, filename = os.path.split(depth_path)
#     filename = filename.replace("rgb", "depth_zbuffer")
    
#     return os.path.join(dir_name, filename)



# def rgb_to_pointinfo_path(rgb_path: str) -> str:
#     """
#     Convert an RGB file path into a point_info file path by replacing:
#     - the first 'rgb' directory with 'point_info'
#     - the 'rgb' in the filename with 'point_info'
#     - the file extension to .json
#     """
#     # Replace the first 'rgb' directory
#     pointinfo_path = rgb_path.replace("/rgb/", "/point_info/", 1)

#     # Replace 'rgb' in the filename
#     dir_name, filename = os.path.split(pointinfo_path)
#     filename = filename.replace("rgb", "point_info")

#     # Change extension to .json
#     filename = os.path.splitext(filename)[0] + ".json"
    
#     return os.path.join(dir_name, filename)


# def load_png_image(image_path: str) -> np.ndarray:
#     """
#     Load a PNG image as a NumPy array.

#     Args:
#         image_path (str): Path to the PNG image.

#     Returns:
#         np.ndarray: Image array (H, W) for grayscale or (H, W, 3) for RGB.
#     """
#     img = iio.imread(image_path)
    

#     return img


# def collect_scene_files(root_dir: str, max_images: int = 1000) -> Dict[str, List[str]]:
#     """
#     Collect image file paths for MegaDepth structure.

#     Returns:
#         Dict[str, List[str]]
#             key   -> "5018_dense0"
#             value -> list of up to max_images image paths (evenly subsampled)
#     """

#     scene_files = {}

#     # iterate over scene folders like 5018, 0000, etc.
#     for scene in sorted(os.listdir(root_dir)):
#         scene_path = os.path.join(root_dir, scene)
#         if not os.path.isdir(scene_path):
#             continue

#         # iterate over dense folders like dense0
#         for dense_subdir in sorted(os.listdir(scene_path)):
#             dense_path = os.path.join(scene_path, dense_subdir)
#             if not os.path.isdir(dense_path):
#                 continue

#             imgs_path = os.path.join(dense_path, "imgs")
#             if not os.path.isdir(imgs_path):
#                 continue  # skip if no imgs folder

#             # Recursively collect all image files inside imgs/
#             image_files = []
#             for root, _, files in os.walk(imgs_path):
#                 for f in files:
#                     if f.lower().endswith((".JPG", ".jpg" ,".jpeg", ".JPEG", ".PNG", ".png")):
#                         image_files.append(os.path.join(root, f))

#             image_files = sorted(image_files)

#             if not image_files:
#                 continue

#             # Subsample evenly if needed
#             if len(image_files) > max_images:
#                 indices = np.linspace(0, len(image_files) - 1, max_images, dtype=int)
#                 image_files = [image_files[i] for i in indices]

#             key = f"{scene}_{dense_subdir}"
#             scene_files[key] = image_files

#     return scene_files



# def rgb_to_mask_path(rgb_path: str) -> str:
#     """
#     Convert an RGB file path into a Euclidean depth file path by replacing:
#     - the first 'rgb' directory with 'depth_euclidean'
#     - the 'rgb' in the filename with 'depth_euclidean'
#     """
#     # point_9_view_9_domain_mask_valid.png
#     # Replace the first 'rgb' directory
#     depth_path = rgb_path.replace("/rgb/", "/mask_valid/", 1)

#     # Replace 'rgb' in the filename
#     dir_name, filename = os.path.split(depth_path)
#     filename = filename.replace("rgb", "mask_valid")
    
#     return os.path.join(dir_name, filename)


# def rgbd_to_pointcloud(rgb, depth, K, RT_c2w, mask=None,   scale = 10.00, save_path="replica_point_cloud.ply"):
#     """
#     Project RGB-D image to colored world point cloud.
    
#     Args:
#         rgb: (H, W, 3) RGB image, uint8 [0-255] or float [0-1]
#         depth: (H, W) depth map
#         K: (3, 3) camera intrinsic matrix
#         RT_c2w: (4, 4) cam2world matrix
#         mask: (H, W) optional binary mask, True/1 = valid
#         save_path: optional path to save PLY file
    
#     Returns:
#         points: (N, 4) world coordinates (homogeneous)
#         colors: (N, 3) RGB colors [0-1]
#     """
#     H, W = depth.shape
    
#     # Create pixel grid (homogeneous)
#     u, v = np.meshgrid(np.arange(W), np.arange(H))
#     ones = np.ones((H, W))
#     pixels = np.stack([u, v, ones], axis=-1).reshape(-1, 3)  # (N, 3)
#     z = depth.flatten() 
    
#     # Filter valid depth
#     valid = z > 0
    
#     # Apply mask if provided
#     if mask is not None:
#         valid = valid & mask.flatten().astype(bool)
    
#     pixels = pixels[valid]
#     z = z[valid]
    
#     # Unproject: p_cam = z * K_inv @ [u, v, 1]
#     K_inv = np.linalg.inv(K)
#     pts_cam = (K_inv @ pixels.T).T * z[:, None]  # (N, 3)
    
#     # Homogeneous camera coordinates
#     pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])  # (N, 4)
    
#     # Transform to world
#     pts_world = (RT_c2w @ pts_cam_h.T).T  # (N, 4)
    
#     # Get colors
#     colors = rgb.reshape(-1, 3)[valid]
#     if colors.max() > 1:
#         colors = colors / 255.0
    
#     # Save with Open3D
#     if save_path:
#         print(f"Saving colored point cloud - file {save_path}")
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts_world[:, :3])
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#         o3d.io.write_point_cloud(save_path, pcd)


# def pad_to_size_with_intrinsics(img, depth, mask, K, new_H, new_W):
#     """
#     Pad image, depth, and mask to (new_H, new_W) via replicate padding,
#     and adjust intrinsics accordingly.

#     Args:
#         img   (np.ndarray): HxWx3 image
#         depth (np.ndarray): HxW depth map
#         mask  (np.ndarray): HxW mask
#         K     (np.ndarray): 3x3 intrinsic matrix
#         new_H (int): target height
#         new_W (int): target width

#     Returns:
#         img_padded, depth_padded, mask_padded, K_new
#     """

#     # ----------------------------
#     # Get input resolution
#     # ----------------------------
#     H, W = depth.shape  # depth is HxW, avoids ambiguity
 
#     assert img.shape[0] == H and img.shape[1] == W, "Image and depth shape mismatch"
#     assert mask.shape == (H, W), "Mask shape mismatch"

#     # ----------------------------
#     # Compute padding (symmetric)
#     # ----------------------------
#     pad_h = (new_H - H)
#     pad_w = (new_W - W)

#     assert pad_h >= 0 and pad_w >= 0, "New size must be >= input size"

#     # split evenly on both sides
#     top = pad_h // 2
#     bottom = pad_h - top
#     left = pad_w // 2
#     right = pad_w - left

#     # ----------------------------
#     # Replicate padding
#     # ----------------------------
#     def pad_replicate(x):
#         if x.ndim == 2:  # depth or mask
#             return np.pad(
#                 x,
#                 ((top, bottom), (left, right)),
#                 mode='edge'
#             )
#         else:  # image HxWx3
#             return np.pad(
#                 x,
#                 ((top, bottom), (left, right), (0, 0)),
#                 mode='edge'
#             )

#     img_padded = pad_replicate(img)
#     depth_padded = pad_replicate(depth)
#     mask_padded = pad_replicate(mask)

#     # ----------------------------
#     # Adjust intrinsics
#     # ----------------------------
#     K_new = K.copy().astype(np.float32)
#     K_new[0, 2] += left   # shift cx
#     K_new[1, 2] += top    # shift cy

#     return img_padded, depth_padded, mask_padded, K_new

# def load_camera_matrices(pose_path):
   
#     with open(pose_path, "r") as f:
#         camera_data = json.load(f)

#     return camera_data

# def _downscale_image(camera_intrinsics, image, depthmap, resolution_out=(512, 384)):
#     H, W = image.shape[:2]
#     resolution_out = sorted(resolution_out)[:: +1 if W < H else -1]

#     image, depthmap, intrinsics_out = cropping.rescale_image_depthmap(
#         image, depthmap, camera_intrinsics, resolution_out, force=False
#     )


#     R_in2out = np.eye(3)

#     return image, depthmap, intrinsics_out, R_in2out

# def _crop_resize_if_necessary(image, depthmap, intrinsics, resolution, aug_crop = 16,  mask = None, rng=None, info=None):
#         """ This function:
#             - first downsizes the image with LANCZOS inteprolation,
#               which is better than bilinear interpolation in
#         """
#         if not isinstance(image, PIL.Image.Image):
#             image = PIL.Image.fromarray(image)

#         # downscale with lanczos interpolation so that image.size == resolution
#         # cropping centered on the principal point
#         W, H = image.size
#         cx, cy = intrinsics[:2, 2].round().astype(int)
#         min_margin_x = min(cx, W-cx)
#         min_margin_y = min(cy, H-cy)

#         # assert min_margin_x > W/5, f'Bad principal point in view={info}'
#         # assert min_margin_y > H/5, f'Bad principal point in view={info}'
#         # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
#         l, t = cx - min_margin_x, cy - min_margin_y
#         r, b = cx + min_margin_x, cy + min_margin_y
#         crop_bbox = (l, t, r, b)
#         image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)

#         # transpose the resolution if necessary
#         W, H = image.size  # new size
#         assert resolution[0] >= resolution[1]
#         if H > 1.1*W:
#             # image is portrait mode
#             resolution = resolution[::-1]
#         elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
#             # image is square, so we chose (portrait, landscape) randomly
#             if rng.integers(2):
#                 resolution = resolution[::-1]

#         # high-quality Lanczos down-scaling
#         target_resolution = np.array(resolution)
#         if aug_crop > 1:
#             target_resolution += rng.integers(0, aug_crop)
#         image, depthmap, intrinsics, mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, mask = mask)

#         # actual cropping (if necessary) with bilinear interpolation
#         intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
#         crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
#         image, depthmap, intrinsics2, mask  = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)
        
#         return image, depthmap, intrinsics2, mask

# def is_valid_image(image_path, valid_pixels_frac=0.05):
#     """
#     Check if the fraction of black pixels in an RGB image is < valid_pixels_frac.

#     Args:
#         image_path (str): Path to the RGB image.
#         valid_pixels_frac (float): Maximum allowed fraction of black pixels.

#     Returns:
#         bool: True if black pixel fraction < valid_pixels_frac, else False.
#     """

#     # Load image as RGB
#     image = np.array(Image.open(image_path).convert("RGB"))

#     # Ensure it's RGB
#     if image.ndim != 3 or image.shape[2] != 3:
#         raise ValueError("Image must be an RGB image of shape (H, W, 3).")

#     black_mask = np.all(image < 6, axis=-1)
#     black_pixels = np.sum(black_mask)

#     total_pixels = image.shape[0] * image.shape[1]
#     fraction_black = black_pixels / total_pixels

#     return fraction_black < valid_pixels_frac



# def points_inside_image(uv, image_size):
#         """
#         Check which projected points fall inside an image.

#         Args:
#             uv:
#                 Projected image coordinates, shape (N, 2).
#                 uv[:, 0] = x / column
#                 uv[:, 1] = y / row

#             image_size:
#                 (H, W)

#         Returns:
#             inside:
#                 Boolean tensor of shape (N,).
#         """
#         H, W = image_size[:2]

#         # Make sure uv is a tensor
#         if not torch.is_tensor(uv):
#             uv = torch.as_tensor(uv)

#         x = uv[:, 0]
#         y = uv[:, 1]

#         inside = (
#             torch.isfinite(x)
#             & torch.isfinite(y)
#             & (x >= 0)
#             & (x < W)
#             & (y >= 0)
#             & (y < H)
#         )

#         return inside


# def interpolate_depth(depth, uv):
#         """
#         Interpolate depth values at projected image coordinates.

#         Args:
#             depth:
#                 (H, W) depth map.

#             uv:
#                 (N, 2) coordinates in NDC:
#                 [-1, 1] x [-1, 1]
#                 where uv[:, 0] = x and uv[:, 1] = y.

#         Returns:
#             sampled_depth:
#                 (N,) depth values.
#         """
#         uv  = pixel_to_ndc(uv,depth.shape)

#         if not torch.is_tensor(depth):
#             depth = torch.from_numpy(depth)

#         if not torch.is_tensor(uv):
#             uv = torch.from_numpy(uv)

#         depth = depth.float()
#         uv = uv.float()

#         device = depth.device
#         uv = uv.to(device)

#         H, W = depth.shape

#         # grid_sample expects:
#         #   (N, H_out, W_out, 2)
#         #
#         # We want one sample for each uv point:
#         #   (1, N, 1, 2)
#         grid = uv.view(1, -1, 1, 2)

#         depth_input = depth.unsqueeze(0).unsqueeze(0)
#         # (1, 1, H, W)

#         sampled = F.grid_sample(
#             depth_input,
#             grid,
#             mode="bilinear",
#             padding_mode="zeros",
#             align_corners=True,
#         )

#         # (1, 1, N, 1) -> (N,)
#         sampled_depth = sampled[0, 0, :, 0]

#         return sampled_depth

# def pixel_to_ndc(uv, image_size):
#         """
#         Convert pixel coordinates to NDC coordinates expected by
#         grid_sample(..., align_corners=True).

#         Args:
#             uv: (N, 2), pixel coordinates (u, v)
#             image_size: (H, W)

#         Returns:
#             uv_ndc: (N, 2), coordinates in [-1, 1]
#         """
#         H, W = image_size

#         u = uv[:, 0]
#         v = uv[:, 1]

#         u_ndc = 2.0 * u / (W - 1) - 1.0
#         v_ndc = 2.0 * v / (H - 1) - 1.0

#         return np.stack([u_ndc, v_ndc], axis=1)


# def compute_scene_camera_overlaps(
#     scene_files: Dict[str, Dict[str, List[str]]],
#     start_idx: int = 0,
#     end_idx: int = None,
# ):
#     """
#     Compute camera overlaps and covisibility for a subset of scenes.

#     Args:
#         scene_files: dict mapping scene_name -> {'rgb': [...], ...}
#         start_idx: start index of scenes to process
#         end_idx: end index (exclusive) of scenes to process
#     """
#     scene_names = list(scene_files.keys())
#     if end_idx is None or end_idx > len(scene_names):
#         end_idx = len(scene_names)

#     # Loop over only the specified range
#     for scene_name in scene_names[start_idx:end_idx]:
#         list_rgbs = scene_files[scene_name]
#         N = len(list_rgbs)
#         depths = []
#         masks = []
#         cam_params = []
#         valid_indices = []

#         overlap_basedir = os.path.join("/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/megadepth_overlaps/")

#         camera_poses_path = "/projects/rps/dff6142/fouheylab/shared_datasets/MegaDepth/pose_data/"
#         overlaps_dir = os.path.join(overlap_basedir, scene_name)
        
#         overlap_file = os.path.join(overlaps_dir, "camera_overlap.npy")
#         occlusion_file = os.path.join(overlaps_dir, "occlusion_matrix.npy")

#         # Skip already processed scenes
#         if os.path.exists(overlap_file) and os.path.exists(occlusion_file):
#             print("Already processed. Skipping.")
#             continue


#         os.makedirs(overlaps_dir, exist_ok=True)

#         skip_flag = False

#         for i in tqdm(range(N), desc=f"Preloading {scene_name}"):

#             file_parts = list_rgbs[i].split(".")
    
#             p = Path(list_rgbs[i])
#             main_dir = p.parents[2].name    # '0000'
#             dense_dir = p.parents[1].name   # 'dense0'
#             subdir = dense_dir.replace("dense", "")  # '0'
#             image_name = list_rgbs[i].split("/")[-1]
            
#             pose_path = os.path.join(camera_poses_path, main_dir, "sparse/manhattan",subdir, "camera_matrices.json")

#             if not os.path.isfile(pose_path):
#                 print(f"[SKIP] Missing camera_matrices.json in {subdir_path}")
#                 skip_flag = True
#                 continue  # skip this subfolder safely
           
#             depth_path = f"{file_parts[0].replace('/imgs/', '/depths/')}.h5"

#             depthmap, mask   = load_depth(depth_path)

#             rgb_image = load_image(list_rgbs[i])

#             camera_info = load_camera_matrices(pose_path)
#             camera_dict  = camera_info[image_name]
#             K_pre = np.array(camera_dict['K'], dtype=np.float32)
#             RT_world2cam = np.array(camera_dict['RT'], dtype=np.float32)
#             distortion = np.array(camera_dict['distortion'], dtype=np.float32)
#             imsize_pre = np.array(camera_dict['size'], dtype=np.int32)


#             imsize_post = rgb_image.shape[1::-1]

#             K_post = cv2.getOptimalNewCameraMatrix(
#                     K_pre,
#                     distortion,
#                     imsize_pre,
#                     alpha=0,
#                     newImgSize=imsize_post,
#                     centerPrincipalPoint=True,
#                 )[0]




#             rng = np.random.default_rng(seed=42)
#             rgb_image_new, depth_new, K_new , _ = _crop_resize_if_necessary(image = rgb_image, depthmap=depthmap, intrinsics=K_post, mask = mask.astype(np.int32), resolution=(518, 518),rng=rng,info=None)
#             rgb_image_new = np.asarray(rgb_image_new)
#             RT_cam2world = np.linalg.inv(RT_world2cam)
#             RT_cam2world[:3, :3] = RT_cam2world[:3, :3] @ np.eye(3).T                                

#             if has_sufficient_valid_depth(depth_new, threshold=0.10, min_fraction=0.6) and is_valid_image(list_rgbs[i]):
#                 valid_indices.append(i)
#                 depths.append(depth_new)
#                 depth_mask = (depth_new > 0).astype(np.bool_)
#                 masks.append(depth_mask)
#                 cam_params.append((K_new, RT_cam2world))
#                 # rgbd_to_pointcloud(rgb = rgb_image_new, depth = depth_new,  mask = None,  K = K_new, RT_c2w = RT_cam2world,  save_path=f"megadepth_point_cloud_{i}.ply")
#                 # breakpoint()

            
    
#         # Initialize matrices
#         camera_overlap = np.full((N, N), -1.0, dtype=np.float32)
#         occlusion_matrix = np.full((N, N), -1.0, dtype=np.float32)


#         if skip_flag:
#             continue

#         for i in valid_indices:
#             camera_overlap[i, i] = 1.0
#             occlusion_matrix[i, i] = 0.0
    

#         for idx_i, target_idx in enumerate(tqdm(valid_indices, desc=f"Computing overlaps {scene_name}")):
#             depth_i = depths[idx_i] 
#             K_i, RT_c2w_i = cam_params[idx_i]
#             mask_i =  masks[idx_i]

#             for idx_j in range(idx_i + 1, len(valid_indices)):
#                 src_idx = valid_indices[idx_j]
#                 depth_j = depths[idx_j]
#                 K_j, RT_c2w_j = cam_params[idx_j]
#                 mask_j =  masks[idx_j]

#                 overlap_dict = compute_frustum_overlap(
#                     K1=K_i,
#                     K2=K_j,
#                     mask1 = mask_i,
#                     mask2 = mask_j,
#                     Rt1_c2w=RT_c2w_i,
#                     depth1=depth_i,
#                     Rt2_c2w=RT_c2w_j,
#                     depth2=depth_j,
#                 )
                
#                 # Fill symmetric entries
#                 camera_overlap[target_idx, src_idx] = overlap_dict["overlap1"]
#                 camera_overlap[src_idx, target_idx] = overlap_dict["overlap2"]
#                 occlusion_matrix[src_idx, target_idx] = overlap_dict["occlusion1"]
#                 occlusion_matrix[target_idx, src_idx] = overlap_dict["occlusion2"]
#                 # breakpoint()
#         # breakpoint()
#         np.save(overlap_file, camera_overlap)
#         np.save(occlusion_file, occlusion_matrix)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--start_idx", type=int, default=0, help="Start index of scenes to process")
#     parser.add_argument("--end_idx", type=int, default=None, help="End index of scenes to process (exclusive)")
#     parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/megadepth_scene_files_1000.pkl", help="Pickle file containing scene files")
#     args = parser.parse_args()

#     rgb_dir = "/projects/rps/dff6142/fouheylab/shared_datasets/MegaDepth/phoenix/S6/zl548/MegaDepth_v1/"

#     if not os.path.exists(args.scene_file):

#         all_scene_files = collect_scene_files(root_dir = rgb_dir)

#         with open(f"{args.scene_file}", "wb") as f:
#             pkl.dump(all_scene_files, f)


#     with open(args.scene_file, "rb") as f:
#         all_scene_files = pkl.load(f)
    

#     compute_scene_camera_overlaps(all_scene_files, start_idx=args.start_idx, end_idx=args.end_idx)



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
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import imageio.v2 as imageio
from typing import Dict, List
from collections import defaultdict
import numpy as np
import open3d as o3d
import h5py
from pathlib import Path
import utils.cropping as cropping
import torch.nn.functional as F
def compute_opengl_projection_matrix(
        fov,           # Taskonomy horizontal FOV
        znear, zfar,   # clipping planes
        width_pixels,
        height_pixels,
        degrees=False
    ):
    """
    Compute OpenGL projection matrix for Taskonomy cameras.
    Taskonomy provides ONLY the horizontal FOV (fov).

    Args:
        fov            : horizontal field of view (radians or degrees)
        znear, zfar    : near/far clipping planes
        width_pixels   : image width
        height_pixels  : image height
        degrees        : whether fov is in degrees

    Returns:
        4×4 OpenGL projection matrix
    """

    if degrees:
        fov = np.radians(fov)

    # Horizontal FOV from dataset
    fovx = fov

    # Derive vertical FOV using aspect ratio
    aspect = width_pixels / height_pixels
    fovy = 2.0 * np.arctan(np.tan(fovx / 2.0) / aspect)

    # Near plane extents
    f_h = np.tan(fovy / 2.0) * znear  # top/bottom
    f_w = np.tan(fovx / 2.0) * znear  # left/right

    left   = -f_w
    right  =  f_w
    bottom = -f_h
    top    =  f_h

    # OpenGL projection matrix
    K = np.zeros((4, 4), dtype=np.float32)
    K[0, 0] = 2.0 * znear / (right - left)
    K[1, 1] = 2.0 * znear / (top - bottom)
    K[0, 2] = (right + left) / (right - left)
    K[1, 2] = (top + bottom) / (top - bottom)
    K[2, 2] = -(zfar + znear) / (zfar - znear)
    K[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    K[3, 2] = -1.0

    return K


def opengl_kndc_to_opencv_K(K_ndc, W, H):
    """
    Convert an OpenGL KNDC (NDC) projection matrix to an OpenCV-style intrinsic matrix.

    Args:
        K_ndc: (4,4) OpenGL projection matrix (z in [-1,1])
        W: image width in pixels
        H: image height in pixels

    Returns:
        K: (3,3) OpenCV intrinsic matrix
    """
    # Horizontal and vertical focal lengths
    fx = K_ndc[0, 0] * W / 2.0
    fy = K_ndc[1, 1] * H / 2.0

    # Principal point
    cx = (1.0 - K_ndc[0, 2]) * W / 2.0
    cy = (1.0 + K_ndc[1, 2]) * H / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K





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


def load_valid_mask(png_path: str) -> np.ndarray:
    """
    Load a 1-channel 8-bit PNG as a boolean mask.
    0 -> False (missing geometry)
    255 -> True (valid geometry)
    
    Args:
        png_path: Path to the PNG file.
    
    Returns:
        mask: np.ndarray of shape (H, W), dtype=bool
    """
    mask = imageio.imread(png_path)  # shape (H, W), dtype=uint8
    mask = mask.astype(bool)      # 0->False, 255->True
    return mask




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




def load_image(file_path):

    """
    Load a JPEG image and return as a NumPy array (H, W, 3) in RGB order.
    """

    img = Image.open(file_path).convert("RGB")

    return np.array(img)



def load_depth(depth_path):
    """
    Loads a depth map from:
        root/depths/<tag_without_ext>.h5

    Parameters
    ----------
    root : str
        Dataset root directory

    Returns
    -------
    depthmap : np.ndarray
        Depth map as float32 array (H, W)
    """
    with h5py.File(depth_path, "r") as hd5:
        depthmap = np.asarray(hd5["depth"])

    mask = depthmap >0.00

    return depthmap, mask





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
    Convert an RGB file path into a Euclidean depth file path by replacing:
    - the first 'rgb' directory with 'depth_euclidean'
    - the 'rgb' in the filename with 'depth_euclidean'
    """
    # Replace the first 'rgb' directory
    depth_path = rgb_path.replace("/rgb/", "/depth_zbuffer/", 1)

    # Replace 'rgb' in the filename
    dir_name, filename = os.path.split(depth_path)
    filename = filename.replace("rgb", "depth_zbuffer")
    
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
    filename = filename.replace("rgb", "point_info")

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


def collect_scene_files(root_dir: str) -> Dict[str, List[str]]:
    """
    Map each scene directory of the CROPPED dataset to its rgb files.

    Cropped layout (dump_megadepth_crops.py):
        root_dir/<scene>_<denseX>/rgb/<name>.jpg
        root_dir/<scene>_<denseX>/depth/<name>.npy
        root_dir/<scene>_<denseX>/cameras.pkl

    Returns:
        Dict[str, List[str]]:
            scene directory name -> sorted list of rgb file paths in <scene>/rgb/
    """
    scene_files = {}

    for scene_name in sorted(os.listdir(root_dir)):
        rgb_dir = os.path.join(root_dir, scene_name, "rgb")
        if not os.path.isdir(rgb_dir):
            continue

        rgb_paths = [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir))
                     if os.path.isfile(os.path.join(rgb_dir, f))]
        if rgb_paths:
            scene_files[scene_name] = rgb_paths

    return scene_files



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


def rgbd_to_pointcloud(rgb, depth, K, RT_c2w, mask=None,   scale = 10.00, save_path="replica_point_cloud.ply"):
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

    return img_padded, depth_padded, mask_padded, K_new

def load_camera_matrices(pose_path):
   
    with open(pose_path, "r") as f:
        camera_data = json.load(f)

    return camera_data

def _downscale_image(camera_intrinsics, image, depthmap, resolution_out=(512, 384)):
    H, W = image.shape[:2]
    resolution_out = sorted(resolution_out)[:: +1 if W < H else -1]

    image, depthmap, intrinsics_out = cropping.rescale_image_depthmap(
        image, depthmap, camera_intrinsics, resolution_out, force=False
    )


    R_in2out = np.eye(3)

    return image, depthmap, intrinsics_out, R_in2out

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
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)

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
        
        return image, depthmap, intrinsics2, mask

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


def load_dumped_scene(scene_dir, rgb_paths, resolution=(518, 518)):
    """
    Read one scene written by dump_megadepth_crops.py, in the order of `rgb_paths`
    (the scene's list from collect_scene_files), and crop/resize every frame to `resolution`
    (default 518 x 518) with _crop_resize_if_necessary: centered on the principal point,
    K updated to the crop. RT_c2w is unchanged by cropping.

    Returns:
        names:      [rgb file names]                      -> row/col i of the overlap matrices
        depths:     [(518, 518) float32, 0 = invalid]
        masks:      [(518, 518) bool]                     valid depth after the crop
        cam_params: [(K (3,3), RT_c2w (4,4))]             K at the cropped resolution
    """
    with open(os.path.join(scene_dir, "cameras.pkl"), "rb") as f:
        cameras = pkl.load(f)

    rng = np.random.default_rng(seed=42)
    names = [os.path.basename(p) for p in rgb_paths]
    depths, masks, cam_params = [], [], []
    for rgb_path, rgb_name in zip(rgb_paths, names):
        cam = cameras[rgb_name]
        image = load_image(rgb_path)
        depth = np.load(os.path.join(scene_dir, "depth", cam["depth"])).astype(np.float32)
        K = np.asarray(cam["K"], np.float32)
        RT_c2w = np.asarray(cam["RT_c2w"], np.float32)
        mask = (depth > 0).astype(np.int32)

        # aug_crop=-1: deterministic crop (no random size jitter) for overlap computation
        _, depth_new, K_new, mask_new = _crop_resize_if_necessary(
            image=image, depthmap=depth, intrinsics=K, resolution=resolution,
            aug_crop=-1, mask=mask, rng=rng, info=rgb_name)

        depth_new = np.asarray(depth_new, np.float32)
        depths.append(depth_new)
        masks.append((np.asarray(mask_new) > 0) & (depth_new > 0))
        cam_params.append((np.asarray(K_new, np.float32), RT_c2w))
    return names, depths, masks, cam_params


def compute_scene_camera_overlaps(
    scene_files: Dict[str, List[str]],
    dump_root: str,
    overlap_basedir: str,
    start_idx: int = 0,
    end_idx: int = None
):
    """
    Compute camera overlaps and covisibility for a subset of DUMPED scenes.
    Images, depths and cameras come from dump_megadepth_crops.py (already resized, validated,
    undistorted K and RT_c2w already processed), so no camera preprocessing happens here.

    Args:
        scene_files: scene directory name -> rgb file paths (collect_scene_files on the cropped data)
        dump_root: folder with <scene>/{rgb, depth, cameras.pkl}
        overlap_basedir: output folder, one sub-folder per scene
        start_idx: start index of scenes to process
        end_idx: end index (exclusive) of scenes to process
    """
    scene_names = list(scene_files.keys())
    if end_idx is None or end_idx > len(scene_names):
        end_idx = len(scene_names)

    # Loop over only the specified range (end exclusive)
    for scene_name in scene_names[start_idx:end_idx]:

        scene_dir = os.path.join(dump_root, scene_name)
        overlaps_dir = os.path.join(overlap_basedir, scene_name)
        file_path = os.path.join(overlaps_dir, "camera_overlap.npy")

        if os.path.exists(file_path):
            print(f"{scene_name}: already processed!")
            continue
        if not os.path.exists(os.path.join(scene_dir, "cameras.pkl")):
            print(f"{scene_name}: not dumped yet, skipping")
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



               # Fill symmetric entries
                camera_overlap[target_idx, src_idx] = overlap_dict["overlap1"]
                camera_overlap[src_idx, target_idx] = overlap_dict["overlap2"]
                camera_occlusion[target_idx, src_idx] =  overlap_dict["occlusion1"]
                camera_occlusion[src_idx, target_idx] =  overlap_dict["occlusion2"]
            
            # breakpoint()
        # # row/col i of both matrices <-> names[i] (rgb file name, also the key in cameras.pkl)
        # with open(os.path.join(overlaps_dir, "frame_names.pkl"), "wb") as f:
        #     pkl.dump(names, f)
        np.save(os.path.join(overlaps_dir, "camera_occlusion.npy"), camera_occlusion)
        np.save(file_path, camera_overlap)            # written last: marks the scene as done



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of scenes to process")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of scenes to process (exclusive)")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/megadepth_low_res_overlap_scene_files.pkl", help="Pickle containing the scene files of the cropped dataset")
    parser.add_argument("--dump_root", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_megadepth", help="Output folder of compute_low_res_megadepth.py (the cropped dataset)")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/low_res_megadepth_overlaps/", help="Where to save the overlap matrices")
    args = parser.parse_args()

    # scene dictionary from the CROPPED dataset: scene directory name -> its rgb file paths
    if not os.path.exists(args.scene_file):

        all_scene_files = collect_scene_files(root_dir = args.dump_root)

        with open(f"{args.scene_file}", "wb") as f:
            pkl.dump(all_scene_files, f)


    with open(args.scene_file, "rb") as f:
        all_scene_files = pkl.load(f)

    # breakpoint()
    compute_scene_camera_overlaps(
        all_scene_files,
        dump_root=args.dump_root,
        overlap_basedir=args.overlap_basedir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )