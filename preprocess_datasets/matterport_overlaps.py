import argparse
import os
import os.path as osp
import pdb
import pickle as pkl
import shutil
import tempfile
from re import A

try:
    import imageio.v3 as imageio
except:
    import imageio

from pathlib import Path

import numpy as np
import torch
import trimesh
import time
from loguru import logger
from PIL import Image
import PIL
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import math
from einops import rearrange
import utils.matterport_parse as matterport_parse
# from occ.vggt.utils.geometry import closed_form_inverse_se3
# import occ.vggt.utils.cropping as cropping
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import json
from scipy.ndimage import median_filter
import torch.nn.functional as F
import utils.cropping as cropping



cachedir = "/projects/rps/dff6142/fouheylab/dma9300/occ_project/preprocessed_training_data/"

class Matterport3D:
    def __init__(
        self,
        root_dir,
    ):
        self.dataset_dir =  root_dir
        self.aug_crop = 16
        self.filter_keys = ["_i1_"]  # we are using only these as the i0 and i2 point to the bottom
        # ["_i0_", "_i1_", "_i2_"],
        split_dir = osp.join(cachedir, "splits")
        # split_file = osp.join(split_dir, "mp3d_split.pkl")
        split_file = osp.join(split_dir, "mp3d_official_split.pkl")
        house_ids = os.listdir(
            osp.join(self.dataset_dir, "undistorted_camera_parameters")
        )
        house_ids.sort()

        with open(split_file, "rb") as f:
            self.splits = pkl.load(f)

        # python -m occ.preprocess_datasets.preprocess_matterport --split train

        house2indexes = {}

        for house_id in house_ids:
            _indexed_items_all = matterport_parse.preload_conf_files(
                matterport_dir=self.dataset_dir,
                house_ids=[house_id],
                group_by_region=False,
            )
            house2indexes[house_id] = {}
            house2indexes[house_id]["all"] = _indexed_items_all
            _indexed_items_ref = matterport_parse.filter_indices(
                _indexed_items_all, self.filter_keys, keep_valid_regions=False
            )
            house2indexes[house_id]["ref"] = _indexed_items_ref

        self.house_ids = house_ids
        self.house2indexes = house2indexes
        self.faces_cachedir = osp.join(cachedir, "mp3d_data_zmx8")
    

        return
    
    def process_matterport_depth(self,depth_path):
        depth = imageio.imread(depth_path)
        depth = depth.astype(float) * 0.25
        depth = depth / 1000
        return depth

    def load_mesh(self, meta_data):
        house_id = meta_data["house_id"]
        mesh = self.houses2mesh[house_id]
        mesh = {
            "faces": mesh["faces"] * 1,
            "vertices": mesh["vertices"] * 1,
        }
    
        (
            fids,
            face_labels,
            face_segments,
            fids_c,
            fids_w,
        ) = matterport_parse.load_face_metadata(
            metadata=meta_data, face_cachedir=self.faces_cachedir
        )

        if fids is not None:
            # mesh['face_labels'] =
            mesh["faces"] = mesh["faces"][fids]
            mesh = matterport_parse.delete_unused_verts(
                mesh,
            )
            mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
            # print('time to load {}'.format(time.time()- sttime))
            return mesh
        else:
            return None

    @staticmethod
    def get_camera(camera_dict):
        cam2world = camera_dict["cam2world"]
        intrinsic = camera_dict["intrinsics"]
    
        Rx = trimesh.transformations.euler_matrix(np.pi, 0, 0, "sxyz")
        cam2world = np.matmul(cam2world, Rx)
        return cam2world, intrinsic

    @staticmethod
    def get_img(dataset_dir, metadata):
        img_path = MatterportPreprocess.get_img_path(
            dataset_dir=dataset_dir, metadata=metadata
        )
        img = imageio.imread(img_path)
        return img


    @staticmethod
    def get_depth_path(dataset_dir, metadata):
        house_id = metadata["house_id"]
        img_name = metadata["image_names"]
        img_dir = osp.join(
            dataset_dir,
            "undistorted_depth_images",
            house_id,
            "undistorted_depth_images",
        )
        img_name = img_name.replace("_i", "_d").replace(".jpg", ".png")
        depth_path = osp.join(img_dir, img_name)
        return depth_path

    @staticmethod
    def get_img_path(dataset_dir, metadata):
        house_id = metadata["house_id"]
        img_name = metadata["image_names"]
        img_dir = osp.join(
            dataset_dir,
            "undistorted_color_images",
            house_id,
            "undistorted_color_images",
        )
        img_path = osp.join(img_dir, img_name)
        return img_path


    def __len__(
        self,
    ):
        return len(self.house_ids)



    
    def has_valid_depth(self, depth_map):
        """
        Check if a depth map is valid based on given conditions.

        Args:
            depth_map (torch.Tensor or np.ndarray): The depth map.

        Returns:
            bool: True if the depth map is valid, False otherwise.
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()  # Convert to NumPy if it's a tensor

        if not isinstance(depth_map, np.ndarray):
            raise TypeError("Input must be a PyTorch tensor or NumPy array")

        depth1d = depth_map.flatten()  # Flatten depth values

        # More than 60% of values are less than 2 and more than 20% of the values are less than 0.0
        if np.mean(depth1d >= 2.0) < 0.6 and np.mean(depth1d == 0.0) > 0.2:
            return False

        return True





    def project_to_world_points(self, depth: np.ndarray, K: np.ndarray, RT_c2w: np.ndarray,depth_mask: np.ndarray,) -> np.ndarray:
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


    def rotation_inverse(self, RT_c2w: np.ndarray) -> np.ndarray:
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

    def project_points(self, K: np.ndarray, RT_c2w: np.ndarray, world_points: np.ndarray, image_size: tuple):
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
        RT_w2c = self.rotation_inverse(RT_c2w)  # (4, 4)

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
        self,
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

    
    def pixel_to_ndc(self, uv, image_size):
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

    



    def interpolate_depth(self, depth, uv):
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
        uv  = self.pixel_to_ndc(uv,depth.shape)

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


    # def compute_frustum_overlap(
    #     self,
    #     K1,
    #     K2,
    #     Rt1_c2w,
    #     depth1,
    #     Rt2_c2w,
    #     depth2,
    #     mask1,
    #     mask2,
    # ):
    #     """
    #     Compute directional camera overlap using depth maps.

    #     overlap1:
    #         Fraction of valid 3D points observed by camera 1
    #         that are also visible in camera 2.

    #     overlap2:
    #         Fraction of valid 3D points observed by camera 2
    #         that are also visible in camera 1.

    #     Returns:
    #         dict:
    #             overlap1: camera 1 -> camera 2
    #             overlap2: camera 2 -> camera 1
    #     """

    #     # Valid depth
    #     mask1 = mask1 & (depth1 > 0.01)
    #     mask2 = mask2 & (depth2 > 0.01)

    #     # ---------------------------------------------------------
    #     # Camera 1 points -> world
    #     # ---------------------------------------------------------
    #     world_points1 = self.project_to_world_points(
    #         depth=depth1,
    #         K=K1,
    #         RT_c2w=Rt1_c2w,
    #         depth_mask=mask1,
    #     )

    #     # Camera 2 points -> world
    #     world_points2 = self.project_to_world_points(
    #         depth=depth2,
    #         K=K2,
    #         RT_c2w=Rt2_c2w,
    #         depth_mask=mask2,
    #     )

        # # ---------------------------------------------------------
        # # Project camera-1 points into camera 2
        # # ---------------------------------------------------------
        # uv1, z1 = self.project_points(
        #     K=K2,
        #     RT_c2w=Rt2_c2w,
        #     world_points=world_points1,
        #     image_size=depth2.shape,
        # )

        # # Project camera-2 points into camera 1
        # # ---------------------------------------------------------
        # uv2, z2 = self.project_points(
        #     K=K1,
        #     RT_c2w=Rt1_c2w,
        #     world_points=world_points2,
        #     image_size=depth1.shape,
        # )

    #     # ---------------------------------------------------------
    #     # Visibility
    #     # ---------------------------------------------------------
    #     vis1 = self.compute_points_visibility(
    #         uv=uv1,
    #         z=z1,
    #         depthmap=depth2,
    #         depth_mask=mask2,
    #     )

    #     vis2 = self.compute_points_visibility(
    #         uv=uv2,
    #         z=z2,
    #         depthmap=depth1,
    #         depth_mask=mask1,
    #     )

    #     # ---------------------------------------------------------
    #     # Directional overlap
    #     # ---------------------------------------------------------

    #     num_valid1 = np.sum(mask1)
    #     num_valid2 = np.sum(mask2)

    #     if num_valid1 > 0:
    #         overlap1 = np.sum(vis1) / num_valid1
    #     else:
    #         overlap1 = 0.0

    #     if num_valid2 > 0:
    #         overlap2 = np.sum(vis2) / num_valid2
    #     else:
    #         overlap2 = 0.0

    #     return {
    #         "overlap1": float(overlap1),  # camera 1 -> camera 2
    #         "overlap2": float(overlap2),  # camera 2 -> camera 1
    #     }


    def points_inside_image(self, uv, image_size):
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

    def compute_frustum_overlap(
        self,
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

        world_points1 = self.project_to_world_points(
            depth=depth1,
            K=K1,
            RT_c2w=Rt1_c2w,
            depth_mask=mask1,
        )

        world_points2 = self.project_to_world_points(
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

        uv12, z12 = self.project_points(
            K=K2,
            RT_c2w=Rt2_c2w,
            world_points=world_points1,
            image_size=depth2.shape,
        )

        uv12 = np.asarray(uv12)
        z12 = np.asarray(z12).reshape(-1)

        inside2 = np.asarray(
            self.points_inside_image(
                uv12,
                image_size=depth2.shape,
            )
        ).astype(bool)

        # breakpoint()

        target_depth2 = np.asarray(self.interpolate_depth(depth2,uv12)).reshape(-1)

        target_mask2 = (
            np.asarray(
                self.interpolate_depth(
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

        uv21, z21 = self.project_points(
            K=K1,
            RT_c2w=Rt1_c2w,
            world_points=world_points2,
            image_size=depth1.shape,
        )

        uv21 = np.asarray(uv21)
        z21 = np.asarray(z21).reshape(-1)

        inside1 = np.asarray(
            self.points_inside_image(
                uv21,
                image_size=depth1.shape,
            )
        ).astype(bool)

        target_depth1 = np.asarray(self.interpolate_depth(depth1, uv21)).reshape(-1)

        target_mask1 = (
            np.asarray(
                self.interpolate_depth(
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





    def has_sufficient_valid_depth(self, depth: np.ndarray, threshold: float = 0.10, min_fraction: float = 0.6) -> bool:
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


    def rgbd_to_pointcloud(self, rgb, depth, K, RT_c2w, mask=None,  save_path="matterport3d_point_cloud.ply"):
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

        
    def is_valid_image(self, image, valid_pixels_frac=0.05):
        """
        Check if the fraction of black pixels in an RGB image is < valid_pixels_frac.

        Args:
            image  np.ndarray:  RGB image.
            valid_pixels_frac (float): Maximum allowed fraction of black pixels.

        Returns:
            bool: True if black pixel fraction < valid_pixels_frac, else False.
        """


        # Load image as RGB
        image = np.asarray(image)

        # Ensure it's RGB
        if image.ndim != 3 :
            raise ValueError("Image must be an RGB image of shape (H, W, 3).")

        # Black pixel = (0,0,0) in all channels
        black_mask = np.all(image < 5, axis=-1)
        black_pixels = np.sum(black_mask)

        total_pixels = image.shape[0] * image.shape[1]
        fraction_black = black_pixels / total_pixels

        return fraction_black < valid_pixels_frac
            


    def compute_camera_overlaps(self, index_start = 0, end_index = None):

        for index in range(index_start, end_index+1):

            house_id = self.house_ids[index]

            overlap_basedir = "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/mv_drdf_supervision_data/datasets_camera_overlaps_all_views_256/"
            overlaps_dir = os.path.join(overlap_basedir, house_id)
            
            file_path = os.path.join(overlaps_dir, "camera_overlap.npy")
            camera_occlusion_file_path = os.path.join(overlaps_dir, "camera_occlusion.npy")


            if os.path.exists(file_path):
                print("File already processed!")
                continue
            
            os.makedirs(overlaps_dir, exist_ok=True)
            # os.makedirs(camera_occlusion_file_path, exist_ok=True)


            indexed_items_all = self.house2indexes[house_id]["all"]
            indexed_items_all = matterport_parse.filter_indices(
                    indexed_items_all, self.filter_keys, keep_valid_regions=False
                )

            # subsample 1000 cameras
            max_cams = 1000
            num_cams = len(indexed_items_all)
            # breakpoint()

            if num_cams > max_cams:
                indices = np.linspace(0, num_cams - 1, max_cams, dtype=int)
                indexed_items_all = [indexed_items_all[i] for i in indices]


            order = [indexed_item["image_names"] for indexed_item in indexed_items_all]
            num_cams = len(order)

            cameras = [self.get_camera(indexed_item) for indexed_item in indexed_items_all]
            img_size = [1024, 1280]
            intrinsics = [camera[1] for camera in cameras]
            RTs = [camera[0] for camera in cameras]  # cam2world

            rgb_img_paths = [self.get_img_path(self.dataset_dir, indexed_item) for indexed_item in indexed_items_all]
            depth_img_paths = [self.get_depth_path(self.dataset_dir, item) for item in indexed_items_all]
            rgb_images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in rgb_img_paths]
            depth_maps = [self.process_matterport_depth(path) for path in depth_img_paths]

            camera_overlap = np.full((num_cams, num_cams), -1.0)
            camera_occlusion = np.full((num_cams, num_cams), -1.0)
            rng = np.random.default_rng(seed=42)

            valid_rgb_images  = []
            valid_depth_images  = []
            valid_intrinsics  = []
            valid_indices = []
            valid_RTs =[]
            valid_depth_masks = []
            for i in tqdm(range(num_cams), desc=f"Preloading {house_id}"):
                rgb_image = cv2.cvtColor(cv2.imread(rgb_img_paths[i]), cv2.COLOR_BGR2RGB)
                depth_map = depth_maps[i]
                depth_mask = depth_map >= 0.00

                resize_rgb_image, resize_depth_map, new_intriniscs, resize_depth_mask = self.crop_resize(
                    image = rgb_image, depthmap = depth_map, intrinsics = intrinsics[i], mask = depth_mask.astype(np.int32),  resolution=(518, 518), rng=rng, info=None
                )

                if self.has_sufficient_valid_depth(resize_depth_map, threshold = 0.10, min_fraction = 0.6) and self.check_for_less_black_pixels(resize_rgb_image):
                    valid_rgb_images.append(np.array(resize_rgb_image))
                    valid_depth_images.append(resize_depth_map)
                    valid_intrinsics.append(new_intriniscs)
                    valid_indices.append(i)
                    valid_RTs.append(RTs[i])
                    valid_depth_masks.append(resize_depth_mask.astype(np.bool_))


            for idx_i, i in enumerate(tqdm(valid_indices, desc=f"Computing overlaps {house_id}")):
                depth_i = valid_depth_images[idx_i]  # corrected indexing
                mask_i = valid_depth_masks[idx_i]
                K_i = valid_intrinsics[idx_i]
                RT_c2w_i = valid_RTs[idx_i]
                rgb_image_i = valid_rgb_images[idx_i]


                # self.rgbd_to_pointcloud(rgb = rgb_image_i, depth = depth_i, K = K_i, RT_c2w = RT_c2w_i,  save_path=f"matterport3d_point_cloud_{idx_i}.ply")
                # plt.imshow(rgb_image_i)
                # plt.savefig(f"matterport3d_image_{idx_i}.png")
                # plt.close()
                # breakpoint()

                for valid_idx in valid_indices:
                    camera_overlap[valid_idx, valid_idx] = 1.0
                    # covisibility[valid_idx, valid_idx] = 1.0
                    # frustum_iou[valid_idx, valid_idx] = 1.0

                

                for idx_j in range(idx_i + 1, len(valid_indices)):
                    j = valid_indices[idx_j]
                    depth_j = valid_depth_images[idx_j]
                    mask_j = valid_depth_masks[idx_j]
                    K_j = valid_intrinsics[idx_j]
                    RT_c2w_j = valid_RTs[idx_j] 

                    overlap_dict = self.compute_frustum_overlap(
                        K1=K_i,
                        K2=K_j,
                        mask1 = mask_i,
                        mask2 = mask_j,
                        Rt1_c2w = RT_c2w_i, 
                        depth1 = depth_i,
                        Rt2_c2w = RT_c2w_j, 
                        depth2 = depth_j,
                    )

                   
                    # Fill symmetric entries
        
                    camera_overlap[i, j] = overlap_dict["overlap1"]
                    camera_overlap[j, i] = overlap_dict["overlap2"]
                    camera_occlusion[i, j] =  overlap_dict["occlusion1"]
                    camera_occlusion[j, i] =  overlap_dict["occlusion2"]
                # breakpoint()
            np.save(os.path.join(overlaps_dir, "camera_overlap.npy"), camera_overlap)
            np.save(os.path.join(overlaps_dir, "camera_occlusion.npy"), camera_occlusion)

    

    # ------------------------------------------------------------------
    # DUMP: resized rgb / depth / cameras per house (same format as Hypersim)
    # ------------------------------------------------------------------
    def dump_houses(self, out_root, index_start=0, end_index=None, max_images=1000,
                    short_side=518, jpg_quality=95):
        """
        Dump houses [index_start, end_index) (end exclusive) to

            <out_root>/<house_id>/rgb/<name>.jpg      rgb, short side `short_side`, aspect kept
            <out_root>/<house_id>/depth/<name>.npy    depth, float32, meters, 0 = invalid
            <out_root>/<house_id>/cameras.pkl         written last (its presence = house done):
                {"<name>.jpg": {"K":       (3,3) at dumped resolution (OpenCV),
                                "RT_c2w":  (4,4) OpenCV camera-to-world,
                                "hw":      dumped (H, W),
                                "depth":   "<name>.npy",
                                "src_rgb": original rgb path}}

        Per house: the "_i1_" views (self.filter_keys) are ALL checked (valid depth fraction,
        not too many black pixels, not a wall-like depth map, finite camera), then up to
        `max_images` good views are kept, evenly spaced, and resized so the short side is
        `short_side`. Every dumped view passed every check.
        """
        if end_index is None or end_index > len(self.house_ids):
            end_index = len(self.house_ids)

        for index in range(index_start, end_index):
            house_id = self.house_ids[index]
            house_dir = os.path.join(out_root, house_id)
            cameras_file = os.path.join(house_dir, "cameras.pkl")

            if os.path.exists(cameras_file):
                print(f"{house_id}: done, skipping")
                continue

            indexed_items = self.house2indexes[house_id]["all"]
            indexed_items = matterport_parse.filter_indices(
                indexed_items, self.filter_keys, keep_valid_regions=False
            )

            # 1. check every view, keep the good ones
            good_items = []
            for item in tqdm(indexed_items, desc=f"{house_id} [check]"):
                try:
                    depth = self.process_matterport_depth(self.get_depth_path(self.dataset_dir, item))
                    rgb = cv2.cvtColor(cv2.imread(self.get_img_path(self.dataset_dir, item)), cv2.COLOR_BGR2RGB)
                    RT_c2w, K = self.get_camera(item)
                except Exception as e:
                    print(f"  unreadable {item.get('image_names')}: {e}")
                    continue

                # all checks live here, so every dumped view can be used as target or source:
                #   - enough valid depth                      (compute_camera_overlaps)
                #   - not too many black pixels (< 10%)       (compute_camera_overlaps)
                #   - not too many black pixels (< 5%)        (build_overlap_sequences: is_valid_image)
                #   - not a wall-like depth map               (build_overlap_sequences: is_wall_depthmap)
                #   - finite camera
                if (self.has_sufficient_valid_depth(depth, threshold=0.10, min_fraction=0.6)
                        and self.is_valid_image(rgb)
                        and not self.is_wall_depthmap(depth)
                        and np.all(np.isfinite(K)) and np.all(np.isfinite(RT_c2w))):
                    good_items.append(item)

            # 2. keep up to max_images good views, evenly spaced
            if len(good_items) > max_images:
                idxs = np.linspace(0, len(good_items) - 1, max_images, dtype=int)
                good_items = [good_items[i] for i in idxs]

            print(f"{house_id}: {len(indexed_items)} views -> {len(good_items)} good kept")
            if not good_items:
                continue

            # 3. resize + dump
            for sub in ("rgb", "depth"):
                os.makedirs(os.path.join(house_dir, sub), exist_ok=True)

            cameras = {}
            for item in tqdm(good_items, desc=f"{house_id} [dump]"):
                rgb_path = self.get_img_path(self.dataset_dir, item)
                name = self.get_image_name(rgb_path)
                rgb_name = name + ".jpg"
                assert rgb_name not in cameras, f"duplicate frame name {rgb_name} in {house_id}"

                depth = self.process_matterport_depth(self.get_depth_path(self.dataset_dir, item)).astype(np.float32)
                depth[~np.isfinite(depth)] = 0.0
                image = Image.open(rgb_path).convert("RGB")
                RT_c2w, K = self.get_camera(item)
                K = np.asarray(K, np.float32)

                # short side -> short_side, aspect kept (scale = max(out / in)), depth: nearest
                image, depth, K = cropping.rescale_image_depthmap(
                    image, depth, K, (short_side, short_side), force=True)[:3]
                depth = np.asarray(depth, np.float32)


                image.save(os.path.join(house_dir, "rgb", rgb_name), quality=jpg_quality)
                np.save(os.path.join(house_dir, "depth", name + ".npy"), depth)

                cameras[rgb_name] = {
                    "K": np.asarray(K, np.float32),              # (3,3) at dumped resolution, OpenCV
                    "RT_c2w": np.asarray(RT_c2w, np.float32),    # (4,4) OpenCV camera-to-world
                    "hw": tuple(int(x) for x in depth.shape),    # dumped (H, W)
                    "depth": name + ".npy",                      # file in depth/
                    "src_rgb": rgb_path,                         # original rgb path
                }

            # one camera file per house, written last: its presence marks the house as done
            tmp = cameras_file + ".tmp"
            with open(tmp, "wb") as f:
                pkl.dump(cameras, f)
            os.replace(tmp, cameras_file)
            print(f"{house_id}: dumped {len(cameras)} views")


    # ------------------------------------------------------------------
    # OVERLAPS on the DUMPED data (every dumped view already passed all checks)
    # ------------------------------------------------------------------
    @staticmethod
    def collect_dump_scene_files(dump_root):
        """
        Map each house directory of the dump to its rgb files:
            dump_root/<house_id>/rgb/<name>.jpg  ->  {house_id: [sorted rgb paths]}
        Houses without a finished dump (no cameras.pkl) are left out.
        """
        scene_files = {}
        for house_id in sorted(os.listdir(dump_root)):
            house_dir = os.path.join(dump_root, house_id)
            rgb_dir = os.path.join(house_dir, "rgb")
            if not (os.path.isdir(rgb_dir) and os.path.exists(os.path.join(house_dir, "cameras.pkl"))):
                continue
            rgb_paths = [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir))
                         if os.path.isfile(os.path.join(rgb_dir, f))]
            if rgb_paths:
                scene_files[house_id] = rgb_paths
        return scene_files

    def compute_overlaps_from_dump_files(self, dump_root, scene_file, overlap_basedir,
                                         index_start=0, end_index=None, resolution=(518, 518)):
        """
        Compute camera overlap / occlusion matrices on the dumped views.

        1. scene file: load `scene_file` if it exists, otherwise build it from the dump
           ({house_id: [rgb paths]}) and save it
        2. for houses [index_start, end_index) (end exclusive): read rgb, depth (meters), K and
           RT_c2w from the dump, crop/resize every view to `resolution` (default 518 x 518) with
           crop_resize (centered on the principal point, K updated, no random jitter), compute
           compute_frustum_overlap for every pair, and save

            <overlap_basedir>/<house_id>/camera_overlap.npy     (N, N), written last (= done)
            <overlap_basedir>/<house_id>/camera_occlusion.npy   (N, N)
            <overlap_basedir>/<house_id>/frame_names.pkl        row/col i <-> rgb file name i
        """
        if not os.path.exists(scene_file):
            scene_files = self.collect_dump_scene_files(dump_root)
            with open(scene_file, "wb") as f:
                pkl.dump(scene_files, f)
            print(f"wrote {scene_file}: {len(scene_files)} houses")


        with open(scene_file, "rb") as f:
            scene_files = pkl.load(f)

        house_ids = list(scene_files.keys())
        if end_index is None or end_index > len(house_ids):
            end_index = len(house_ids)
        
        for house_id in house_ids[index_start:end_index + 1]:
            house_dir = os.path.join(dump_root, house_id)
            overlaps_dir = os.path.join(overlap_basedir, house_id)
            overlap_file = os.path.join(overlaps_dir, "camera_overlap.npy")

            if os.path.exists(overlap_file):
                print(f"{house_id}: already processed!")
                continue
            os.makedirs(overlaps_dir, exist_ok=True)

            with open(os.path.join(house_dir, "cameras.pkl"), "rb") as f:
                cameras = pkl.load(f)

            # frame order = scene file order
            names = [os.path.basename(p) for p in scene_files[house_id]]

            # crop every view to `resolution`; deterministic crops (no aug_crop jitter) for overlaps
            aug_crop = getattr(self, "aug_crop", -1)
            self.aug_crop = -1
            rng = np.random.default_rng(seed=42)
            depths, masks, Ks, RTs = [], [], [], []
            for rgb_path, n in zip(scene_files[house_id], names):
                image = np.array(Image.open(rgb_path).convert("RGB"))
                depth = np.load(os.path.join(house_dir, "depth", cameras[n]["depth"])).astype(np.float32)
                K = np.asarray(cameras[n]["K"], np.float32)
                mask = (depth > 0).astype(np.int32)

                # breakpoint()

                _, depth_new, K_new, mask_new = self.crop_resize(
                    image=image, depthmap=depth, intrinsics=K, resolution=resolution,
                    mask=mask, rng=rng, info=n)

                depth_new = np.asarray(depth_new, np.float32)
                depths.append(depth_new)
                masks.append((np.asarray(mask_new) > 0) & (depth_new > 0))
                Ks.append(np.asarray(K_new, np.float32))
                RTs.append(np.asarray(cameras[n]["RT_c2w"], np.float32))     # cropping does not change RT
            self.aug_crop = aug_crop
            num_cams = len(names)

            camera_overlap = np.full((num_cams, num_cams), -1.0, dtype=np.float32)
            camera_occlusion = np.full((num_cams, num_cams), -1.0, dtype=np.float32)
            np.fill_diagonal(camera_overlap, 1.0)
            np.fill_diagonal(camera_occlusion, 0.0)

            for i in tqdm(range(num_cams), desc=f"Computing overlaps {house_id}"):
                for j in range(i + 1, num_cams):
                    overlap_dict = self.compute_frustum_overlap(
                        K1=Ks[i],
                        K2=Ks[j],
                        mask1=masks[i],
                        mask2=masks[j],
                        Rt1_c2w=RTs[i],
                        depth1=depths[i],
                        Rt2_c2w=RTs[j],
                        depth2=depths[j],
                    )
                    # Fill symmetric entries (same convention as compute_camera_overlaps)
                    camera_overlap[i, j] = overlap_dict["overlap1"]
                    camera_overlap[j, i] = overlap_dict["overlap2"]
                    camera_occlusion[i, j] = overlap_dict["occlusion1"]
                    camera_occlusion[j, i] = overlap_dict["occlusion2"]

            # with open(os.path.join(overlaps_dir, "frame_names.pkl"), "wb") as f:
            #     pkl.dump(names, f)
            np.save(os.path.join(overlaps_dir, "camera_occlusion.npy"), camera_occlusion)
            np.save(overlap_file, camera_overlap)          # written last: marks the house as done
            print(f"{house_id}: overlaps for {num_cams} views saved to {overlaps_dir}")


    def get_image_name(self, path):
        return path.split("/")[-1].split(".")[0]
        


        
    def crop_resize(self, image, depthmap, intrinsics, resolution, mask = None, rng=np.random.default_rng(seed=42), info=None):
        """
        Crops and resizes an image and its corresponding depth map while adjusting camera intrinsics accordingly.

        Parameters:
        - image (PIL.Image.Image or np.array): The input RGB image.
        - depthmap (np.array): The depth map corresponding to the image.
        - intrinsics (np.array): Camera intrinsic matrix (3x3).
        - resolution (tuple): Desired output resolution (width, height).
        - rng (np.random.Generator, optional): Random number generator (default: fixed seed for reproducibility).
        - info (str, optional): Additional info for debugging (e.g., image ID or view metadata).

        Returns:
        - image (PIL.Image.Image): The cropped and resized image.
        - depthmap (np.array): The corresponding resized depth map.
        - intrinsics2 (np.array): Updated camera intrinsic matrix adjusted for cropping and resizing.
        """

        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Get original image dimensions
        W, H = image.size

        # Compute the principal point
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)

        # Ensure the principal point is valid
        assert min_margin_x > W / 5, f'Bad principal point in view={info}'
        assert min_margin_y > H / 5, f'Bad principal point in view={info}'

        # Define the cropping region centered on the principal point
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        # Crop image and depth map
        if mask is not None:
            image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)
        else:
            image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)


        # Adjust resolution based on image orientation
        W, H = image.size  # Updated image size
        assert resolution[0] >= resolution[1]

        if H > 1.1 * W:
            resolution = resolution[::-1]  # Flip resolution for portrait mode
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # Randomly choose between portrait and landscape for square images
            if rng.integers(2):
                resolution = resolution[::-1]

        # Apply slight augmentation to target resolution if self.aug_crop > 1
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)

        # Rescale image and depth map to the new resolution
        if mask is not None:
            image, depthmap, intrinsics, mask  = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, mask = mask)
        else:
            image, depthmap, intrinsics  = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, mask = mask)



        # Compute final cropping parameters based on adjusted intrinsics
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor = 0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)

        # Apply final cropping
        if mask is not None:
            image, depthmap, intrinsics2, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, mask = mask)

        return image, depthmap, intrinsics2, mask

  

    def check_for_less_black_pixels(self, image, valid_pixels_frac=0.10):
        """
        Check if the fraction of black pixels in an RGB image is < valid_pixels_frac.

        Args:
            image (str): RGB image.
            valid_pixels_frac (float): Maximum allowed fraction of black pixels.

        Returns:
            bool: True if black pixel fraction < valid_pixels_frac, else False.
        """

    
        # Ensure it's RGB
        image = np.asarray(image)
        if  image.shape[2] != 3:
            raise ValueError("Image must be an RGB image of shape (H, W, 3).")

        # Black pixel = (0,0,0) in all channels
        black_mask = np.all(image < 5, axis=-1)
        black_pixels = np.sum(black_mask)

        total_pixels = image.shape[0] * image.shape[1]
        fraction_black = black_pixels / total_pixels

        return fraction_black < valid_pixels_frac


  

    def is_wall_depthmap(self, depth, tol=0.05, wall_frac_thresh=0.5):
        """
        Decide whether a depth map corresponds to a wall-like view.

        Args:
            path :  depth path
            tol: relative tolerance (e.g. 0.05 = ±5% of median)
            wall_frac_thresh: fraction of pixels near median above which we call it a wall

        Returns:
            bool: True if wall-like, False otherwise
        """

        valid = depth > 0.1
        d = depth[valid]

        if len(d) == 0:
            return True  # treat invalid depth as wall/bad

        med = np.median(d)
        close = np.abs(d - med) < tol * med

        frac_close = close.mean()

        return frac_close >= wall_frac_thresh

    def is_wall_heavy(self, depth_map, threshold=0.6):
        """Returns True if frame should be eliminated (too much wall)"""

        
        # Clean the depth map
        cleaned = median_filter(depth_map, size=5)
        
        # Find local median to detect flat surfaces
        local_median = median_filter(cleaned, size=15)
        flatness = np.abs(cleaned - local_median)
        
        # Identify flat regions (likely walls)
        is_flat = flatness < 0.03  # adjust based on your sensor noise
        
        # Calculate percentage of frame that's flat
        valid_pixels = np.sum(depth_map > 0.01)
        flat_pixels = np.sum(is_flat & (depth_map > 0.01))
        wall_percentage = flat_pixels / valid_pixels if valid_pixels > 0 else 0
        
        # Reject if too much wall
        return wall_percentage > threshold



    def build_overlap_sequences(
        self,
        iou_threshold=0.010,
        max_sources=100,
        split="train",
    ):
        """
            Build image-depth-camera sequences from multiple sequences of 1000 images each.

        """

        results = {}
        house_ids = sorted(self.splits[split])
        rng = np.random.default_rng(seed=42)


        for house_id in tqdm(house_ids,
                            desc=f"Processing Matterport {split} dataset",
                            leave=False):

            # ----------------------------------------------------
            # LOAD + FILTER INDICES 
            # ----------------------------------------------------
            indexed_items = self.house2indexes[house_id]["all"]
            indexed_items = matterport_parse.filter_indices(
                indexed_items,
                self.filter_keys,
                keep_valid_regions=False
            )

            # Subsample to 1000 views 
            if len(indexed_items) > 1000:
                idxs = np.linspace(0, len(indexed_items) - 1, 1000, dtype=int)
                indexed_items = [indexed_items[i] for i in idxs]

            num_images = len(indexed_items)

            # ----------------------------------------------------
            # LOAD OVERLAP MATRICES (Replica style)
            # ----------------------------------------------------
            base = f"/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/matterport3d_overlaps/{house_id}"


            if not os.path.exists(f"{base}/camera_occlusion.npy"):
                print("SKIP HOUSE!")
                continue

            occlusion_matrix = np.load(f"{base}/camera_occlusion.npy")
            camera_overlaps = np.load(f"{base}/camera_overlap.npy")
        

            # PRELOAD EVERYTHING
            # Image paths
            img_paths = [
                self.get_img_path(self.dataset_dir, item)
                for item in indexed_items
            ]

            # Depth paths
            depth_paths = [
                self.get_depth_path(self.dataset_dir, item)
                for item in indexed_items
            ]


            cameras = [self.get_camera(indexed_item) for indexed_item in indexed_items]
            img_size = [1024, 1280]
            intrinsics = [camera[1] for camera in cameras]
            RTs = [camera[0] for camera in cameras]  # cam2world


            # no per-view checks here: depth / black-pixel / wall checks are done once in
            # dump_houses, and every dumped view has passed them

            # ----------------------------------------------------
            # MAIN LOOP
            # ----------------------------------------------------
            for target_idx in tqdm(
                range(num_images),
                desc=f"House {house_id}",
                leave=False
            ):

                ious = camera_overlaps[target_idx]
                camera_occlusion = occlusion_matrix[target_idx]

                # Filter by overlap threshold
                valid_indices = np.where(
                    (ious >= iou_threshold)
        
                )[0]



                # Remove self
                valid_indices = valid_indices[valid_indices != target_idx]

                if len(valid_indices) == 0:
                    continue

                # Sort by IoU only (descending)
                sorted_indices = valid_indices[np.lexsort(( -camera_occlusion[valid_indices],-ious[valid_indices]))]
                # Sort by overlap (primary), occlusion as tie-breaker
               
               

                selected_sources = sorted_indices[:max_sources]

                

                # Target info (from cache)
                target_img = img_paths[target_idx]
                target_depth = depth_paths[target_idx]
                K_t = intrinsics[target_idx]
                RT_t = RTs[target_idx]

                sources = []
                for src_idx in selected_sources:

                    sources.append({
                        "image": img_paths[src_idx],
                        "depth": depth_paths[src_idx],
                        "K": intrinsics[src_idx],
                        "extrinsics": RTs[src_idx],
                        "occlusion": float(occlusion_matrix[target_idx, src_idx]),
                        "overlap": float(camera_overlaps[target_idx, src_idx]),
                    })

                if len(sources) == 0:
                    continue

                target_name = img_paths[target_idx].split("/")[-1].split(".")[0]
                key_name = f"{house_id}__{target_name}"

                results[key_name] = {
                    "target": {
                        "image": target_img,
                        "depth": target_depth,
                        "K": K_t,
                        "extrinsics": RT_t,
                    },
                    "sources": sources,
                }

        # ----------------------------------------------------
        # SAVE (Replica style)
        # ----------------------------------------------------
        save_path = (f"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/preprocessed_matterport3d_{split}_set.pkl")

        with open(save_path, "wb") as f:
            pkl.dump(results, f)

        print(f"\nSaved {len(results)} sequences to {save_path}")
        return results


    # ------------------------------------------------------------------
    # SEQUENCES from the DUMPED data + its overlaps (all preprocessing already done)
    # ------------------------------------------------------------------
    @staticmethod
    def build_sequences_from_dump(dump_root, scene_file, overlap_basedir, split="train",
                                  iou_threshold=0.010, max_sources=100,
                                  split_file=osp.join(cachedir, "splits", "mp3d_official_split.pkl")):
        """
        Build target / source sequences from the dump and the overlaps computed on it
        (compute_overlaps_from_dump_files). Nothing is re-checked or re-loaded from the raw data:

            <dump_root>/<house_id>/rgb/<name>.jpg
            <dump_root>/<house_id>/depth/<name>.npy      float32, meters, 0 = invalid
            <dump_root>/<house_id>/cameras.pkl           {"<name>.jpg": {"K", "RT_c2w" (OpenCV), "hw", "depth", "src_rgb"}}
            <overlap_basedir>/<house_id>/camera_overlap.npy, camera_occlusion.npy
        Row / column i of the matrices <-> scene_files[house_id][i] (the scene file the overlaps used).
        """
        with open(split_file, "rb") as f:
            house_ids = sorted(pkl.load(f)[split])
        with open(scene_file, "rb") as f:
            scene_files = pkl.load(f)

        results = {}
        for house_id in tqdm(house_ids, desc=f"Processing Matterport {split} (dump)"):
            house_dir = os.path.join(dump_root, house_id)
            overlaps_dir = os.path.join(overlap_basedir, house_id)

            if house_id not in scene_files or not os.path.exists(os.path.join(overlaps_dir, "camera_overlap.npy")):
                print(f"{house_id}: no dump / overlaps, skipping")
                continue

            # ---------------- overlaps ----------------
            camera_overlaps = np.load(os.path.join(overlaps_dir, "camera_overlap.npy"))
            occlusion_matrix = np.load(os.path.join(overlaps_dir, "camera_occlusion.npy"))

            img_paths = scene_files[house_id]
            names = [os.path.basename(p) for p in img_paths]
            num_images = len(img_paths)
            assert camera_overlaps.shape == (num_images, num_images), \
                f"{house_id}: overlap matrix {camera_overlaps.shape} vs {num_images} images"

            # ---------------- cameras (already processed) ----------------
            with open(os.path.join(house_dir, "cameras.pkl"), "rb") as f:
                cameras = pkl.load(f)
            cams = [cameras[n] for n in names]
            intrinsics = [np.asarray(c["K"], np.float32) for c in cams]
            RTs = [np.asarray(c["RT_c2w"], np.float32) for c in cams]               # cam2world, OpenCV
            depth_paths = [os.path.join(house_dir, "depth", c["depth"]) for c in cams]

            # ---------------- sequences ----------------
            for target_idx in range(num_images):
                ious = camera_overlaps[target_idx]
                camera_occlusion = occlusion_matrix[target_idx]

                # sources: overlap above the threshold, not the target itself
                valid_indices = np.where(ious >= iou_threshold)[0]
                valid_indices = valid_indices[valid_indices != target_idx]
                if len(valid_indices) == 0:
                    continue

                # sort by overlap (primary), occlusion as tie-breaker; keep top-k
                sorted_indices = valid_indices[np.lexsort((-camera_occlusion[valid_indices], -ious[valid_indices]))]
                selected_sources = sorted_indices[:max_sources]

                sources = [{
                    "image": img_paths[src_idx],
                    "depth": depth_paths[src_idx],
                    "K": intrinsics[src_idx],
                    "extrinsics": RTs[src_idx],
                    "occlusion": float(camera_occlusion[src_idx]),
                    "overlap": float(ious[src_idx]),
                } for src_idx in selected_sources]

                target_name = os.path.splitext(names[target_idx])[0]
                results[f"{house_id}__{target_name}"] = {
                    "target": {
                        "image": img_paths[target_idx],
                        "depth": depth_paths[target_idx],
                        "K": intrinsics[target_idx],
                        "extrinsics": RTs[target_idx],
                    },
                    "sources": sources,
                }

        save_path = f"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/preprocessed_low_res_matterport3d_3d_{split}_set.pkl"
        with open(save_path, "wb") as f:
            pkl.dump(results, f)
        print(f"\nSaved {len(results)} sequences to {save_path}")
        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of scenes to process")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of scenes to process (exclusive)")
    parser.add_argument("--split", type=str, default="train", help="Split to use to create sequences")

    parser.add_argument("--root_dir", type=str, default= "/projects/rps/dff6142/fouheylab/dma9300/matterport3D/scans/", help="Path to data")
    parser.add_argument("--mode", type=str, default="sequences", choices=["dump", "dump_overlaps", "dump_sequences", "overlaps", "sequences"],
                        help="dump: resized rgb/depth/cameras per house | dump_overlaps: overlaps on the dumped views | "
                             "dump_sequences: sequences from the dump + its overlaps | "
                             "overlaps: compute_camera_overlaps | sequences: build_overlap_sequences")
    parser.add_argument("--out_root", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_matterport3d/", help="Dump folder (written by --mode dump, read by --mode dump_overlaps)")
    parser.add_argument("--dump_scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/matterport3d_low_res_scene_files.pkl", help="Scene file built from the dump (--mode dump_overlaps)")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/low_res_matterport3d_overlaps/", help="Output folder for --mode dump_overlaps")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    if args.mode == "dump_overlaps":
        # everything comes from the dump: the Matterport index in __init__ is not needed
        # breakpoint()
        dataloader = object.__new__(Matterport3D)
        dataloader.compute_overlaps_from_dump_files(
            dump_root = args.out_root,
            scene_file = args.dump_scene_file,
            overlap_basedir = args.overlap_basedir,
            index_start = int(args.start_idx),
            end_index = args.end_idx,                # Inclusive of end idx
        )
        raise SystemExit

    if args.mode == "dump_sequences":
        # everything comes from the dump + its overlaps: no Matterport index needed
        Matterport3D.build_sequences_from_dump(
            dump_root = args.out_root,
            scene_file = args.dump_scene_file,
            overlap_basedir = args.overlap_basedir,
            split = args.split,
        )
        raise SystemExit

    dataloader = Matterport3D(root_dir = args.root_dir)
    if args.end_idx is None:
        args.end_idx = len(dataloader)
    
    # breakpoint()

    if args.mode == "dump":
        # end_idx is EXCLUSIVE here: houses [start_idx, end_idx)
        dataloader.dump_houses(out_root = args.out_root, index_start = int(args.start_idx), end_index = int(args.end_idx))
    elif args.mode == "overlaps":
        dataloader.compute_camera_overlaps(index_start = int(args.start_idx), end_index = int(args.end_idx))
    else:
        dataloader.build_overlap_sequences(split = args.split )