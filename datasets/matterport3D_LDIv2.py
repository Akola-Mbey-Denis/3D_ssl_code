import collections
import io
import os
import os.path as osp
import pdb
import pickle as pkl
import random
import time
import argparse
import sys
import yaml


import zipfile
from typing import Any, DefaultDict, Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Subset
from einops import rearrange
from .dataset_util import *

from utils import data_sampling, matterport_parse
from trimesh.ray.ray_pyembree import RayMeshIntersector
from einops import rearrange
import matplotlib.pyplot as plt
import open3d as o3d
from .augmentation import get_image_augmentation
import torchvision.transforms as tvf
from .easy_dataset import EasyDataset

curr_dir = osp.dirname(osp.abspath(__file__))

cachedir = "/projects/rps/dff6142/fouheylab/dma9300/occ_project/preprocessed_training_data/"


class MatterportDataLDI(EasyDataset):
    def __init__(self, config_path, split = "train", enable_transform = True):

        with open(config_path, "r") as f:
            opts = CfgNode(yaml.safe_load(f))
        
        self.dataset_dir = opts.MATTERPORT_PATH
        self.opts = opts
        self.aug_crop = self.opts.DATALOADER.AUG_CROP
        self.enable_transform = enable_transform
        self.split = "val" if split == "test" else split

        self.filter_keys = [
            "_i1_",  # "_i2_", "_i0_"
        ]  # we are using only these as the i0 and i2 point to the bottom
        _house_ids = os.listdir(
            osp.join(self.dataset_dir, "undistorted_camera_parameters")
        )
        split_dir = osp.join(cachedir, "splits")
        split_file = osp.join(split_dir, "mp3d_split.pkl")
  
        with open(split_file, "rb") as f:
            splits = pkl.load(f)
    
        self.dataloader_mode = opts.DATALOADER.MODE
        _house_ids = splits[self.split]


        self.meshes = DefaultDict(dict)
        self.mp3d_cachedir = osp.join(cachedir, "mp3d_data_zmx8")
        group_by_region = False

        if self.split == "train" and self.enable_transform:
            self.image_aug = get_image_augmentation()
        else:
            print("Loading test time data augmentation")
            self.image_aug  = tvf.Compose([tvf.ToTensor()])

        _indexed_items = matterport_parse.preload_conf_files(
            matterport_dir=self.dataset_dir,
            house_ids=_house_ids,
            group_by_region=group_by_region,
        )
        _indexed_items = matterport_parse.filter_indices(
            _indexed_items, self.filter_keys, keep_valid_regions=False
        )

        sample_frac = opts.DATALOADER.SAMPLE_FRAC

        val_uuids_file = "./matterport_val_uuid_file.lst"
        if split == "train":
            _indexed_items = data_sampling.matterport_create_subsample(
                indexed_items=_indexed_items,
                sample_frac=sample_frac,
                split= self.split,
            )
            good_inds = self.mark_good_inds(_indexed_items, split = self.split)
            _indexed_items = np.array(_indexed_items)[np.array(good_inds)].tolist()
        elif split == "val" and osp.exists(val_uuids_file):
            logger.info("dropping indices that are not needed for evaluation. ")
            logger.info(f"Loading uuids from {val_uuids_file}")
            with open(val_uuids_file) as f:
                val_uuids = [k.strip() for k in f.readlines()]
            val_indexed_items = []
            val_uuids = set(val_uuids)

            for indexed_item in _indexed_items:
                house_id = indexed_item["house_id"]
                image_names = indexed_item["image_names"].replace(".jpg", "")
                uuid = f"{house_id}_{image_names}"
                if uuid in val_uuids:
                    val_indexed_items.append(indexed_item)
            _indexed_items = val_indexed_items
        else:
            good_inds = self.mark_good_inds(_indexed_items, split = self.split)
            _indexed_items = np.array(_indexed_items)[np.array(good_inds)].tolist()


        self.houses2mesh = matterport_parse.preload_meshes(
            indexed_items=_indexed_items, matterport_dir=self.dataset_dir
        )
        if split == "train":
            house_ids = self.houses2mesh.keys()
            self.sampled_house2mesh_data = data_sampling.matterport_mesh_sample(
                house_ids, sample_frac, split
            )
        else:
            self.sampled_house2mesh_data = {}

        self.indexed_items = _indexed_items

        # indexed_items = self.save_sampled_dict(output_path = f"{self.split}_indexes.pkl", data = self.indexed_items, sample_size = 1000)
        # # breakpoint()

        if self.split == "train":
            with open("train_indexes.pkl", 'rb') as f:
                self.indexed_items = pkl.load(f)

        if self.split == "val":
            with open("val_indexes.pkl", 'rb') as f:
                self.indexed_items = pkl.load(f)
        
        self.num_items = len(self.indexed_items)

    


    def save_sampled_dict(self,data: List[Dict[str, Any]], output_path: str, sample_size: int = 1000 ,seed: int = 42) -> List[Dict[str, Any]]:
        """
        Randomly samples a number of dictionaries from a list and saves the result as a .pkl file.

        Args:
            data (List[Dict]): List of dictionaries to sample from.
            output_path (str): File path to save the .pkl file.
            sample_size (int): Number of items to sample. Defaults to 100.
            seed (int): Optional seed for reproducibility.

        Returns:
            List[Dict]: The sampled list of dictionaries.
        """
        if seed is not None:
            random.seed(seed)

        if sample_size > len(data):
            raise ValueError("Sample size cannot be larger than the list length.")

        sampled_list = random.sample(data, sample_size)

        with open(output_path, 'wb') as f:
            pkl.dump(sampled_list, f)

        print(f"Sampled {sample_size} dictionaries and saved to {output_path}")
        return sampled_list



    def check_example_validity(self, meta_data):
        img_data = self.forward_img(meta_data)
        depth = img_data["depth"]
        depth1d = depth.reshape(-1)

        # More than 60% of values are less than 2 and more than 50% of the values are less than 0.001
        if np.mean(depth1d >= 2.0) < 0.6 and np.mean(depth1d < 0.001) > 0.5:
            return {"empty": True}
    
        return {"empty": False}

    def mark_good_inds(self, indexed_items, split):
        opts = self.opts
        checkpoint_dir = osp.join(self.opts.CACHE_DIR, "matterport_good_list", "all")
        # checkpoint_dir = osp.join(self.opts.CACHE_DIR, 'snapshots', opts.NAME)
        if split == "val":
            good_inds_file = osp.join(checkpoint_dir, f"good_inds_file_{split}.pkl")
        else:
            good_inds_file = osp.join(checkpoint_dir, "good_inds_file.pkl")
        # logger.info(f"Good Inds file {good_inds_file}")
        if osp.exists(good_inds_file):
            # logger.info("Fast loading good inds from file")
            with open(good_inds_file, "rb") as f:
                good_list = pkl.load(f)
        else:
            good_list = []
            for ix in range(len(indexed_items)):
                meta_data = indexed_items[ix]
                elem = self.check_example_validity(meta_data)
                # if ix % 1000 == 0:
                #     # logger.info(f"Checking {ix}/{len(indexed_items)}")
                if not elem["empty"]:
                    uuid = meta_data["uuid"]
                    good_list.append(uuid)

            with open(good_inds_file, "wb") as f:
                pkl.dump(good_list, f)

        good_inds = []

        good_list = set(good_list)
        for ix in range(len(indexed_items)):
            meta_data = indexed_items[ix]
            uuid = meta_data["uuid"]
            if uuid in good_list:
                good_inds.append(ix)

        return good_inds

    def load_mesh(self, meta_data):
        house_id = meta_data["house_id"]

        mesh = self.houses2mesh[house_id]
        mesh = {
            "faces": mesh["faces"] * 1,
            "vertices": mesh["vertices"] * 1,
        }
        if self.opts.DATALOADER.USE_POISSON_MESH and self.split == "train":
            fid_data = np.load(osp.join(self.mp3d_cachedir, meta_data["uuid"] + ".npz"))
            fids = fid_data["fids"]
        else:
            (
                fids,
                face_labels,
                face_segments,
                fids_c,
                fids_w,
            ) = matterport_parse.load_face_metadata(
                metadata=meta_data, face_cachedir=self.mp3d_cachedir
            )

        if fids is not None:
            # mesh['face_labels'] =
            if house_id in self.sampled_house2mesh_data.keys():
                sampled_house_mesh = self.sampled_house2mesh_data[house_id]
                fid_set = set(sampled_house_mesh["faces_ids"])
                # fids = [fid  for fid in fids if fid in sampled_house_mesh['faces_ids_strict']]
                fids = [fid for fid in fids if fid in fid_set]
                fids = np.array(fids)
            mesh["faces"] = mesh["faces"][fids]
            mesh = matterport_parse.delete_unused_verts(
                mesh,
            )
            mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
            return mesh
        else:
            return None

    def forward_mesh(self, meta_data):
        house_id = meta_data["house_id"]
        region_id = meta_data["region_id"]
        region_mesh_pth = osp.join(
            self.dataset_dir,
            "region_segmentations",
            house_id,
            "region_segmentations",
            f"region{region_id}.ply",
        )
        mesh = None  ## caching meshes!
        if house_id in self.meshes.keys():
            if region_id in self.meshes["house_id"].keys():
                mesh = self.meshes["house_id"]["region_id"]
        if mesh is None:
            with open(region_mesh_pth, "rb") as f:
                mesh = trimesh.exchange.ply.load_ply(f)
            self.meshes[house_id][region_id] = mesh

        simple_mesh = trimesh.Trimesh(mesh["vertices"], mesh["faces"])
        return simple_mesh

    def crop_resize(self, image, depthmap, intrinsics, resolution, rng=np.random.default_rng(seed=42), info=None):
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
        image, depthmap, intrinsics = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

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
        image, depthmap, intrinsics = rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # Compute final cropping parameters based on adjusted intrinsics
        intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)

        # Apply final cropping
        image, depthmap, intrinsics2 = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2

    def process_view(self, meta_data) -> Dict[str, Any]:
        opts = self.opts
        cam2world, intrinsic = self.forward_camera(meta_data)
        # Rx = trimesh.transformations.euler_matrix(np.pi, 0, 0, 'sxyz')
        # cam2mp = np.matmul(cam2mp, Rx)
        img_name = meta_data["image_names"]
        house_id = meta_data["house_id"]
        img_dir = osp.join(
            self.dataset_dir,
            "undistorted_color_images",
            house_id,
            "undistorted_color_images",
        )
        depth_dir = osp.join(
            self.dataset_dir,
            "undistorted_depth_images",
            house_id,
            "undistorted_depth_images",
        )

        depth_path = osp.join(
            depth_dir, img_name.replace("_i", "_d").replace(".jpg", ".png")
        )
        img_path = osp.join(img_dir, img_name)

        img = imageio.imread(img_path)
        depth = imageio.imread(depth_path)

        img, depth, intrinsic =  self.crop_resize(img, depth, intrinsic, resolution = (self.opts.DATALOADER.IMAGE_RESOLUTION[0], self.opts.DATALOADER.IMAGE_RESOLUTION[1]), rng=np.random.default_rng(seed=42), info=None)
        
        img = np.array(img)

        depth = depth.astype(float) * 0.25
        depth = depth / 1000
        data = {}
        data["img"] = img
        data["depth"] = depth
        data["cam2world"] = cam2world
        data["intrinsics"] = intrinsic
    
        return data


    def __len__(
        self,
    ):
        return self.num_items

    def forward_camera(self, camera_dict) -> Tuple[np.array, np.array]:
        cam2world = camera_dict["cam2world"]
        intrinsic = camera_dict["intrinsics"]
        Rx = trimesh.transformations.euler_matrix(np.pi, 0, 0, "sxyz")
        cam2world = np.matmul(cam2world, Rx)
        return cam2world, intrinsic
    
    def convert_K_to_KNDC(self, K, img_size):
        K[0, 0] = 1 * K[0, 0]  # Flip the x-axis.
        K[1, 1] = 1 * K[1, 1]  # Flip the y-axis.
        K[2, 2] = 1  # Flip the z-axis.

        Kndc_mult = np.array(
                [[2.0 / img_size[0], 0, -1], [0, 2.0 / img_size[1], -1], [0, 0, 1]]
            )

        Kndc = np.matmul(Kndc_mult, K)
        return Kndc

    def rotate_inverse(self, se3: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of a single SE(3) transformation matrix (4x4).

        Args:
            se3 (np.ndarray): A 4x4 SE(3) transformation matrix.

        Returns:
            np.ndarray: The inverse of the input SE(3) matrix.
        """
        assert se3.shape == (4, 4), f"Expected shape (4, 4), got {se3.shape}"

        R = se3[:3, :3]
        T = se3[:3, 3]

        R_inv = R.T
        T_inv = -R_inv @ T

        se3_inv = np.eye(4)
        se3_inv[:3, :3] = R_inv
        se3_inv[:3, 3] = T_inv

        return se3_inv


    def save_ray_pointclouds(self, ray_origins, ray_points, out_dir="ray_visualizations", sample_rays=10000):
        """
        Save ray origins and points along rays as separate point clouds.

        Args:
            ray_origins: (H*W, N, 3) or (H*W, 3)
            ray_points: (H*W, N, 3)
            out_dir: directory to save .ply files
            sample_rays: how many rays to randomly save for visualization
        """

        os.makedirs(out_dir, exist_ok=True)

        if ray_origins.ndim == 3:
            ray_origins_vis = ray_origins[:, 0, :]  # Use one origin per ray
        else:
            ray_origins_vis = ray_origins  # Already (H*W, 3)

        total_rays = ray_origins_vis.shape[0]
        idx = np.random.choice(total_rays, size=min(sample_rays, total_rays), replace=False)

        sampled_origins = ray_origins_vis[idx]  # (N, 3)
        sampled_points = ray_points[idx].reshape(-1, 3)  # (N * num_samples, 3)

        # Create point cloud for ray origins
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(sampled_origins)
        pcd_orig.paint_uniform_color([0.0, 0.0, 1.0])  # blue
        o3d.io.write_point_cloud(os.path.join(out_dir, "ray_origins.ply"), pcd_orig)

        # Create point cloud for sampled ray points
        pcd_pts = o3d.geometry.PointCloud()
        pcd_pts.points = o3d.utility.Vector3dVector(sampled_points)
        pcd_pts.paint_uniform_color([1.0, 0.0, 0.0])  # red
        o3d.io.write_point_cloud(os.path.join(out_dir, "ray_points.ply"), pcd_pts)

        print(f"Saved ray origins and sampled ray points to '{out_dir}'")


    def sample_points_along_rays(self, K, RT_w2c, H, W, num_samples=256, near=0.1, far=8.0):
        """
        Args:
            K: (3, 3) Camera intrinsics matrix
            RT_w2c: (4, 4) World-to-camera transform
            H, W: Image height and width
            num_samples: Number of points sampled per ray
            near, far: Depth sampling bounds along each ray

        Returns:
            ray_dirs: (H*W, num_samples, 3) normalized ray directions from camera origin to sampled points
            ray_origins: (H*W, num_samples, 3) ray origins (camera center, repeated)
            ray_points: (H*W, num_samples, 3) sampled world points along rays
        """
        # Invert world-to-camera to get camera-to-world matrix
        RT_c2w = self.rotate_inverse(RT_w2c)

        # Camera origin in world coordinates (3,)
        cam_origin_world = RT_c2w[:3, 3]
        cam_origin_world_exp = cam_origin_world[None, None, :]  # (1, 1, 3) for broadcasting

        # Create pixel coordinate grid
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        pixels_homo = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # (3, H*W)

        # Convert pixel coordinates to normalized camera directions
        K_inv = np.linalg.inv(K)
        dirs_cam = (K_inv @ pixels_homo).T  # (H*W, 3)

        # Rotate camera directions into world coordinates
        ray_dirs_world = (RT_c2w[:3, :3] @ dirs_cam.T).T  # (H*W, 3)
        ray_dirs_world /= np.linalg.norm(ray_dirs_world, axis=1, keepdims=True)  # normalize

        # Sample depths along each ray
        z_vals = np.linspace(near, far, num_samples, dtype=np.float32)  # (num_samples,)
        z_vals = z_vals[None, :, None]  # (1, num_samples, 1)

        # Expand ray dirs and origins to shape (H*W, num_samples, 3)
        ray_dirs_exp = ray_dirs_world[:, None, :]  # (H*W, 1, 3)
        cam_origin_exp = np.broadcast_to(cam_origin_world_exp, (ray_dirs_world.shape[0], num_samples, 3))  # (H*W, num_samples, 3)

        # Compute sampled points along each ray
        ray_points = cam_origin_exp + z_vals * ray_dirs_exp  # (H*W, num_samples, 3)

        # Compute normalized ray directions from camera origin to sampled points (redundant but consistent)
        ray_dirs = ray_points - cam_origin_exp  # (H*W, num_samples, 3)
        ray_dirs /= np.linalg.norm(ray_dirs, axis=2, keepdims=True)

        # ray_origins is camera center repeated for each sample point
        ray_origins = cam_origin_exp  # (H*W, num_samples, 3)

        return ray_dirs, ray_origins, ray_points

    
    def generate_rays(self, K, RT_w2c, H, W, num_samples=256, z_min=0.10, z_far=8.00):
        """
        Generate rays and sample 3D points along them in world coordinates.

        Args:
            K: (3, 3) Intrinsics
            RT_w2c: (4, 4) World-to-camera matrix
            H, W: Image size
            num_samples: Number of samples per ray
            z_min, z_far: Depth sampling range

        Returns:
            ray_origins: (H*W, num_samples, 3)
            ray_dirs: (H*W, num_samples, 3)
            ray_points: (H*W, num_samples, 3)
        """
        RT_c2w = self.rotate_inverse(RT_w2c)

        # Create pixel grid
        
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # (3, H*W)

        # Directions in camera frame
        K_inv = np.linalg.inv(K)
        dirs_cam = (K_inv @ pixels).T  # (H*W, 3)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)

        # Rotate directions to world frame
        R = RT_c2w[:3, :3]
        T = RT_c2w[:3, 3]
        ray_dirs = (R @ dirs_cam.T).T  # (H*W, 3)
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)

        #  Ray origins in world frame
        ray_origins = np.broadcast_to(T, ray_dirs.shape)  # (H*W, 3)

        # Sample depths along rays
        z_vals = np.linspace(z_min, z_far, num_samples).astype(np.float32)  # (num_samples,)
        z_vals = z_vals.reshape(1, num_samples, 1)  # (1, N, 1)

    
        ray_dirs_exp = ray_dirs[:, None, :]  # (H*W, 1, 3)
        ray_origins_exp = ray_origins[:, None, :]  # (H*W, 1, 3)

        # Sample 3D points along each ray
        ray_points = ray_origins_exp + z_vals * ray_dirs_exp  # (H*W, N, 3)

        # Repeat ray_dirs and ray_origins per sample
        ray_dirs = np.repeat(ray_dirs_exp, num_samples, axis=1)      # (H*W, N, 3)
        ray_origins = np.repeat(ray_origins_exp, num_samples, axis=1)  # (H*W, N, 3)

        return ray_origins, ray_dirs, ray_points


    def generate_layered_depths(self, mesh, ray_origins, ray_dirs, H, W, max_layers=4, min_depth = 0.10, max_depth = 8.0):
        """
        Generate a layered depth image (LDI) by intersecting rays with a mesh.

        Args:
            mesh (trimesh.Trimesh): Mesh to intersect.
            ray_origins (np.ndarray): (H*W, 3) Ray origins in world coordinates.
            ray_dirs (np.ndarray): (H*W, 3) Normalized ray directions.
            H (int): Image height.
            W (int): Image width.
            max_layers (int): Number of depth layers to store per pixel.
            max_depth (float): Maximum depth to include.

        Returns:
            layered_depth (np.ndarray): (H, W, max_layers) Layered depth image.
        """
        # Intersect rays with mesh 

        mesh_intersector = RayMeshIntersector(mesh)
        locations, index_ray, index_tri = mesh_intersector.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=True)
       
        if len(locations) == 0:
            print("Warning: No ray hits found.")
            return np.full((H, W, max_layers), np.inf, dtype=np.float32)

        # Compute depth per hit

        depths = np.linalg.norm(locations - ray_origins[index_ray], axis=1)


        #  Filter out depths beyond max depth
        valid_mask  = (depths >= min_depth) & (depths <= max_depth)

        depths = depths[valid_mask]
        index_ray = index_ray[valid_mask]

        # Initialize layered depth image 
        layered_depth = np.full((H * W, max_layers), np.inf, dtype=np.float32)

        # Populate layered depth image
        for ray_idx, depth in zip(index_ray, depths):
            layer = np.where(np.isinf(layered_depth[ray_idx]))[0]
            if len(layer) > 0:
                layered_depth[ray_idx, layer[0]] = depth

        # Reshape to (H, W, max_layers)
        layered_depth = layered_depth.reshape(H, W, max_layers)

        return layered_depth


    def save_ldi_outputs(self, output_dir, image_name, rgb_image, gt_depth, layered_depths, inf_replacement=np.inf, cmap='Spectral_r'):
        """
        Saves the RGB image, ground truth depth, and layered depth images using the same colormap.

        Args:
            output_dir (str): Directory to save the outputs.
            image_name (str): Base name for the saved images (e.g., "frame_001").
            rgb_image (np.ndarray): (H, W, 3) RGB image in uint8 format.
            gt_depth (np.ndarray): (H, W) Ground truth depth map (float32), with 0 = invalid.
            layered_depths (np.ndarray): (H, W, L) Layered depth image (float32), with inf = invalid.
            inf_replacement (float): Value indicating invalid depth in LDI.
            cmap (str): Matplotlib colormap name (e.g., 'Spectral_r').
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save RGB image
        rgb_image  = (rgb_image * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"{image_name}_rgb.png"), rgb_image)

        # Save GT depth map (mask where depth == 0)
        gt_mask = gt_depth == 0
        gt_vis = self.visualize_depth(np.ma.masked_where(gt_mask, gt_depth), cmap)
        imageio.imwrite(os.path.join(output_dir, f"{image_name}_gt_depth.png"), gt_vis)

        # Save each LDI layer
        num_layers = layered_depths.shape[2]
        for i in range(num_layers):
            layer = layered_depths[:, :, i]
            mask = layer == inf_replacement
            layer_vis = self.visualize_depth(np.ma.masked_where(mask, layer), cmap)
            imageio.imwrite(os.path.join(output_dir, f"{image_name}_layer_{i}.png"), layer_vis)

    def visualize_depth(self, depth_map, cmap='Spectral_r', vmin=None, vmax=None):
        """
        Converts a depth map (or masked depth map) to an RGB image using the given colormap.

        Args:
            depth_map (np.ndarray or np.ma.MaskedArray): Depth map to visualize.
            cmap (str): Matplotlib colormap name.
            vmin (float): Min value for normalization. If None, inferred from valid values.
            vmax (float): Max value for normalization. If None, inferred from valid values.

        Returns:
            np.ndarray: (H, W, 3) RGB image in uint8.
        """
        plt.ioff()
        cmap_func = plt.get_cmap(cmap)
        
        if isinstance(depth_map, np.ma.MaskedArray):
            data = depth_map.data
            mask = depth_map.mask
            valid = ~mask
        else:
            data = depth_map
            valid = np.isfinite(data)

        if not np.any(valid):
            return np.zeros((*data.shape, 3), dtype=np.uint8)

        if vmin is None:
            vmin = np.min(data[valid])
        if vmax is None:
            vmax = np.max(data[valid])

        normed = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1)
        rgba = cmap_func(normed)
        rgba[..., 3] = valid.astype(float)  # set alpha to 0 for invalid
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return rgb


    def plot_ldi_and_gt_depth(self, ldi_depths, gt_depth, save_path='ldis_gt_depthmap.png', inf_replacement=np.inf):
        """
        Plots each LDI layer and ground-truth depth, then saves the figure.

        Args:
            ldi_depths (np.ndarray): (H, W, num_layers) layered depth image.
            gt_depth (np.ndarray): (H, W) ground truth depth map.
            save_path (str): Path to save the final figure.
            inf_replacement (float): Value used for invalid depth in LDI.
        """

        H, W, num_layers = ldi_depths.shape
        fig, axs = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))

        for i in range(num_layers):
            layer = ldi_depths[:, :, i]
            mask = layer == inf_replacement

            axs[i].imshow(np.ma.masked_where(mask, layer), cmap='Spectral_r', vmin=0, vmax=1)
            axs[i].set_title(f"LDI Layer {i+1}")
            axs[i].axis('off')

        axs[-1].imshow(gt_depth, cmap='Spectral_r')
        axs[-1].set_title("Ground Truth Depth")
        axs[-1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    

    def pixel_to_camera(self, K_inv, u, v, d):
        """
        Project 2D pixel coordinates with depth into 3D camera space using intrinsics (homogeneous).

        Args:
            K_inv: (3, 3) inverse of camera intrinsic matrix
            u: (N,) pixel x coordinates
            v: (N,) pixel y coordinates
            d: (N,) depth values (assumed to be distances along the ray)

        Returns:
            xyz_cam: (N, 3) 3D points in camera coordinates
        """
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)  # (N, 3)

        ray_dirs = (K_inv @ uv1.T).T  # (N, 3)

        ray_dirs_norm = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)

        xyz_cam = ray_dirs_norm * d[:, None]  # (N, 3)

        return xyz_cam




    def camera_to_world(self, xyz_cam, RT_w2c):
        """
        Transform camera-space 3D points to world-space using 4x4 camera extrinsics.
        
        Args:
            xyz_cam: (N, 3) points in camera space
            RT_w2c: (4, 4) world-to-camera matrix
        
        Returns:
            xyz_world: (N, 3) points in world coordinates
        """
        # Invert to get camera-to-world matrix
        RT_c2w = self.rotate_inverse(RT_w2c)

        # Convert to homogeneous coordinates (N, 4)
        N = xyz_cam.shape[0]
        xyz_cam_h = np.concatenate([xyz_cam, np.ones((N, 1))], axis=1)  # (N, 4)

        # Apply transformation: (N, 4) @ (4, 4).T = (N, 4)
        xyz_world_h = (RT_c2w @ xyz_cam_h.T).T

        # Convert back to Cartesian coordinates (N, 3)
        xyz_world = xyz_world_h[:, :3] / xyz_world_h[:, 3:]

        return xyz_world

    def generate_faces(self, H, W, verts_flat):
        """
        Generate triangle faces for a grid of (H, W) vertices, skipping invalid ones.

        Args:
            H, W: image height and width
            verts_flat: (H*W, 3) flattened vertex positions

        Returns:
            faces: (N, 3) list of triangle vertex indices
        """
        verts = verts_flat.reshape(H, W, 3)
        valid_mask = np.isfinite(verts).all(axis=2) & (np.linalg.norm(verts, axis=2) > 1e-6)

        # Check for valid 2x2 quad patches
        valid_00 = valid_mask[:-1, :-1]
        valid_01 = valid_mask[:-1, 1:]
        valid_10 = valid_mask[1:, :-1]
        valid_11 = valid_mask[1:, 1:]

        quad_valid = valid_00 & valid_01 & valid_10 & valid_11
        ys, xs = np.where(quad_valid)

        # Compute flat indices for each corner of the quads
        i0 = ys * W + xs
        i1 = ys * W + (xs + 1)
        i2 = (ys + 1) * W + xs
        i3 = (ys + 1) * W + (xs + 1)

        # Two triangles per quad
      
        faces = np.vstack([
            np.stack([i0, i1, i3], axis=1),
            np.stack([i0, i3, i2], axis=1)
            ])

    
        return faces

    

    def mesh_in_frustum(self, mesh: trimesh.Trimesh, RT_w2c: np.ndarray, K: np.ndarray, H: int, W: int,
                        near=0.1, far = 8.0):
        """
        Extract submesh inside camera frustum defined by RT, K, h, w.

        Parameters:
        - mesh: trimesh.Trimesh (world coords)
        - RT: (4,4) world-to-camera transform
        - K: (3,3) intrinsics
        - H, W: image size
        - near, far: clip planes

        Returns:
        - trimesh.Trimesh cropped to frustum (subset of original mesh)
        """

        vertices = mesh.vertices
        faces = mesh.faces

        # Transform vertices to camera coords
        verts_h = np.hstack([vertices, np.ones((vertices.shape[0],1))])
        verts_cam = (RT_w2c @ verts_h.T).T[:, :3]

        z = verts_cam[:, 2]
        proj = (K @ verts_cam.T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

        # Point inside frustum mask
        mask = (z > near) & (z < far) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Which faces have all vertices inside frustum
        face_mask = mask[faces].all(axis=1)

        # Keep those faces and remap vertices to those used by them
        kept_faces = faces[face_mask]

        # Get unique vertices used
        unique_vids, new_indices = np.unique(kept_faces, return_inverse=True)

        cropped_vertices = vertices[unique_vids]
        cropped_faces = new_indices.reshape((-1, 3))

        return trimesh.Trimesh(vertices=cropped_vertices, faces=cropped_faces, process=False)


    def convert_ldi_to_trimesh(self, depths, K, RT_w2c):
        """
        Convert a Layered Depth Image to a mesh using Open3D (Poisson) and return a Trimesh object.

        Args:
            depths: (H, W, L) layered depth map
            K: (3, 3) camera intrinsics
            RT_w2c: (4, 4) world-to-camera matrix

        Returns:
            Trimesh mesh with geometry only (no color)
        """
        H, W, L = depths.shape
        K_inv = np.linalg.inv(K)

        vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        us_flat = us.flatten()
        vs_flat = vs.flatten()

        all_points = []

        for l in range(L):
            d = depths[..., l].flatten()
            valid_depth = np.isfinite(d) & (d >= 0.10) & (d <= 8.0)
            if np.count_nonzero(valid_depth) == 0:
                continue

            # Project pixels to world coordinates
            xyz_cam = self.pixel_to_camera(K_inv, us_flat[valid_depth], vs_flat[valid_depth], d[valid_depth])
            xyz_world = self.camera_to_world(xyz_cam, RT_w2c)
            all_points.append(xyz_world)

        if len(all_points) == 0:
            return None

        # Combine all LDI layers into one point cloud
        all_points = np.concatenate(all_points, axis=0)

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)

        # Estimate normals for meshing
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)

        # Run Poisson reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)

        # Remove low-density vertices
        densities = np.asarray(densities)
        density_thresh = np.quantile(densities, 0.05)
        keep_indices = np.where(densities > density_thresh)[0]
        mesh_o3d = mesh_o3d.select_by_index(keep_indices)

        # Convert to trimesh and flip triangle winding order to fix normals
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)[:, ::-1]  # Flip face winding

        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return mesh_trimesh


    def export_point_cloud(self, xyz_world, filename="xyz_output.ply"):
        # Remove invalid (0,0,0) or NaN/Inf points
        mask = np.isfinite(xyz_world).all(axis=1) & (xyz_world != 0).any(axis=1)
        xyz_world_clean = xyz_world[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_world)
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved point cloud to {filename}")

    def export_mesh_as_glb(self, mesh: trimesh.Trimesh, filepath: str):
        glb_data = mesh.export(file_type='glb')
        with open(filepath, 'wb') as f:
            f.write(glb_data)
        print(f"Exported mesh to {filepath}")

    def __getitem__(self, index):

        meta_data = self.indexed_items[index]
        
        # Load the mesh
        mesh = self.load_mesh(meta_data)

        # load image, depth, camera intrinsics and extrinsics
        view_data = self.process_view(meta_data)

        elem = {}
        image = view_data["img"]
        depth  = view_data["depth"]
        extrinsics_world2cam = self.rotate_inverse(view_data["cam2world"])
        img_size = (image.shape[2], image.shape[1])
        intrinsic = view_data["intrinsics"]

        image_name = meta_data["image_names"]

        house_id = meta_data["house_id"]

        uuid = f"{house_id}_{image_name.replace('.jpg','')}"

        elem["house_id"] = house_id
        elem["image_name"] = image_name
        
        save_folder = f'/scratch/dma9300/matterport3d_ldis_{self.split}'
        # files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]

        os.makedirs(save_folder, exist_ok=True)
        ldi_path = f"{save_folder}/{image_name.split('.')[0]}.npy"
        H, W, _  = image.shape
        save_path = os.path.join(save_folder, f"{image_name.split('.')[0]}.npy")
        if os.path.exists(ldi_path):
            ldis = np.load(ldi_path)
        else:
            ray_origins, ray_dirs, ray_points = self.generate_rays(K = intrinsic, RT_w2c = extrinsics_world2cam.astype(np.float32) , H = H, W = W, z_far = 8.0)
            ldis =  self.generate_layered_depths(mesh = mesh, ray_origins =ray_origins[:, 0,:], ray_dirs=ray_dirs[:, 0,:], H = H, W = W, max_layers = 4, max_depth = 8.0)
            np.save(ldi_path, ldis)

        if self.opts.DATALOADER.VIS:
            
            output_dir = "layered_depths_with_ldi_mesh"

            self.save_ldi_outputs(output_dir = output_dir, image_name = image_name.split(".")[0], rgb_image = image, gt_depth = depth, layered_depths = ldis, inf_replacement=np.inf, cmap='Spectral_r')
        
            # Convert LDI to trimesh mesh
            ldi_mesh = self.convert_ldi_to_trimesh(
                depths = ldis,
                K = intrinsic,
                RT_w2c = extrinsics_world2cam.astype(np.float32)
            )

            basename = image_name.split(".")[0]

            # Export GT mesh
            gt_mesh_path = os.path.join(output_dir, f"{basename}_gt_mesh.glb")
            gt_mesh  = self.mesh_in_frustum(mesh = mesh, RT_w2c = extrinsics_world2cam.astype(np.float32), K = intrinsic, H = H, W = W)
            self.export_mesh_as_glb(gt_mesh, gt_mesh_path)

            # Export LDI mesh if it exists
            if ldi_mesh is not None:
                ldi_mesh_path = os.path.join(output_dir, f"{basename}_ldi_mesh.glb")
                self.export_mesh_as_glb(ldi_mesh, ldi_mesh_path)
            else:
                print("LDI mesh is None, skipping export.")

        ldis_mask = np.isfinite(ldis) 
        
        image  = self.image_aug(image)

        elem["image"] = image.unsqueeze(0)
        elem["depth"] = torch.from_numpy(view_data["depth"]).to(torch.float32)
        elem["intrinsic"] = torch.from_numpy(intrinsic).to(torch.float32)
        elem["extrinsic"] = torch.from_numpy(extrinsics_world2cam).to(torch.float32)
        elem["ldis"] = torch.from_numpy(ldis).to(torch.float32)
        elem["ldis_mask"] = torch.from_numpy(ldis_mask).to(torch.bool)
        return elem






import argparse
import sys
from yacs.config import CfgNode as CN

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a LDI model")
    parser.add_argument(
        "--cfg",  help="optional config file", required=True, default=None, type=str
    )
    parser.add_argument(
        "--split",
        help=" dataset split",
        default="train"    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

if __name__ == "__main__":

    from tqdm import tqdm
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    cmd_args = parse_args()

    dataset = MatterportDataLDI(config_path = cmd_args.cfg , split = cmd_args.split)

    indices = list(range(len(dataset)))

    # Loop over dataset using the indices
    for idx in tqdm(indices, desc="Processing dataset samples"):
        sample = dataset[idx]
        # breakpoint()
            
