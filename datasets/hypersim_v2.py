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
from tqdm import tqdm

import zipfile
from typing import Any, DefaultDict, Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Subset
from .dataset_util import *
import matplotlib.pyplot as plt
import open3d as o3d
from fvcore.common.config import CfgNode
import utils.cropping as cropping


import torchvision.transforms as tvf
from .base_dataset import BaseDataset

from PIL import Image
import json

class Hypersim_NVS(BaseDataset): 
    def __init__(self,split, config_path, overfit = False, in_domain_eval = False,  stage =1, img_size = 518, n_target_views = 1, max_views = 2,  *args, **kwargs):
        with open(config_path, "r") as f:
            opts = CfgNode(yaml.safe_load(f))
        self.common_conf = opts.COMMON
        self.common_conf['IMG_PER_SEQ'] = max_views - 1
        self.common_conf["IMG_SIZE"] = img_size 
        self.overfit = overfit
        self.in_domain_eval = in_domain_eval
        # self.img_size = img_size

        self.n_target_views = int(n_target_views)
        self.split =  split
        self.opts = opts
        self.root_dir = opts.ROOT_DIR
        self.aug_crop = opts.AUG_CROP
        super().__init__(self.common_conf, *args, **kwargs)
        self.training = self.split == "train"
        self.sequences = None

        self.data_path = f"{self.root_dir}/preprocessed_hypersim_3d_{self.split}_set.pkl"

        self.prob = 0.30
        with open(self.data_path, "rb") as f:
            self.sequences = pkl.load(f)
        
        self.image_transform  = tvf.Compose([tvf.ToTensor()])

        self.sequences = self.filter_results_by_overlap(self.sequences,  min_overlap = self.opts.DEPTH_FRUSTRUM_OVERLAP_THRESHOLD_MIN, max_overlap =self.opts.DEPTH_FRUSTRUM_OVERLAP_THRESHOLD_MAX, min_occlusion =self.opts.OCCLUSION_THRESHOLD_MIN, max_occlusion = self.opts.OCCLUSION_THRESHOLD_MAX,  seq_len = 6)

        # self.save_random_dict_samples(self.sequences, save_path = f"{self.root_dir}/hypersim_stage2_evaluation_100.pkl", num_samples = 100)

        if stage == 2 and self.split =="test":
            with open(f"{self.root_dir}/hypersim_stage2_evaluation_100.pkl", 'rb') as f:
                self.sequences = pkl.load(f)
        
        # self.save_random_dict_samples(self.sequences, save_path = f"{self.root_dir}/hypersim_stage2_evaluation_50.pkl", num_samples = 50)

        # breakpoint()


        if stage == 2 and self.split == "test" and self.in_domain_eval:
            with open(f"{self.root_dir}/hypersim_stage2_evaluation_50.pkl", 'rb') as f:
                self.sequences = pkl.load(f)



        # self.save_random_dict_samples(self.sequences, save_path = f"{self.root_dir}/hypersim_overfit_split_100.pkl", num_samples = 100)
        # breakpoint()
        
        if self.overfit and (self.split == "train" or self.split =="val"):
            with open(f"{self.root_dir}/hypersim_overfit_split_100.pkl", 'rb') as f:
                self.sequences = pkl.load(f)

        if self.split == "val" and not self.overfit:
            with open(f"{self.root_dir}/hypersim_val_set_1000samples.pkl", 'rb') as f:
                self.sequences = pkl.load(f)
        
    
        if self.split == "test" and stage == 1:
            with open(f"{self.root_dir}/hypersim_test_set_200samples.pkl", 'rb') as f:
                self.sequences = pkl.load(f)



        self.keys = list(self.sequences.keys())


    def save_random_dict_samples(self,data_dict, save_path, num_samples=1000):
        """
        Randomly selects and saves a subset of a dictionary to a pickle file.

        Args:
            data_dict (dict): Dictionary to sample from.
            save_path (str): Path to save the .pkl file.
            num_samples (int): Number of samples to save.
        """
        keys = list(data_dict.keys())
        sampled_keys = random.sample(keys, min(num_samples, len(keys)))
        sampled_dict = {k: data_dict[k] for k in sampled_keys}


        # Save the sampled dictionary
        with open(save_path, 'wb') as f:
            pkl.dump(sampled_dict, f)

        print(f"Saved {len(sampled_dict)} samples to {save_path}")

    
    def deterministic_even_sampling(self, sequence, k):
        """
        Deterministically sample `k` sequence evenly spaced across a list.

        This function is useful when you want:
        - reproducible behavior (no randomness)
        - uniform coverage across the entire list
        - no duplicates
        - consistent sampling independent of machine or seed

        Args:
            sequence (list):
                Input list of any sequence (dicts, numbers, etc.).
            k (int):
                Number of samples to return.

        Returns:
            list:
                A list of `k` evenly spaced elements from `sequence`.
                If k >= len(sequence), returns the entire list.
        
        Example:
            sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            deterministic_even_sample(sequence, 4)
            → [0, 3, 6, 9]
        """
        n = len(sequence)
        
        # Case 1: N <= k → return full list
        if n <= k :
            return sequence

        # Case 2: N > k → even deterministic sampling
        indices = np.linspace(0, n - 1, k, dtype=int)

        return [sequence[i] for i in indices]

        

    def process_scene_sequence(
        self,
        scene_item: dict,
        scene_id: str,
        covis_thresh: float = 0.5,
        overlap_thresh: float = 0.5,
        max_length: int = 8,
    ):
        """
        Process a single scene item  into target and source views,
        filtering source views by  overlap thresholds.

        Args:
            scene_item (dict): A dictionary with keys:
                - 'target': dict with 'image', 'depth', 'K', 'extrinsics'
                - 'sources': list of dicts with same keys plus 'occlusion' and 'overlap'
            scene_id (str): Scene or house identifier.
           
            max_length (int): Maximum number of source views to load.

        Returns:
            list[dict]: Sequence of views (target first, then filtered and sorted source views).
        """

        target_dict = scene_item["target"]
        sources = scene_item["sources"]

        # ---- Load target ----
        target_img_name = target_dict["image"].split("/")[-1].split(".")[0]

        t_depthmap, t_depthmap_mask = self.load_z_buffer_depth_and_depth_mask(target_dict["depth"])

        target_view = {
            "image": self.load_image(target_dict["image"]),
            "depth_distance": t_depthmap,
            "intrinsics": target_dict["K"],
            "extrinsics": target_dict["extrinsics"], #cam2world
            "image_name": f"{scene_id}_{target_img_name}",
            "mask" :  t_depthmap_mask,
            "image_path": target_dict["image"]
        }

        sequence = [target_view]

        # ---- Sort sources by  overlap -----
        sorted_sources = sorted(sources, key=lambda s: (s["overlap"]),reverse=True)


        # ---- Sample N samples evenly ----
        # selected_sources  = self.deterministic_even_sampling(sorted_sources, k = max_length)
        selected_sources  = sorted_sources[:max_length]




        # ---- Load selected sources ----
        for src_dict in selected_sources:
            src_img_name = src_dict["image"].split("/")[-1].split(".")[0]

            src_depthmap, src_depthmap_mask = self.load_z_buffer_depth_and_depth_mask(src_dict["depth"])


            source_view = {
                "image": self.load_image(src_dict["image"]),
                "depth_distance":  src_depthmap,
                "intrinsics": src_dict["K"],
                "extrinsics": src_dict["extrinsics"],
                "occlusion": src_dict["occlusion"],
                "overlap": src_dict["overlap"],
                "image_name": f"{scene_id}_{src_img_name}",
                "scene_id": scene_id,
                "mask" : src_depthmap_mask,
                "image_path": src_dict["image"]


            }

            sequence.append(source_view)

        return sequence

    def load_valid_mask(self, png_path: str) -> np.ndarray:
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

    

    def load_image(self, file_path):

        """
        Load a JPEG image and return as a NumPy array (H, W, 3) in RGB order.
        """

        img = Image.open(file_path).convert("RGB")

        return np.array(img)


    def load_z_buffer_depth_and_depth_mask(self, path):
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
        depth_m[~mask] = 0.0

        return depth_m, mask



    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a JPEG image using OpenCV in RGB order.

        Args:
            image_path (str): Path to the .jpeg image file.

        Returns:
            np.ndarray: Image array of shape (H, W, 3), dtype uint8.
        """
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def filter_results_by_overlap(
        self,
        results: dict,
        min_overlap: float = 0.10,
        max_overlap: float = 0.40,
        min_occlusion: float = 0.40,
        max_occlusion: float = 0.70,
        seq_len: int = 6,
    ):
        """
        Filter results by overlap and occlusion ranges.

        Args:
            results (dict): Original results dictionary containing targets and sources.
            min_overlap (float): Minimum allowed overlap.
            max_overlap (float): Maximum allowed overlap.
            min_occlusion (float): Minimum allowed occlusion.
            max_occlusion (float): Maximum allowed occlusion.
            seq_len (int): Minimum number of valid sources required per target.

        Returns:
            dict: Filtered results dictionary.
        """
        filtered_results = {}

        for key, data in results.items():
            sources = data["sources"]

            valid_sources = [
                s for s in sources
                if min_overlap <= s["overlap"] <= max_overlap
                and min_occlusion <= s["occlusion"] <= max_occlusion
            ]


            if len(valid_sources) < seq_len:
                continue

            # Highest occlusion first
            valid_sources = sorted(
                valid_sources,
                key=lambda s: s["occlusion"],
                reverse=True,
            )

            filtered_results[key] = {
                "target": data["target"],
                "sources": valid_sources[:seq_len],
            }

        print(
            f"Filtered results: kept "
            f"{len(filtered_results)} of {len(results)} samples"
        )

        return filtered_results

    # def filter_results_by_overlap(
    #     self,
    #     results: dict,
    #     min_overlap : float = 0.10,
    #     max_overlap: float = 0.40,
    #     seq_len: int = 6,
    # ):
    #     """
    #     Filter a results dictionary by overlap threshold and
    #     sort sources by occlusion (descending).

    #     Args:
    #         results (dict): Original results dictionary containing targets and sources.
    #         overlap_thresh (float): Minimum overlap threshold.
    #         seq_len (int): Number of top sources to keep per target.

    #     Returns:
    #         dict: Filtered results dictionary.
    #     """
    #     filtered_results = {}

    #     for key, data in results.items():
    #         sources = data["sources"]

    #         overlap_values = np.array([s["overlap"] for s in sources])
    #         occlusion_values = np.array([s["occlusion"] for s in sources])

    #         breakpoint()

    #         # Filter by overlap
    #         valid_sources = [s for s in sources if s["overlap"] <= overlap_thresh and 0.60 < s["occlusion"] < 0.80]

    #         if len(valid_sources) > 0:
    #             print(
    #                 key,
    #                 "valid:", len(valid_sources),
    #                 "overlap:",
    #                 [(s["overlap"], s["occlusion"]) for s in valid_sources[:10]]
    #             )
    #             break

    #         if len(valid_sources) < seq_len:
    #             continue

    #         # Sort by occlusion (descending)
    #         valid_sources = sorted(
    #             valid_sources,
    #             key=lambda s: s["occlusion"],
    #             reverse=True,
    #         )

    #         # Keep top-k sources
    #         filtered_results[key] = {
    #             "target": data["target"],
    #             "sources": valid_sources,
    #         }

    #     print(f"Filtered results: kept {len(filtered_results)} of {len(results)} samples")
    #     return filtered_results


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

    def compute_relative_pose(self, ref_pose: np.ndarray, src_pose: np.ndarray) -> np.ndarray:
        """
        Compute the relative transformation from the source camera to the reference camera.

        Args:
            ref_pose (np.ndarray): (4, 4) world-to-reference camera extrinsic matrix (w2c).
            src_pose (np.ndarray): (4, 4) world-to-source camera extrinsic matrix (w2c).
            normalize_translation (bool): Whether to normalize the translation vector to unit length.

        Returns:
            np.ndarray: (4, 4) transformation from source camera to reference camera.
        """
        # Compute relative transform: cam_src → cam_ref
        src_to_ref = ref_pose @ self.rotate_inverse(src_pose)

        return  src_to_ref.astype(np.float32)


    
    def compute_all_relative_poses(self, pose_list: List[np.ndarray],reference_camera_index:int = 1) -> List[np.ndarray]:
        """
        Compute relative poses from the first camera to all cameras in the list.

        Args:
            pose_list (List[np.ndarray]): List of (4, 4) world-to-camera extrinsic matrices.
            normalize_translation (bool): Whether to normalize the translation vector.

        Returns:
            List[np.ndarray]: List of relative (4, 4) transformations to the first camera.
        """
        assert all(p.shape == (4, 4) for p in pose_list), "All input poses must be 4x4 matrices."

        ref_pose = pose_list[reference_camera_index]
        # ref_pose = 1

        rel_poses = [
            self.compute_relative_pose(ref_pose = ref_pose, src_pose = src_pose)
            for src_pose in pose_list
        ]
        return rel_poses

    
    def __len__(self):
        
        return len(self.keys)


    def hypersim_distance_to_depth(self, distance, width=1024, height=768, focal=886.81):
        """
        Convert Hypersim "distance" (from camera) to depth along Z.
        see https://github.com/apple/ml-hypersim/issues/9


        Args:
            distance: np.ndarray of shape (H, W) with distance values
            width: image width (default 1024)
            height: image height (default 768)
            focal: camera focal length (default 886.81)

        Returns:
            depth: np.ndarray of shape (H, W)
        """
        # Create image plane coordinates centered at 0
        x = np.linspace(-0.5 * width + 0.5, 0.5 * width - 0.5, width)
        y = np.linspace(-0.5 * height + 0.5, 0.5 * height - 0.5, height)
        xv, yv = np.meshgrid(x, y)

        # 3D vectors from camera to pixels
        image_plane = np.stack([xv, yv, np.full_like(xv, focal)], axis=-1)  # H x W x 3

        # Compute depth along Z
        norm = np.linalg.norm(image_plane, axis=2)
        depth = distance / norm * focal

        return depth.astype(np.float32)

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, aug_crop = 16,  mask = None, rng=None, info=None):
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


    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 1,
        aspect_ratio: float = 1.0,
        rng = np.random.default_rng(seed=42)
    ) -> dict:
        """
        Retrieve a data batch for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence including target.
            aspect_ratio (float): Aspect ratio for resizing.

        Returns:
            dict: Data batch with images, depths, intrinsics, extrinsics, etc.
        """

        key = self.keys[seq_index]

        scene_id, target_view_name = key.split("__")

        scene_sequence = self.sequences.get(key)

        # Process sequence views, with max_length limiting source views
        sequence_views = self.process_scene_sequence(scene_sequence, scene_id, max_length=img_per_seq)

        # Compute target shape
        target_shape = self.get_target_shape(aspect_ratio)

        # Prepare containers
        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []
        iou_depth_frustrums, iou_dino_sims = [], []
        depth_masks = []
        rays = []
        pose_encoding = []  # unused in original, kept for future
        image_names = []
        occlusion_scores = []
        overlaps = []

        for idx, view in enumerate(sequence_views):
            image_name = view["image_name"]
            mask = view["mask"]
            depth_distance = view["depth_distance"]
            image = view["image"]

            K = view["intrinsics"]
            H, W = depth_distance.shape
            
            depth_map = self.hypersim_distance_to_depth(depth_distance, width = W, height = H, focal = (view["intrinsics"][0, 0]+ view["intrinsics"][1, 1])/2)

            extri = self.rotate_inverse(np.array(view["extrinsics"]))  # world2cam
            intri = view["intrinsics"]

            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
            original_size = np.array(image.shape[:2])

            #  Process one image to get world , masks etc.

            image, depth_map, intri, mask = self._crop_resize_if_necessary(image = image, depthmap = depth_map, intrinsics = intri,  mask = mask.astype(np.int32), resolution=(self.img_size, self.img_size), rng=rng, info=None)

            # breakpoint()

            image = np.asarray(image).copy()

            # rgbd_to_pointcloud(rgb = image, depth = depth_map,  mask = None,  K = intri, RT_c2w = view["extrinsics"],  save_path=f"hypersim_point_cloud_dataloader_{idx}.ply")

            # plt.imshow(image)
            # plt.savefig(f"hypersim_image_dataloader{idx}.png")
            # plt.close()
            # breakpoint()

            depth_mask = mask.astype(np.bool_)

            image = self.image_transform(image)

            # Append data
            image_names.append(image_name)
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri.astype(np.float32))
            intrinsics.append(intri)
            depth_masks.append(depth_mask.astype(np.bool_))
            original_sizes.append(original_size)


            if idx != 0:
                # breakpoint()
                occlusion_scores.append(view["occlusion"])
                overlaps.append(view["overlap"])


        n_target_views = self.n_target_views

        # Compute relative extrinsics (world2cam) with respect to target view
        extrinsics_rel = self.compute_all_relative_poses(pose_list=extrinsics, reference_camera_index=n_target_views)

        # Split into target and source views
        t_images = images[:n_target_views]
        t_image_names = image_names[:n_target_views]
        t_depths = depths[:n_target_views]
        t_extrinsics = extrinsics_rel[:n_target_views]
        # t_extri = extrinsics[:n_target_views]
        t_intrinsics = intrinsics[:n_target_views]
        t_depth_masks = depth_masks[:n_target_views]
        t_original_sizes = original_sizes[:n_target_views]
        t_extri = [self.rotate_inverse(camera) for camera in extrinsics[:n_target_views]]


        s_images = images[n_target_views:]
        s_depths = depths[n_target_views:]
        s_image_names = image_names[n_target_views:]
        s_extrinsics = extrinsics_rel[n_target_views:]
        # s_extri = extrinsics[n_target_views:]
        s_extri =  [self.rotate_inverse(camera) for camera in extrinsics[n_target_views:]] #cam2world


        s_intrinsics = intrinsics[n_target_views:]
        s_depth_masks = depth_masks[n_target_views:]
        s_original_sizes = original_sizes[n_target_views:]

        # breakpoint()

        return {
            "seq_name": key,
            "frame_num": len(images),

            # Target views
            "t_images": t_images,
            "t_depths": t_depths,
            "t_depth_masks": t_depth_masks,
            "t_extrinsics": t_extrinsics,
            "t_extri": t_extri,
            "t_intrinsics": t_intrinsics,
            "t_image_names": t_image_names,

            # Source views
            "s_images": s_images,
            "s_depths": s_depths,
            "s_depth_masks": s_depth_masks,
            "s_extrinsics": s_extrinsics,
            "s_intrinsics": s_intrinsics,
            "s_extri": s_extri,
            "s_image_names": s_image_names,
            "s_overlaps" : overlaps,
            "s_occlusion": occlusion_scores
        }

def combined_pointcloud_ply(
    images, depths, masks, intrinsics, src2ref_poses, ply_path
):
    """
    Build a world point cloud from multiple RGB-D frames and save as .ply
    
    Args:
        images:     (S, H, W, 3) float in [0,1]
        depths:     (S, H, W) in meters
        masks:      (S, H, W) bool or 0/1
        intrinsics: (S, 3, 3)
        src2ref_poses:  (S, 4, 4)
        ply_path:   output file path
    """

    S, C, H, W  = images.shape
    device = images.device

    images = (images * 255).permute((0, 2, 3, 1 )).clamp(0,255).to(torch.uint8)

    # pixel coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(xx)
    pix = torch.stack([xx, yy, ones], dim=-1).reshape(1, H*W, 3).float()  # (1,HW,3)

    xyz_world_list = []
    rgb_list = []

    for i in range(S):
        K = intrinsics[i]
        depth = depths[i]
        mask = masks[i].bool()
        rgb = images[i]

        depth_flat = depth.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid = mask_flat & (depth_flat > 0)

        if valid.sum() == 0:
            continue

        px = pix[0][valid]                 # (N,3)
        d = depth_flat[valid]              # (N,)
        rgb_pixels = rgb.reshape(-1,3)[valid]  # (N,3)

        Kinv = torch.inverse(K)

        # backproject to camera coordinates
        cam_xyz = (Kinv @ px.T).T * d[:, None]     # (N,3)

        cam_xyz_h = torch.cat([cam_xyz, torch.ones(len(cam_xyz),1,device=device)], dim=1)
        world_xyz = (src2ref_poses[i] @ cam_xyz_h.T).T[:, :3]

        xyz_world_list.append(world_xyz)
        rgb_list.append(rgb_pixels)

    if not xyz_world_list:
        raise ValueError("No valid points found in input frames.")

    xyz_world = torch.cat(xyz_world_list, dim=0).cpu().numpy()
    rgb_all   = torch.cat(rgb_list, dim=0).cpu().numpy()

    # ---- Open3D export ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_world)
    pcd.colors = o3d.utility.Vector3dVector(rgb_all / 255.0)

    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud with {xyz_world.shape[0]} points to: {ply_path}")

    # return xyz_world, rgb_all
def show_target_and_sources(
    t_images,
    s_images,
    index=0,
    title="Target and Source Images",
    n_cols=4,
):
    """
    Visualize all target images and all source images.

    Args:
        t_images (Tensor): (T, 3, H, W) tensor of target images in [0,1].
        s_images (Tensor): (S, 3, H, W) tensor of source images in [0,1].
        index (int): Sample index used in the output filename.
        title (str): Overall figure title.
        n_cols (int): Number of images per row.
    """
    assert t_images.ndim == 4, (
        f"Expected t_images with shape (T,3,H,W), got {t_images.shape}"
    )
    assert s_images.ndim == 4, (
        f"Expected s_images with shape (S,3,H,W), got {s_images.shape}"
    )

    def to_numpy(img_tensor):
        """Convert (3,H,W) float tensor in [0,1] to (H,W,3) uint8."""
        img = img_tensor.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    num_targets = len(t_images)
    num_sources = len(s_images)
    total = num_targets + num_sources

    if total == 0:
        return

    # Convert all images
    target_imgs = [to_numpy(img) for img in t_images]
    source_imgs = [to_numpy(img) for img in s_images]

    # Dynamically determine number of rows
    n_rows = int(np.ceil(total / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    axes = axes.flatten()

    # Plot all target views
    for i, img in enumerate(target_imgs):
        axes[i].imshow(img)
        axes[i].set_title(f"Target View {i + 1}", fontsize=12)
        axes[i].axis("off")

    # Plot all source views
    offset = num_targets

    for i, img in enumerate(source_imgs):
        axes[offset + i].imshow(img)
        axes[offset + i].set_title(f"Source View {i + 1}", fontsize=12)
        axes[offset + i].axis("off")

    # Hide unused axes
    for i in range(total, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(
        f"hypersim__nvs_samples_{index}.png",
        bbox_inches="tight",
    )
    plt.close(fig)

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


def save_image_pairs(
    t_images,
    s_images,
    target_view_name,
    save_dir,
    max_sources=None,
):
    """
    Save target and source image pairs to disk.

    Args:
        t_images (Tensor): (T, 3, H, W) target images in [0,1]
        s_images (Tensor): (S, 3, H, W) source images in [0,1]
        target_view_name (str): name/id of the target view
        save_dir (str): directory to save images
        max_sources (int, optional): limit number of sources saved
    """

    assert t_images.ndim == 4 and s_images.ndim == 4, \
        "Expected (T,3,H,W) and (S,3,H,W) tensors"

    os.makedirs(save_dir, exist_ok=True)

    def to_uint8(img_tensor):
        """(3,H,W) float tensor [0,1] → uint8 (H,W,3)."""
        img = img_tensor.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    # --------------------------------------------------
    # Save target (always index 0, same as your viz)
    # --------------------------------------------------
    target_img = to_uint8(t_images[0])
    target_path = os.path.join(
        save_dir, f"target__{target_view_name}.png"
    )
    imageio.imwrite(target_path, target_img)

    # --------------------------------------------------
    # Save sources
    # --------------------------------------------------
    num_sources = len(s_images)
    if max_sources is not None:
        num_sources = min(num_sources, max_sources)

    for i in range(num_sources):
        src_img = to_uint8(s_images[i])
        src_path = os.path.join(
            save_dir,
            f"source_{i}__{target_view_name}.png"
        )
        imageio.imwrite(src_path, src_img)


def get_args():
        parser = argparse.ArgumentParser(description="HM3D dataset config")

        parser.add_argument('--split', type=str, default='train',
                            help="Dataset split to use (e.g., train, val, test)")
        parser.add_argument('--config_path', type=str, required=True,
                            help="Path to dataset configuration file")
        parser.add_argument('--resolution', type=int, default=518,
                            help="Image resolution (assumed square, e.g., 512 for 518x518)")
        parser.add_argument('--seed', type=int, default=42,
                            help="Random seed for reproducibility")

        return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    dataset = Hypersim_NVS(split = args.split, n_target_views = 4, max_views = 6, overfit = False, config_path= args.config_path)
    data_dict = {}
    for sample_idx, views in enumerate(tqdm(dataset, desc="Processing dataset")):

        show_target_and_sources(views["t_images"], views["s_images"], sample_idx)
        # save_image_pairs(t_images = views["t_images"], s_images=views['s_images'], target_view_name = views["t_image_names"][0], save_dir="./hypersim_pairs/images")
        breakpoint()

        # data_dict[views["t_image_names"][0]] = { "iou" : views["s_overlaps"][0], "occlusion" : views["s_covisibilities"][0]}

        # with open("./hypersim_pairs/data.json", "w") as f:
        #     json.dump(data_dict, f, indent=2)

        
        # if sample_idx == 1000:
        #     breakpoint()








