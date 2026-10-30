# # === Standard Library ===
# import copy
# import io
# import json
# import os
# import os.path as osp
# import random
# import sys
# import time
# import zipfile
# from collections import defaultdict
# from pathlib import Path
# from typing import Any, DefaultDict, Dict, List, Tuple

# # === Third-Party Libraries ===
# import cv2
# import argparse
# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torchvision.transforms as tvf
# import trimesh
# import yaml
# from PIL import Image
# from tqdm import tqdm
# import pickle as pkl
# from einops import rearrange
# from scipy.spatial.transform import Rotation as R
# from fvcore.common.config import CfgNode
# from torch.utils.data import DataLoader, Dataset, Subset
# from torchvision.transforms.functional import to_pil_image
# from yacs.config import CfgNode

# # === Local Imports ===
# from .augmentation import get_image_augmentation
# from .base_dataset import BaseDataset
# from .dataset_util import *
# from utils import data_sampling, matterport_parse
# from vggt.utils.geometry import closed_form_inverse_se3
# import dust3r.datasets.utils.cropping as cropping




# # === Conditional Import ===
# try:
#     import imageio.v3 as imageio
# except ImportError:
#     import imageio


# class Matterport3D(BaseDataset):
#     def __init__(self,split, config_path,  *args, **kwargs):
#         # super().__init__()
#         with open(config_path, "r") as f:
#             opts = CfgNode(yaml.safe_load(f))
#         self.common_conf = opts.COMMON
#         self.ROOT = opts.MATTERPORT_PATH
#         self.base_dir = opts.BASE_DIR
#         self.split = "val" if split == "test" else split
#         self.opts = opts
#         self.aug_crop = opts.AUG_CROP
#         super().__init__(self.common_conf, *args, **kwargs)
#         self.training = self.split == "train"


#         self.filter_keys = [
#             "_i1_",  # "_i2_", "_i0_"
#         ]  # we are using only these as the i0 and i2 point to the bottom
#         self._house_ids = os.listdir(osp.join(self.ROOT, 'undistorted_camera_parameters'))

#         if self.training:
#             self.image_aug = get_image_augmentation()
#         else:
#             self.image_aug  = tvf.Compose([tvf.ToTensor()])

#         split_dir = osp.join(self.opts.CACHE_DIR, "splits")
#         split_file = osp.join(split_dir, "mp3d_split.pkl")
#         house_ids = os.listdir(osp.join(self.ROOT, 'undistorted_camera_parameters'))
        
#         house2indexes = {}

#         for house_id in house_ids:
#             _indexed_items_all = matterport_parse.preload_conf_files(
#                 matterport_dir=self.ROOT,
#                 house_ids=[house_id],
#                 group_by_region=False,
#             )
#             house2indexes[house_id] = {}
#             house2indexes[house_id]["all"] = _indexed_items_all
#             _indexed_items_ref = matterport_parse.filter_indices(
#                 _indexed_items_all, self.filter_keys, keep_valid_regions=False
#             )
#             house2indexes[house_id]["ref"] = _indexed_items_ref

#         self.house2indexes = house2indexes
#         with open(split_file, "rb") as f:
#             splits = pkl.load(f)
#         self._house_ids = splits[self.split]

#         self.pairs = {}
#         group_by_region = False
#         _indexed_items_all = matterport_parse.preload_conf_files(
#             matterport_dir=self.ROOT,
#             house_ids=self._house_ids,
#             group_by_region=group_by_region,
#         )
#         _indexed_items = matterport_parse.filter_indices(
#             _indexed_items_all, self.filter_keys, keep_valid_regions=False
#         )

#         self.filtered_indexed_items = _indexed_items
#         self.split = self.split

#         self.tar_file_reader = None
#         val_uuids_file = "./matterport_val_uuid_file.lst"

#         if self.split == "train":

#             good_inds = self.mark_good_inds(_indexed_items, split=self.split)

#             _indexed_items = np.array(_indexed_items)[np.array(good_inds)].tolist()

#             sample_files = data_sampling.matterport_sample_files
#             sample_file = sample_files[str(opts.DATALOADER.SAMPLE_FRAC)]

#             if sample_file != "matterport_100" and self.split == "train":
#                 sample_file = osp.join(
#                     self.opts.CACHE_DIR,
#                     "sampling_splits",
#                     sample_file,
#                     f"{self.split}.pkl",
#                 )
#                 sampled_indexes = self.select_sampled_items(
#                     indexed_items=_indexed_items, sample_file=sample_file
#                 )
#                 _indexed_items = np.array(_indexed_items)[
#                     np.array(sampled_indexes)
#                 ].tolist()
      
#         elif self.split == "val" and osp.exists(val_uuids_file):
#             logger.info("dropping indices that are not needed for evaluation. ")
#             logger.info(f"Loading uuids from {val_uuids_file}")
#             with open(val_uuids_file) as f:
#                 val_uuids = [k.strip() for k in f.readlines()]
#             val_indexed_items = []
#             val_uuids = set(val_uuids)

#             for indexed_item in _indexed_items:
#                 house_id = indexed_item["house_id"]
#                 image_names = indexed_item["image_names"].replace(".jpg", "")
#                 uuid = f"{house_id}_{image_names}"
#                 if uuid in val_uuids:
#                     val_indexed_items.append(indexed_item)
#             _indexed_items = val_indexed_items
         
#         else:
#             good_inds = self.mark_good_inds(_indexed_items, split=self.split)
#             _indexed_items = np.array(_indexed_items)[np.array(good_inds)].tolist()

#         self.houses2mesh = matterport_parse.preload_meshes(
#             indexed_items=_indexed_items, matterport_dir=self.ROOT
#         )

#         self.faces_cachedir = osp.join(self.opts.CACHE_DIR, "mp3d_data_zmx8")

#         self.indexed_items = _indexed_items
#         self.num_items = len(self.indexed_items)

#         self.sequence_path  = f"vggt_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_subset.pkl"

#         if os.path.exists(self.sequence_path):
#             with open(self.sequence_path, "rb") as file:
#                 print(f"Loading data from {self.sequence_path}...")
#                 self.sequences = pkl.load(file)
#         else:

#             self.dino_overlap_data = self.load_dino_overlap_data(opts.DINO_SIM_DATA, self._house_ids)
#             self.depth_overlap_data = self.load_depth_overlap_data(opts.DEPTH_OVERLAPS_DATA, self._house_ids)
#             self.frustrum_overlap_data  = self.load_frustrum_overlap_data(self.opts.FRUSTUM_OVERLAP_DATA, self._house_ids)
#             # self.compute_normalized_global_histogram()

#             if self.opts.DINO_SIM and not (self.opts.FRUSTRUM_DEPTH_OVERLAP):
#                 self.overlap_pairs_path = f"{self.base_dir}/vggt_pairs_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_subset.pkl"
#             elif not (self.opts.DINO_SIM) and self.opts.FRUSTRUM_DEPTH_OVERLAP:
#                 self.overlap_pairs_path =f"{self.base_dir}/vggt_pairs_matterport3d_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_subset.pkl"
#             elif self.opts.DINO_SIM and self.opts.FRUSTRUM_DEPTH_OVERLAP:
#                 self.overlap_pairs_path = f"{self.base_dir}/vggt_pairs_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_subset.pkl"
#             # else:
#             #     self.overlap_pairs_path = f"pairs_matterport3d_dino_sim_50_to_75_frustrum_overlap_10_to_100_percent_window_size0_{self.split}_fixed_subset.pkl"
            
        
#             if os.path.exists(self.overlap_pairs_path):
#                 with open(self.overlap_pairs_path, "rb") as file:
#                     print(f"Loading data from {self.overlap_pairs_path}...")
#                     self.pairs = pkl.load(file)
#                     # self.create_train_val_subsets(self.pairs)
#             else:
#                 # self.compute_normalized_global_histogram()
#                 print("computing overlapping pairs")
#                 self.pairs = self.process_overlapping_pairs()
#             # breakpoint()

#             save_path = f"{self.base_dir}/vggt_scene_sequences_matterport3d_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_subset.pkl"

#             self.pairs  = self.filter_symmetric_pairs(self.pairs)


#             self.pairs = self.collect_all_matches(self.pairs, save_path)

#             # breakpoint()


#             # self.pairs = self.filter_pairs_by_rotation_angle(self.pairs)

#             # print("preparing training data")
#             # self.processed_sequences(self.pairs,self.sequence_path)

#         # self.sequence_list = list(self.data_store.keys())
#         # self.sequence_list_len = len(self.sequence_list)
#         # self.total_frame_num = total_frame_num

#         # self.create_subsets(self.sequences)
#         # breakpoint()
    
       
#         # pkl_path =f"pairs_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_overfit_subset.pkl"
#         # if os.path.exists(pkl_path):
#         #         with open(pkl_path, "rb") as file:
#         #             print(f"Loading data from  {self.split} subset")
#         #             self.sequences = pkl.load(file)
        
#         # # # breakpoint()

#         self.keys = list(self.sequences.keys())
#         self.seed  = opts.DATALOADER.SEED

#     def rotation_angle_deg(self, R1, R2):
#         """Compute the relative rotation angle (in degrees) between two 3x3 rotation matrices."""
#         R_rel = R2 @ R1.T
#         angle = R.from_matrix(R_rel).magnitude() * (180.0 / np.pi)
#         return angle

#     def filter_pairs_by_rotation_angle(self, pairs_data, min_angle_deg=90, max_angle_deg=120):
#         """
#         Filter a dictionary of image pairs based on the relative rotation angle between views.

#         Args:
#             pairs_data: dict, with keys like 'house__img1__img2' and values with RT_view1/RT_view2.
#             min_angle_deg: minimum rotation angle (inclusive)
#             max_angle_deg: maximum rotation angle (inclusive)

#         Returns:
#             filtered_pairs: dict with only the keys where the rotation angle is within the given range.
#         """
#         filtered_pairs = {}
        
#         for key, value in pairs_data.items():
#             RT1 = value["RT_view1"]
#             RT2 = value["RT_view2"]
#             R1 = np.array(RT1)[:3, :3]
#             R2 = np.array(RT2)[:3, :3]
            
#             angle = self.rotation_angle_deg(R1, R2)
            
#             if min_angle_deg <= angle <= max_angle_deg:
#                 filtered_pairs[key] = value
#                 filtered_pairs[key]["rotation_angle_deg"] = angle  # Optional: add angle info

#         return filtered_pairs


#     def processed_sequences(self, pairs_data, output_path):
#         """
#         Converts pairs_data into a structured dictionary of pairwise matches.

#         Args:
#             pairs_data: dict with keys "<house_id>__<img1>__<img2>", each mapping to:
#                 {
#                     "house_id": str,
#                     "RT_view1": array,
#                     "RT_view2": array,
#                     "intrinsic_view1": array,
#                     "intrinsic_view2": array,
#                     "img1_path": str,
#                     "img2_path": str,
#                     "iou_frustrum": float,
#                     "iou_dino_sim": float,
#                 }
#             output_path: Path to save the processed results as .pkl

#         Returns:
#             results: dict with keys "<house_id>__<img1>" mapping to:
#                 {
#                     "img1_name": {
#                         "img_path": str,
#                         "intrinsics": array,
#                         "extrinsics": array,
#                     },
#                     "img2_name": {
#                         "img_path": str,
#                         "intrinsics": array,
#                         "extrinsics": array,
#                     },
#                     "iou_dino_sim": float,
#                     "iou_frustrum": float,
#                 }
#         """
#         matches = {}

#         for key, v in pairs_data.items():
#             house_id, img1, img2 = key.split("__")

#             match_key = f"{house_id}__{img1}__{img2}"

#             matches[match_key] = {
#                 img1: {
#                     "img_path": v["img1_path"],
#                     "intrinsics": v["intrinsic_view1"],
#                     "extrinsics": v["RT_view1"],
#                 },
#                 img2: {
#                     "img_path": v["img2_path"],
#                     "intrinsics": v["intrinsic_view2"],
#                     "extrinsics": v["RT_view2"],
#                 },
#                 "iou_dino_sim": v["iou_dino_sim"],
#                 "iou_frustrum": v["iou_frustrum"],
#             }
#         # breakpoint()
#         with open(output_path, "wb") as f:
#             pkl.dump(matches, f)

#         print(f"Saved {len(matches)} matches to {output_path}")
#         self.sequences = matches
#         # breakpoint()
#         return


#     def collect_all_matches(self, pairs_data, output_path):
#         """
#         Processes all house/image_name pairs in pairs_data and stores structured match data.

#         Args:
#             pairs_data: dict with keys in format <house_id>__<img1>__<img2>
#             output_path: path to save the final results as .pkl

#         Returns:
#             results: dict mapping "<house_id>__<image_name>" to {
#                 "target": ...,
#                 "matches": [...]
#             }
#         """
#         house_image_keys = defaultdict(list)

#         for key in pairs_data:
#             house_id, img1, img2 = key.split("__")
#             house_image_keys[(house_id, img1)].append(key)
#             house_image_keys[(house_id, img2)].append(key)

#         results = {}

#         for (house_id, image_name), keys in house_image_keys.items():
#             entry = {"target": None, "matches": []}

#             for key in keys:
#                 v = pairs_data[key]
#                 _, img1, img2 = key.split("__")

#                 data1 = {
#                     "img_path": v["img1_path"],
#                     "intrinsics": v["intrinsic_view1"],
#                     "extrinsics": v["RT_view1"],
#                 }

#                 data2 = {
#                     "img_path": v["img2_path"],
#                     "intrinsics": v["intrinsic_view2"],
#                     "extrinsics": v["RT_view2"],
#                 }

#                 if img1 == image_name:
#                     if entry["target"] is None:
#                         entry["target"] = data1
#                     other = data2
#                 elif img2 == image_name:
#                     if entry["target"] is None:
#                         entry["target"] = data2
#                     other = data1
#                 else:
#                     continue

#                 other.update({
#                     "iou_dino_sim": v["iou_dino_sim"],
#                     "iou_frustrum": v["iou_frustrum"],
#                 })
#                 entry["matches"].append(other)

#             # Add the target entry itself to the start of matches
#             if entry["target"] is not None:
#                 entry["matches"].insert(0, copy.deepcopy(entry["target"]))

#             results[f"{house_id}__{image_name}"] = entry
        
#         # breakpoint()
#         with open(output_path, "wb") as f:
#             pkl.dump(results, f)

#         print(f"Saved {len(results)} entries to {output_path}")
#         return results



#     def get_image_name(self, path):
#         return path.split("/")[-1].split(".")[0]

#     def filter_symmetric_pairs(self, pairs_data):
#         filtered = {}
#         seen = set()

#         for key, value in pairs_data.items():
#             try:
#                 house_id, img1, img2 = key.split("__")
#             except ValueError:
#                 continue  # skip malformed keys

#             # Create a canonical pair key for symmetry check
#             canonical_key = (house_id, tuple(sorted([img1, img2])))

#             if canonical_key not in seen:
#                 seen.add(canonical_key)
#                 filtered[key] = value

#         return filtered


    
#     def create_subsets(self, pairs_dict):
#         """
#         Randomly select 10 key-value pairs from the given dictionary and save them 
#         as both train and validation subsets.

#         Args:
#             pairs_dict (dict): The loaded dictionary of pairs from a .pkl file.


#         """

#         sample_size = 100
#         if len(pairs_dict) < sample_size:
#             raise ValueError(f"Dataset contains only {len(pairs_dict)} samples, but {sample_size} were requested.")

#         # Randomly select 10 keys
#         selected_keys = random.sample(list(pairs_dict.keys()), sample_size)
        
#         # Create subsets with the selected keys
#         selected_samples = {key: pairs_dict[key] for key in selected_keys}

#         # Save function
#         def save_subset(pairs_subset, filename):
#             with open(filename, "wb") as f:
#                 pkl.dump(pairs_subset, f)

#         # File name
#         train_pkl =f"pairs_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_train_overfit_subset.pkl"
#         val_pkl =f"pairs_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_val_overfit_subset.pkl"

        

#         # Save val subsets
#         save_subset(selected_samples, val_pkl)
#         save_subset(selected_samples, train_pkl)

#         # print(f"Saved {sample_size} samples to '{val_pkl}'.")
#         print(f"Saved {sample_size} samples to '{train_pkl}'.")
#         print(f"Saved {sample_size} samples to '{val_pkl}'.")

#         return selected_samples


#     def create_val_subsets(self, pairs_dict):
#         """
#         Randomly select 10 key-value pairs from the given dictionary and save them 
#         as both train and validation subsets.

#         Args:
#             pairs_dict (dict): The loaded dictionary of pairs from a .pkl file.


#         """


#         sample_size = 2000
#         if len(pairs_dict) < sample_size:
#             raise ValueError(f"Dataset contains only {len(pairs_dict)} samples, but {sample_size} were requested.")

#         # Randomly select 10 keys
#         selected_keys = random.sample(list(pairs_dict.keys()), sample_size)
        
#         # Create subsets with the selected keys
#         selected_samples = {key: pairs_dict[key] for key in selected_keys}

#         # Save function
#         def save_subset(pairs_subset, filename):
#             with open(filename, "wb") as f:
#                 pkl.dump(pairs_subset, f)

#        # File name
#         # val_pkl = "pairs_matterport3d_dino_sim_65_to_85_frustrum_overlap_10_to_100_percent_window_size0_val_fixed_subset.pkl"
#         val_pkl =f"pairs_matterport3d_dino_sim_{int(self.opts.DINO_SIM_THRESHOLD[0]*100)}_to_{int(self.opts.DINO_SIM_THRESHOLD[1]*100)}_frustrum_overlap_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0]*100)}_to_{int(self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]*100)}_percent_window_size{self.opts.WINDOW_SIZE if self.opts.WINDOW_SIZE is not None else 0}_{self.split}_fixed_subset.pkl"


#         # Save val subsets
#         save_subset(selected_samples, val_pkl)
        
#         print(f"Saved {sample_size} samples to '{val_pkl}'.")
#         return selected_samples

#     def mark_good_inds(self, indexed_items, split):
#         opts = self.opts
#         checkpoint_dir = osp.join(self.opts.CACHE_DIR, "matterport_good_list", "all")
#         # checkpoint_dir = osp.join(self.opts.CACHE_DIR, 'snapshots', opts.NAME)
#         if split == "val":
#             good_inds_file = osp.join(checkpoint_dir, f"good_inds_file_{split}.pkl")
#         else:
#             good_inds_file = osp.join(checkpoint_dir, "good_inds_file.pkl")
#         # logger.info(f"Good Inds file {good_inds_file}")
#         if osp.exists(good_inds_file):
#             # logger.info("Fast loading good inds from file")
#             with open(good_inds_file, "rb") as f:
#                 good_list = pkl.load(f)
#         else:
#             good_list = []
#             for ix in range(len(indexed_items)):
#                 meta_data = indexed_items[ix]
#                 elem = self.check_example_validity(meta_data)
#                 # if ix % 1000 == 0:
#                 #     # logger.info(f"Checking {ix}/{len(indexed_items)}")
#                 if not elem["empty"]:
#                     uuid = meta_data["uuid"]
#                     good_list.append(uuid)

#             with open(good_inds_file, "wb") as f:
#                 pkl.dump(good_list, f)

#         good_inds = []

#         good_list = set(good_list)
#         for ix in range(len(indexed_items)):
#             meta_data = indexed_items[ix]
#             uuid = meta_data["uuid"]
#             if uuid in good_list:
#                 good_inds.append(ix)

#         return good_inds

#     @staticmethod
#     def get_camera(camera_dict):
#         cam2world = camera_dict["cam2world"]
#         intrinsic = camera_dict["intrinsics"]
#         Rx = trimesh.transformations.euler_matrix(np.pi, 0, 0, "sxyz")
#         cam2world = np.matmul(cam2world, Rx)
#         return cam2world, intrinsic

#     @staticmethod
#     def get_img(dataset_dir, metadata):
#         img_path = MatterportPreprocess.get_img_path(
#             dataset_dir=dataset_dir, metadata=metadata
#             )
#         img = imageio.imread(img_path)
#         return img

#     @staticmethod
#     def get_depth_path(dataset_dir, metadata):
#         house_id = metadata["house_id"]
#         img_name = metadata["image_names"]
#         img_dir = osp.join(
#                 dataset_dir,
#                 "undistorted_depth_images",
#                 house_id,
#                 "undistorted_depth_images",
#             )
#         img_name = img_name.replace("_i", "_d").replace(".jpg", ".png")
#         depth_path = osp.join(img_dir, img_name)
#         return depth_path

#     @staticmethod
#     def get_img_path(dataset_dir, metadata):
#         house_id = metadata["house_id"]
#         img_name = metadata["image_names"]
#         img_dir = osp.join(
#                 dataset_dir,
#                 "undistorted_color_images",
#                 house_id,
#                 "undistorted_color_images",
#             )
#         img_path = osp.join(img_dir, img_name)
#         return img_path



#     def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
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
#         assert min_margin_x > W/5, f'Bad principal point in view={info}'
#         assert min_margin_y > H/5, f'Bad principal point in view={info}'
#         # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
#         l, t = cx - min_margin_x, cy - min_margin_y
#         r, b = cx + min_margin_x, cy + min_margin_y
#         crop_bbox = (l, t, r, b)
#         image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

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
#         if self.aug_crop > 1:
#             target_resolution += rng.integers(0, self.aug_crop)
#         image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

#         # actual cropping (if necessary) with bilinear interpolation
#         intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
#         crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
#         image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

#         return image, depthmap, intrinsics2

#     def process_overlapping_pairs(self):
#         pairs_data = {}

#         for house_id in self.dino_overlap_data:
#             indexed_items_all = self.house2indexes[house_id]["all"]
#             indexed_items_all = matterport_parse.filter_indices(
#                 indexed_items_all, self.filter_keys, keep_valid_regions=False)

#             num_cams = len(indexed_items_all)

#             # Precompute and cache data
#             cameras = [self.get_camera(item) for item in indexed_items_all]
#             RTs = [np.linalg.inv(camera[0]) for camera in cameras]
#             intrinsics = [camera[1] for camera in cameras]
#             depth_img_paths = [self.get_depth_path(self.ROOT, item) for item in indexed_items_all]
#             rgb_img_paths = [self.get_img_path(self.ROOT, item) for item in indexed_items_all]
    
#             house_dino_overlaps = self.dino_overlap_data[house_id]
#             house_depth_frustrum_overlaps =  self.depth_overlap_data[house_id]
            
#             valid_indices = []
#             rng = np.random.default_rng(seed=42)

#             for i in range(num_cams):
#                 rgb_image = cv2.cvtColor(cv2.imread(rgb_img_paths[i]), cv2.COLOR_BGR2RGB)
#                 depth_map = self.process_matterport_depth(depth_img_paths[i])
#                 _, depth_map, _ = self._crop_resize_if_necessary(
#                     rgb_image, depth_map, intrinsics[i], resolution=(512, 512), rng=rng, info=None
#                 )

#                 if self.has_valid_depth(depth_map):
#                     valid_indices.append(i)

#             # # Now apply the logarithmic window for valid indices
#             for ix in valid_indices:
#                 # Logarithmic window offsets
#                 pairsid = set()
#                 if self.opts.WINDOW_SIZE:
#                     offsets = [2**i +(i%2) for i in range(self.opts.WINDOW_SIZE)]  # Example window size (can be adjusted)
                   

#                     # Loop through the offsets
#                     for offset in offsets:
#                         # Calculate the indices from the logarithmic window
#                         idx_l = ix - offset
#                         idx_r = ix + offset
#                         # Ensure valid IoU and in range
#                         for jx in [idx_l, idx_r]:
#                             if 0 <= jx < num_cams and ix != jx:
#                                 dino_sim_score = house_dino_overlaps["ious"][ix, jx]
#                                 iou_depth_frustrum  = house_depth_frustrum_overlaps["ious"][ix, jx]
#                                 if self.opts.DINO_SIM and not (self.opts.FRUSTRUM_DEPTH_OVERLAP):
#                                     if (self.opts.DINO_SIM_THRESHOLD[0] <= dino_sim_score <= self.opts.DINO_SIM_THRESHOLD[1]):
#                                         pairsid.add((ix, jx))
#                                 elif self.opts.FRUSTRUM_DEPTH_OVERLAP and not (self.opts.DINO_SIM):
#                                     if (self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0] <= iou_depth_frustrum <= self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]):
#                                         pairsid.add((ix, jx))
#                                 elif self.opts.DINO_SIM and self.opts.FRUSTRUM_DEPTH_OVERLAP:
#                                     if (self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0] <= iou_depth_frustrum <= self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]) and (self.opts.DINO_SIM_THRESHOLD[0] <= dino_sim_score <= self.opts.DINO_SIM_THRESHOLD[1]):
#                                         pairsid.add((ix, jx))
#                 else:
#                     # breakpoint()
#                     for jx in valid_indices:
                        
#                         if ix != jx:

#                             dino_sim_score = house_dino_overlaps["ious"][ix, jx]
#                             iou_depth_frustrum  = house_depth_frustrum_overlaps["ious"][ix, jx]
#                             if self.opts.DINO_SIM and not (self.opts.FRUSTRUM_DEPTH_OVERLAP):
#                                 if (self.opts.DINO_SIM_THRESHOLD[0] < dino_sim_score <= self.opts.DINO_SIM_THRESHOLD[1]):
#                                     pairsid.add((ix, jx))
#                             elif self.opts.FRUSTRUM_DEPTH_OVERLAP and not (self.opts.DINO_SIM):
#                                 if (self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0] <= iou_depth_frustrum <= self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]):
#                                     pairsid.add((ix, jx))
#                             elif self.opts.DINO_SIM and self.opts.FRUSTRUM_DEPTH_OVERLAP:
#                                 if (self.opts.FRUSTRUM_OVERLAP_THRESHOLD[0] <= iou_depth_frustrum <= self.opts.FRUSTRUM_OVERLAP_THRESHOLD[1]) and (self.opts.DINO_SIM_THRESHOLD[0] <= dino_sim_score <= self.opts.DINO_SIM_THRESHOLD[1]):
#                                     pairsid.add((ix, jx))

#                 # Loop over the selected pairs and store them
#                 for i, j in pairsid:
#                     pairs_data[f"{house_id}__{self.get_image_name(rgb_img_paths[i])}__{self.get_image_name(rgb_img_paths[j])}"] = {
#                             "house_id": house_id,
#                             "RT_view1": RTs[i],
#                             "RT_view2": RTs[j],
#                             "intrinsic_view1": intrinsics[i],
#                             "intrinsic_view2": intrinsics[j],
#                             "img1_path": rgb_img_paths[i],
#                             "img2_path": rgb_img_paths[j],
#                             "iou_frustrum": house_depth_frustrum_overlaps["ious"][i, j],
#                             "iou_dino_sim": house_dino_overlaps["ious"][i, j]
#                         }
                                    
#         with open(self.overlap_pairs_path, "wb") as file:
#             pkl.dump(pairs_data, file)
        
#         return pairs_data



#     def process_matterport_depth(self,depth_path):
#         depth = imageio.imread(depth_path)
#         depth = depth.astype(float) * 0.25
#         depth = depth / 1000
#         return depth


#     def load_image(self, house_id, img_name):
#         """
#         Load an undistorted color image from the dataset.

#         Args:
#              img_path

#         Returns:
#             numpy.ndarray: The loaded image as a NumPy array.

#         """
#         img_dir = osp.join(
#             self.ROOT,
#             "undistorted_color_images",
#             house_id,
#             "undistorted_color_images",
#         )

#         # Append the '.jpg' extension to the image name
#         img_name = f"{img_name}.jpg"

#         # Construct the full path to the image file
#         image_path = osp.join(img_dir, img_name)

#         # Read and return the image
#         img = imageio.imread(image_path)
        
#         return img


#     def load_depth(self, house_id, depth_name):
#         """
#         Load and process a depth image from the dataset.

#         Args:
#             house_id (str): The unique identifier for the house.
#             depth_name (str): The name of the depth image file (without extension).

#         Returns:
#             numpy.ndarray: The processed depth image as a NumPy array with metric depth values (in meters).

#         """
#         # Construct the directory path containing the depth images
#         img_dir = osp.join(
#             self.ROOT,
#             "undistorted_depth_images",
#             house_id,
#             "undistorted_depth_images",
#         )

#         depth_name = f"{depth_name}.png"

#         # Construct the full path to the depth image file
#         depth_path = osp.join(img_dir, depth_name)

#         # Read the depth image
#         depthmap = imageio.imread(depth_path)

#         # Convert depth values to meters
#         depthmap = depthmap.astype(float) * 0.25  # Scale by 0.25
#         depthmap = depthmap / 1000  # Convert from millimeters to meters
#         depthmap[~np.isfinite(depthmap)] = 0.00  # invalid
#         return depthmap
    

#     def load_dino_overlap_data(self,path, house_ids):
#         dino_overlap_data = {}
#         for house_id in house_ids:
#             with open(osp.join(path, f"{house_id}.pkl"), "rb") as f:
#                 data = pkl.load(f)
#                 dino_overlap_data[house_id] = data
#         return dino_overlap_data

    

#     def load_depth_overlap_data(self,path, house_ids):
#         overlap_data = {}
#         for house_id in house_ids:
#             with open(osp.join(path, f"{house_id}.pkl"), "rb") as f:
#                 data = pkl.load(f)
#                 overlap_data[house_id] = data
#         return overlap_data

    
#     def load_frustrum_overlap_data(self,path, house_ids):
#         overlap_data = {}
#         for house_id in house_ids:
#             with open(osp.join(path, f"{house_id}.pkl"), "rb") as f:
#                 data = pkl.load(f)
#                 overlap_data[house_id] = data
#         return overlap_data

#     def has_valid_depth(self, depth_map):
#         """
#         Check if a depth map is valid based on given conditions.

#         Args:
#             depth_map (torch.Tensor or np.ndarray): The depth map.

#         Returns:
#             bool: True if the depth map is valid, False otherwise.
#         """
#         if isinstance(depth_map, torch.Tensor):
#             depth_map = depth_map.cpu().numpy()  # Convert to NumPy if it's a tensor

#         if not isinstance(depth_map, np.ndarray):
#             raise TypeError("Input must be a PyTorch tensor or NumPy array")

#         depth1d = depth_map.flatten()  # Flatten depth values

#         # More than 60% of values are less than 2 and more than 20% of the values are less than 0.0
#         if np.mean(depth1d >= 2.0) < 0.6 and np.mean(depth1d == 0.0) > 0.2:
#             return False

#         return True


#     def create_colored_point_cloud_from_depth(self, image: np.ndarray,
#                                             depth: np.ndarray,
#                                             RT_world2cam: np.ndarray,
#                                             K: np.ndarray,
#                                             filename: str):
#         """
#         Generate and save a colored world point cloud from an image, depth map, and world-to-camera transform.

#         Args:
#             image (np.ndarray): (H, W, 3) RGB image (uint8).
#             depth (np.ndarray): (H, W) depth map in meters.
#             RT_world2cam (np.ndarray): (4, 4) world-to-camera transformation matrix.
#             K (np.ndarray): (3, 3) camera intrinsics matrix.
#             filename (str): Base filename (should NOT include '_projected.glb').

#         Returns:
#             None. Saves the colored point cloud as a .glb file.
#         """
#         assert image.shape[:2] == depth.shape, "Image and depth must have same height and width"
#         assert RT_world2cam.shape == (4, 4), "RT_world2cam must be 4x4"
#         assert K.shape == (3, 3), "Intrinsics K must be 3x3"

#         H, W = depth.shape
#         fx, fy = K[0, 0], K[1, 1]
#         cx, cy = K[0, 2], K[1, 2]

#         # Generate a pixel grid
#         u, v = np.meshgrid(np.arange(W), np.arange(H))
#         z = depth

#         # Backproject to camera coordinates
#         x = (u - cx) * z / fx
#         y = (v - cy) * z / fy
#         cam_points_hom = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)  # (H*W, 4)

#         # Transform to world coordinates
#         RT_cam2world = closed_form_inverse_se3(RT_world2cam[None])[0]
#         world_points = (RT_cam2world @ cam_points_hom.T).T[:, :3]  # (H*W, 3)

#         # Flatten and filter invalid points
#         valid = np.isfinite(world_points).all(axis=1)
#         world_points = world_points[valid]
#         colors = image.reshape(-1, 3)[valid].astype(np.uint8)

#         # Build and export point cloud
#         pct = trimesh.PointCloud(vertices=world_points, colors=colors)

#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         save_path = f"{filename}_projected.glb"
#         pct.export(save_path, file_type='glb')
#         print(f"Saved projected point cloud to: {save_path}")



#     def save_colored_point_cloud_glb(self, world_points: np.ndarray, image: np.ndarray, filename: str, output_dir: str = ""):
#         """
#         Save a colored point cloud as a .glb file using trimesh.

#         Args:
#             world_points (np.ndarray): (H, W, 3) array of 3D world coordinates.
#             image (np.ndarray): (H, W, 3) RGB image with pixel values in [0, 255] and dtype uint8.
#             filename (str): Name of the output file (should end with '.glb').
#             output_dir (str): Subdirectory under '/scratch/projects/fouheylab/dma9300/occ_project/occ/dust3r/'.
#         """
#         assert world_points.shape == image.shape, "Shape mismatch between world_points and image"
#         assert filename.endswith(".glb"), "Filename should end with '.glb'"

#         # Flatten and mask invalid points
#         H, W, _ = world_points.shape
#         points = world_points.reshape(-1, 3)
#         colors = image.reshape(-1, 3).astype(np.uint8)
#         valid = np.isfinite(points).all(axis=1)
#         points = points[valid]
#         colors = colors[valid]

#         if len(points) == 0:
#             print(f" No valid points to export for {filename}")
#             return

#         # Create point cloud
#         pct = trimesh.PointCloud(vertices=points, colors=colors)

#         # Create full output path
#         full_dir = os.path.join("./")
#         os.makedirs(full_dir, exist_ok=True)
#         full_path = os.path.join(full_dir, filename)

#         # Export as .glb
#         pct.export(full_path, file_type='glb')
#         print(f"Exported point cloud to {full_path}")


#     def get_data(
#         self,
#         seq_index: int = None,
#         img_per_seq: int = 2,
#         aspect_ratio: float = None
#     ) -> dict:
#         """
#         Retrieve a data batch for a specific sequence.

#         Args:
#             seq_index (int): Index of the sequence to retrieve.
#             img_per_seq (int): Number of images per sequence.
#             aspect_ratio (float): Aspect ratio for resizing.

#         Returns:
#             dict: Data batch with images, depths, intrinsics, extrinsics, etc.
#         """
#         key = self.keys[seq_index]
#         house_id, view1_name, view2_name = key.split("__")
#         data = self.sequences[key]

#         if aspect_ratio is None:
#             aspect_ratio = random.uniform(0.33, 1.0)

#         target_shape = self.get_target_shape(aspect_ratio)

#         images, depths, extrinsics, intrinsics = [], [], [], []
#         cam_points, world_points, point_masks = [], [], []
#         image_paths, original_sizes = [], []

#         for view_name in [k for k in data if k not in ("iou_dino_sim", "iou_frustrum")]:
#             view = data[view_name]
#             img_path = view["img_path"]
#             image = self.load_image(house_id, view_name)

#             # Depth loading and processing
#             if self.load_depth:
#                 depth_name = view_name.replace("_i", "_d")
#                 depth_map = self.load_depth(house_id, depth_name)
#                 depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
#             else:
#                 depth_map = None

#             original_size = np.array(image.shape[:2])
#             extri = np.array(view["extrinsics"])  # world2cam
#             intri = np.array(view["intrinsics"])

#             image, depth_map, extri, intri, world_pts, cam_pts, mask, _ = self.process_one_image(
#                 image, depth_map, extri, intri, original_size, target_shape, filepath=img_path
#             )
#             # self.save_colored_point_cloud_glb(world_pts, image, f'{view_name}.glb')
#             # self.create_colored_point_cloud_from_depth(image, depth_map, extri, intri, view_name)
     
#             image = self.image_aug(image)

#             images.append(image)
#             depths.append(depth_map)
#             extrinsics.append(extri)
#             intrinsics.append(intri)
#             cam_points.append(cam_pts)
#             world_points.append(world_pts)
#             point_masks.append(mask)
#             image_paths.append(img_path)
#             original_sizes.append(original_size)

#         return {
#             "seq_name": key,
#             "frame_num": len(images),
#             "images": images,
#             "depths": depths,
#             "extrinsics": extrinsics,
#             "intrinsics": intrinsics,
#             "cam_points": cam_points,
#             "world_points": world_points,
#             "point_masks": point_masks,
#             "original_sizes": original_sizes,
#             "iou_dino_sim": data["iou_dino_sim"],
#             "iou_frustrum": data["iou_frustrum"],
#             "view_name":  view_name,

#         }



#     def __len__(self):
#         return len(self.keys)

    
   
#     def select_sampled_items(self, indexed_items, sample_file):
#         with open(sample_file, "rb") as f:
#             sample_data = pkl.load(f)
#         sampled_indexes = []
#         for ix, index_item in enumerate(indexed_items):
#             house_id = index_item["house_id"]
#             sample_img_ids = sample_data[house_id]
#             img_id = index_item["image_names"]
#             if img_id.replace(".jpg", "") in sample_img_ids:
#                 sampled_indexes.append(ix)
#         return sampled_indexes
 




# def get_args():
#         parser = argparse.ArgumentParser(description="Matterport3D dataset config")

#         parser.add_argument('--split', type=str, default='train',
#                             help="Dataset split to use (e.g., train, val, test)")
#         parser.add_argument('--config_path', type=str, required=True,
#                             help="Path to dataset configuration file")
#         parser.add_argument('--aug_crop', action='store_true',
#                             help="Whether to apply data augmentation via cropping")
#         parser.add_argument('--resolution', type=int, default=512,
#                             help="Image resolution (assumed square, e.g., 512 for 512x512)")
#         parser.add_argument('--seed', type=int, default=42,
#                             help="Random seed for reproducibility")

#         return parser.parse_args()

# def save_views_from_dataset(dataset, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     metadata_records = []

#     for sample_idx, views in enumerate(tqdm(dataset, desc="Processing dataset")):
#         view1, view2 = views

#         # Extract image data
#         img1 = view1['img']  # torch.Tensor or np.ndarray (C, H, W)
#         img2 = view2['img']

#         # Get relevant metadata
#         dino_iou = view1['dino_iou']
#         depth_frustrum_iou = view1['depth_iou_frustrum']
#         pair_name = view1['pair_name']
#         target_name = view1['instance']
#         source_name = view2['instance']

#         # Save images
#         img1_path = os.path.join(output_dir, f"{target_name}.jpeg")
#         img2_path = os.path.join(output_dir, f"{source_name}.jpeg")

#         # Convert to PIL Image and save
#         img1_pil = to_pil_image(img1) if not isinstance(img1, Image.Image) else img1
#         img2_pil = to_pil_image(img2) if not isinstance(img2, Image.Image) else img2
#         img1_pil.save(img1_path, format="JPEG")
#         img2_pil.save(img2_path, format="JPEG")

#         # Record metadata
#         record = {
#             "pair_name": pair_name,
#             "target_image": f"{target_name}.jpeg",
#             "source_image": f"{source_name}.jpeg",
#             "dino_iou": dino_iou,
#             "depth_frustrum_iou": depth_frustrum_iou
#         }
#         metadata_records.append(record)

#     # Save metadata to JSON
#     with open(os.path.join(output_dir, "metadata.json"), "w") as f:
#         json.dump(metadata_records, f, indent=2)

#     print(f"Saved {len(metadata_records)} pairs and metadata to {output_dir}")
# if __name__ == "__main__":
#     args = get_args()
#     dataset = Matterport3D(split= args.split, config_path= args.config_path)
#     dataset[0]
#     breakpoint()
#     # save_views_from_dataset(dataset, "matterport3d_dino_depth_overlaps")

    

   