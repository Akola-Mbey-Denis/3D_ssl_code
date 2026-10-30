# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *
from .easy_dataset import EasyDataset

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(EasyDataset):
    """
    Base dataset class for VGGT and VGGSfM training.
    
    This abstract class handles common operations like image resizing, 
    augmentation, and coordinate transformations. Concrete dataset 
    implementations should inherit from this class.
    
    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """
    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.
        
        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 518
                - patch_size: Default is 14
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - landscape_check: Default is True
        """
        super().__init__()
        self.common_conf = common_conf
        self.img_size = self.common_conf.IMG_SIZE 
        self.patch_size = self.common_conf.PATCH_SIZE
        self.aug_scale = self.common_conf.AUGS.SCALES
        self.rescale = self.common_conf.RESCALE
        self.rescale_aug = self.common_conf.RESCALE_AUG
        self.landscape_check = self.common_conf.LANDSCAPE_CHECK
        self.img_per_seq = self.common_conf.IMG_PER_SEQ

    def __len__(self):
        return self.len_train



    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Sequence index

        Returns:
            dict: Sample with selected fields converted to torch tensors if present.
        """
        data = self.get_data(seq_index=idx_N, img_per_seq=self.img_per_seq)


        if "images" in data:
            data["images"] = torch.stack(data["images"]).to(torch.float32)

        if "sampled_pix_locs" in data:
            data["sampled_pix_locs"] = torch.stack(data["sampled_pix_locs"]).to(torch.long)

        if "depths" in data:
            data["depths"] = torch.from_numpy(np.stack(data["depths"])).to(torch.float32)

        if "extrinsics" in data:
            data["extrinsics"] = torch.from_numpy(np.stack(data["extrinsics"])).to(torch.float32)

        if "intrinsics" in data:
            data["intrinsics"] =  torch.from_numpy(np.stack(data["intrinsics"])).to(torch.float32)

        if "cam_points" in data:
            data["cam_points"] = torch.from_numpy(np.stack(data["cam_points"])).to(torch.float32)

        if "world_points" in data:
            data["world_points"] =  torch.from_numpy(np.stack(data["world_points"])).to(torch.float32)

        if "point_masks" in data:
            data["point_masks"] = torch.from_numpy(np.stack(data["point_masks"]))

        
        if "s_images" in data:
            data["s_images"] = torch.stack(data["s_images"]).to(torch.float32)

        if "s_depths" in data:
             data["s_depths"] = torch.from_numpy(np.stack(data["s_depths"])).to(torch.float32)
        
        if "s_intrinsics" in data:
            data["s_intrinsics"] = torch.from_numpy(np.stack(data["s_intrinsics"])).to(torch.float32)

        if "s_extrinsics" in data:
            data["s_extrinsics"] = torch.from_numpy(np.stack(data["s_extrinsics"])).to(torch.float32)
        
        if "s_extri" in data:
            data["s_extri"] = torch.from_numpy(np.stack(data["s_extri"])).to(torch.float32)

        if "s_cam_points" in data:
            data["s_cam_points"] = torch.from_numpy(np.stack(data["s_cam_points"])).to(torch.float32)

        if "s_world_points" in data:
            data["s_world_points"] = torch.from_numpy(np.stack(data["s_world_points"])).to(torch.float32)
        
        if "s_point_masks" in data:
            data["s_point_masks"] = torch.from_numpy(np.stack(data["s_point_masks"]))
        
        if "s_depth_masks" in data:
            data["s_depth_masks"] = torch.from_numpy(np.stack(data["s_depth_masks"]))
        
        for k in ("t_overlaps", "s_overlaps", "t_occlusions", "s_occlusions"):
            if k in data:
                data[k] = torch.as_tensor([float(x) for x in data[k]], dtype=torch.float32)

        if "t_images" in data:
            if isinstance(data["t_images"], list):
                data["t_images"] = torch.stack(data["t_images"]).to(torch.float32)
            else:
                data["t_images"] = data["t_images"].to(torch.float32).unsqueeze(0)

        if "event_segments" in data:
            if isinstance(data["event_segments"], list):
                data["event_segments"] = torch.stack(data["event_segments"]).to(torch.float32)
            else:
                data["event_segments"] = data["event_segments"].to(torch.float32).unsqueeze(0)

        if "event_segments_mask" in data:
            if isinstance(data["event_segments_mask"], list):
                data["event_segments_mask"] = torch.stack(data["event_segments_mask"]).to(torch.float32)
            else:
                data["event_segments_mask"] = data["event_segments_mask"].to(torch.float32).unsqueeze(0)

        if "t_depths" in data:
            if isinstance(data["t_depths"], list):
                data["t_depths"] = torch.from_numpy(np.stack(data["t_depths"])).to(torch.float32)
            else:
                data["t_depths"] = torch.from_numpy(data["t_depths"]).to(torch.float32).unsqueeze(0)

        if "t_extrinsics" in data:
            if isinstance(data["t_extrinsics"], list):
                data["t_extrinsics"] = torch.from_numpy(np.stack(data["t_extrinsics"])).to(torch.float32)
            else:
                data["t_extrinsics"] = torch.from_numpy(data["t_extrinsics"]).to(torch.float32).unsqueeze(0)

        if "t_extri" in data:
            if isinstance(data["t_extri"], list):
                data["t_extri"] = torch.from_numpy(np.stack(data["t_extri"])).to(torch.float32)
            else:
                data["t_extri"] = torch.from_numpy(data["t_extri"]).to(torch.float32).unsqueeze(0)

        if "t_intrinsics" in data:
            if isinstance(data["t_intrinsics"], list):
                data["t_intrinsics"] = torch.from_numpy(np.stack(data["t_intrinsics"])).to(torch.float32)
            else:
                data["t_intrinsics"] = torch.from_numpy(data["t_intrinsics"]).to(torch.float32).unsqueeze(0)
    
        if "t_cam_points" in data:
            if isinstance(data["t_cam_points"], list):
                data["t_cam_points"] = torch.from_numpy(np.stack(data["t_cam_points"])).to(torch.float32)
            else:
                data["t_cam_points"] = torch.from_numpy(data["t_cam_points"]).to(torch.float32).unsqueeze(0)

        if "t_world_points" in data:
            if isinstance(data["t_world_points"], list):
                data["t_world_points"] = torch.from_numpy(np.stack(data["t_world_points"])).to(torch.float32)
            else:
                data["t_world_points"] = torch.from_numpy(data["t_world_points"]).to(torch.float32).unsqueeze(0)

        if "t_point_mask" in data:
            if isinstance(data["t_point_mask"], list):
                data["t_point_mask"] = torch.from_numpy(np.stack(data["t_point_mask"])).to(torch.float32)
            else:
                data["t_point_mask"] = torch.from_numpy(data["t_point_mask"]).to(torch.float32).unsqueeze(0)
        
        if "t_depth_masks" in data:
            if isinstance(data["t_depth_masks"], list):
                data["t_depth_masks"] = torch.from_numpy(np.stack(data["t_depth_masks"])).to(torch.float32)
            else:
                data["t_depth_masks"] = torch.from_numpy(data["t_depth_masks"]).to(torch.float32).unsqueeze(0)
        
        
        return data
   


    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.
        
        Args:
            seq_index (int, optional): Index of the sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio. 
            
        Returns:
            Dataset-specific data
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )



    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.
        
        Args:
            aspect_ratio: Target aspect ratio
            
        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        mask = None,
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data.
        
        This method handles image transformations, depth processing, and coordinate conversions.
        
        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.
            
        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # # Apply random scale augmentation during training if enabled
        # if self.training and self.aug_scale:
        #     breakpoint()
        #     random_h_scale, random_w_scale = np.random.uniform(self.aug_scale[0], self.aug_scale[1], 2)
        #     # Avoid random padding by capping at 1.0
        #     random_h_scale = min(random_h_scale, 1.0)
        #     random_w_scale = min(random_w_scale, 1.0)
        #     aug_size = original_size * np.array([random_h_scale, random_w_scale])
        #     aug_size = aug_size.astype(np.int32)
        # else:
        aug_size = original_size


        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv, track, mask = crop_image_depth_and_intrinsic_by_pp(image, depth_map, intri_opencv, aug_size, track=track,  mask = mask,  filepath=filepath)

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Handle landscape vs. portrait orientation
        # rotate_to_portrait = False
        # if self.landscape_check:
        #     # Switch between landscape and portrait if necessary
        #     if original_size[0] > 1.25 * original_size[1]:
        #         if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
        #             target_shape = np.array([target_image_shape[1], target_image_shape[0]])
        #             rotate_to_portrait = True

        # Resize images and update intrinsics
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track, 
                safe_bound=safe_bound, 
                rescale_aug=self.rescale_aug
            )
        else:
            print("Not rescaling the images")

        # # Ensure final crop to target shape
        image, depth_map, intri_opencv, track, mask  = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, mask = mask, filepath=filepath, strict=True,
        )

        # Apply 90-degree rotation if needed
        # if rotate_to_portrait:
        #     assert self.landscape_check
        #     clockwise = np.random.rand() > 0.5
        #     image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
        #         image,
        #         depth_map,
        #         extri_opencv,
        #         intri_opencv,
        #         clockwise=clockwise,
        #         track=track,
        #     )

        # Convert depth to world and camera coordinates
        # world_coords_points, cam_coords_points, point_mask = (
        #     depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        # )
        # if mask is not None:
        #     mask  = mask 
        #     & point_mask
        # else:
        #     mask = point_mask

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            # world_coords_points,
            # cam_coords_points,
            mask,
            track,
        )

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        TODO: add the function to sample the ids by pose similarity ranking.
        
        Sample a set of IDs from a sequence close to a given start index.
        
        You can specify the range either as a ratio of the number of input IDs 
        or as a fixed integer window.
        

        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which the number of IDs expands 
                around the start index. Default is 2.0 if neither expand_ratio nor 
                expand_range is provided.
            expand_range (int, optional): Fixed number of items to expand around the 
                start index. If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the 
                original start index.

        Examples:
            # Using expand_ratio (default behavior)
            # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
            # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

            # Using expand_range directly
            # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
            # IDs are sampled from [90...110] (if boundaries allow).
            
        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # Default behavior

        total_ids = len(ids)
        start_idx = ids[0]

        # Determine the actual expand_range
        if expand_range is None:
            # Use ratio to determine range
            expand_range = int(total_ids * expand_ratio)

        # Calculate valid boundaries
        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # we accept the situation that some sampled ids are the same
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids

