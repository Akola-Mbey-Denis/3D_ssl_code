import csv 
import os
import json 
import numpy as np 
import pickle as pkl
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image
import imageio.v2 as imageio

def load_split_ids(csv_path):
    split_dict = {}
    train_ids, val_ids, test_ids = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)   # reads using header row: id,train,val,test

        for row in reader:
            _id = row["id"]

            if row["train"] == "1":
                train_ids.append(_id)
            if row["val"] == "1":
                val_ids.append(_id)
            if row["test"] == "1":
                test_ids.append(_id)
        split_dict["train"] = train_ids 
        split_dict["val"] = val_ids
        split_dict["test"] = test_ids

    return split_dict

def rgb_to_depth_path(rgb_path: str) -> str:
    """
    Convert an RGB file path into a Euclidean depth file path by replacing:
    - the first 'rgb' directory with 'depth_zbuffer'
    - the 'rgb' in the filename with 'depth_zbuffer'
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
    filename = filename.replace("rgb", "fixatedpose")

    # Change extension to .json
    filename = os.path.splitext(filename)[0] + ".json"
    
    return os.path.join(dir_name, filename)


def rgb_to_mask_path(rgb_path: str) -> str:
    """
    Convert an RGB file path into a Euclidean depth file path by replacing:
    - the first 'rgb' directory with 'depth_zbuffer'
    - the 'rgb' in the filename with 'depth_zbuffer'
    """
    # point_9_view_9_domain_mask_valid.png
    # Replace the first 'rgb' directory
    depth_path = rgb_path.replace("/rgb/", "/mask_valid/", 1)

    # Replace 'rgb' in the filename
    dir_name, filename = os.path.split(depth_path)
    filename = filename.replace("rgb", "mask_valid")
    
    return os.path.join(dir_name, filename)


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

def is_valid_image(image_path, valid_pixels_frac=0.10):
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

    # Black pixel = (0,0,0) in all channels
    black_mask = np.all(image == 0, axis=-1)
    black_pixels = np.sum(black_mask)

    total_pixels = image.shape[0] * image.shape[1]
    fraction_black = black_pixels / total_pixels

    return fraction_black < valid_pixels_frac



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


def is_wall_depthmap(depth, tol=0.05, wall_frac_thresh=0.4):
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


def load_z_buffer_depth_and_depth_mask(path):
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


def build_overlap_sequences(
    dataset_root,
    split_csv_file,
    scene_dict,
    overlap_thres=0.01,
    max_sources=100,
    split = "train"
):
    """
    Build image-depth-camera sequences from multiple sequences of 1000 images each.

    Args:
        sequences_root (str): Path containing multiple subfolders (one per sequence).
        overlap_thres (float): Minimum overlap to consider.
        max_sources (int): Max number of sources per target.

    Returns:
        dict: Mapping {sceneid_targetid: {"target": {...}, "sources": [...]}}
    """

    results = {}

    dataset_splits = load_split_ids(split_csv_file)

    scene_ids = sorted(dataset_splits[split])
    
    overlaps_dir = "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/replica_overlaps/"

    
    for scene_id in tqdm(scene_ids, desc =f"Processing replica {split} dataset", leave = False): 
        if scene_id in scene_dict.keys():
        
            camera_overlap_path = f"{overlaps_dir}/{scene_id}/camera_overlap.npy"
            camera_occlusion_path = f"{overlaps_dir}/{scene_id}/occlusion_matrix.npy"

            # Load overlap and occlusion matrices
            camera_overlap_matrix = np.load(camera_overlap_path)
            occlusion_matrix = np.load(camera_occlusion_path)

            scene_files = scene_dict[scene_id]
            num_images = occlusion_matrix.shape[0]

            # ----------------------------------------------------
            # PRELOAD EVERYTHING
            # ----------------------------------------------------

            # Precompute all image paths 
            img_paths = [scene_files[i] for i in range(num_images)]

            depth_paths = [rgb_to_depth_path(rgb_path) for rgb_path in img_paths]

            depth_masks_paths = [rgb_to_mask_path(rgb_path) for rgb_path in img_paths]
            depthmaps_list = []
            wall_depthmaps =[]
            good_depths = []
            
            # Cache image validity
            image_valid_cache = [is_valid_image(p) for p in tqdm(
                img_paths, leave=False, desc=f"Validating images for {scene_id}")]

            # Cache camera parameters 
            K_cache = [None] * num_images
            RT_cache = [None] * num_images
            for i in range(num_images):
                cameras_dict = load_camera_params(rgb_to_pointinfo_path(scene_files[i]))
                RT_i, K_i = view_camera_parameters(cameras_dict) # cam2world
                K_cache[i] = K_i
                RT_cache[i] = RT_i
                depthmap, mask = load_z_buffer_depth_and_depth_mask(depth_paths[i])
                good_depth_flag = has_sufficient_valid_depth(depth= depthmap)
                good_depths.append(good_depth_flag)
                wall_depth_flag = is_wall_depthmap(depth= depthmap)
                wall_depthmaps.append(wall_depth_flag)

            # tqdm for per-target progress
            for target_idx in tqdm(range(occlusion_matrix.shape[0]),
                                desc=f"Scene {scene_id}",
                                leave=False):

                if not image_valid_cache[target_idx]:
                    continue

                if not good_depths[target_idx]:
                    continue
                
                if good_depths[target_idx] and wall_depthmaps[target_idx]:
                    continue


                occlusion_values = occlusion_matrix[target_idx]      # (N,)
                overlap_values = camera_overlap_matrix[target_idx]          # (N,)
                

                # Filter by overlap threshold
                valid_indices = np.where(overlap_values >= overlap_thres)[0]

                # Remove self
                valid_indices = valid_indices[valid_indices != target_idx]


                if len(valid_indices) == 0:
                    continue

                # Sort by overlap (primary), occlusion as tie-breaker
                sorted_indices = valid_indices[
                    np.lexsort((
                        -occlusion_values[valid_indices],  # secondary key
                        -overlap_values[valid_indices],              # primary key
                    ))
                ]



                # Select top-k
                # breakpoint()
                selected_sources = sorted_indices[:max_sources]

                # Target info (from cache)
                target_img = img_paths[target_idx]
                target_depth = depth_paths[target_idx]
                target_depth_mask = depth_masks_paths[target_idx]
                K_t = K_cache[target_idx]
                RT_c2w_t = RT_cache[target_idx]


                sources = []
                for src_idx in selected_sources:

                    if not image_valid_cache[src_idx]:
                        continue
                    if not good_depths[src_idx]:
                        continue
                    
                    if good_depths[src_idx] and wall_depthmaps[src_idx]:
                        continue

                    sources.append({
                        "image": img_paths[src_idx],
                        "depth": depth_paths[src_idx],
                        "K": K_cache[src_idx],
                        "mask": depth_masks_paths[src_idx],
                        "extrinsics": RT_cache[src_idx],
                        "occlusion": float(occlusion_matrix[target_idx, src_idx]),
                        "overlap": float(camera_overlap_matrix[target_idx, src_idx]),
                    })

                if len(sources) == 0:
                    continue

                key_name = f"{scene_id}__{target_idx:05d}"
                results[key_name] = {
                    "target": {
                        "image": target_img,
                        "depth": target_depth,
                        "K": K_t,
                        "mask": target_depth_mask,
                        "extrinsics": RT_c2w_t
                    },
                    "sources": sources
                }
    # Save as pickle
    save_path = f"/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/preprocessed_replica_3d_{split}_set.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(results, f)

    print(f"\n Saved {len(results)} sequences to {save_path}")
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", help="dataset split")
    parser.add_argument("--dataset_root", type=str, default="/scratch/dma9300/replica/", help="dataset split")
    parser.add_argument("--scene_splits_csv", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/occ_project/occ/preprocess_datasets/replica_splits.csv", help="CSV file containing scene splits")
    parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/occ_project/preprocessed_training_data/replica_scene_files_1000.pkl", help="Pickle file containing scene files")
    args = parser.parse_args()

    with open(f"{args.scene_file}", "rb") as f:
        all_scene_dict = pkl.load(f)
    # breakpoint()
    build_overlap_sequences(dataset_root = args.dataset_root, split_csv_file = args.scene_splits_csv, scene_dict = all_scene_dict,split = args.split)

