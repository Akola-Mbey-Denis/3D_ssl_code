import os
import json 
import csv 
import numpy as np 
import pickle as pkl
from scipy.spatial.transform import Rotation as R
from typing import Dict, List
from collections import defaultdict



def get_cam_to_world_RT(point_info, euler_x_offset_deg=90.0):
    """
    Convert point_info with camera location, rotation, and fov to camera-to-world 3x4 RT matrix.

    Args:
        point_info: dict with keys
            - 'camera_location': (Tx, Ty, Tz)
            - 'camera_rotation_final': (ex, ey, ez) in radians
        euler_x_offset_deg: X rotation offset in degrees (default 90°)
    
    Returns:
        RT: 3x4 camera-to-world matrix
    """
    location = point_info['camera_location']
    rotation = point_info['camera_rotation_final']

    # Adjust Euler angles
    ex, ey, ez = rotation
    ex -= np.deg2rad(euler_x_offset_deg)
    ey, ez = -ey, -ez

    # Rotation matrix in XZY order
    r = R.from_euler('XZY', [ex, ez, ey])
    R_matrix = r.as_matrix()  # 3x3

    # Translation vector
    Tx, Ty, Tz = location
    T = np.array([[-Tx, Tz, Ty]]).T  # shape 3x1

    # Combine R and T to 3x4 RT matrix
    RT = np.hstack([R_matrix, T])

    return RT


def get_intrinsics(fov_x, fov_y, W, H):
    """
    Recover camera intrinsic matrix from fov and image size.

    Args:
        fov_x: horizontal FOV in radians
        fov_y: vertical FOV in radians
        W: image width in pixels
        H: image height in pixels

    Returns:
        K: 3x3 intrinsic matrix
    """
    fx = W / (2 * np.tan(fov_x / 2))
    fy = H / (2 * np.tan(fov_y / 2))
    cx = W / 2
    cy = H / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def load_split_ids(csv_path):
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

    return train_ids, val_ids, test_ids


def collect_scene_files(root_dir: str, max_images = 1000) -> Dict[str, List[str]]:
    """
    Collects file paths for each unique scene, merging all sub-scene directories.
    Subdirectories are processed in sorted order.

    Args:
        root_dir (str): Root directory containing subdirectories like 'ai_055_009-cam_00'.

    Returns:
        Dict[str, List[str]]: 
            scene_name -> [list of file paths for all sub-scenes, sorted by subdir]
    """
    scene_files = defaultdict(list)

    # Sort the subdirectories
    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        # Extract unique scene name
        scene_name = subdir.split('-')[0]

        # Get all files in the subdir, sorted
        files_in_subdir = sorted([
            os.path.join(subdir_path, f)
            for f in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, f))
        ])

        # Subsample if needed
        if len(files_in_subdir) > max_images:
            indices = np.linspace(0, len(files_in_subdir) - 1, max_images, dtype=int)
            files_in_subdir = [files_in_subdir[i] for i in indices]

        
        scene_files[scene_name].extend(files_in_subdir)
        # breakpoint()

    return scene_files

            
if __name__ == "__main__":
    root_rgb_paths = "/vast/dma9300/taskonomy/rgb/taskonomy"
    # breakpoint()
    all_scene_files = collect_scene_files(root_dir = root_rgb_paths)

    with open("taskonomy_scene_files.pkl", "wb") as f:
        pkl.dump(all_scene_files, f)

    breakpoint()

