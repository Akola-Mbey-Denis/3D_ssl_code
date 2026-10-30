import json 
import csv 
import numpy as np 
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import os
from collections import defaultdict
from typing import Dict, List

import os
from collections import defaultdict
from typing import Dict, List
import numpy as np


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



# def compute_overlaps_per_scene(root_dir):
#     point_infos = []
#     for root, dirs, files in os.walk(root_dir): 
#         for file in files:
#             file_path = os.path.join(root, file)
#             breakpoint()

import os
from collections import defaultdict
from typing import Dict, List

import os
from collections import defaultdict
from typing import Dict, List

def collect_scene_files(root_dir: str) -> Dict[str, List[str]]:
    """
    Collects all file paths for each unique scene across multiple subdirectories.

    Args:
        root_dir (str): Root directory containing multiple scene subfolders.

    Returns:
        Dict[str, List[str]]: 
            scene_name -> [list of file paths for all sub-scenes]
    """
    scene_files = defaultdict(list)

    # Iterate through all subdirectories in root_dir
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        print("dir ::", subdir)
        # Walk inside each subdirectory
        for root, _, files in os.walk(subdir_path):
            breakpoint()
            for file in files:
                # Extract unique scene name from filename (before first '-')
                scene_name = file.split('-')[0]
                file_path = os.path.join(root, file)
                scene_files[scene_name].append(file_path)

    return scene_files




def collect_scene_files(root_dir: str, max_images: int = 500) -> Dict[str, List[str]]:
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

        # Subsample if needed
        if len(files_in_subdir) > max_images:
            indices = np.linspace(0, len(files_in_subdir) - 1, max_images, dtype=int)
            files_in_subdir = [files_in_subdir[i] for i in indices]

        # Append to the scene list
        scene_files[scene_name].extend(files_in_subdir)

    return scene_files





# Example usage
root_dir = "/vast/dma9300/replica/point_info/replica/"

            
if __name__ == "__main__":
    data_dir = "/vast/dma9300/replica/"
    depth_dir = "/vast/dma9300/replica/depth_euclidean/replica/"
    rgb_dir = "/vast/dma9300/replica/rgb/replica/"
    all_scene_files = collect_scene_files(root_dir = rgb_dir)

    with open("replica_scene_files.pkl", "wb") as f:
        pkl.dump(all_scene_files, f)

    # csv_path = "/scratch/projects/fouheylab/dma9300/occ_project/occ/vggt/datasets/replica_splits.csv"
    # train_ids, val_ids, test_ids = load_split_ids(csv_path)

    # breakpoint()

