import os
import json 
import csv 
import numpy as np 
import pickle as pkl
from scipy.spatial.transform import Rotation as R
from typing import Dict, List
from collections import defaultdict


def collect_scene_files(root_dir: str, max_images: int = 300) -> Dict[str, List[str]]:
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

        # Loop through files in the subdirectory and append
        for file_name in sorted(os.listdir(subdir_path)):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path):
                # breakpoint()
                scene_files[scene_name].append(file_path)
                
                
        if len(scene_files[scene_name]) > max_images:
            files = scene_files[scene_name]
            idxs = np.linspace(0, len(files) - 1, max_images, dtype=int)
            scene_files[scene_name] = [files[i] for i in idxs]

    return scene_files



# Example usage
root_dir = "/vast/dma9300/hypersim/point_info/hypersim/"

            
if __name__ == "__main__":
    data_dir = "/vast/dma9300/hypersim/"
    depth_dir = "/vast/dma9300/hypersim/depth_euclidean/hypersim/"
    point_info = "/vast/dma9300/hypersim/rgb/hypersim/"
    all_scene_files = collect_scene_files(root_dir = point_info)

    with open("hypersim_scene_files.pkl", "wb") as f:
        pkl.dump(all_scene_files, f)





    # csv_path = "/scratch/projects/fouheylab/dma9300/occ_project/occ/vggt/datasets/hypersim_splits.csv"
    # train_ids, val_ids, test_ids = load_split_ids(csv_path)

    breakpoint()

