# import csv 
# import os
# import json 
# import numpy as np 
# import pickle as pkl
# import argparse
# from tqdm import tqdm
# from scipy.spatial.transform import Rotation as R
# from PIL import Image
# import utils.cropping as cropping
# import imageio.v2 as imageio
# import PIL
# from pathlib import Path
# import h5py
# import cv2




# def load_split_ids(csv_path):
#     split_dict = {}
#     train_ids, val_ids, test_ids = [], [], []

#     with open(csv_path, "r") as f:
#         reader = csv.DictReader(f)   # reads using header row: id,train,val,test

#         for row in reader:
#             _id = row["id"]

#             if row["train"] == "1":
#                 train_ids.append(_id)
#             if row["val"] == "1":
#                 val_ids.append(_id)
#             if row["test"] == "1":
#                 test_ids.append(_id)
#         split_dict["train"] = train_ids 
#         split_dict["val"] = val_ids
#         split_dict["test"] = test_ids

#     return split_dict

# def rgb_to_depth_path(rgb_path: str) -> str:
#     """
#     Convert an RGB file path into a Euclidean depth file path by replacing:
#     - the first 'rgb' directory with 'depth_euclidean'
#     - the 'rgb' in the filename with 'depth_euclidean'
#     """
#     # Replace the first 'rgb' directory
#     depth_path = rgb_path.replace("/rgb/", "/depth_euclidean/", 1)

#     # Replace 'rgb' in the filename
#     dir_name, filename = os.path.split(depth_path)
#     filename = filename.replace("rgb", "depth_euclidean")
    
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
# def get_intrinsics(fov_x, fov_y, W, H):
#     """
#     Recover camera intrinsic matrix from fov and image size.

#     Args:
#         fov_x: horizontal FOV in radians
#         fov_y: vertical FOV in radians
#         W: image width in pixels
#         H: image height in pixels

#     Returns:
#         K: 3x3 intrinsic matrix
#     """
#     fx = W / (2 * np.tan(fov_x / 2))
#     fy = H / (2 * np.tan(fov_y / 2))
#     cx = W / 2
#     cy = H / 2

#     K = np.array([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1]
#     ])
#     return K




# def view_camera_parameters(point_info, cam_settings=None):
#     """
#     Returns: (RT_cam2world, K) both in OpenGL coordinate system:
#         +X right, +Y up, -Z forward.
#     """

#     # ---- Extract metadata ----
#     location = point_info["location"]      # (Tx, Tz, Ty) from dataset
#     rotation = point_info["rotation"]      # Blender XYZ euler
#     W = point_info["width"]
#     H = point_info["height"]


#     # Build rotation matrix from Euler angles in 'sxyz' order
#     R_euler = R.from_euler('xyz', rotation).as_matrix()  # shape (3,3)

#     # Multiply by Rx(pi) to flip the camera coordinate frame
#     Rx = R.from_euler('x', np.pi).as_matrix()
#     R_cam2world = R_euler @ Rx


#     # Build 4x4 cam2world matrix
#     RT_4x4 = np.eye(4, dtype=np.float32)
#     RT_4x4[:3, :3] = R_cam2world
#     RT_4x4[:3, 3] = location 

#     # ---- Intrinsics from OpenGL projection ----
#     M_proj = cam_settings["M_proj"]
#     K = opengl_to_intrinsics(
#         proj_matrix=M_proj, 
#         width_pixels=W, 
#         height_pixels=H
#     )

#     return RT_4x4, K



# def is_valid_image(image, valid_pixels_frac=0.10):
#     """
#     Check if the fraction of black pixels in an RGB image is < valid_pixels_frac.

#     Args:
#         image:  RGB image.
#         valid_pixels_frac (float): Maximum allowed fraction of black pixels.

#     Returns:
#         bool: True if black pixel fraction < valid_pixels_frac, else False.
#     """

#     # Load image as RGB
#     image = np.asarray(image)
#     # Ensure it's RGB
#     if image.ndim != 3 :
#         raise ValueError("Image must be an RGB image of shape (H, W, 3).")

#     # Black pixel = (0,0,0) in all channels
#     black_mask = np.all(image == 0, axis=-1)
#     black_pixels = np.sum(black_mask)

#     total_pixels = image.shape[0] * image.shape[1]
#     fraction_black = black_pixels / total_pixels

#     return fraction_black < valid_pixels_frac


# def opengl_to_intrinsics(proj_matrix, width_pixels, height_pixels):
#     # https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/datasets_preprocess/preprocess_hypersim.py#L50
#     # Extract parameters from the projection matrix.
#     K00 = proj_matrix[0, 0] * width_pixels / 2.0
#     K01 = -proj_matrix[0, 1] * width_pixels / 2.0
#     K02 = (1.0 - proj_matrix[0, 2]) * width_pixels / 2.0
#     K11 = proj_matrix[1, 1] * height_pixels / 2.0
#     K12 = (1.0 + proj_matrix[1, 2]) * height_pixels / 2.0
#     return np.array([[K00, K01, K02], [0.0, K11, K12], [0.0, 0.0, 1.0]])


# def load_all_hypersim_cameras(metadata_camera_parameters_csv_file):
#     """Load camera parameters for all scenes without pandas."""
#     cameras = {}
    
#     with open(metadata_camera_parameters_csv_file, 'r') as f:
#         reader = csv.DictReader(f)
        
#         for row in reader:
#             scene_name = row['scene_name']

#             metric_scale = row['settings_units_info_meters_scale']
            
#             # Extract parameters (same as before)
#             width_pixels = int(float(row["settings_output_img_width"]))
#             height_pixels = int(float(row["settings_output_img_height"]))
            
#             use_camera_physical = row["use_camera_physical"].lower() in ['true', '1', 'yes']
#             if use_camera_physical:
#                 fov_x = float(row["camera_physical_fov"])
#             else:
#                 fov_x = float(row["settings_camera_fov"])

#             M_proj = np.array(
#             [
#                 [float(row["M_proj_00"]), float(row["M_proj_01"]), float(row["M_proj_02"]), float(row["M_proj_03"])],
#                 [float(row["M_proj_10"]), float(row["M_proj_11"]), float(row["M_proj_12"]), float(row["M_proj_13"])],
#                 [float(row["M_proj_20"]), float(row["M_proj_21"]), float(row["M_proj_22"]), float(row["M_proj_23"])],
#                 [float(row["M_proj_30"]), float(row["M_proj_31"]), float(row["M_proj_32"]), float(row["M_proj_33"])],
#             ],
#             dtype=np.float32)

    
#             # Store as dictionary indexed by scene_name
#             cameras[scene_name] = {
#                 'width': width_pixels,
#                 'height': height_pixels,
#                 'metric_scale': metric_scale,
#                 'M_proj': M_proj
#             }
    
#     return cameras  # Returns dict, not DictReader

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
#         # breakpoint()
#         cx, cy = intrinsics[:2, 2].round().astype(int)
#         min_margin_x = min(cx, W-cx)
#         min_margin_y = min(cy, H-cy)
#         # breakpoint()
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
        
#         # breakpoint()
#         return image, depthmap, intrinsics2, mask

# def load_depth_distance(path):
#     """
#     Load quantized raycast depth and return metric depth in meters.
#     Depth is expected to be stored as integers with 1/512 m resolution.

#     Args:
#         path (str): Path to the depth file (e.g., PNG, uint16).

#     Returns:
#         depth_m (np.ndarray, float32): Depth map in meters.
#     """
#     # Read image (uint16 expected)
#     depth_int = imageio.imread(path)

#     invalid_val = (2**16 - 1)
#     mask = depth_int < invalid_val

#     # Convert to float (metric depth)
#     depth_m = depth_int.astype(np.float32) / 512

#     depth_m[~mask] = 0.00

#     return depth_m, mask


# def load_image(file_path):

#     """
#     Load a JPEG image and return as a NumPy array (H, W, 3) in RGB order.
#     """

#     img = Image.open(file_path).convert("RGB")


#     return np.array(img)


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


# def is_wall_depthmap(depth, tol=0.05, wall_frac_thresh=0.4):
#     """
#         Decide whether a depth map corresponds to a wall-like view.

#         Args:
#             path :  depth path
#             tol: relative tolerance (e.g. 0.05 = ±5% of median)
#             wall_frac_thresh: fraction of pixels near median above which we call it a wall

#         Returns:
#             bool: True if wall-like, False otherwise
#     """

#     valid = depth > 0.1
#     d = depth[valid]

#     if len(d) == 0:
#         return True  # treat invalid depth as wall/bad

#     med = np.median(d)
#     close = np.abs(d - med) < tol * med

#     frac_close = close.mean()

#     return frac_close >= wall_frac_thresh



# def hypersim_distance_to_depth(distance, width=1024, height=768, focal=886.81):
#     """
#     Convert Hypersim "distance" (from camera) to depth along Z.
#     see https://github.com/apple/ml-hypersim/issues/9


#     Args:
#         distance: np.ndarray of shape (H, W) with distance values
#         width: image width (default 1024)
#         height: image height (default 768)
#         focal: camera focal length (default 886.81)

#     Returns:
#         depth: np.ndarray of shape (H, W)
#     """
#     # Create image plane coordinates centered at 0
#     x = np.linspace(-0.5 * width + 0.5, 0.5 * width - 0.5, width)
#     y = np.linspace(-0.5 * height + 0.5, 0.5 * height - 0.5, height)
#     xv, yv = np.meshgrid(x, y)

#     # 3D vectors from camera to pixels
#     image_plane = np.stack([xv, yv, np.full_like(xv, focal)], axis=-1)  # H x W x 3

#     # Compute depth along Z
#     norm = np.linalg.norm(image_plane, axis=2)
#     depth = distance / norm * focal

#     return depth.astype(np.float32)


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



# def load_camera_matrices(pose_path):
   
#     with open(pose_path, "r") as f:
#         camera_data = json.load(f)

#     return camera_data


# def build_overlap_sequences(
#     dataset_root,
#     scene_dict,
#     iou_threshold=0.20,
#     max_sources=100,
#     split="full"
# ):


#     results = {}

#     overlaps_dir = "/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/megadepth_overlaps/"
#     camera_poses_path = "/projects/rps/dff6142/fouheylab/shared_datasets/MegaDepth/pose_data/"

#     for scene_name in tqdm(scene_dict.keys(), desc="Processing sequences"):

#         # -----------------------------
#         # Load overlap + covisibility matrices (FAST)
#         # -----------------------------
#         camera_overlap_path = f"{overlaps_dir}/{scene_name}/camera_overlap.npy"
#         camera_occlusion = f"{overlaps_dir}/{scene_name}/occlusion_matrix.npy"

#         if not os.path.isfile(camera_overlap_path):
#             print(f"Invalid sequences in {scene_name}")
#             # breakpoint()
#             continue  # skip this subfolder safely

    
#         camera_overlap_matrix = np.load(camera_overlap_path)
#         occlusion_matrix = np.load(camera_occlusion)

#         scene_files = scene_dict[scene_name]
#         num_images = len(scene_files)
#         images_list = []
#         depthmaps_list = []
#         RT_list = []
#         K_list = []

#         depth_paths_list = []
        

#         for i in tqdm(range(num_images), desc=f"Preloading {scene_name}"):
#             p = Path(scene_files[i])
#             main_dir = p.parents[2].name    # '0000'
#             dense_dir = p.parents[1].name   # 'dense0'
#             subdir = dense_dir.replace("dense", "")  # '0'
#             image_name = scene_files[i].split("/")[-1]
                
#             pose_path = os.path.join(camera_poses_path, main_dir, "sparse/manhattan",subdir, "camera_matrices.json")

#             file_parts = scene_files[i].split(".")

#             depth_path = f"{file_parts[0].replace('/imgs/', '/depths/')}.h5"

#             depthmap, mask   = load_depth(depth_path)

#             rgb_image = load_image(scene_files[i])

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
             
#             image_new, depth_new, _,  _ = _crop_resize_if_necessary(image = rgb_image, depthmap=depthmap, intrinsics=K_post, mask = mask.astype(np.int32),resolution=(518, 518),rng=rng,info=None)
#             rgb_image_new = np.asarray(image_new)
#             RT_cam2world = np.linalg.inv(RT_world2cam)
#             RT_cam2world[:3, :3] = RT_cam2world[:3, :3] @ np.eye(3).T                                
#             depthmaps_list.append(depth_new)
#             images_list.append(image_new)
#             K_list.append(K_post)
#             RT_list.append(RT_cam2world)
#             depth_paths_list.append(depth_path)


#         # Preload validity of each image
#         image_valid_cache = [is_valid_image(image = image) for image in tqdm(images_list, leave =False, desc="Validating images")]
#         good_depths = [has_sufficient_valid_depth(depth = depth_map) for depth_map in tqdm(depthmaps_list, leave=False, desc="Validationg images")]
#         # wall_depthmaps = [is_wall_depthmap(depth = depth) for depth in tqdm(depthmaps_list, leave =False, desc="Checking wall depths")]


#         #  Precompute basename/view name
#         view_names = [
#             os.path.splitext(os.path.basename(p))[0]
#             for p in scene_files
#         ]



#         # -----------------------------
#         # MAIN LOOP — target view
#         # -----------------------------
#         for target_idx in tqdm(range(num_images),
#                                desc=f"Scene {scene_name}",
#                                leave=False):

#             if not image_valid_cache[target_idx]:
#                 continue

#             if not good_depths[target_idx]:
#                 continue

#             # if good_depths[target_idx] and wall_depthmaps[target_idx]:
#             #     continue

#             occlusion_values = occlusion_matrix[target_idx]      # (N,)
#             camera_overlap_values = camera_overlap_matrix[target_idx]          # (N,)

#             # Filter by overlap threshold
#             valid_indices = np.where(camera_overlap_values >= iou_threshold)[0]

#             # Remove self
#             valid_indices = valid_indices[valid_indices != target_idx]

#             if len(valid_indices) == 0:
#                 continue

#             sorted_indices = valid_indices[
#                     np.lexsort((
#                         -occlusion_values[valid_indices],  # secondary key
#                         -camera_overlap_values[valid_indices],              # primary key
#                     ))
#                 ]

#             # Select top-k
#             selected_sources = sorted_indices[:max_sources]

#             sources = []
#             for src_idx in selected_sources:

#                 if not image_valid_cache[src_idx]:
#                     continue

#                 if not good_depths[src_idx]:
#                     continue
                
#                 # if good_depths[src_idx] and wall_depthmaps[src_idx]:
#                 #     continue

#                 occlusion_score = float(occlusion_matrix[target_idx, src_idx])
#                 overlap_score = float(camera_overlap_matrix[target_idx, src_idx])

#                 sources.append({
#                     "image": scene_files[src_idx],
#                     "depth": depth_paths_list[src_idx],
#                     "K": K_list[src_idx],
#                     "extrinsics": RT_list[src_idx],
#                     "occlusion": occlusion_score,
#                     "overlap": overlap_score
#                 })

#             # Target view setup
#             target_view_name = view_names[target_idx]

#             key_name = f"{scene_name}__{target_view_name}"

#             results[key_name] = {
#                 "target": {
#                     "image": scene_files[target_idx],
#                     "depth": depth_paths_list[target_idx],
#                     "K": K_list[target_idx],
#                     "extrinsics": RT_list[target_idx]
#                 },
#                 "sources": sources
#             }
#             # breakpoint()
#         del good_depths, image_valid_cache, images_list, depthmaps_list
        
#     # -----------------------------
#     # SAVE FINAL RESULT
#     # -----------------------------
#     save_path = f"/projects/rps/dff6142/fouheylab/dma9300/occ_project/preprocessed_training_data/preprocessed_megadepth_3d_{split}_set.pkl"
#     with open(save_path, "wb") as f:
#         pkl.dump(results, f)

#     print(f"\n Saved {len(results)} sequences to {save_path}")

#     return results



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--split", type=str, default="all", help="dataset split")
#     parser.add_argument("--dataset_root", type=str, default="/scratch/dma9300/hypersim/", help="dataset split")
#     parser.add_argument("--scene_file", type=str, default="/projects/rps/dff6142/fouheylab/dma9300/occ_project/preprocessed_training_data/megadepth_scene_files.pkl", help="Pickle file containing scene files")
#     args = parser.parse_args()

#     with open(f"{args.scene_file}", "rb") as f:
#         all_scene_dict = pkl.load(f)
#     # breakpoint()
#     build_overlap_sequences(dataset_root = args.dataset_root, scene_dict = all_scene_dict,split = args.split)


import os
import json
import argparse
import pickle as pkl
from collections import defaultdict

import numpy as np
from tqdm import tqdm


MIN_IMAGES = 30                                           # keys with fewer images are dropped
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 0
SAVE_DIR = "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data"


def make_scene_splits(scene_dict, overlap_basedir, split_file):
    """
    MegaDepth has no official split. Keep the valid keys (>= MIN_IMAGES images and overlaps computed),
    then split them 80 / 10 / 10 by BASE scene ("0015" for "0015_dense0", "0015_dense1", ...), so all
    reconstructions of one place land in the same split. Saved to `split_file` (json + one txt per split)
    and reused if it exists, so every run gets the same split.
    """
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            splits = json.load(f)
    else:
        valid_keys = sorted(
            k for k, files in scene_dict.items()
            if len(files) >= MIN_IMAGES and os.path.exists(os.path.join(overlap_basedir, k, "camera_overlap.npy"))
        )
        print(f"Valid keys: {len(valid_keys)} of {len(scene_dict)} (>= {MIN_IMAGES} images, overlaps computed)")

        keys_by_scene = defaultdict(list)
        for k in valid_keys:
            keys_by_scene[k.split("_dense")[0]].append(k)
        scenes = sorted(keys_by_scene)
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(scenes))
        scenes = [scenes[i] for i in perm]

        n_val = int(round(SPLIT_FRACTIONS["val"] * len(scenes)))
        n_test = int(round(SPLIT_FRACTIONS["test"] * len(scenes)))
        split_scenes = {
            "val": scenes[:n_val],
            "test": scenes[n_val:n_val + n_test],
            "train": scenes[n_val + n_test:],
        }
        splits = {name: sorted(k for sc in sc_list for k in keys_by_scene[sc]) for name, sc_list in split_scenes.items()}

        with open(split_file, "w") as f:
            json.dump(splits, f, indent=2)
        for name, keys in splits.items():
            with open(split_file.replace(".json", f"_{name}.txt"), "w") as f:
                f.write("\n".join(keys) + "\n")

    # no key and no base scene in more than one split
    base = {name: {k.split("_dense")[0] for k in keys} for name, keys in splits.items()}
    assert not (base["train"] & base["val"]) and not (base["train"] & base["test"]) and not (base["val"] & base["test"]), \
        "scene splits overlap"

    print({name: f"{len(keys)} keys / {len(base[name])} scenes" for name, keys in splits.items()}, "->", split_file)
    return splits


def build_overlap_sequences(scene_dict, scene_names, dump_root, overlap_basedir, overlap_thres=0.20, max_sources=100, split=""):
    """
    Build target / source sequences from the dumped MegaDepth scenes and their overlap matrices.

    Everything is already preprocessed:
      <dump_root>/<key>/rgb/<name>.jpg
      <dump_root>/<key>/depth/<name>.npy       float32, MegaDepth SfM units, 0 = invalid
      <dump_root>/<key>/cameras.pkl            {"<name>.jpg": {"K", "RT_c2w" (OpenCV), "hw", "depth", "src_rgb"}}
      <overlap_basedir>/<key>/camera_overlap.npy, camera_occlusion.npy, frame_names.pkl
    Row / column i of the matrices <-> scene_dict[key][i] (checked against frame_names.pkl).
    Frames were validated by the dump, so no image / depth checks here.
    """
    results = {}

    for scene_name in tqdm(scene_names, desc=f"Processing megadepth {split}"):
        scene_dir = os.path.join(dump_root, scene_name)
        overlaps_dir = os.path.join(overlap_basedir, scene_name)

        if not os.path.exists(os.path.join(overlaps_dir, "camera_overlap.npy")):
            continue                                   # overlaps not computed (yet)

        # ---------------- overlaps ----------------
        camera_overlap_matrix = np.load(os.path.join(overlaps_dir, "camera_overlap.npy"))
        occlusion_matrix = np.load(os.path.join(overlaps_dir, "camera_occlusion.npy"))

        scene_files = scene_dict[scene_name]
        names = [os.path.basename(p) for p in scene_files]
        num_images = len(scene_files)
        assert camera_overlap_matrix.shape == (num_images, num_images), \
            f"{scene_name}: overlap matrix {camera_overlap_matrix.shape} vs {num_images} images"

        frame_names_file = os.path.join(overlaps_dir, "frame_names.pkl")
        if os.path.exists(frame_names_file):
            with open(frame_names_file, "rb") as f:
                assert pkl.load(f) == names, f"{scene_name}: overlap matrix order differs from the scene file"

        # ---------------- cameras (already processed) ----------------
        with open(os.path.join(scene_dir, "cameras.pkl"), "rb") as f:
            cameras = pkl.load(f)

        cams = [cameras[n] for n in names]
        K_list = [np.asarray(c["K"], np.float32) for c in cams]
        RT_list = [np.asarray(c["RT_c2w"], np.float32) for c in cams]
        depth_paths = [os.path.join(scene_dir, "depth", c["depth"]) for c in cams]
        view_names = [os.path.splitext(n)[0] for n in names]

        # ---------------- sequences ----------------
        for target_idx in range(num_images):
            overlap_scores = camera_overlap_matrix[target_idx]      # (N,)
            occlusion_scores = occlusion_matrix[target_idx]         # (N,)

            # sources: overlap above the threshold, not the target itself
            valid_indices = np.where(overlap_scores >= overlap_thres)[0]
            valid_indices = valid_indices[valid_indices != target_idx]
            if len(valid_indices) == 0:
                continue

            # sort by overlap (primary), occlusion as tie-breaker; keep top-k
            sorted_indices = valid_indices[np.lexsort((-occlusion_scores[valid_indices],
                                                       -overlap_scores[valid_indices]))]
            selected_sources = sorted_indices[:max_sources]

            sources = [{
                "image": scene_files[src_idx],
                "depth": depth_paths[src_idx],
                "K": K_list[src_idx],
                "extrinsics": RT_list[src_idx],
                "occlusion": float(occlusion_scores[src_idx]),
                "overlap": float(overlap_scores[src_idx]),
            } for src_idx in selected_sources]

            results[f"{scene_name}__{view_names[target_idx]}"] = {
                "target": {
                    "image": scene_files[target_idx],
                    "depth": depth_paths[target_idx],
                    "K": K_list[target_idx],
                    "extrinsics": RT_list[target_idx],
                },
                "sources": sources,
            }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_file", type=str, default=f"{SAVE_DIR}/megadepth_low_res_overlap_scene_files.pkl", help="Scene files of the dumped dataset (same file used for the overlaps)")
    parser.add_argument("--dump_root", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/low_res_megadepth", help="Dumped dataset: <key>/{rgb, depth, cameras.pkl}")
    parser.add_argument("--overlap_basedir", type=str, default="/projects/rps/dff6142/fouheylab/shared_datasets/dma9300/dataset_overlaps/low_res_megadepth_overlaps/", help="Overlap matrices, one folder per scene key")
    args = parser.parse_args()

    with open(args.scene_file, "rb") as f:
        all_scene_dict = pkl.load(f)

    # disjoint train / val / test keys (created once, then reused)
    splits = make_scene_splits(all_scene_dict, args.overlap_basedir, split_file=f"{SAVE_DIR}/megadepth_low_res_scene_splits.json")

    for split in ("train", "val", "test"):
        results = build_overlap_sequences(scene_dict=all_scene_dict, scene_names=splits[split], dump_root=args.dump_root,
                                          overlap_basedir=args.overlap_basedir, split=split)

        save_path = f"{SAVE_DIR}/preprocessed_low_res_megadepth_3d_{split}_set.pkl"
        with open(save_path, "wb") as f:
            pkl.dump(results, f)
        print(f"\n Saved {len(results)} {split} sequences to {save_path}")