import os
import json
import argparse
import pickle as pkl
import random

from typing import List

import numpy as np
import torch
import yaml
from fvcore.common.config import CfgNode
from tqdm import tqdm

from .dataset_util import *
from .augmentation import PhotometricAug
from .base_dataset import BaseDataset
import utils.cropping as cropping
import PIL
from PIL import Image, ImageFile
import cv2

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# DTU evaluation scans (MVSNet split, as in Pi3 / Poincar3)
TEST_SCAN_NUMBERS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]


def load_cam_mvsnet(words: str):
    """DTU MVSNet-format camera txt -> (intrinsic (3,3), extrinsic (4,4) world2cam)."""
    words = words.split()
    cam = np.zeros((2, 4, 4))
    for i in range(4):
        for j in range(4):
            cam[0][i][j] = words[4 * i + j + 1]
    for i in range(3):
        for j in range(3):
            cam[1][i][j] = words[3 * i + j + 18]
    return cam[1][:3, :3].astype(np.float32), cam[0].astype(np.float32)


class DTU_NVS(BaseDataset):
    """
    DTU multi-view samples, read directly from the files (no LMDB), returned in the same format as
    Sevenscenes_NVS / ETH3D_NVS (t_* targets, s_* sources).

    Files (MVSNet DTU test set, as used by Pi3 / Poincar3), only the 22 evaluation scans:
        <ROOT>/scan<N>/images/<id:08d>.jpg
        <ROOT>/scan<N>/depths/<id:08d>.npy          depth in MILLIMETRES, nan = invalid
        <ROOT>/scan<N>/binary_masks/<id:08d>.png    object mask
        <ROOT>/scan<N>/cams/<id:08d>_cam.txt        MVSNet camera: extrinsic world2cam, intrinsic at depth resolution
    As in Pi3: the image is resized to the depth resolution, the object mask is binarised, resized and
    eroded (10x10), and depth outside the mask is set to 0.

    DTU is evaluation-only. A sample is `max_views` views of one scan:
      - seq_id_map=<json>  (Pi3 / Poincar3 format, {scan: [frame ids]}): one sample per scan, its ids in
        that order (only the first max_views are used);
      - otherwise: every window of `max_views` consecutive views (a window starts every `stride` views,
        default 1), then a fixed random subset of `n_test_samples` windows (default 200; seed 0;
        None = all windows), like the 7-Scenes test set.
    The sample list is saved once to <ROOT>/dtu_test_set__<n>_samples__<max_views>views.pkl and reused,
    so every model and every run sees the same sequences.
    The first `n_target_views` views of a sample are the targets, the rest the sources.

    DTU is in millimetres: use --pm_icp_threshold 100 in eval_ssl.py (Pi3 uses 100 for DTU, 0.1 elsewhere);
    Acc / Comp are then in mm.

    Root folder: root_dir=..., else ROOT_DIR in the config.
    """

    def __init__(self, split="test", config_path=None, overfit=False, img_size=224, in_domain_eval=False,
                 stage=1, max_views=6, n_target_views=2, load_depth=True, root_dir=None,
                 seq_id_map=None, stride=1, n_test_samples=200, *args, **kwargs):
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        opts = CfgNode(raw_cfg)
        self.opts = opts
        self.common_conf = opts.COMMON
        self.common_conf["IMG_SIZE"] = img_size
        self.common_conf["IMG_PER_SEQ"] = max_views - 1
        super().__init__(self.common_conf, *args, **kwargs)

        if split not in ("test", "eval"):
            raise ValueError("DTU is evaluation-only: use split='test'")
        self.split = split
        self.load_depth = load_depth
        self.training = False
        self.overfit = overfit
        self.is_test = True
        self.in_domain_eval = in_domain_eval
        self.stage = stage
        self.root_dir = os.path.expandvars(root_dir or opts.ROOT_DIR)

        self.max_views = int(max_views)
        self.n_target_views = int(n_target_views)
        assert self.n_target_views >= 1, "need at least one target view"
        assert self.max_views > self.n_target_views, \
            f"max_views ({max_views}) must exceed n_target_views ({n_target_views}) to leave >= 1 source"
        self.num_source_views = self.max_views - 1          # same meaning as in Sevenscenes_NVS

        self.image_aug = PhotometricAug()

        # ---------------- frames per scene ----------------
        self.scene_frames = {}
        for n in TEST_SCAN_NUMBERS:
            img_dir = os.path.join(self.root_dir, f"scan{n}", "images")
            if os.path.isdir(img_dir):
                self.scene_frames[f"scan{n}"] = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        missing = [n for n in TEST_SCAN_NUMBERS if f"scan{n}" not in self.scene_frames]
        if missing:
            print(f"DTU_NVS: scans {missing} not found in {self.root_dir}")
        if not self.scene_frames:
            raise RuntimeError(f"no DTU scans (scan<N>/images) in {self.root_dir}")

        # ---------------- fixed sample list ----------------
        self.samples = self.build_samples(seq_id_map, stride, n_test_samples)
        self.seq_keys = [f"{scene}_{'-'.join(map(str, ids))}" for scene, ids in self.samples]
        self.keys = self.seq_keys

        print(f"DTU_NVS [{split}] {len(self.samples):,} sequences from {len(self.scene_frames)} scenes in {self.root_dir}")
        print(f"DTU_NVS views/sample: {self.max_views} "
              f"({self.n_target_views} target + {self.max_views - self.n_target_views} source)")

    def build_samples(self, seq_id_map, stride, n_test_samples):
        """[(scene, [frame indices])], fixed for every model / run."""
        if seq_id_map is not None:
            with open(seq_id_map, "r") as f:
                id_map = json.load(f)
            samples = []
            for scene, ids in id_map.items():
                if scene not in self.scene_frames:
                    raise KeyError(f"scene '{scene}' from {seq_id_map} not in {self.root_dir}")
                if len(ids) < self.max_views:
                    raise RuntimeError(f"{scene}: {len(ids)} ids in {seq_id_map}, max_views={self.max_views}")
                samples.append((scene, [int(i) for i in ids[:self.max_views]]))
            return samples

        stride = int(stride)
        n_tag = "all" if n_test_samples is None else n_test_samples
        cache = os.path.join(self.root_dir, f"dtu_test_set__{n_tag}_samples__{self.max_views}views.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pkl.load(f)

        windows = []
        for scene, frames in self.scene_frames.items():
            for start in range(0, len(frames) - self.max_views + 1, stride):
                windows.append((scene, list(range(start, start + self.max_views))))
        if n_test_samples is None:
            samples = windows
        else:
            if len(windows) < n_test_samples:
                print(f"DTU: only {len(windows)} windows (stride {stride}), using all of them")
            picked = random.Random(0).sample(range(len(windows)), min(n_test_samples, len(windows)))
            samples = [windows[i] for i in sorted(picked)]          # sorted: grouped by scene, in order
        with open(cache, "wb") as f:
            pkl.dump(samples, f)
        print(f"Saved {len(samples)} DTU samples (of {len(windows)} windows) to {cache}")
        return samples

    # ----------------------------------------------------------------------
    # file access
    # ----------------------------------------------------------------------
    def load_view(self, scene, frame_id):
        name = self.scene_frames[scene][frame_id]
        stem = name.split(".")[0]                       # "00000012"
        scene_dir = os.path.join(self.root_dir, scene)

        depth = np.load(os.path.join(scene_dir, "depths", f"{stem}.npy"))
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
        H, W = depth.shape

        # image resized to the depth resolution (the intrinsics are given at that resolution)
        image = Image.open(os.path.join(scene_dir, "images", name)).convert("RGB")
        image = np.array(image.resize((W, H), Image.LANCZOS))

        if self.load_depth:
            mask = (cv2.imread(os.path.join(scene_dir, "binary_masks", f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                    / 255.0).astype(np.float32)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0.5).astype(np.float32)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1)
            depth = depth * mask
            depth[depth < 1e-4] = 0.0                   # Pi3 / Poincar3: valid = depth > 1e-4
        else:
            depth = np.zeros((H, W), np.float32)

        with open(os.path.join(scene_dir, "cams", f"{stem}_cam.txt"), "r") as f:
            K, w2c = load_cam_mvsnet(f.read())

        return {
            "image": image,
            "depth": depth,
            "intrinsics": K,
            "extrinsics": w2c.astype(np.float64),       # world2cam
            "image_name": stem,
        }

    # ----------------------------------------------------------------------
    # geometry helpers (same as Sevenscenes_NVS)
    # ----------------------------------------------------------------------
    def rotate_inverse(self, se3: np.ndarray) -> np.ndarray:
        assert se3.shape == (4, 4), f"Expected shape (4, 4), got {se3.shape}"
        R, T = se3[:3, :3], se3[:3, 3]
        se3_inv = np.eye(4)
        se3_inv[:3, :3] = R.T
        se3_inv[:3, 3] = -R.T @ T
        return se3_inv

    def compute_relative_pose(self, ref_pose, src_pose):
        return (ref_pose @ self.rotate_inverse(src_pose)).astype(np.float32)

    def compute_all_relative_poses(self, pose_list: List[np.ndarray], reference_camera_index: int = 0):
        """World-to-camera poses -> world-to-camera poses relative to the reference view (as MegaDepth_NVS):
        E_i' = E_i @ E_ref^-1, so the reference view is the identity."""
        ref_pose = pose_list[reference_camera_index]
        return [self.compute_relative_pose(p, ref_pose) for p in pose_list]

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, aug_crop=-1,
                                  mask=None, rng=None, info=None):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # crop centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, (l, t, r, b), mask=mask)

        W, H = image.size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            if rng.integers(2):
                resolution = resolution[::-1]

        # Lanczos down-scaling
        target_resolution = np.array(resolution)
        if aug_crop > 1:
            target_resolution += rng.integers(0, aug_crop)
        image, depthmap, intrinsics, mask = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution, mask=mask)

        # final crop
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, mask = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox, mask=mask)
        return image, depthmap, intrinsics2, mask

    # ----------------------------------------------------------------------
    # dataset API
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=1.0, rng=None):
        """
        img_per_seq: number of source views besides the first view (default max_views - 1).
                     Total views = 1 + img_per_seq; the first n_target_views are targets.
        """
        if seq_index is None:
            seq_index = random.randrange(len(self.samples))
        if img_per_seq is None:
            img_per_seq = self.num_source_views
        if rng is None:
            rng = np.random.default_rng(seed=42)
        n_target_views = self.n_target_views
        assert 1 + img_per_seq > n_target_views, \
            f"{1 + img_per_seq} views cannot hold {n_target_views} targets and >= 1 source"

        scene, ids = self.samples[seq_index]
        ids = ids[:1 + img_per_seq]
        key = self.seq_keys[seq_index]

        res = (self.common_conf["IMG_SIZE"], self.common_conf["IMG_SIZE"])
        images, depths, extrinsics, intrinsics, depth_masks, image_names = [], [], [], [], [], []

        for frame_id in ids:
            view = self.load_view(scene, frame_id)
            depth_map = view["depth"]
            mask = (depth_map > 0.0).astype(np.int32)
            extri = view["extrinsics"]                                   # world2cam
            intri = view["intrinsics"].copy()

            image, depth_map, intri, mask = self._crop_resize_if_necessary(
                image=view["image"], depthmap=depth_map, intrinsics=intri, resolution=res,
                mask=mask, rng=rng, info=None)

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri.astype(np.float32))
            intrinsics.append(intri)
            depth_masks.append(mask)
            image_names.append(view["image_name"])

        T = n_target_views

        images = self.image_aug(images, train=self.training)

        # if self.load_depth:
        #     extrinsics = self.compute_all_relative_poses(pose_list=extrinsics, reference_camera_index=T)

        # no overlap matrices for DTU: placeholders so the keys match the other datasets
        overlaps = [1.0] + [float("nan")] * (len(ids) - 1)
        occlusions = [0.0] + [float("nan")] * (len(ids) - 1)

        return {
            "seq_name": key,
            "scene_name": scene,                 # camera metrics are averaged per scene
            "frame_num": len(images),

            "t_images": images[:T],
            "t_depths": depths[:T],
            "t_depth_masks": depth_masks[:T],
            "t_extrinsics": extrinsics[:T],
            "t_intrinsics": intrinsics[:T],
            "t_image_names": image_names[:T],
            "t_overlaps": overlaps[:T],
            "t_occlusions": occlusions[:T],

            "s_images": images[T:],
            "s_depths": depths[T:],
            "s_depth_masks": depth_masks[T:],
            "s_extrinsics": extrinsics[T:],
            "s_intrinsics": intrinsics[T:],
            "s_image_names": image_names[T:],
            "s_overlaps": overlaps[T:],
            "s_occlusions": occlusions[T:],
        }


def get_args():
    parser = argparse.ArgumentParser(description="DTU dataset check")
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--seq_id_map', type=str, default=None)
    parser.add_argument('--max_views', type=int, default=6)
    parser.add_argument('--n_target_views', type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    from .sevenscenes_NVS import show_target_and_sources, combined_pointcloud_ply
    args = get_args()
    dataset = DTU_NVS(split=args.split, config_path=args.config_path, root_dir=args.root_dir,
                        seq_id_map=args.seq_id_map, load_depth=True,
                        max_views=args.max_views, n_target_views=args.n_target_views)
    for sample_idx, views in enumerate(tqdm(dataset, desc="Processing dataset")):     # dataset[i] -> tensors
        show_target_and_sources(views["t_images"], views["s_images"], sample_idx)
        combined_pointcloud_ply(images=views["s_images"], depths=views["s_depths"], masks=views["s_depth_masks"],
                                intrinsics=views["s_intrinsics"], src2ref_poses=views["s_extrinsics"],
                                use_pose_as_is=False, ply_path=f"combined_dtu_point_cloud-{sample_idx}.ply")
        breakpoint()