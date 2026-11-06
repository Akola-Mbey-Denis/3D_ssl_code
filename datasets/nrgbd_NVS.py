
import os
import argparse
import pickle as pkl
import random

from typing import List

import cv2
import numpy as np
import torch
import yaml
from fvcore.common.config import CfgNode
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import torchvision.transforms as tvf

from .dataset_util import *
from .augmentation import PhotometricAug
from .base_dataset import BaseDataset
import utils.cropping as cropping
import PIL
from PIL import Image


class NRGBD_NVS(BaseDataset):
    """
    NRGBD multi-view samples loaded from preprocessed_nrgbd_3d_<split>_set.pkl.

    Pickle layout:
        seq_key:
            target: image, depth, K, extrinsics
            sources: list of image, depth, K, extrinsics, overlap, occlusion

    Stored poses:
        OpenCV camera-to-world (c2w).

    Returned poses:
        OpenCV world-to-camera (w2c).

    Each sample contains one stored target plus max_views - 1 selected
    source candidates. The first n_target_views views are returned as
    targets, and the remaining views are returned as sources.
    """

    def __init__(
        self,
        split,
        config_path,
        overfit=False,
        img_size=518,
        in_domain_eval=False,
        stage=1,
        max_views=6,
        n_target_views=2,
        pkl_path=None,
        load_depth=True,
        *args,
        **kwargs,
    ):
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        opts = CfgNode(raw_cfg)
        self.opts = opts
        self.common_conf = opts.COMMON
        self.common_conf["IMG_SIZE"] = img_size
        self.common_conf["IMG_PER_SEQ"] = max_views - 1

        super().__init__(self.common_conf, *args, **kwargs)

        self.split = split
        self.training = split == "train"
        self.overfit = overfit
        self.load_depth = load_depth

        if self.overfit:
            self.split = "train"

        self.in_domain_eval = in_domain_eval
        self.stage = stage
        self.aug_crop = opts.AUG_CROP
        self.root_dir = opts.ROOT_DIR

        self.max_views = int(max_views)
        self.n_target_views = int(n_target_views)

        assert self.n_target_views >= 1, "need at least one target view"
        assert self.max_views > self.n_target_views, (
            f"max_views ({max_views}) must exceed "
            f"n_target_views ({n_target_views}) to leave >= 1 source"
        )

        # Number of stored candidate sources selected for each sample.
        self.num_source_views = self.max_views - 1

        self.prob = 0.30
        self.image_aug = PhotometricAug()

        # ---------------------------------------------------------------
        # Pickle path
        # ---------------------------------------------------------------

       
        if pkl_path is None:
            pkl_path = os.path.join(self.root_dir,f"preprocessed_nrgbd_3d_all_set.pkl")

        self.pkl_path = pkl_path

        with open(self.pkl_path, "rb") as f:
            all_data = pkl.load(f)

        # Keep entries whose target and enough source files exist.
        valid_data = {}

        for key, scene_item in all_data.items():
            target = scene_item.get("target")
            sources = scene_item.get("sources", [])

            if target is None or not self._view_files_exist(target):
                continue

            valid_sources = [
                view for view in sources
                if self._view_files_exist(view)
            ]

            if len(valid_sources) < self.num_source_views:
                continue

            valid_data[key] = {
                "target": target,
                "sources": valid_sources,
            }

        self.seq_keys = sorted(valid_data.keys())
        self.data = valid_data

        # Deterministic test subsampling happens inside the dataloader.
        if self.split == "test":
            n_test_samples = 200
            sample_seed = 40
            if len(self.seq_keys) > n_test_samples:
                rng = random.Random(sample_seed)
                self.seq_keys = sorted(
                    rng.sample(self.seq_keys, n_test_samples)
                )

    
        if self.split == "val":
            val_path = os.path.join(
                self.root_dir,
                "nrgbd_validation_set__200_samples.pkl",
            )

            if not os.path.exists(val_path):
                self.save_random_keys(
                    self.seq_keys,
                    save_path=val_path,
                    num_samples=200,
                )

            with open(val_path, "rb") as f:
                self.seq_keys = pkl.load(f)

        self.keys = self.seq_keys

        print(
            f"NRGBD_NVS [{split}] {len(self.seq_keys):,} sequences "
            f"from {self.pkl_path}"
        )
        print(
            f"NRGBD_NVS views/sample: {self.max_views} "
            f"({self.n_target_views} target + "
            f"{self.max_views - self.n_target_views} source)"
        )

    # ------------------------------------------------------------------
    # File and sequence access
    # ------------------------------------------------------------------
    @staticmethod
    def _view_files_exist(view):
        return (
            os.path.isfile(view.get("image", ""))
            and os.path.isfile(view.get("depth", ""))
        )

    def save_random_keys(self, keys, save_path, num_samples=1000):
        keys = list(keys)
        sampled_keys = random.Random(42).sample(
            keys,
            min(num_samples, len(keys)),
        )

        with open(save_path, "wb") as f:
            pkl.dump(sampled_keys, f)

        print(f"Saved {len(sampled_keys)} keys to {save_path}")

    def _get_sequence(self, key):
        return self.data[key]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def rotate_inverse(self, se3: np.ndarray) -> np.ndarray:
        """Invert a rigid 4x4 pose."""
        assert se3.shape == (4, 4), (
            f"Expected shape (4, 4), got {se3.shape}"
        )

        R, T = se3[:3, :3], se3[:3, 3]

        se3_inv = np.eye(4, dtype=np.float32)
        se3_inv[:3, :3] = R.T
        se3_inv[:3, 3] = -R.T @ T

        return se3_inv

    def compute_relative_pose(self, ref_pose, src_pose):
        return (
            ref_pose @ self.rotate_inverse(src_pose)
        ).astype(np.float32)

    def compute_all_relative_poses(
        self,
        pose_list: List[np.ndarray],
        reference_camera_index: int = 0,
    ):
        ref_pose = pose_list[reference_camera_index]
        return [
            self.compute_relative_pose(pose, ref_pose)
            for pose in pose_list
        ]

    # ------------------------------------------------------------------
    # Original project cropping method
    # ------------------------------------------------------------------
    def _crop_resize_if_necessary(
        self,
        image,
        depthmap,
        intrinsics,
        resolution,
        aug_crop=-1,
        mask=None,
        rng=None,
        info=None,
    ):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        if rng is None:
            rng = np.random.default_rng(seed=42)

        # Crop centered on the principal point.
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)

        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)

        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y

        image, depthmap, intrinsics, mask = (
            cropping.crop_image_depthmap(
                image,
                depthmap,
                intrinsics,
                (l, t, r, b),
                mask=mask,
            )
        )

        W, H = image.size
        assert resolution[0] >= resolution[1]

        if H > 1.1 * W:
            resolution = resolution[::-1]
        elif (
            0.9 < H / W < 1.1
            and resolution[0] != resolution[1]
        ):
            if rng.integers(2):
                resolution = resolution[::-1]

        # Lanczos down-scaling.
        target_resolution = np.array(resolution)

        if aug_crop > 1:
            target_resolution += rng.integers(0, aug_crop)

        image, depthmap, intrinsics, mask = (
            cropping.rescale_image_depthmap(
                image,
                depthmap,
                intrinsics,
                target_resolution,
                mask=mask,
            )
        )

        # Final crop, preserving the original camera intrinsics.
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics,
            image.size,
            resolution,
            offset_factor=0.5,
        )

        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics,
            intrinsics2,
            resolution,
        )

        image, depthmap, intrinsics2, mask = (
            cropping.crop_image_depthmap(
                image,
                depthmap,
                intrinsics,
                crop_bbox,
                mask=mask,
            )
        )

        return image, depthmap, intrinsics2, mask

    # ------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------
    def select_sources(self, sources, n):
        """
        Training: random source subset.
        Evaluation: evenly spaced sources across the sequence.
        """
        if len(sources) < n:
            raise RuntimeError(
                f"Sequence has {len(sources)} sources but {n} are needed "
                f"(max_views={self.max_views}); rebuild the sequence pickle "
                "with more sources or lower max_views."
            )

        sources = list(sources)

        if self.training:
            return random.sample(sources, n)

        indices = np.linspace(0, len(sources) - 1, n, dtype=int)
        return [sources[i] for i in indices]

    def process_scene_sequence(self, scene_item, scene_id, max_length):
        """Return the stored target followed by selected source views."""
        target = scene_item["target"]
        views = [target] + self.select_sources(
            scene_item["sources"],
            max_length,
        )

        sequence = []

        for i, view in enumerate(views):
            if self.load_depth:
                depth_raw = np.asarray(
                    Image.open(view["depth"]),
                    dtype=np.float32,
                )

                # NRGBD depth PNG is uint16 millimeters.
                depth = depth_raw / 1000.0

                depth_mask = (
                    np.isfinite(depth)
                    & (depth > 0)
                )
                depth[~depth_mask] = 0.0
            else:
                depth = None
                depth_mask = None

            sequence.append(
                {
                    "image": cv2.cvtColor(
                        cv2.imread(view["image"], cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB,
                    ),
                    "depth": depth,
                    "mask": depth_mask,
                    "intrinsics": view["K"],
                    "extrinsics": view["extrinsics"], # cam2world
                    "image_name": os.path.splitext(
                        os.path.basename(view["image"])
                    )[0],
                    "scene_id": scene_id,
                    "overlap": (
                        1.0 if i == 0
                        else view.get("overlap", 0.0)
                    ),
                    "occlusion": (
                        0.0 if i == 0
                        else view.get("occlusion", 0.0)
                    ),
                }
            )

        return sequence

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.seq_keys)

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        aspect_ratio=1.0,
        rng=None,
    ):
        """
        img_per_seq is the number of stored source views to load.
        Total views = 1 + img_per_seq.

        The first n_target_views become targets.
        """
        if seq_index is None:
            seq_index = random.randrange(len(self.seq_keys))

        if img_per_seq is None:
            img_per_seq = self.num_source_views

        if rng is None:
            rng = np.random.default_rng(seed=42)

        T = self.n_target_views

        assert 1 + img_per_seq > T, (
            f"{1 + img_per_seq} views cannot hold {T} targets "
            "and at least one source"
        )

        key = self.seq_keys[seq_index]
        scene_id = key.rsplit("__", 1)[0] if "__" in key else key

        sequence_views = self.process_scene_sequence(
            self._get_sequence(key),
            scene_id,
            max_length=img_per_seq,
        )

        resolution = (
            self.common_conf["IMG_SIZE"],
            self.common_conf["IMG_SIZE"],
        )

        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        depth_masks = []
        image_names = []
        overlaps = []
        occlusions = []

        for view in sequence_views:
            if self.load_depth:
                depth_map = threshold_depth_map(
                    view["depth"],
                    min_percentile=-1,
                    max_percentile=98,
                )
                mask = (depth_map > 0.00).astype(np.int32)
            else:
                H, W = view["image"].shape[:2]
                depth_map = np.zeros((H, W), np.float32)
                mask = np.zeros((H, W), np.int32)

            # Stored pose is OpenCV c2w. Convert to w2c once here.
            c2w = np.asarray(
                view["extrinsics"],
                dtype=np.float32,
            )
            extri = self.rotate_inverse(c2w) #world2cam

            intri = np.asarray(
                view["intrinsics"],
                dtype=np.float32,
            ).copy()

            image, depth_map, intri, mask = (
                self._crop_resize_if_necessary(
                    image=view["image"],
                    depthmap=depth_map,
                    intrinsics=intri,
                    resolution=resolution,
                    aug_crop=-1,
                    mask=mask,
                    rng=rng,
                    info=None,
                )
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri.astype(np.float32))
            intrinsics.append(intri.astype(np.float32))
            depth_masks.append(mask)
            image_names.append(view["image_name"])
            # overlaps.append(view["overlap"])
            # occlusions.append(view["occlusion"])

        images = self.image_aug(
            images,
            train=self.training,
        )

        return {
            "seq_name": key,
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def show_target_and_sources(
    t_images,
    s_images,
    index=0,
    title="Target and Source Images",
    max_sources=8,
):
    """Display target views and up to max_sources source views."""

    def to_numpy(img):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        else:
            img = np.array(img)

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        return np.clip(img * 255, 0, 255).astype(np.uint8)

    targets = [to_numpy(img) for img in t_images]
    sources = [
        to_numpy(s_images[i])
        for i in range(min(len(s_images), max_sources))
    ]

    total = len(targets) + len(sources)
    n_cols = 3
    n_rows = int(np.ceil(total / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
    )
    axes = np.array(axes).reshape(-1)

    for i, img in enumerate(targets):
        axes[i].imshow(img)
        axes[i].set_title(f"Target View {i + 1}")
        axes[i].axis("off")

    for i, img in enumerate(sources):
        ax = axes[len(targets) + i]
        ax.imshow(img)
        ax.set_title(f"Source View {i + 1}")
        ax.axis("off")

    for j in range(total, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(
        f"nrgbd_nvs_samples_{index}.png",
        bbox_inches="tight",
    )
    plt.close()


def combined_pointcloud_ply(images, depths, masks, intrinsics, src2ref_poses, use_pose_as_is, ply_path):
    S, C, H, W = images.shape
    device = images.device
    images = (images * 255).permute((0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(1, H * W, 3).float()
    xyz_world_list, rgb_list = [], []
    for i in range(S):
        depth_flat = depths[i].reshape(-1)
        valid = masks[i].bool().reshape(-1) & (depth_flat > 0)
        if valid.sum() == 0:
            continue
        cam_xyz = (torch.inverse(intrinsics[i]) @ pix[0][valid].T).T * depth_flat[valid][:, None]
        cam_xyz_h = torch.cat([cam_xyz, torch.ones(len(cam_xyz), 1, device=device)], dim=1)
        if use_pose_as_is:
            xyz_world_list.append((src2ref_poses[i] @ cam_xyz_h.T).T[:, :3])
        else:
            xyz_world_list.append((torch.inverse(src2ref_poses[i]) @ cam_xyz_h.T).T[:, :3])

        rgb_list.append(images[i].reshape(-1, 3)[valid])
    if not xyz_world_list:
        raise ValueError("No valid points found in input frames.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(torch.cat(xyz_world_list).cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(torch.cat(rgb_list).cpu().numpy() / 255.0)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud to: {ply_path}")

def get_args():
    parser = argparse.ArgumentParser(
        description="NRGBD dataset configuration"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--pkl_path", type=str, default=None)
    parser.add_argument("--max_views", type=int, default=6)
    parser.add_argument("--n_target_views", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--load_depth", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dataset = NRGBD_NVS(
        split=args.split,
        overfit=False,
        load_depth=args.load_depth,
        config_path=args.config_path,
        pkl_path=args.pkl_path,
        img_size=args.img_size,
        max_views=args.max_views,
        n_target_views=args.n_target_views,
    )

    for sample_idx, views in enumerate(
        tqdm(dataset, desc="Processing NRGBD")
    ):
        show_target_and_sources(
            views["t_images"],
            views["s_images"],
            sample_idx,
        )
        # breakpoint()
        combined_pointcloud_ply(images = views["s_images"], depths = views["s_depths"],use_pose_as_is = False, masks = views["s_depth_masks"], intrinsics = views["s_intrinsics"], src2ref_poses =views["s_extrinsics"], ply_path = f"combined_point_cloud_{sample_idx}_nrgbd.ply")
        breakpoint()

