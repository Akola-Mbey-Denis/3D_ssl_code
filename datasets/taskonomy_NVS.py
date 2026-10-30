import os
import argparse
import pickle as pkl
import random

from typing import List

import cv2
import lmdb
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


# ---------------------------------------------------------------------------
# LMDB handles
#
# One read-only handle per (process, file). py-lmdb refuses to open the same file twice in one
# process, and a handle must not be used across fork(): a DataLoader worker drops the copy it
# inherited from the parent and opens its own.
# ---------------------------------------------------------------------------
_LMDB_ENVS = {}

def _open_lmdb(path):
    pid = os.getpid()
    key = (pid, path)
    if key not in _LMDB_ENVS:
        for k in [k for k in _LMDB_ENVS if k[1] == path and k[0] != pid]:
            _LMDB_ENVS.pop(k).close()
        _LMDB_ENVS[key] = lmdb.open(path, subdir=False, readonly=True, lock=False,
                                    readahead=False, meminit=False, max_readers=512)
    return _LMDB_ENVS[key]


def _decode_rgb(buf):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("failed to decode RGB bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _decode_z_buffer_depth(buf):
    """
    Taskonomy z-buffer depth from PNG bytes (uint16, 1/512 m; 65535 = invalid), as
    load_z_buffer_depth_and_depth_mask did from the file.

    Returns:
        depth_m (H, W) float32 meters (0 where invalid), mask (H, W) bool (True = valid)
    """
    depth_int = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth_int is None:
        raise RuntimeError("failed to decode depth bytes")
    depth_int = depth_int.astype(np.uint16)
    mask = depth_int < (2 ** 16 - 1)
    depth_m = depth_int.astype(np.float32) / 512.0
    depth_m[~mask] = 0.0
    return depth_m, mask


class Taskonomy_NVS(BaseDataset):
    """
    Taskonomy multi-view samples read from an LMDB built from preprocessed_taskonomy_3d_<split>_set.pkl
    (same layout as the Habitat LMDB: seq:<key>, rgb:<path>, depth:<path>, __seq_keys__, __info__).

    One LMDB per split, set in the config:
        LMDB_PATHS:
          train: /scratch/.../taskonomy_train.lmdb
          val:   /scratch/.../taskonomy_val.lmdb
          test:  /scratch/.../taskonomy_test.lmdb
    (lmdb_path=... overrides it; "vis" uses the test LMDB unless LMDB_PATHS.vis is set.)

    Each sample = 1 stored target + (max_views - 1) source views. The first `n_target_views` of
    those views are returned as targets ("t_*"), the rest as sources ("s_*"):
        n_target_views=1: [target]            | [src1, src2, ...]
        n_target_views=2: [target, src1]      | [src2, src3, ...]
    Source order: training -> random subset of the stored candidates; eval -> top by overlap.
    """

    def __init__(self, split, config_path, overfit=False, img_size=224, in_domain_eval=False,
                 stage=1, max_views=2, n_target_views=1, lmdb_path=None, load_depth = False, *args, **kwargs):
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
            self.split = "train"                 # overfit: val/test read the training data

        self.in_domain_eval = in_domain_eval
        self.stage = stage
        self.aug_crop = opts.AUG_CROP
        self.root_dir = opts.ROOT_DIR

        self.max_views = int(max_views)
        self.n_target_views = int(n_target_views)
        assert self.n_target_views >= 1, "need at least one target view"
        assert self.max_views > self.n_target_views, \
            f"max_views ({max_views}) must exceed n_target_views ({n_target_views}) to leave >= 1 source"
        self.num_source_views = self.max_views - 1          # stored sources loaded per sample

        # self.prob = 0.30
        # self.train_image_aug = get_image_augmentation()
        # self.image_transform = tvf.Compose([tvf.ToTensor()])
        self.image_aug = PhotometricAug()


        # ---------------- LMDB ----------------
        # if lmdb_path is None:
        #     # 1) LMDB_PATHS.<split> from the yaml (read from the raw dict); "vis" falls back to test
        #     # 2) otherwise <ROOT_DIR>/taskonomy_<split>.lmdb
        #     paths = raw_cfg.get("LMDB_PATHS") or {}
        #     lmdb_split = self.split
        #     if lmdb_split == "vis" and "vis" not in paths:
        #         lmdb_split = "test"
        #     lmdb_path = paths.get(lmdb_split) or os.path.join(
        #         str(raw_cfg.get("ROOT_DIR", "")), f"taskonomy_{lmdb_split}.lmdb")

        if lmdb_path is None:
            # 1) LMDB_PATHS.<split> from the yaml (read from the raw dict)
            # 2) otherwise <ROOT_DIR>/dl3dv_<split>.lmdb
            paths = raw_cfg.get("LMDB_PATHS") or {}
            lmdb_path = paths.get(self.split) or os.path.join(str(raw_cfg.get("ROOT_DIR", "")), f"taskonomy_{self.split}.lmdb")
            # 3) the job copied the LMDBs to $LMDB_DIR: same file name, local folder
            if os.environ.get("LMDB_DIR"):
                lmdb_path = os.path.join(os.environ["LMDB_DIR"], os.path.basename(lmdb_path))
        
        if not os.path.isfile(lmdb_path):
            raise FileNotFoundError(
                f"LMDB for split '{self.split}' not found: {lmdb_path}\n"
                f"  config: {os.path.abspath(config_path)}\n"
                f"  LMDB_PATHS in config: {raw_cfg.get('LMDB_PATHS')}")
        self.lmdb_path = lmdb_path

        with self._lmdb().begin() as txn:
            keys = txn.get(b"__seq_keys__")
            info = txn.get(b"__info__")
        if keys is None:
            raise RuntimeError(f"{self.lmdb_path} has no __seq_keys__")
        self.seq_keys = pkl.loads(keys)
        self.lmdb_info = pkl.loads(info) if info is not None else {}
        if len(self.seq_keys) == 0:
            raise RuntimeError(f"{self.lmdb_path} contains no sequences")


        if self.overfit:
            overfit_path = f"{self.root_dir}/taskonomy_overfit__100_samples.pkl"
            if not os.path.exists(overfit_path):
                self.save_random_keys(self.seq_keys, save_path=overfit_path, num_samples=100)
                
            with open(f"{self.root_dir}/taskonomy_overfit__100_samples.pkl", 'rb') as f:
                self.seq_keys = pkl.load(f)
        
        if self.split == "val":
            val_path = f"{self.root_dir}/taskonomy_validation_set__2000_samples.pkl"
            if not os.path.exists(val_path):
                self.save_random_keys(self.seq_keys, save_path=val_path, num_samples=2000)
            with open(val_path, 'rb') as f:
                self.seq_keys = pkl.load(f)

        self.keys = self.seq_keys                            # name used by the old code
        print(f"Taskonomy_NVS [{split}] {len(self.seq_keys):,} sequences from {self.lmdb_path}")
        print(f"Taskonomy_NVS views/sample: {self.max_views} "
              f"({self.n_target_views} target + {self.max_views - self.n_target_views} source)")

    # ----------------------------------------------------------------------
    # LMDB access
    # ----------------------------------------------------------------------
    def _lmdb(self):
        return _open_lmdb(self.lmdb_path)

    def _lmdb_get(self, key):
        with self._lmdb().begin() as txn:
            buf = txn.get(key.encode("utf-8"))
        if buf is None:
            raise KeyError(f"'{key}' not in {self.lmdb_path}")
        return buf

    def save_random_keys(self, keys, save_path, num_samples=1000):
        """
        Randomly selects and saves a subset of LMDB sequence keys to a pickle file.

        Args:
            keys (list): Sequence keys to sample from.
            save_path (str): Path to save the .pkl file.
            num_samples (int): Number of keys to save.
        """
        keys = list(keys)
        sampled_keys = random.sample(keys, min(num_samples, len(keys)))

        # Save the sampled keys
        with open(save_path, 'wb') as f:
            pkl.dump(sampled_keys, f)

        print(f"Saved {len(sampled_keys)} keys to {save_path}")

    def _get_sequence(self, key):
        return pkl.loads(self._lmdb_get(f"seq:{key}"))

    def _load_rgb(self, view):
        # bytes inside the sequence record, or a separate "rgb:<path>" key
        buf = view.get("rgb_bytes")
        return _decode_rgb(buf if buf is not None else self._lmdb_get("rgb:" + view["image"]))

    def _load_depth_and_mask(self, view):
        buf = view.get("depth_bytes")
        return _decode_z_buffer_depth(buf if buf is not None else self._lmdb_get("depth:" + view["depth"]))

    # ----------------------------------------------------------------------
    # geometry helpers
    # ----------------------------------------------------------------------
    def rotate_inverse(self, se3: np.ndarray) -> np.ndarray:
        assert se3.shape == (4, 4), f"Expected shape (4, 4), got {se3.shape}"
        R, T = se3[:3, :3], se3[:3, 3]
        se3_inv = np.eye(4)
        se3_inv[:3, :3] = R.T
        se3_inv[:3, 3] = -R.T @ T
        return se3_inv

    # def compute_relative_pose(self, ref_pose, src_pose):
    #     return (ref_pose @ self.rotate_inverse(src_pose)).astype(np.float32)

    # def compute_all_relative_poses(self, pose_list: List[np.ndarray], reference_camera_index: int = 1):
    #     ref_pose = pose_list[reference_camera_index]
    #     return [self.compute_relative_pose(ref_pose, p) for p in pose_list]

    def compute_relative_pose(self, ref_pose, src_pose):
        return (ref_pose @ self.rotate_inverse(src_pose)).astype(np.float32)

    def compute_all_relative_poses(self, pose_list: List[np.ndarray], reference_camera_index: int = 0):
        """World-to-camera poses -> world-to-camera poses relative to the reference view (VGGT convention).

        E_i' = E_i @ E_ref^-1 maps the reference camera's coordinates into camera i, so the reference
        view becomes the identity and the "world" is the reference camera's frame.
        (The previous version returned E_ref @ E_i^-1: camera-to-reference, i.e. camera-to-world.)
        """
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
    # sequence processing
    # ----------------------------------------------------------------------
    def select_sources(self, sources, n):
        """Training: random subset of the stored candidates. Eval: top-n by overlap (deterministic)."""
        if len(sources) < n:
            raise RuntimeError(f"sequence has {len(sources)} sources but {n} are needed "
                               f"(max_views={self.max_views}); rebuild the LMDB with more sources "
                               f"or lower max_views")
        if self.training:
            return random.sample(list(sources), n)
        return sorted(sources, key=lambda s: s["overlap"], reverse=True)[:n]

    def process_scene_sequence(self, scene_item, scene_id, max_length):
        """Decoded views: stored target first, then `max_length` selected sources."""
        target = scene_item["target"]
        views = [target] + self.select_sources(scene_item["sources"], max_length)

        sequence = []
        for i, v in enumerate(views):
            if self.load_depth:
                depth, depth_mask = self._load_depth_and_mask(v)
            else:
                depth = None
                depth_mask = None
            sequence.append({
                "image": self._load_rgb(v),
                "depth": depth,
                "mask": depth_mask,                            # valid z-buffer pixels
                "intrinsics": v["K"],
                "extrinsics": v["extrinsics"],                 # cam2world
                "image_name": v["image"].split("/")[-1].split(".")[0],
                "scene_id": scene_id,
                # the stored target has no overlap with itself in the pkl: use 1.0 / 0.0
                "overlap": 1.0 if i == 0 else v["overlap"],
                "occlusion": 0.0 if i == 0 else v["occlusion"],
            })
        return sequence

    # ----------------------------------------------------------------------
    # dataset API
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.seq_keys)

    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=1.0, rng=None):
        """
        img_per_seq: number of stored source views to load (default max_views - 1).
                     Total views = 1 + img_per_seq; the first n_target_views are targets.
        """
        if seq_index is None:
            seq_index = random.randrange(len(self.seq_keys))
        if img_per_seq is None:
            img_per_seq = self.num_source_views
        if rng is None:
            rng = np.random.default_rng(seed=42)
        n_target_views = self.n_target_views
        assert 1 + img_per_seq > n_target_views, \
            f"{1 + img_per_seq} views cannot hold {n_target_views} targets and >= 1 source"

        key = self.seq_keys[seq_index]
        scene_id = key.rsplit("_", 1)[0] if "_" in key else key

        sequence_views = self.process_scene_sequence(self._get_sequence(key), scene_id, max_length=img_per_seq)

        res = (self.common_conf["IMG_SIZE"], self.common_conf["IMG_SIZE"])
        images, depths, extrinsics, intrinsics, depth_masks = [], [], [], [], []
        image_names, overlaps, occlusions = [], [], []

        for view in sequence_views:
            
            if self.load_depth:
                depth_map = threshold_depth_map(view["depth"], min_percentile=-1, max_percentile=98)
                mask = view["mask"].astype(np.int32) 
            else:
                H, W = view["image"].shape[:2]
                depth_map = np.zeros((H, W), np.float32)
                mask = np.zeros((H, W), np.int32)
                
            # depth_map = threshold_depth_map(view["depth"], min_percentile=-1, max_percentile=98)
            extri = self.rotate_inverse(np.array(view["extrinsics"]))                  # world2cam
            intri = np.array(view["intrinsics"])
            # mask = view["mask"].astype(np.int32)                                       # z-buffer valid mask

            image, depth_map, intri, mask = self._crop_resize_if_necessary(
                image=view["image"], depthmap=depth_map, intrinsics=intri, resolution=res,
                mask=mask, rng=rng, info=None)
            # image = np.array(image)

            # if self.training and np.random.rand() < self.prob:
            #     image = self.train_image_aug(image)
            # else:
            #     image = self.image_transform(image)

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri.astype(np.float32))
            intrinsics.append(intri)
            depth_masks.append(mask)
            image_names.append(view["image_name"])
            overlaps.append(view["overlap"])
            occlusions.append(view["occlusion"])
        
        images = self.image_aug(images, train=self.training)

        T = n_target_views

        # if self.load_depth:
        # extrinsics = self.compute_all_relative_poses(pose_list = extrinsics, reference_camera_index = T)

        return {
            "seq_name": key,
            "frame_num": len(images),

            # targets: stored target + the first (T - 1) selected sources
            "t_images": images[:T],
            "t_depths": depths[:T],
            "t_depth_masks": depth_masks[:T],
            "t_extrinsics": extrinsics[:T],
            "t_intrinsics": intrinsics[:T],
            "t_image_names": image_names[:T],
            "t_overlaps": overlaps[:T],          # w.r.t. the stored target (1.0 for itself)
            "t_occlusions": occlusions[:T],

            # sources: the remaining views (overlap/occlusion aligned with s_images)
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
# visualization helpers
# ---------------------------------------------------------------------------

def rgbd_to_pointcloud(rgb, depth, K, RT_c2w, mask=None, save_path="taskonomy_point_cloud.ply"):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([u, v, np.ones((H, W))], axis=-1).reshape(-1, 3)
    z = depth.flatten()
    valid = z > 0
    if mask is not None:
        valid = valid & mask.flatten().astype(bool)
    pixels, z = pixels[valid], z[valid]
    pts_cam = (np.linalg.inv(K) @ pixels.T).T * z[:, None]
    pts_cam_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_world = (RT_c2w @ pts_cam_h.T).T
    colors = rgb.reshape(-1, 3)[valid]
    if colors.max() > 1:
        colors = colors / 255.0
    if save_path:
        print(f"Saving colored point cloud - file {save_path}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_path, pcd)


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


def show_target_and_sources(t_images, s_images, index=0, title='Target and Source Images', max_sources=8):
    """t_images: (T,3,H,W), s_images: (S,3,H,W) in [0,1]. Shows all targets and up to max_sources sources."""
    def to_numpy(img_tensor):
        img = img_tensor.detach().cpu().numpy() if torch.is_tensor(img_tensor) else np.array(img_tensor)
        return np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

    targets = [to_numpy(t) for t in t_images]
    sources = [to_numpy(s_images[i]) for i in range(min(len(s_images), max_sources))]
    total = len(targets) + len(sources)
    n_cols = 3
    n_rows = int(np.ceil(total / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    for i, img in enumerate(targets):
        axes[i].imshow(img); axes[i].set_title(f"Target View {i + 1}", fontsize=12); axes[i].axis("off")
    for i, img in enumerate(sources):
        ax = axes[len(targets) + i]
        ax.imshow(img); ax.set_title(f"Source View {i + 1}", fontsize=12); ax.axis("off")
    for j in range(total, len(axes)):
        axes[j].axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"taskonomy_nvs_samples_{index}.png", bbox_inches="tight")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Taskonomy dataset config")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--lmdb_path', type=str, default=None)
    parser.add_argument('--max_views', type=int, default=6)
    parser.add_argument('--n_target_views', type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dataset = Taskonomy_NVS(split=args.split, overfit=False, config_path=args.config_path,  load_depth = True, lmdb_path=args.lmdb_path,
                            max_views=args.max_views, n_target_views=args.n_target_views)
    for sample_idx, views in enumerate(tqdm(dataset, desc="Processing dataset")):     # dataset[i] -> tensors
        show_target_and_sources(views["t_images"], views["s_images"], sample_idx)
        combined_pointcloud_ply(images=views["s_images"], depths=views["s_depths"], masks=views["s_depth_masks"], use_pose_as_is = False,
                                intrinsics=views["s_intrinsics"], src2ref_poses=views["s_extrinsics"],
                                ply_path=f"combined_point_cloud_{sample_idx}_taskonomy.ply")
        breakpoint()





