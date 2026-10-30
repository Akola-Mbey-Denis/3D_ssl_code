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


class Mapfree_NVS(BaseDataset):
    """
    Mapfree_NVS multi-view samples read from an LMDB

    Map-free has images + cameras only: no depth, no masks, no occlusion.
    Targets are seq0 frames, sources are seq1 frames of the same scene (preprocess_mapfree_sequences.py).
    Every source carries, w.r.t. the stored target:
        overlap    SfM co-visibility score in [0, 1] provided by Map-free (overlaps.npz)
        seq, frame sequence id (1 for sources) and frame index of the view

    One LMDB per split, set in the config:
        LMDB_PATHS:
          train: /scratch/.../mapfree_train.lmdb
          val:   /scratch/.../mapfree_val.lmdb
          test:  /scratch/.../mapfree_test.lmdb
    (lmdb_path=... overrides it.)

    Each sample = 1 stored target + (max_views - 1) source views. The first `n_target_views` of
    those views are returned as targets ("t_*"), the rest as sources ("s_*"):
        n_target_views=1: [target]            | [src1, src2, ...]
        n_target_views=2: [target, src1]      | [src2, src3, ...]
        n_target_views=3: [target, src1, src2]| [src3, ...]
    Source order: training -> random subset of the stored candidates; eval -> highest overlap first.

    Depth keys ("t_depths", "s_depths", "*_depth_masks") are returned as zeros so batches keep the
    same structure as the depth datasets; "has_depth" is False -> skip depth losses / metrics.
    """

    def __init__(self, split, config_path, overfit=False, img_size=224, in_domain_eval=False,
                 stage=1, max_views=2, n_target_views=1, lmdb_path=None, *args, **kwargs):
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

        if self.overfit:
            self.split = "train"

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

        self.image_aug = PhotometricAug()

   


        # ---------------- LMDB ----------------
        # if lmdb_path is None:
        #     # 1) LMDB_PATHS.<split> from the yaml (read from the raw dict)
        #     # 2) otherwise <ROOT_DIR>/mapfree_<split>.lmdb
        #     paths = raw_cfg.get("LMDB_PATHS") or {}
        #     lmdb_path = paths.get(self.split) or os.path.join(str(raw_cfg.get("ROOT_DIR", "")), f"mapfree_{self.split}.lmdb")
        if lmdb_path is None:
            # 1) LMDB_PATHS.<split> from the yaml (read from the raw dict)
            # 2) otherwise <ROOT_DIR>/dl3dv_<split>.lmdb
            paths = raw_cfg.get("LMDB_PATHS") or {}
            lmdb_path = paths.get(self.split) or os.path.join(str(raw_cfg.get("ROOT_DIR", "")), f"mapfree_{self.split}.lmdb")
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
            overfit_path = f"{self.root_dir}/mapfree_overfit__100_samples.pkl"
            if not os.path.exists(overfit_path):
                self.save_random_keys(self.seq_keys, save_path=overfit_path, num_samples=100)
            with open(overfit_path, 'rb') as f:
                self.seq_keys = pkl.load(f)
        
        if self.split == "val":
            val_path = f"{self.root_dir}/mapfree_validation_set__2000_samples.pkl"
            if not os.path.exists(val_path):
                self.save_random_keys(self.seq_keys, save_path=val_path, num_samples=2000)
            with open(val_path, 'rb') as f:
                self.seq_keys = pkl.load(f)
        
        # if self.split == "train":
        #     train_path = f"{self.root_dir}/mapfree_train_set__50_000_samples.pkl"
        #     if not os.path.exists(train_path):
        #         self.save_random_keys(self.seq_keys, save_path=train_path, num_samples=50000)
        #     with open(train_path, 'rb') as f:
        #         self.seq_keys = pkl.load(f)

        self.keys = self.seq_keys                            # name used by the old code
        print(f"Mapfree_NVS [{self.split}] {len(self.seq_keys):,} sequences from {self.lmdb_path}")
        print(f"Mapfree_NVS views/sample: {self.max_views} "
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
        return _decode_rgb(self._lmdb_get("rgb:" + view["image"]))

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
        """Training: random subset of the stored candidates. Eval: highest overlap with the target first (deterministic)."""
        if len(sources) < n:
            raise RuntimeError(f"sequence has {len(sources)} sources but {n} are needed "
                               f"(max_views={self.max_views}); rebuild the LMDB with more sources "
                               f"or lower max_views")
        if self.training:
            return random.sample(list(sources), n)
        return sorted(sources, key=lambda s: (-s["overlap"], s.get("frame", 0)))[:n]

    def process_scene_sequence(self, scene_item, scene_id, max_length):
        """Decoded views: stored target first, then `max_length` selected sources."""
        target = scene_item["target"]
        views = [target] + self.select_sources(scene_item["sources"], max_length)

        sequence = []
        for i, v in enumerate(views):
            sequence.append({
                "image": self._load_rgb(v),
                "intrinsics": v["K"],
                "extrinsics": v["extrinsics"],                 # cam2world (OpenCV)
                "image_name": v["image"].split("/")[-1].split(".")[0],
                "scene_id": scene_id,
                # w.r.t. the stored target (target itself: overlap 1)
                "overlap": 1.0 if i == 0 else float(v["overlap"]),
                "seq": int(v["image"].split("/seq")[-1].split("/")[0]),          # .../seq0/frame_00012.jpg
                "frame": int(v["image"].split("frame_")[-1].split(".")[0]),
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
        scene_id = key.rsplit("__", 1)[0]                    # keys are "<scene>__seq0_<frame>"

        sequence_views = self.process_scene_sequence(self._get_sequence(key), scene_id, max_length=img_per_seq)

        res = (self.common_conf["IMG_SIZE"], self.common_conf["IMG_SIZE"])
        images, extrinsics, intrinsics, depth_masks, depths, occlusions = [], [], [], [], [], []
        image_names, overlaps, seqs, frames = [], [], [], []

        for view in sequence_views:
            extri = self.rotate_inverse(np.array(view["extrinsics"]))                  # world2cam
            intri = np.array(view["intrinsics"])

            # no depth in Mapfree_NVS: crop / resize a zero placeholder alongside the image
            H, W = view["image"].shape[:2]
            depth_map = np.zeros((H, W), dtype=np.float32)
            mask = np.zeros((H, W), dtype=np.int32)

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
            occlusions.append(-1.00)
            seqs.append(view["seq"])
            frames.append(view["frame"])
        
        images = self.image_aug(images, train=self.training)

        T = n_target_views
        return {
            "seq_name": key,
            "frame_num": len(images),
            # "has_depth": False,

            # targets: stored target + the first (T - 1) selected sources
            "t_images": images[:T],
            "t_depths": depths[:T],              # zeros (no depth)
            "t_depth_masks": depth_masks[:T],    # zeros (no valid depth)
            "t_extrinsics": extrinsics[:T],
            "t_intrinsics": intrinsics[:T],
            "t_image_names": image_names[:T],
            "t_overlaps": overlaps[:T],          # w.r.t. the stored target (1 for itself)
            "t_occlusions": occlusions[:T],
            # "t_seqs": seqs[:T],
            # "t_frames": frames[:T],

            # sources: the remaining views (overlaps aligned with s_images)
            "s_images": images[T:],
            "s_depths": depths[T:],
            "s_depth_masks": depth_masks[T:],
            "s_extrinsics": extrinsics[T:],
            "s_intrinsics": intrinsics[T:],
            "s_image_names": image_names[T:],
            "s_overlaps": overlaps[T:],
            "s_occlusions": occlusions[T:],

            # "s_seqs": seqs[T:],
            # "s_frames": frames[T:],
        }


# ---------------------------------------------------------------------------
# visualization helpers
# ---------------------------------------------------------------------------

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
    plt.savefig(f"mapfree_nvs_samples_{index}.png", bbox_inches="tight")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Mapfree_NVS dataset config")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--lmdb_path', type=str, default=None)
    parser.add_argument('--max_views', type=int, default=6)
    parser.add_argument('--n_target_views', type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dataset = Mapfree_NVS(split=args.split, overfit=False, img_size = 224, config_path=args.config_path, lmdb_path=args.lmdb_path,
                        max_views=args.max_views, n_target_views=args.n_target_views)
    for sample_idx, views in enumerate(tqdm(dataset, desc="Processing dataset")):     # dataset[i] -> tensors
        show_target_and_sources(views["t_images"], views["s_images"], sample_idx)
        breakpoint()
       