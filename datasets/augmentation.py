import numpy as np
import torch
from typing import Optional, Dict
from torchvision import transforms
from PIL import Image



def get_image_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])



def get_image_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])




import random
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
from PIL import Image


class PhotometricAug:
    @dataclass(frozen=True)
    class Cfg:
        # VGGT's first dataset block (VGGT's `training/config/default_dataset.yaml`) uses:
        # ColorJitter(0.5, 0.5, 0.5, 0.1) p=0.9, RandomGrayscale p=0.05, GaussianBlur OFF, and
        # cojitter=True / ratio=0.3 (one shared jitter per sequence ~70% of the time, per-frame 30%).
        enabled: bool = True
        no_aug_prob: float = 0.30  # sequences returned as plain tensors (no augmentation at all)
        brightness: float = 0.5  # VGGT: 0.5
        contrast: float = 0.5  # VGGT: 0.5
        saturation: float = 0.5  # VGGT: 0.5
        hue: float = 0.1  # VGGT: 0.1
        jitter_prob: float = 0.8  # VGGT: 0.9
        grayscale_prob: float = 0.1  # VGGT: 0.05
        blur_prob: float = 0.1  # VGGT: 0.0 (off)
        # Probability of applying ONE shared jitter to all frames in a sequence
        # (vs. independent per-frame jitter). VGGT: ~0.7 (cojitter).
        cojitter_prob: float = 0.0

    def __init__(self, cfg: Cfg | None = None) -> None:
        self.cfg = cfg or PhotometricAug.Cfg()
        ops: list = []
        if self.cfg.jitter_prob > 0:
            ops.append(
                TF.RandomApply(
                    [
                        TF.ColorJitter(
                            brightness=self.cfg.brightness,
                            contrast=self.cfg.contrast,
                            saturation=self.cfg.saturation,
                            hue=self.cfg.hue,
                        )
                    ],
                    p=self.cfg.jitter_prob,
                )
            )
        if self.cfg.grayscale_prob > 0:
            ops.append(TF.RandomGrayscale(p=self.cfg.grayscale_prob))
        if self.cfg.blur_prob > 0:
            ops.append(
                TF.RandomApply([TF.GaussianBlur(5, sigma=(0.1, 2.0))], p=self.cfg.blur_prob)
            )
        self.transform = TF.Compose(ops) if ops else None

    @staticmethod
    def to_tensor(frame) -> torch.Tensor:
        """PIL image / uint8 or float numpy (H, W, 3) / tensor -> (3, H, W) float tensor in [0, 1]."""
        if isinstance(frame, torch.Tensor):
            return frame.float() / 255.0 if frame.dtype == torch.uint8 else frame.float()
        if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
            return torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).float()
        return TFF.to_tensor(frame)                       # PIL or uint8 numpy: (3, H, W) in [0, 1]

    def __call__(self, frames: list, train: bool = True) -> list[torch.Tensor]:
        """
        frames: list of images (PIL, numpy (H, W, 3) or (3, H, W) tensors).
        Returns a list of (3, H, W) float tensors in [0, 1].
            inference (train=False)                -> tensors only
            training, with prob no_aug_prob (0.30) -> tensors only
            training, otherwise                    -> photometric augmentation
        """
        frames = [self.to_tensor(f) for f in frames]
        if not train or not self.cfg.enabled or self.transform is None:
            return frames
        if random.random() < self.cfg.no_aug_prob:
            return frames
        # Shared jitter for the whole sequence: one transform call on the stack
        # applies identical random params to every frame.
        if self.cfg.cojitter_prob > 0 and random.random() < self.cfg.cojitter_prob:
            return list(self.transform(torch.stack(frames)).unbind(0))
        # Independent per-frame jitter (re-samples params each call).
        return [self.transform(f) for f in frames]
        
# def get_image_augmentation(
#     color_jitter: Optional[Dict[str, float]] = None,
#     gray_scale: bool = True,
#     gau_blur: bool = True,
# ) -> Optional[transforms.Compose]:
#     """Same augmentations as before, but applied on the PIL image; ToTensor is LAST (fast PIL code path)."""
#     default_jitter = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05, "p": 0.8}
#     if color_jitter is not None:
#         if not isinstance(color_jitter, dict):
#             raise ValueError("color_jitter must be a dictionary or None")
#         effective_jitter = {**default_jitter, **color_jitter}
#     else:
#         effective_jitter = default_jitter

#     transform_list = [
#         transforms.RandomApply(
#             [transforms.ColorJitter(
#                 brightness=effective_jitter["brightness"],
#                 contrast=effective_jitter["contrast"],
#                 saturation=effective_jitter["saturation"],
#                 hue=effective_jitter["hue"],
#             )],
#             p=effective_jitter["p"],
#         )
#     ]
#     if gray_scale:
#         transform_list.append(transforms.RandomGrayscale(p=0.05))
#     if gau_blur:
#         transform_list.append(transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05))

#     transform_list.append(transforms.ToTensor())          # last: PIL -> float tensor in [0,1]
#     return transforms.Compose(transform_list)


class FastToTensor:
    """Drop-in for tvf.ToTensor(): PIL / numpy HxWxC (uint8 or float [0,1]) -> float32 tensor (C,H,W) in [0,1]."""
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)                                   # uint8 HxWxC, no copy
        if img.ndim == 2:
            img = img[:, :, None]
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if t.dtype == torch.uint8:
            return t.float().div_(255.0)
        return t.float()                                            # float64 -> float32, already [0,1]
