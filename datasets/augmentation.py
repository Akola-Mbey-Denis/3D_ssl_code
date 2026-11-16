import random
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
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
        
class SimpleAug:
    """
    Sequence-consistent augmentation for SSL pixel-reconstruction pre-training.

    Horizontal flip, colour jitter (brightness / contrast / saturation / hue) and random
    grayscale. Every random decision is drawn ONCE per sequence and applied identically to
    all frames, so a target view is always augmented exactly like its sources and stays
    predictable from them. No crop / resize and no ImageNet Normalize: outputs stay in [0, 1].

    IMAGES ONLY: depth maps and intrinsics are not updated.
    """

    @dataclass(frozen=True)
    class Cfg:
        enabled: bool = True
        hflip_prob: float = 0.5
        jitter_prob: float = 0.8
        brightness: float = 0.4
        contrast: float = 0.4
        saturation: float = 0.4
        hue: float = 0.1
        grayscale_prob: float = 0.2

    def __init__(self, cfg: "SimpleAug.Cfg | None" = None) -> None:
        self.cfg = cfg or SimpleAug.Cfg()

    @staticmethod
    def to_tensor(img) -> torch.Tensor:
        """PIL / numpy (H, W, 3) / tensor (3, H, W) or (H, W, 3) -> float tensor (3, H, W) in [0, 1]."""
        if isinstance(img, torch.Tensor):
            t = img.float()
            if t.ndim == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            return t / 255.0 if t.max() > 1.0 else t
        if isinstance(img, Image.Image):
            img = np.asarray(img.convert("RGB"))
        arr = np.asarray(img)
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        return t / 255.0 if arr.dtype == np.uint8 or t.max() > 1.0 else t

    @staticmethod
    def _factor(strength: float):
        return random.uniform(max(0.0, 1.0 - strength), 1.0 + strength) if strength > 0 else None

    def _sample_params(self) -> dict:
        c = self.cfg
        p = {"flip": random.random() < c.hflip_prob, "jitter": None,
             "gray": random.random() < c.grayscale_prob}
        if random.random() < c.jitter_prob:
            ops = []
            if (f := self._factor(c.brightness)) is not None:
                ops.append((TFF.adjust_brightness, f))
            if (f := self._factor(c.contrast)) is not None:
                ops.append((TFF.adjust_contrast, f))
            if (f := self._factor(c.saturation)) is not None:
                ops.append((TFF.adjust_saturation, f))
            if c.hue > 0:
                ops.append((TFF.adjust_hue, random.uniform(-c.hue, c.hue)))
            random.shuffle(ops)  # random order, same order for every frame
            p["jitter"] = ops
        return p

    @staticmethod
    def _apply(img: torch.Tensor, p: dict) -> torch.Tensor:
        if p["flip"]:
            img = TFF.hflip(img)
        if p["jitter"]:
            for fn, v in p["jitter"]:
                img = fn(img, v)
        if p["gray"]:
            img = TFF.rgb_to_grayscale(img, num_output_channels=3)
        return img.clamp(0.0, 1.0)

    def __call__(self, frames: list, train: bool = True) -> list[torch.Tensor]:
        """frames: list of PIL / numpy / tensor images. Returns list of (3, H, W) tensors in [0, 1]."""
        frames = [self.to_tensor(f) for f in frames]
        if not train or not self.cfg.enabled:
            return frames
        p = self._sample_params()  # one draw per sequence
        return [self._apply(f, p) for f in frames]