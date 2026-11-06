# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import os
import json
import tqdm
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import matplotlib.cm as cm

class Latex_MetricLogger:
    def __init__(self, json_path="metrics.json"):
        self.json_path = json_path
        if os.path.exists(json_path):
            self.load()
        else:
            self.results = {}  # dataset -> metric_group -> metric_name -> model -> value

    def add_metric(self, dataset, metric_group, metric_name, model, value):
        """Add a metric value for a given dataset, group, metric, and model."""
        self.results.setdefault(dataset, {})
        self.results[dataset].setdefault(metric_group, {})
        self.results[dataset][metric_group].setdefault(metric_name, {})
        self.results[dataset][metric_group][metric_name][model] = value

    def save(self):
        """Save metrics to JSON file."""
        with open(self.json_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def load(self):
        """Load metrics from JSON file."""
        with open(self.json_path, "r") as f:
            self.results = json.load(f)

    def generate_latex_tables(self, datasets=None, point_models_order=None, depth_models_order=None):
        """Generate LaTeX tables for point cloud and depth metrics, rounded to 4 decimals."""
        if datasets is None:
            datasets = list(self.results.keys())

        latex_output = ""

        # --- POINT CLOUD TABLE ---
        metric_group = "point_cloud"
        point_metrics = ["Acc", "Comp"]  # Only mean & median for these

        if point_models_order is None:
            point_models_order = set()
            for d in datasets:
                mg = self.results[d].get(metric_group, {})
                for mname in mg:
                    point_models_order.update(mg[mname].keys())
            point_models_order = sorted(point_models_order)

        latex_output += "\\begin{table}[t]\n\\centering\n\\vspace{8pt}\n"
        latex_output += "\\caption{Point Cloud Metrics across datasets (Mean and Median).}\n"
        latex_output += "\\resizebox{\\textwidth}{!}{%\n"
        total_cols = 1 + len(datasets) * len(point_metrics) * 2
        latex_output += "\\begin{tabular}{" + "l " + "cc " * ((total_cols-1)//2) + "}\n"
        latex_output += "\\toprule\n"
        latex_output += "& " + " & ".join([f"\\multicolumn{{{len(point_metrics)*2}}}{{c}}{{{d}}}" for d in datasets]) + " \\\\\n"
        latex_output += "& " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{m} $\\downarrow$}}" for _ in datasets for m in point_metrics]) + " \\\\\n"
        latex_output += "\\textbf{Model} " + " & " + " & ".join(["Mean & Med." for _ in datasets for _ in point_metrics]) + " \\\\\n\\midrule\n"

        for model in point_models_order:
            latex_output += model
            for d in datasets:
                mg = self.results[d].get(metric_group, {})
                for mname in point_metrics:
                    val = mg.get(mname, {}).get(model, {"mean": "-", "median": "-"})
                    mean_val = val.get("mean", "-") if isinstance(val, dict) else val
                    med_val = val.get("median", "-") if isinstance(val, dict) else "-"
                    # Round numeric values to 4 decimals
                    mean_val = f"{mean_val:.4f}" if isinstance(mean_val, (float, int)) else mean_val
                    med_val = f"{med_val:.4f}" if isinstance(med_val, (float, int)) else med_val
                    latex_output += f" & {mean_val} & {med_val}"
            latex_output += " \\\\\n"
        latex_output += "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n\n"

        # --- DEPTH TABLE ---
        metric_group = "depth"
        depth_metrics = ["AbsRel", "\\delta<1.25"]

        if depth_models_order is None:
            depth_models_order = set()
            for d in datasets:
                mg = self.results[d].get(metric_group, {})
                for mname in mg:
                    depth_models_order.update(mg[mname].keys())
            depth_models_order = sorted(depth_models_order)

        latex_output += "\\begin{table}[t]\n\\centering\n\\vspace{8pt}\n"
        latex_output += "\\caption{Depth Metrics (AbsRel and $\\delta < 1.25$ pixels).}\n"
        latex_output += "\\resizebox{\\textwidth}{!}{%\n"
        total_cols = 1 + len(datasets) * len(depth_metrics)
        latex_output += "\\begin{tabular}{" + "l " + "c "*(total_cols-1) + "}\n"
        latex_output += "\\toprule\n"
        latex_output += "& " + " & ".join([f"\\multicolumn{{{len(depth_metrics)}}}{{c}}{{{d}}}" for d in datasets]) + " \\\\\n"
        latex_output += "& " + " & ".join([m for _ in datasets for m in depth_metrics]) + " \\\\\n"
        latex_output += "\\midrule\n"

        for model in depth_models_order:
            latex_output += model
            for d in datasets:
                mg = self.results[d].get(metric_group, {})
                for mname in depth_metrics:
                    key = mname.replace("\\delta<1.25","Error<1.25")  # map LaTeX symbol
                    val = mg.get(key, {}).get(model, "-")
                    # Round numeric values
                    if isinstance(val, (float, int)):
                        val = f"{val:.4f}"
                    latex_output += f" & {val}"
            latex_output += " \\\\\n"
        latex_output += "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"

        return latex_output

 


def latents_to_rgb_images(latents, percentile_clip=True):
    """
    Convert (B, S, P, C) latents to (B, S, 3, P, C) RGB images
    where (P, C) are treated as image height and width
    
    Args:
        latents: (B, S, P, C) tensor
        method: 'colormap', 'normalize', or 'minmax'
        percentile_clip: if True, use 5th and 95th percentiles for normalization
    
    Returns:
        images: (B, S, 3, P, C) RGB images normalized to [0, 1]
    """
    B, S, P, C = latents.shape
    device = latents.device

        
    # Normalize to [0, 1]
    if percentile_clip:
        # Use 5th and 95th percentiles
        v_min = torch.quantile(latents, 0.05)
        v_max = torch.quantile(latents, 0.95)
            
        # Clip and normalize
        latents_clipped = torch.clamp(latents, v_min, v_max)
        latents_norm = (latents_clipped - v_min) / (v_max - v_min + 1e-8)
    else:
        # Use global min/max
        latents_norm = (latents - latents.min()) / (latents.max() - latents.min() + 1e-8)
        
    # Apply colormap
    cmap = cm.get_cmap('viridis')
    latents_np = latents_norm.cpu().numpy()
        
    # cmap returns (B, S, P, C, 4) with RGBA, we take RGB
    images_rgba = cmap(latents_np)  # (B, S, P, C, 4)
    images_rgb = images_rgba[..., :3]  # (B, S, P, C, 3)
        
    # Transpose to (B, S, 3, P, C)
    images = np.transpose(images_rgb, (0, 1, 4, 2, 3))
        
    return torch.from_numpy(images).float().to(device)

def latents_to_rgb_custom_colormap(latents, cmap_name='viridis'):
    """
    Convert (B, S, P, C) latents to (B, S, 3, P, C) RGB images using custom colormap
    
    Args:
        latents: (B, S, P, C) tensor
        cmap_name: matplotlib colormap name ('viridis', 'plasma', 'hot', 'cool', 'jet', etc.)
    
    Returns:
        images: (B, S, 3, P, C) RGB images normalized to [0, 1]
    """
    from matplotlib import cm
    
    B, S, P, C = latents.shape
    device = latents.device
    
    # Normalize to [0, 1]
    latents_norm = (latents - latents.min()) / (latents.max() - latents.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(cmap_name)
    latents_np = latents_norm.cpu().numpy()
    images_rgba = cmap(latents_np)  # (B, S, P, C, 4)
    images_rgb = images_rgba[..., :3]  # (B, S, P, C, 3)
    
    # Transpose to (B, S, 3, P, C)
    images = np.transpose(images_rgb, (0, 1, 4, 2, 3))
    
    return torch.from_numpy(images).float().to(device)


def prepare_latents_comparison_for_tensorboard(pred_latents, gt_latents, cmap='viridis'):
    """
    Prepare pred, gt, and diff images for TensorBoard logging
    
    Args:
        pred_latents: (B, S, P, C) predicted latents
        gt_latents: (B, S, P, C) ground truth latents
        cmap: colormap name
    
    Returns:
        pred_images: (B, S, 3, P, C)
        gt_images: (B, S, 3, P, C)
        diff_images: (B, S, 3, P, C)
    """
    # Convert to RGB
    pred_images = latents_to_rgb_custom_colormap(pred_latents, cmap)
    gt_images = latents_to_rgb_custom_colormap(gt_latents, cmap)
    
    # Compute absolute difference
    diff = torch.abs(pred_latents - gt_latents)
    diff_images = latents_to_rgb_custom_colormap(diff, 'hot')  # Use 'hot' for differences
    
    return pred_images, gt_images, diff_images


def log_latents_to_tensorboard(pred_latents, gt_latents, writer, step, prefix='val', max_samples=2):
    """
    Log latent visualizations to TensorBoard
    
    Args:
        pred_latents: (B, S, P, C) predicted latents
        gt_latents: (B, S, P, C) ground truth latents
        writer: SummaryWriter instance
        step: global step
        prefix: tag prefix
        max_samples: maximum number of samples to log per batch
    """
    B, S, P, C = pred_latents.shape
    
    # Convert to RGB images
    pred_images, gt_images, diff_images = prepare_latents_comparison_for_tensorboard(
        pred_latents, gt_latents
    )
    
    # Log images
    for b in range(min(B, 2)):  # Log first 2 batches
        for s in range(min(S, max_samples)):
            # Extract (3, P, C) - already in correct format for TensorBoard
            pred_img = pred_images[b, s]  # (3, P, C)
            gt_img = gt_images[b, s]      # (3, P, C)
            diff_img = diff_images[b, s]  # (3, P, C)
            
            writer.add_image(f'{prefix}/latent_gt_b{b}_s{s}', gt_img, step)
            writer.add_image(f'{prefix}/latent_pred_b{b}_s{s}', pred_img, step)
            writer.add_image(f'{prefix}/latent_diff_b{b}_s{s}', diff_img, step)
    
    # Also log metrics
    mse = torch.mean((pred_latents - gt_latents) ** 2).item()
    mae = torch.mean(torch.abs(pred_latents - gt_latents)).item()
    writer.add_scalar(f'{prefix}/latent_mse', mse, step)
    writer.add_scalar(f'{prefix}/latent_mae', mae, step)


def latents_to_rgb_side_by_side(pred_latents, gt_latents, batch_idx=0, sample_idx=0):
    """
    Create side-by-side comparison image for a single sample
    
    Args:
        pred_latents: (B, S, P, C) predicted latents
        gt_latents: (B, S, P, C) ground truth latents
        batch_idx: which batch to visualize
        sample_idx: which sample to visualize
    
    Returns:
        combined_image: (3, P, C*3) RGB image with GT | Pred | Diff side-by-side
    """
    # Extract single sample
    pred = pred_latents[batch_idx, sample_idx:sample_idx+1]  # (1, 1, P, C)
    gt = gt_latents[batch_idx, sample_idx:sample_idx+1]      # (1, 1, P, C)
    diff = torch.abs(pred - gt)
    
    # Convert to RGB
    pred_rgb = latents_to_rgb_custom_colormap(pred, 'viridis')[0, 0]  # (3, P, C)
    gt_rgb = latents_to_rgb_custom_colormap(gt, 'viridis')[0, 0]      # (3, P, C)
    diff_rgb = latents_to_rgb_custom_colormap(diff, 'hot')[0, 0]      # (3, P, C)
    
    # Concatenate horizontally
    combined = torch.cat([gt_rgb, pred_rgb, diff_rgb], dim=2)  # (3, P, C*3)
    
    return combined



def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2

def move_selected_keys_to_device(batch, keys, device="cuda"):
    for key in keys:
        value = batch.get(key)
        if value is not None:
            if isinstance(value, list):
                # Convert list of tensors or arrays to a stacked tensor
                tensor = torch.stack(value) if isinstance(value[0], torch.Tensor) else torch.from_numpy(np.stack(value))
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                tensor = value  # assume it's already a tensor

            batch[key] = tensor.to(dtype=torch.float32, device=device, non_blocking=True)
    return batch


def loss_of_one_batch_depth_diffusion(batch, model, criterion, device, ret=None):

    tensor_keys = [
        "t_depths",
        "t_depth_masks",
        "s_depths_mask",
        "s_depths",
        "s_images",
        "t_images",
            ]   

    batch = move_selected_keys_to_device(batch, tensor_keys, device=device)

    images  = batch["t_images"]
    diffused_tokens_list = batch["diffused_tokens"]
    patch_start_idx = batch["patch_start"]


    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.cuda.amp.autocast(dtype=dtype):
        # breakpoint()
        pred = model(images = images, diffused_tokens_list  = diffused_tokens_list, patch_start_idx = patch_start_idx)
    with torch.cuda.amp.autocast(enabled=False):
        loss = criterion(pred, batch) if criterion is not None else None

    result = dict(gt = batch, pred = pred, loss = loss)
    return result[ret] if ret else result
    return result


def loss_of_one_batch_nvs(batch, model, criterion, device, ret=None):

    tensor_keys = [
        "t_extrinsics",
        "t_depths",
        "t_depth_masks",
        "s_depths_mask",
        "s_depths",
        "t_intrinsics",
        "s_images",
        "t_images",
        "s_extrinsics"
        ]

    batch = move_selected_keys_to_device(batch, tensor_keys, device=device)

    images  = batch["s_images"]

    rel_poses = batch["t_extrinsics"]

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images = images, poses = rel_poses)
    
    with torch.cuda.amp.autocast(enabled=False):
        loss = criterion(pred, batch) if criterion is not None else None

    result = dict(gt = batch, pred = pred, loss = loss)
    return result[ret] if ret else result
    return result


def loss_of_one_batch_ldi(batch, model, criterion, device, ret=None):

    tensor_keys = [
        "image",
        "depth",
        "extrinsic",
        "intrinsic",
        "ldis",
        "ldis_mask"
        ]

    batch = move_selected_keys_to_device(batch, tensor_keys, device=device)

    images  = batch["image"]

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)
    
    with torch.cuda.amp.autocast(dtype=dtype):
        loss = criterion(pred, batch) if criterion is not None else None

    result = dict(gt = batch, pred = pred, loss = loss)
    return result[ret] if ret else result
    return result


def loss_of_one_batch(batch, model, criterion, device, ret=None):
    tensor_keys = [
        "images",
        "depths",
        "extrinsics",
        "intrinsics",
        "cam_points",
        "world_points",
        "point_masks"
    ]
    batch = move_selected_keys_to_device(batch, tensor_keys, device=device)

    images  = batch["images"]

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)

    with torch.cuda.amp.autocast(dtype=dtype):
        loss = criterion(pred, batch) if criterion is not None else None

    # breakpoint()
    result = dict(gt = batch, pred = pred, loss = loss)
    return result[ret] if ret else result
    return result




def undo_normalization_batch(image, mean, std):
    """
    Undo normalization for a batch of images.

    Args:
        image (Tensor): The image tensor (B, C, H, W) or (C, H, W), on any device.
        mean (tuple): The mean used for normalization.
        std (tuple): The standard deviation used for normalization.

    Returns:
        Tensor: Denormalized image tensor (same device as input).
    """
    # Ensure mean and std are on the same device as image
    mean = torch.tensor(mean, device=image.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=image.device).view(1, -1, 1, 1)
    
    return image * std + mean  # Undo normalization


def generate_batch_images(output_dict, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Converts a batch of normalized image tensors to NumPy format after denormalization.

    Args:
        images (Tensor): A batch of image tensors of shape (B, C, H, W).
        mean (tuple): The mean used for normalization.
        std (tuple): The standard deviation used for normalization.

    Returns:
        list: A list of NumPy arrays representing denormalized images.
    """
    images  =  output_dict["view1"]["img"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("Input images must be a PyTorch tensor")
    
    if images.ndim != 4:
        raise ValueError("Expected input shape (B, C, H, W), but got shape {}".format(images.shape))
    
    # Undo normalization
    images_denormalized = undo_normalization_batch(images, mean=mean, std=std)

    # Convert images to NumPy format and rescale to [0, 255]
    images_np = images_denormalized.detach().cpu().numpy()
    images_np = (images_np * 255).astype(np.uint8)

    # Rearrange (B, C, H, W) -> (B, H, W, C) for RGB images
    if images_np.shape[1] == 3:  # Check if the input has 3 channels (RGB)
        images_np = images_np.transpose(0, 2, 3, 1)  # Convert from (B, C, H, W) to (B, H, W, C)
    
    return images_np  # Returns a list of NumPy arrays



def vis_depth_maps_nvs(
    depth_batch: torch.Tensor,            # (B, H, W)
    conf_batch: Optional[torch.Tensor] = None,  # (B, H, W)
    mask: Optional[torch.Tensor] = None,        # (B, H, W)
    percentile: float = 25.0
) -> np.ndarray:
    """
    Visualize depth maps with spectral_r colormap, optionally masked by confidence or binary mask.

    Args:
        depth_batch (torch.Tensor): (B, H, W), float32 or float16.
        conf_batch (torch.Tensor, optional): (B, H, W) confidence scores.
        mask (torch.Tensor, optional): (B, H, W) binary mask.
        percentile (float): Confidence threshold percentile.

    Returns:
        np.ndarray: (B, H, W, 3) RGB visualization.
    """
    depth_batch = depth_batch.cpu().numpy()
    if conf_batch is not None:
        conf_batch = conf_batch.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy()

    B, H, W = depth_batch.shape
    colored_maps = np.zeros((B, H, W, 3), dtype=np.uint8)
    cmap = cm.get_cmap('Spectral_r')

    for b in range(B):
        depth = depth_batch[b]

        # Mask logic
        if mask is not None:
            valid = mask[b]
        elif conf_batch is not None:
            conf = conf_batch[b]
            thresh = np.percentile(conf, percentile)
            valid = conf > thresh
        else:
            valid = np.ones_like(depth, dtype=bool)

        # Normalize valid depth values
        valid_depth = depth[valid]
        if valid_depth.size > 0:
            v_min, v_max = np.nanmin(valid_depth), np.nanmax(valid_depth)
            norm_depth = (depth - v_min) / (v_max - v_min + 1e-8)
        else:
            norm_depth = np.zeros_like(depth)

        color = cmap(norm_depth)  # RGBA
        rgb = (color[..., :3] * 255).astype(np.uint8)
        
        rgb[~valid] = 0                          # invalid pixels really black

        colored_maps[b] = rgb

    return colored_maps


def _fit_alignment(pred, gt, valid, mode):
    """Scale (and shift) mapping pred -> gt on the valid pixels: (s, t) with aligned = s * pred + t."""
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    if p.size < 10:
        return 1.0, 0.0
    if mode == "scale":                                   # median ratio: robust, keeps 0 at 0
        return float(np.median(g / np.maximum(p, 1e-6))), 0.0
    if mode == "scale_shift":                             # least squares (LaRI, z-shift only)
        A = np.stack([p, np.ones_like(p)], axis=1)
        (s, t), *_ = np.linalg.lstsq(A, g, rcond=None)
        return float(s), float(t)
    raise ValueError(f"unknown align mode {mode!r}")


def vis_layered_depth_maps(
    depth_batch: torch.Tensor,
    conf_batch: torch.Tensor = None,
    mask: torch.Tensor = None,
    percentile: float = 25.0,
    gt_batch: torch.Tensor = None,
    gt_mask: torch.Tensor = None,
    align: str = "scale",
    return_alignment: bool = False,
    mask_to_gt: bool = True,
):
    """
    Visualize layered depth maps using a colormap, with optional confidence and validity masking.
    With GT layered depth, the prediction is aligned to the GT scale and coloured on the GT's colour range, so
    predicted and GT maps are directly comparable.

    Args:
        depth_batch (torch.Tensor): Tensor of shape (B, H, W, L), float32 (predicted, or GT when visualizing GT).
        conf_batch (torch.Tensor, optional): Confidence scores of shape (B, H, W, L).
        mask (torch.Tensor, optional): Boolean mask of shape (B, H, W, L), True = valid pixel to draw.
            Black (invalid) is always: depth <= 0 / nan / inf, outside `mask`, below the confidence percentile,
            and, when gt_batch is given, where the GT layer does not exist.
        percentile (float): Percentile threshold for confidence masking.
        gt_batch (torch.Tensor, optional): GT layered depth (B, H, W, L). If given:
            - the prediction is aligned to GT per image (align), fitted on LAYER 0 pixels where GT is valid and
              applied to every layer (the hidden layers keep their position relative to the visible surface);
            - every layer is coloured with the GT's min / max of that layer, so the same colour = the same depth
              in the GT and in the prediction (visualize the GT with the same call and gt_batch=None,
              mask=gt_mask: it uses the same per-layer range).
        gt_mask (torch.Tensor, optional): (B, H, W, L) bool, True where the GT layer exists (default: gt > 0).
        align (str or None): "scale" (median gt / pred ratio), "scale_shift" (least squares, as LaRI) or None
            (metric prediction, only the shared colour range).
        return_alignment (bool): also return the per-image (scale, shift), shape (B, 2).
        mask_to_gt (bool): with gt_batch, black out the pixels where the GT layer does not exist (default). False
            keeps the alignment and the GT colour range but draws EVERY predicted pixel, to see what the model
            predicts where GT has no surface (only depth <= 0 / nan / inf, `mask` and confidence stay black).

    Returns:
        np.ndarray: Array of shape (B, L, H, W, 3), dtype uint8 (invalid pixels black);
                    with return_alignment, (colored_maps, alignment (B, 2)).
    """
    to_np = lambda x: x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    depth_batch = np.nan_to_num(to_np(depth_batch).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    conf_batch = to_np(conf_batch) if conf_batch is not None else None
    mask = to_np(mask).astype(bool) if mask is not None else None
    if gt_batch is not None:
        gt_batch = np.nan_to_num(to_np(gt_batch).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        gt_mask = to_np(gt_mask).astype(bool) if gt_mask is not None else gt_batch > 0
        gt_mask &= gt_batch > 0

    B, H, W, L = depth_batch.shape
    colored_maps = np.zeros((B, L, H, W, 3), dtype=np.uint8)
    alignment = np.tile(np.array([1.0, 0.0]), (B, 1))
    cmap = cm.get_cmap('Spectral_r')

    for b in range(B):
        if gt_batch is not None and align is not None:
            fit_px = gt_mask[b, :, :, 0] & (depth_batch[b, :, :, 0] > 0)
            alignment[b] = _fit_alignment(depth_batch[b, :, :, 0], gt_batch[b, :, :, 0], fit_px, align)
        s, t = alignment[b]

        for l in range(L):
            depth = depth_batch[b, :, :, l] * s + t

            # Validity mask for this sample/layer
            valid_mask = depth_batch[b, :, :, l] > 0              # no depth (0 / nan / inf) -> black
            if mask is not None:
                valid_mask &= mask[b, :, :, l]
            if gt_batch is not None and mask_to_gt:
                valid_mask &= gt_mask[b, :, :, l]                  # no GT layer -> black
            if conf_batch is not None and np.any(valid_mask):
                conf = conf_batch[b, :, :, l]
                valid_mask &= conf > np.percentile(conf[valid_mask], percentile)

            # Colour range: the GT's range of this layer if GT is given, else this map's own range
            if gt_batch is not None and gt_mask[b, :, :, l].any():
                ref = gt_batch[b, :, :, l][gt_mask[b, :, :, l]]
            else:
                ref = depth[valid_mask]
            v_min, v_max = (float(ref.min()), float(ref.max())) if ref.size > 0 else (0.0, 1.0)

            if v_max > v_min:
                norm_depth = np.clip((depth - v_min) / (v_max - v_min), 0.0, 1.0)
            else:
                norm_depth = np.zeros_like(depth)

            rgb = (cmap(norm_depth)[..., :3] * 255).astype(np.uint8)
            rgb[~valid_mask] = 0                          # invalid pixels really black
            colored_maps[b, l] = rgb

    return (colored_maps, alignment) if return_alignment else colored_maps
    
# def vis_layered_depth_maps(
#     depth_batch: torch.Tensor,
#     conf_batch: torch.Tensor = None,
#     mask: torch.Tensor = None,
#     percentile: float = 25.0
# ) -> np.ndarray:
#     """
#     Visualize layered depth maps using a colormap, with optional confidence and validity masking.
    
#     Args:
#         depth_batch (torch.Tensor): Tensor of shape (B, H, W, L), float32.
#         conf_batch (torch.Tensor, optional): Confidence scores of shape (B, H, W, L).
#         mask (torch.Tensor, optional): Boolean mask of shape (B, H, W, L), where True indicates valid pixels.
#         percentile (float): Percentile threshold for confidence masking.

#     Returns:
#         np.ndarray: Array of shape (B, L, H, W, 3), dtype uint8.
#     """
#     depth_batch = depth_batch.cpu().numpy()
#     if conf_batch is not None:
#         conf_batch = conf_batch.cpu().numpy()
#     if mask is not None:
#         mask = mask.cpu().numpy()

#     B, H, W, L = depth_batch.shape
#     colored_maps = np.zeros((B, L, H, W, 3), dtype=np.uint8)
#     cmap = cm.get_cmap('Spectral_r')

#     for b in range(B):
#         for l in range(L):
#             depth = depth_batch[b, :, :, l]
#             depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

#             # Validity mask for this sample/layer
#             valid_mask = np.ones((H, W), dtype=bool)
#             if mask is not None:
#                 valid_mask &= mask[b, :, :, l].astype(bool)

#             if conf_batch is not None:
#                 conf = conf_batch[b, :, :, l]
#                 if np.any(valid_mask):
#                     thresh = np.percentile(conf[valid_mask], percentile)
#                     valid_mask &= conf > thresh

#             valid_depth = depth[valid_mask]
#             if valid_depth.size > 0:
#                 v_min, v_max = np.min(valid_depth), np.max(valid_depth)
#             else:
#                 v_min, v_max = 0, 1  # fallback

#             if v_max > v_min:
#                 norm_depth = (depth - v_min) / (v_max - v_min)
#             else:
#                 norm_depth = np.zeros_like(depth)

#             norm_depth[~valid_mask] = 0.0  # Black out invalid pixels
#             color = cmap(norm_depth)
#             rgb = (color[..., :3] * 255).astype(np.uint8)
#             colored_maps[b, l] = rgb

#     return colored_maps


import numpy as np

def metric_to_relative(depth_metric):
    d_min = np.min(depth_metric)
    d_max = np.max(depth_metric)
    return (depth_metric - d_min) / (d_max - d_min + 1e-8)  # add eps for safety


def vis_depth_maps(
    depth_batch: torch.Tensor,
    conf_batch: torch.Tensor = None,
    mask_batch: torch.Tensor = None,
    percentile: float = 25.0,
    v_min = None,
    v_max = None

) -> np.ndarray:
    """
    Visualize depth maps with Spectral_r colormap, using optional confidence and binary masks.

    Args:
        depth_batch (torch.Tensor): Tensor of shape (B, S, H, W), dtype float32.
        conf_batch (torch.Tensor, optional): Tensor of same shape as depth_batch for confidence masking.
        mask_batch (torch.Tensor, optional): Binary tensor of same shape as depth_batch for validity masking.
        percentile (float): Percentile threshold for confidence map.

    Returns:
        np.ndarray: RGB visualization of shape (B, S, H, W, 3), dtype uint8.
    """
    depth_batch = depth_batch.detach().cpu().numpy()
    conf_batch = conf_batch.detach().cpu().numpy() if conf_batch is not None else None
    mask_batch = mask_batch.detach().cpu().numpy() if mask_batch is not None else None

    B, S, H, W = depth_batch.shape
    colored_maps = np.zeros((B, S, H, W, 3), dtype=np.uint8)
    cmap = cm.get_cmap('Spectral_r')

    for b in range(B):
        for s in range(S):
            depth = depth_batch[b, s]

            # Base mask (all True)
            mask = np.ones_like(depth, dtype=bool)

            # Apply confidence mask
            if conf_batch is not None:
                conf = conf_batch[b, s]
                thresh = np.percentile(conf, percentile)
                mask &= (conf > thresh)

            # Apply external binary mask
            if mask_batch is not None:
                mask &= mask_batch[b, s].astype(bool)

            valid_depth = depth[mask]
            valid_depth = np.nan_to_num(valid_depth, nan=0.0, posinf=0.0, neginf=0.0)

            if v_min is not None and v_max is not None:
                valid_depth = np.clip(valid_depth, v_min, v_max)
                
            if v_min is None:
                v_min = np.nanmin(valid_depth)
            if v_max is None:
                v_max = np.nanmax(valid_depth)
            
            if v_max > v_min:
                norm_depth = (depth - v_min) / (v_max - v_min)
            else:
                norm_depth = np.zeros_like(depth)

            # norm_depth[~mask] = 0.0  # Zero out masked regions for visualization

            color = cmap(norm_depth)  # RGBA
            rgb = (color[..., :3] * 255).astype(np.uint8)
            rgb[~mask] = 0
            colored_maps[b, s] = rgb

    return colored_maps


def align_pred_to_gt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_pixels: int = 100,
) -> tuple[float, float, np.ndarray]:
    """
    Aligns a predicted depth map to a ground truth depth map using scale and shift.
    The alignment is: gt_aligned_to_pred ≈ scale * pred_depth + shift.

    Args:
        pred_depth (np.ndarray): The HxW predicted depth map.
        gt_depth (np.ndarray): The HxW ground truth depth map.
        min_gt_depth (float): Minimum valid depth value for GT.
        max_gt_depth (float): Maximum valid depth value for GT.
        min_pred_depth (float): Minimum valid depth value for predictions.
        min_valid_pixels (int): Minimum number of valid overlapping pixels
                                 required to perform the alignment.
        robust_median_scale (bool): If True, uses median(gt/pred) for scale and then
                                    median(gt - scale*pred) for shift. Otherwise, uses
                                    least squares for both scale and shift simultaneously.

    Returns:
        tuple[float, float, np.ndarray]:
            - scale (float): The calculated scale factor. (NaN if alignment failed)
            - shift (float): The calculated shift offset. (NaN if alignment failed)
            - aligned_pred_depth (np.ndarray): The HxW predicted depth map after
                                               applying scale and shift. (Original pred_depth
                                               if alignment failed)
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    # Extract valid depth values
    gt_masked = gt_depth[valid_mask]
    pred_masked = pred_depth[valid_mask]

    if len(gt_masked) < min_valid_pixels:
        print(
            f"Warning: Not enough valid pixels ({len(gt_masked)} < {min_valid_pixels}) to align. "
            "Using all pixels."
        )
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)


    # Handle case where pred_masked has no variance (e.g., all zeros or a constant value)
    if np.std(pred_masked) < 1e-6: # Small epsilon to check for near-constant values
        print(
            "Warning: Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        shift = np.mean(gt_masked) - np.mean(pred_masked) # or np.median(gt_masked) - np.median(pred_masked)
    else:
        A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, residuals, rank, s_values = np.linalg.lstsq(A, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as e:
            print(f"Warning: Least squares alignment failed ({e}). Returning original prediction.")
            return np.nan, np.nan, pred_depth.copy()


    aligned_pred_depth = scale * pred_depth + shift
    
    return scale, shift, aligned_pred_depth

 
def vis_images(img_batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of [0,1] range images (from ToTensor) to RGB uint8 format.

    Args:
        img_batch (torch.Tensor): Tensor of shape (B, S, 3, H, W) with values in [0.0, 1.0].

    Returns:
        np.ndarray: RGB images of shape (B, S, H, W, 3), dtype uint8.
    """
    img_batch = img_batch.clamp(0, 1)  # Ensure values are in [0, 1]
    img_batch = img_batch.permute(0, 1, 3, 4, 2)  # (B, S, H, W, 3)
    img_batch = (img_batch * 255).to(torch.uint8)

    return img_batch.detach().cpu().numpy()

def vis_images_ldi(img_batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of [0,1] range RGB images to uint8 format.

    Args:
        img_batch (torch.Tensor): Tensor of shape (B, 3, H, W) with values in [0.0, 1.0].

    Returns:
        np.ndarray: RGB images of shape (B, H, W, 3), dtype uint8.
    """
    
    img_batch = img_batch.clamp(0, 1)  # Ensure values are in [0, 1]
    img_batch = img_batch.permute(0, 2, 3, 1)  # (B, H, W, 3)
    img_batch = (img_batch * 255).to(torch.uint8)

    return img_batch.cpu().numpy()


def generate_depth_from_point_map(point_map, mask, RT_c2w, min_depth = 0.10, max_depth = 8.0):
    """
    Transforms a point map back to the camera coordinate frame using RT_c2w and extracts a colored depth map.

    Parameters:
    point_map (torch.Tensor): A 4D tensor of shape (B, H, W, 3), representing (X, Y, Z) in world coordinates.
    mask (torch.Tensor): A 3D boolean tensor of shape (B, H, W), where `True` values indicate valid depth regions.
    RT_c2w (torch.Tensor): A 3D tensor of shape (B, 4, 4), representing the camera-to-world transformation matrices.

    Returns:
    numpy.ndarray: A masked and colorized depth map of shape (B, H, W, 3) in RGB format.
    """
    if not isinstance(point_map, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(RT_c2w, torch.Tensor):
        raise TypeError("Inputs must be PyTorch tensors")

    B, H, W, C = point_map.shape
    if C != 3:
        raise ValueError("point_map should have shape (B, H, W, 3)")

    if mask.shape != (B, H, W):
        raise ValueError("mask should have shape (B, H, W) matching point_map")

    if RT_c2w.shape != (B, 4, 4):
        raise ValueError("RT_c2w should have shape (B, 4, 4)")

    # Convert point_map to homogeneous coordinates (B, H, W, 4)
    ones = torch.ones((B, H, W, 1), device=point_map.device)
    point_map_h = torch.cat([point_map, ones], dim=-1)  # (B, H, W, 4)

    # Compute world-to-camera transform (RT^-1)
    RT_w2c = torch.inverse(RT_c2w)  # (B, 4, 4)

    # Reshape for batched matrix multiplication
    point_map_h = point_map_h.view(B, H * W, 4).permute(0, 2, 1)  # (B, 4, H*W)

    # Apply transformation
    camera_coords = torch.bmm(RT_w2c, point_map_h)  # (B, 4, H*W)

    # Reshape back to (B, H, W, 3)
    camera_coords = camera_coords.permute(0, 2, 1).view(B, H, W, 4)

    # Extract Z-depth values
    depth_map = camera_coords[..., 2]  # (B, H, W)

    # Apply mask
    depth_map = depth_map * mask

    # Normalize depth values for colormap application
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)

    # Apply spectral_r colormap
    cmap = cm.get_cmap("Spectral_r")
    depth_colored = cmap(normalized_depth.detach().cpu().numpy())  # (B, H, W, 4) RGBA

    # Convert to RGB by removing alpha channel
    depth_colored = (depth_colored[..., :3] * 255).astype(np.uint8)  # (B, H, W, 3)

    return depth_colored


def generate_depths(model_output_dict):
    """
    Extracts ground truth and predicted depth maps from the model output dictionary.

    Parameters:
    model_output_dict (dict): A dictionary containing model outputs with keys:
                              - "view1": Dictionary with "pts3d" for ground truth depth.
                              - "view2": Dictionary with "pts3d" for ground truth depth.
                              - "pred1": Dictionary with "pts3d" for predicted depth.
                              - "pred2": Dictionary with "pts3d_in_other_view" for predicted depth.

    Returns:
    tuple: A tuple containing four numpy arrays:
           - gt_depths_view1: Ground truth depth for view1.
           - gt_depths_view2: Ground truth depth for view2.
           - pred_depths_view1: Predicted depth for view1.
           - pred_depths_view2: Predicted depth for view2.
    """
    view1_dict = model_output_dict["view1"]

    view1_pred = model_output_dict["pred1"]
    # breakpoint()
    
    gt_depths_view1 = generate_depth_from_point_map(view1_dict["pts3d"], view1_dict["valid_mask"],view1_dict["camera_pose"], min_depth = 0.10, max_depth = 8.0)

    pred_depths_view1 = generate_depth_from_point_map(view1_pred["pts3d"], view1_dict["valid_mask"],view1_dict["camera_pose"],min_depth = 0.10, max_depth = 8.0)
    
    return gt_depths_view1, pred_depths_view1

def generate_confidence_colormap(confidence_maps, mask, min_confidence=0.0, max_confidence=1.0):
    """
    Normalizes a batch of confidence maps and applies the Spectral_r colormap to each map.
    Each confidence map is colorized using the colormap and returned in RGB format.

    Parameters:
    confidence_maps (torch.Tensor): A 4D tensor of shape (B, H, W), representing confidence maps.
    mask (torch.Tensor): A 3D boolean tensor of shape (B, H, W), where `True` values indicate valid regions.
    min_confidence (float): The minimum confidence value for normalization.
    max_confidence (float): The maximum confidence value for normalization.

    Returns:
    numpy.ndarray: A batch of colorized confidence maps in RGB format (B, H, W, 3).
    """
    if not isinstance(confidence_maps, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Inputs must be PyTorch tensors")

    B, H, W = confidence_maps.shape
    if mask.shape != (B, H, W):
        raise ValueError("mask should have shape (B, H, W) matching confidence_maps")

    # Apply mask
    confidence_maps = confidence_maps * mask  # Mask out invalid regions

    # Normalize confidence maps to [0, 1] range
    normalized_confidence = (confidence_maps - min_confidence) / (max_confidence - min_confidence)

    # Apply Spectral_r colormap
    cmap = cm.get_cmap("Spectral_r")
    confidence_colored = cmap(normalized_confidence.detach().cpu().numpy())  # (B, H, W, 4) RGBA

    # Convert to RGB by removing the alpha channel
    confidence_colored = (confidence_colored[..., :3] * 255).astype(np.uint8)  # (B, H, W, 3)

    return confidence_colored


def colored_confidence_map(confidence_map):
    """
    Normalizes the confidence map using its min and max values, applies the Spectral_r colormap,
    and returns the colored map as an HxWx3 RGB image.

    Args:
        confidence_map (np.ndarray): Input confidence map (H x W).
    
    Returns:
        np.ndarray: Colored map with shape (H, W, 3) in RGB format.
    """
    # Normalize confidence map to [0,1]
    min_val, max_val = np.min(confidence_map), np.max(confidence_map)
    normalized_map = (confidence_map - min_val) / (max_val - min_val)

    # Apply the 'Spectral_r' colormap to the normalized map
    cmap = cm.get_cmap('Spectral_r')
    colored_map = cmap(normalized_map)  # Apply the colormap
    
    # The colormap returns an (H, W, 4) array with RGBA, so we drop the alpha channel
    colored_map_rgb = (colored_map[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255] range and to uint8
    
    return colored_map_rgb


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)





