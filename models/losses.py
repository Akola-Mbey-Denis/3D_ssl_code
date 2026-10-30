import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


from models.utils.pose_enc import extri_intri_to_pose_encoding

def normalize_pose_scale(poses: torch.Tensor) -> torch.Tensor:
    """Recenter each sequence's extrinsics on frame 0, then scale translation
    by the mean camera-center distance from frame 0 -- the camera-loss GT
    target's own normalization, *not* `normalize_batch_scale`'s depth-derived
    `scale` above.

    Why a separate function: this training mixture spans wildly different
    depth regimes batch to batch (aerial/landmark/indoor/object-scan scenes
    mixed by `DynamicBatchSampler`) -- `normalize_batch_scale`'s mean-GT-depth
    `scale` swung ~300x across batches of a real run (3.75 to 1139.8, see
    `experiments/ffrecon/train_poincar3.py`'s wandb `scale` history), which is
    fine for `depths_norm` (each batch's own depth is self-normalized to mean
    ~1 regardless of domain) but corrupts `compute_camera_loss`'s flat-L1
    translation target: the target's effective units become a function of
    ambient scene depth, decoupled from actual camera motion, so a fixed-
    magnitude translation error costs 300x more loss in a close-up batch than
    a far-away one. That shows up as `loss_camera`/AUC@30 plateauing early
    while RRA (rotation, untouched by this scale factor) keeps climbing
    cleanly and RTA (translation) stalls/degrades -- confirmed against a real
    run's `relpose_megadepth/RRA@15` vs `RTA@15` curves.

    Mean camera-center distance is bounded ~O(1) by construction (it is
    exactly what gets divided out) and depends only on camera motion, not on
    scene content, so it cannot blow up the way a depth-derived scale can
    across a heterogeneous dataset mixture.

    Deliberately does *not* touch `depths_norm`/`normalize_batch_scale`'s own
    `poses_norm`: those two stay on the shared depth-derived scale, since
    anything composing depth with pose translation needs them on one scale.
    Only `compute_camera_loss`'s GT argument uses this function's output.

    Args:
        poses: `(B, N, 4, 4)` cam-from-world extrinsics.

    Returns:
        `(B, N, 4, 4)` recentered + rescaled extrinsics.
    """
    batch_size = poses.shape[0]
    world_from_cam0 = torch.linalg.inv(poses[:, 0])
    poses_norm = poses @ world_from_cam0.unsqueeze(1)

    R = poses_norm[..., :3, :3]
    t = poses_norm[..., :3, 3]
    # Camera centers in the recentered frame: c_i = -R_i^T t_i (c_0 == 0).
    centers = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)  # (B, N, 3)
    scale = centers.norm(dim=-1).mean(dim=1).clamp_min(1e-6)  # (B,)

    poses_norm = poses_norm.clone()
    poses_norm[..., :3, 3] = t / scale.view(batch_size, 1, 1)
    return poses_norm

def compute_camera_loss(
    pred_pose_encoding: torch.Tensor,
    gt_poses: torch.Tensor,
    gt_K: torch.Tensor,
    image_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Flat L1 over the whole 9D `absT_quaR_FoV` pose encoding, which is more
    stable here than VGGT's Huber with separately-weighted
    translation/rotation/FOV components and per-component outlier clamping.
    `CameraHead` is single-shot (see its docstring), so there is only ever one
    refinement stage to weight.

    Args:
        pred_pose_encoding: `CameraHead.forward`'s single-element list,
            unwrapped by the caller (`pred_pose_enc_list[-1]`).
        gt_poses: `(B, N, 4, 4)` cam-from-world extrinsics.
        gt_K: `(B, N, 3, 3)` intrinsics.
        image_hw: `(height, width)` `pred_pose_encoding` was computed at.
    """
    gt_pose_encoding = extri_intri_to_pose_encoding(gt_poses[..., :3, :], gt_K, image_hw)
    loss = F.l1_loss(pred_pose_encoding, gt_pose_encoding)
    return {"loss_camera": loss}


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


class MultitaskLoss(nn.Module):
    """Combines camera, depth, and pointmap losses."""

    def __init__(
        self,
        camera=None,
        depth=None,
        layered_depth=None,
        # point=None,
    ):
        super().__init__()

        self.camera_loss = CameraLoss(**(camera or {}))
        self.depth_loss = DepthLoss(**(depth or {}))
        self.layered_depth_loss = LayeredDepthLoss(**(layered_depth or {}))
        self.layered_depth_enabled = layered_depth is not None
        # self.pointmap_loss = PointmapLoss(**(point or {}))

        self.camera_enabled = camera is not None
        self.depth_enabled = depth is not None
        # self.pointmap_enabled = point is not None

        # if track is not None:
        #     raise NotImplementedError("Tracking loss is not cleaned up yet.")

    def forward(self, predictions, gt_depths, masks, gt_intrinsics, gt_extrinsics, images,
                gt_layers=None, gt_layer_masks=None):
        losses = {}
        objective =  0.0

        if "pose_enc_list" in predictions and self.camera_enabled:
            camera_losses = self.camera_loss(predictions = predictions, gt_depth_masks = masks,  gt_extrinsics = gt_extrinsics, gt_intrinsics = gt_intrinsics, images = images)

            objective = objective + camera_losses["loss_camera"]
            losses.update(camera_losses)

        if "depth" in predictions and self.depth_enabled:
            depth_losses = self.depth_loss(predictions = predictions, gt_depths = gt_depths, masks = masks)
            objective = objective + depth_losses["loss_depth"]

            losses.update(depth_losses)

        if self.layered_depth_enabled:
            if "layered_depth" not in predictions:
                raise KeyError("layered_depth loss enabled but the model returned no 'layered_depth'")
            if gt_layers is None or gt_layer_masks is None:
                raise ValueError("layered_depth loss needs gt_layers and gt_layer_masks")
            layered_losses = self.layered_depth_loss(predictions, gt_layers, gt_layer_masks)
            objective = objective + layered_losses["loss_layered_depth"]
            losses.update(layered_losses)

        # if "world_points" in predictions and self.pointmap_enabled:
        #     point_losses = self.pointmap_loss(predictions, batch)
        #     objective = objective + point_losses["loss_pointmap"]
        #     losses.update(point_losses)

        losses["objective"] = objective

        return losses


# -----------------------------------------------------------------------------
# Camera loss
# -----------------------------------------------------------------------------


class CameraLoss(nn.Module):
    """
    Camera pose loss over translation, rotation, and focal length.

    Expected predictions:
        pose_enc_list: list[(B, S, D)]

    Expected batch:
        extrinsics: (B, S, ...)
        intrinsics: (B, S, ...)
        images:      (B, S, C, H, W)
        point_masks: (B, S, H, W)
    """

    def __init__(
        self,
        weight=1.0,
        loss_type="l1",
        gamma=0.6,
        pose_encoding_type="absT_quaR_FoV",
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=0.5,
        min_valid_points=100,
        **kwargs,
    ):
        super().__init__()

        self.weight = weight
        self.loss_type = loss_type
        self.gamma = gamma
        self.pose_encoding_type = pose_encoding_type

        self.weight_trans = weight_trans
        self.weight_rot = weight_rot
        self.weight_focal = weight_focal
        self.min_valid_points = min_valid_points

    def forward(self, predictions, gt_extrinsics, gt_intrinsics, images, gt_depth_masks):
        pred_stages = predictions["pose_enc_list"]

        valid_frames = gt_depth_masks[:, 0].sum(dim=(-1, -2)) > self.min_valid_points

        # Normalize the GT poses first (Poincar3's normalize_pose_scale): recenter on frame 0 (identity),
        # translations divided by the mean camera-centre distance.
        with torch.autocast(device_type=gt_extrinsics.device.type, enabled=False):
            gt_w2c = gt_extrinsics.float()
            if gt_w2c.shape[-2] == 3:                                   # (B, S, 3, 4) -> (B, S, 4, 4)
                bottom = torch.zeros_like(gt_w2c[..., :1, :])
                bottom[..., 0, 3] = 1.0
                gt_w2c = torch.cat([gt_w2c, bottom], dim=-2)
            gt_w2c = normalize_pose_scale(gt_w2c)

        gt_pose = extri_intri_to_pose_encoding(
            gt_w2c[..., :3, :],
            gt_intrinsics,
            images.shape[-2:],
            pose_encoding_type=self.pose_encoding_type,
        )

        loss_T = 0.0
        loss_R = 0.0
        loss_FL = 0.0

        num_stages = len(pred_stages)

        for stage_idx, pred_pose in enumerate(pred_stages):
            stage_weight = self.gamma ** (num_stages - stage_idx - 1)

            if valid_frames.any():
                stage_T, stage_R, stage_FL = self._single_stage(
                    pred_pose[valid_frames],
                    gt_pose[valid_frames],
                )
            else:
                zero = pred_pose.mean() * 0.0
                stage_T = zero
                stage_R = zero
                stage_FL = zero

            loss_T = loss_T + stage_weight * stage_T
            loss_R = loss_R + stage_weight * stage_R
            loss_FL = loss_FL + stage_weight * stage_FL

        loss_T = loss_T / num_stages
        loss_R = loss_R / num_stages
        loss_FL = loss_FL / num_stages

        loss_camera = (
            self.weight_trans * loss_T
            + self.weight_rot * loss_R
            + self.weight_focal * loss_FL
        )

        loss_camera = loss_camera * self.weight

        return {
            "loss_camera": loss_camera,
            "loss_T": loss_T,
            "loss_R": loss_R,
            "loss_FL": loss_FL,
        }

    def _single_stage(self, pred, gt):
        if self.loss_type == "l1":
            loss_T = (pred[..., :3] - gt[..., :3]).abs().mean()
            loss_R = (pred[..., 3:7] - gt[..., 3:7]).abs().mean()
            loss_FL = (pred[..., 7:] - gt[..., 7:]).abs().mean()

        elif self.loss_type == "l2":
            loss_T = (pred[..., :3] - gt[..., :3]).norm(dim=-1).mean()
            loss_R = (pred[..., 3:7] - gt[..., 3:7]).norm(dim=-1).mean()
            loss_FL = (pred[..., 7:] - gt[..., 7:]).norm(dim=-1).mean()

        else:
            raise ValueError(f"Unknown camera loss type: {self.loss_type}")

        loss_T = check_and_fix_inf_nan(loss_T, "loss_T").clamp(max=100)
        loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
        loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

        return loss_T, loss_R, loss_FL


# -----------------------------------------------------------------------------
# Depth loss  (VGGT recipe: conf-weighted L1 + plain L1 + multi-scale gradient L1, no conf in the grad term)
# -----------------------------------------------------------------------------


def filter_by_quantile(loss, valid_range, min_elements=1000, hard_max=100.0):
    """
    VGGT's filter_by_quantile: clamp to `hard_max`, then drop the worst (1 - valid_range) fraction.
    loss: flat per-element losses (already indexed to valid pixels). valid_range <= 0 disables filtering.
    """
    loss = loss.clamp(max=hard_max)
    if valid_range <= 0 or loss.numel() <= min_elements:
        return loss

    threshold = torch_quantile(loss.detach(), valid_range).clamp(max=hard_max)   # kthvalue: no 16M-element limit
    keep = loss < threshold
    if keep.sum() > min_elements:
        return loss[keep]
    return loss



def depth_gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and GT spatial gradients (adjacent-pixel
    differences of the `pred - gt` residual, x and y), masked to pixels valid
    on both sides of each finite difference.

    Ported from VGGT's `gradient_loss` (`VGGT's `training/loss.py``),
    dropping its optional confidence-weighting branch: VGGT's own default
    recipe (`gradient_loss_fn="grad"`, no "conf" in the name) never exercises
    it, so plain unweighted masked L1 is what "matching VGGT" actually means
    here.

    Args:
        pred, gt: `(B, H, W, C)`.
        mask: `(B, H, W)` bool, True = valid.
    """
    mask = mask[..., None].expand(-1, -1, -1, pred.shape[-1])
    diff = (pred - gt) * mask

    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * (mask[:, :, 1:] & mask[:, :, :-1])
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * (mask[:, 1:, :] & mask[:, :-1, :])
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    divisor = mask.sum()
    if divisor == 0:
        return (0.0 * pred).mean()
    return (grad_x.sum() + grad_y.sum()) / divisor



def depth_gradient_loss_multi_scale(pred, gt, mask, scales=4):
    """VGGT's gradient_loss_multi_scale_wrapper: average over `scales` 2x-subsampled resolutions."""
    total = pred.new_zeros(())
    for scale in range(scales):
        step = 2 ** scale
        total = total + depth_gradient_loss(pred[:, ::step, ::step], gt[:, ::step, ::step], mask[:, ::step, ::step])
    return total / scales


class DepthLoss(nn.Module):
    """
    VGGT depth loss:
        loss_conf = gamma * |pred - gt| * conf - alpha * log(conf)     (valid pixels, quantile-filtered)
        loss_reg  = |pred - gt|                                        (valid pixels, quantile-filtered)
        loss_grad = multi-scale L1 on spatial gradients of (pred - gt) (no confidence)
        loss_depth = (loss_conf.mean() + loss_reg.mean() + loss_grad) * weight

    Predictions:
        predictions["depth"]:      (B, S, H, W) or (B, S, H, W, 1)
        predictions["depth_conf"]: (B, S, H, W)
    Ground truth:
        gt_depths: (B, S, H, W)
        masks:     (B, S, H, W)
    """

    def __init__(
        self,
        weight=1.0,
        gamma=1.0,
        alpha=0.2,
        gradient_loss_fn="grad",      # "grad" or None (no gradient term)
        valid_range=0.98,             # VGGT default; <= 0 disables quantile filtering
        min_valid_points=1,           # VGGT returns a zero loss only when no pixel is valid
        gradient_scales=4,            # VGGT default for the plain gradient loss
        **kwargs,
    ):
        super().__init__()
        if gradient_loss_fn not in (None, "grad"):
            raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")

        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.use_gradient_loss = gradient_loss_fn is not None
        self.valid_range = valid_range
        self.min_valid_points = min_valid_points
        self.gradient_scales = gradient_scales

    def forward(self, predictions, gt_depths, masks):
        pred = predictions["depth"]
        if pred.ndim == 5:
            pred = pred.squeeze(-1)
        conf = predictions["depth_conf"]
        gt = gt_depths
        mask = masks.bool()

        if mask.sum() < self.min_valid_points:
            zero = (0.0 * pred).mean()
            return {"loss_depth": zero, "loss_conf_depth": zero, "loss_reg_depth": zero, "loss_grad_depth": zero}

        loss_reg = (pred[mask] - gt[mask]).abs()
        confidence = conf[mask]
        loss_conf = self.gamma * loss_reg * confidence - self.alpha * torch.log(confidence)

        loss_conf = filter_by_quantile(loss_conf, self.valid_range).mean()
        loss_reg = filter_by_quantile(loss_reg, self.valid_range).mean()

        if self.use_gradient_loss:
            B, S, H, W = pred.shape
            loss_grad = depth_gradient_loss_multi_scale(
                pred.reshape(B * S, H, W, 1),
                gt.reshape(B * S, H, W, 1),
                mask.reshape(B * S, H, W),
                scales=self.gradient_scales,
            )
        else:
            loss_grad = (0.0 * pred).mean()

        loss_depth = (loss_conf + loss_reg + loss_grad) * self.weight
        return {
            "loss_depth": loss_depth,
            "loss_conf_depth": loss_conf,
            "loss_reg_depth": loss_reg,
            "loss_grad_depth": loss_grad,
        }


# -----------------------------------------------------------------------------
# Layered depth loss  (DepthLoss per layer + ordering + layer existence)
# -----------------------------------------------------------------------------

# class LayeredDepthLoss(nn.Module):
#     """
#     Loss for LayeredDPTHead (no confidence): N depths per pixel, sorted near -> far (layer 0 = visible surface),
#     plus an existence logit per layer.
 
#     Per layer k, on the pixels where GT layer k exists:
#         reg_k  = |d_k - gt_k|                                           (metres, quantile-filtered, mean)
#         grad_k = multi-scale L1 on spatial gradients of (d_k - gt_k)    (if gradient_loss_fn="grad")
#     Each layer is averaged on its own pixels, then layers are combined with layer_weights, so the
#     rarer back layers are not drowned out by layer 0 (which has far more valid pixels).
 
#     order = mean relu(d_k - d_{k+1} + margin)       on pixels where GT layers k and k+1 both exist (metres)
#     exist = BCE(exist_logit_k, [GT layer k exists])  on pixels where GT layer 0 exists (pixels with no
#             valid geometry at all carry no label)
 
#     loss_layered_depth = weight * ( sum_k w_k (reg_k + grad_weight * grad_k) / sum_k w_k
#                                     + order_weight * order + exist_weight * exist )
 
#     Predictions:
#         layered_depth:       (B, S, N, H, W)  > 0
#         layered_depth_exist: (B, S, N, H, W) logits, or None / missing (then no existence term)
#     Ground truth:
#         gt_layers:      (B, S, L, H, W), L >= N, sorted near -> far, same depth type as the head (z-depth)
#         gt_layer_masks: (B, S, L, H, W) bool, True where layer k exists (layers stack front to back:
#                         if layer k+1 exists, layer k exists)
#     """
 
#     def __init__(
#         self,
#         weight=1.0,
#         gradient_loss_fn=None,        # "grad" or None
#         grad_weight=1.0,
#         valid_range=0.98,             # quantile filter on the per-pixel errors (1.0 = keep all)
#         gradient_scales=4,
#         layer_weights=None,           # list of N floats; default all 1
#         order_weight=1.0,
#         order_margin=0.05,            # metres; 0 = only penalise d_{k+1} < d_k
#         exist_weight=0.1,
#         **kwargs,                     # gamma / alpha / depth_loss from older configs: ignored
#     ):
#         super().__init__()
#         if gradient_loss_fn not in (None, "grad"):
#             raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")
#         self.weight = weight
#         self.use_gradient_loss = gradient_loss_fn is not None
#         self.grad_weight = grad_weight
#         self.valid_range = valid_range
#         self.gradient_scales = gradient_scales
#         self.layer_weights = layer_weights
#         self.order_weight = order_weight
#         self.order_margin = order_margin
#         self.exist_weight = exist_weight
 
#     def forward(self, predictions, gt_layers, gt_layer_masks):
#         pred = predictions["layered_depth"].float()
#         exist = predictions.get("layered_depth_exist")
#         B, S, N, H, W = pred.shape
#         if gt_layers.shape[2] < N:
#             raise ValueError(f"head predicts {N} layers but GT has only {gt_layers.shape[2]}")
#         gt = gt_layers[:, :, :N].float()
#         mask = gt_layer_masks[:, :, :N].bool() & (gt > 0)
#         lw = self.layer_weights or [1.0] * N
#         zero = (0.0 * pred).mean()
 
#         out = {}
#         total, wsum = zero, 0.0
#         for k in range(N):
#             m = mask[:, :, k]
#             if not m.any():                       # no pixel has a k-th surface in this batch
#                 out[f"loss_depth_L{k}"] = zero.detach()
#                 continue
#             d, g = pred[:, :, k], gt[:, :, k]
#             reg = filter_by_quantile((d[m] - g[m]).abs(), self.valid_range).mean()
#             if self.use_gradient_loss:
#                 grad = depth_gradient_loss_multi_scale(
#                     d.reshape(B * S, H, W, 1), g.reshape(B * S, H, W, 1), m.reshape(B * S, H, W),
#                     scales=self.gradient_scales)
#             else:
#                 grad = zero
#             layer_loss = reg + self.grad_weight * grad
#             out[f"loss_depth_L{k}"] = layer_loss.detach()
#             total = total + lw[k] * layer_loss
#             wsum += lw[k]
#         loss_layers = total / wsum if wsum > 0 else zero
 
#         # ordering (metres): consecutive predicted layers must stay sorted where both GT layers exist
#         both = mask[:, :, :-1] & mask[:, :, 1:]
#         if N > 1 and both.any():
#             loss_order = F.relu(pred[:, :, :-1] - pred[:, :, 1:] + self.order_margin)[both].mean()
#         else:
#             loss_order = zero
 
#         # existence of each layer, on pixels with any valid geometry
#         valid_px = mask[:, :, :1].expand_as(mask)
#         if exist is not None and valid_px.any():
#             loss_exist = F.binary_cross_entropy_with_logits(
#                 exist[valid_px].float(), mask[valid_px].float())
#         else:
#             loss_exist = zero
 
#         loss = (loss_layers + self.order_weight * loss_order + self.exist_weight * loss_exist) * self.weight
#         out.update({
#             "objective": loss,
#             "loss_layers": loss_layers.detach(),
#             "loss_order": loss_order.detach(),
#             "loss_exist": loss_exist.detach(),
#         })
#         return out



# class LayeredDepthLoss(nn.Module):
#     """
#     Loss for LayeredDPTHead (no confidence): N depths per pixel, sorted near -> far (layer 0 = visible surface),
#     plus an existence logit per layer.

#     Per layer k, on the pixels where GT layer k exists:
#         l1_k    = mean |d_k - gt_k|                                   (metres)
#         grad_k  = multi-scale L1 on spatial gradients of (d_k - gt_k)  (if gradient_loss_fn="grad")
#         layer_k = l1_k + grad_weight * grad_k
#     Each layer is averaged on its own pixels, then layers are combined with layer_weights, so the
#     rarer back layers are not drowned out by layer 0.

#     order = mean relu(d_k - d_{k+1} + margin)       on pixels where GT layers k and k+1 both exist (metres)
#     exist = BCE(exist_logit_k, [GT layer k exists])  on pixels where GT layer 0 exists

#     loss = weight * ( sum_k w_k layer_k / sum_k w_k + order_weight * order + exist_weight * exist )

#     Predictions:
#         layered_depth:       (B, S, N, H, W)  > 0
#         layered_depth_exist: (B, S, N, H, W) logits, or None / missing (then no existence term)
#     Ground truth:
#         gt_layers:      (B, S, L, H, W), L >= N, sorted near -> far, same depth type as the head (z-depth)
#         gt_layer_masks: (B, S, L, H, W) bool, True where layer k exists
#     """

#     def __init__(
#         self,
#         weight=1.0,
#         gradient_loss_fn="grad",        # "grad" or None
#         grad_weight=0.10,
#         gradient_scales=4,
#         layer_weights=None,           # list of N floats; default all 1
#         order_weight=0.05,
#         order_margin=0.0,             # metres; 0 = only penalise d_{k+1} < d_k
#         exist_weight=0.0,             # 0 = existence head gets no gradient (BCE still logged)
#         **kwargs,                     # gamma / alpha / depth_loss / valid_range from older configs: ignored
#     ):
#         super().__init__()
#         if gradient_loss_fn not in (None, "grad"):
#             raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")
#         self.weight = weight
#         self.use_gradient_loss = gradient_loss_fn is not None
#         self.grad_weight = grad_weight
#         self.gradient_scales = gradient_scales
#         self.layer_weights = layer_weights
#         self.order_weight = order_weight
#         self.order_margin = order_margin
#         self.exist_weight = exist_weight

#     def forward(self, predictions, gt_layers, gt_layer_masks):
#         pred = predictions["layered_depth"].float()
#         exist = predictions.get("layered_depth_exist")
#         B, S, N, H, W = pred.shape
#         if gt_layers.shape[2] < N:
#             raise ValueError(f"head predicts {N} layers but GT has only {gt_layers.shape[2]}")
#         gt = gt_layers[:, :, :N].float()
#         mask = gt_layer_masks[:, :, :N].bool() & (gt > 0)
#         lw = self.layer_weights or [1.0] * N
#         zero = (0.0 * pred).mean()

#         out = {}
#         total, wsum = zero, 0.0
#         for k in range(N):
#             m = mask[:, :, k]
#             d, g = pred[:, :, k], gt[:, :, k]

#             # ---- per-layer logging stats (no grad) ----
#             with torch.no_grad():
#                 cnt = m.reshape(B * S, -1).sum(dim=1).float()      # valid pixels per image
#                 has = cnt > 0
#                 out[f"valid_pct_L{k}"] = m.float().mean() * 100.0  # % of all pixels with a valid layer-k GT
#                 out[f"n_imgs_valid_L{k}"] = has.sum().float()      # images with >= 1 valid pixel
#                 out[f"avg_valid_px_L{k}"] = cnt.mean()             # avg valid pixels per image (all images)
#                 if m.any():
#                     err = (d[m] - g[m]).abs()
#                     out[f"mae_L{k}"] = err.mean()                  # metres
#                     out[f"absrel_L{k}"] = (err / g[m]).mean()
#                 else:
#                     out[f"mae_L{k}"] = zero.detach()
#                     out[f"absrel_L{k}"] = zero.detach()

#             if not m.any():                       # no pixel has a k-th surface in this batch
#                 out[f"loss_depth_L{k}"] = zero.detach()
#                 continue
#             l1 = (d[m] - g[m]).abs().mean()
#             if self.use_gradient_loss:
#                 grad = depth_gradient_loss_multi_scale(
#                     d.reshape(B * S, H, W, 1), g.reshape(B * S, H, W, 1),
#                     m.reshape(B * S, H, W), scales=self.gradient_scales)
#             else:
#                 grad = zero
#             layer_loss = l1 + self.grad_weight * grad
#             out[f"loss_depth_L{k}"] = layer_loss.detach()
#             total = total + lw[k] * layer_loss
#             wsum += lw[k]
#         loss_layers = total / wsum if wsum > 0 else zero

#         # ordering (metres): consecutive predicted layers must stay sorted where both GT layers exist
#         both = mask[:, :, :-1] & mask[:, :, 1:]
#         if N > 1 and both.any():
#             loss_order = F.relu(pred[:, :, :-1] - pred[:, :, 1:] + self.order_margin)[both].mean()
#         else:
#             loss_order = zero

#         # existence of each layer, on pixels with any valid geometry
#         valid_px = mask[:, :, :1].expand_as(mask)
#         if exist is not None and valid_px.any():
#             loss_exist = F.binary_cross_entropy_with_logits(
#                 exist[valid_px].float(), mask[valid_px].float())
#         else:
#             loss_exist = zero

#         loss = (loss_layers + self.order_weight * loss_order + self.exist_weight * loss_exist) * self.weight
#         out.update({
#             "objective": loss,
#             "loss_layers": loss_layers.detach(),
#             "loss_order": loss_order.detach(),
#             "loss_exist": loss_exist.detach(),
#         })
#         return out
# # -----------------------------------------------------------------------------
# Spatial gradient loss used by PointmapLoss (unchanged, supports confidence)
# # -----------------------------------------------------------------------------

# class LayeredDepthLoss(nn.Module):
#     """
#     Loss for LayeredDPTHead (no confidence): N depths per pixel, sorted near -> far (layer 0 = visible surface),
#     plus an existence logit per layer.

#     Per layer k, on the pixels where GT layer k exists:
#         l1_k    = mean |d_k - gt_k|            (metres)   or   mean |log d_k - log gt_k|   (log_space=True)
#         grad_k  = multi-scale L1 on spatial gradients of (d_k - gt_k)  (if gradient_loss_fn="grad")
#         layer_k = l1_k + grad_weight * grad_k
#     Layers are combined with layer_weights (default [3, 1, 1, 1] = loss_L0 + mean(L1..L3), normalised),
#     so layer 0 trains like the plain depth probe and the back layers still get half the weight.

#     order = mean relu(d_k - d_{k+1} + margin)       on pixels where GT layers k and k+1 both exist (metres)
#     exist = BCE(exist_logit_k, [GT layer k exists])  on pixels where GT layer 0 exists,
#             pos_weight per layer (= (1 - p_k) / p_k from the layer frequencies) if exist_pos_weight is given

#     loss = weight * ( sum_k w_k layer_k / sum_k w_k + order_weight * order + exist_weight * exist )

#     Extra logging (no grad):
#         absrel_aligned_L{k}: AbsRel after scaling every image by the median gt/pred ratio of its LAYER 0
#                              (same scale for all layers) -> separates "wrong metric scale" from "wrong geometry"
#         absrel_gapbase_L{k}: AbsRel of the trivial predictor gt_0 + median(gt_k - gt_0) over the batch (k >= 1)
#                              -> the floor a hidden layer has to beat to show the features know what's behind
#     """

#     def __init__(
#         self,
#         weight=1.0,
#         gradient_loss_fn="grad",        # "grad" or None
#         grad_weight=0.10,
#         gradient_scales=4,
#         layer_weights=None,           # list of N floats; default [3, 1, ..., 1]
#         log_space=True,               # L1 on log-depth (scale-balanced, matches the exp activation)
#         order_weight=0.05,
#         order_margin=0.0,             # metres; 0 = only penalise d_{k+1} < d_k
#         exist_weight=0.0,             # 0 = existence head gets no gradient (BCE still logged)
#         exist_pos_weight=None,        # list of N floats, e.g. [0.07, 3.2, 9.2, 39]
#         **kwargs,                     # gamma / alpha / depth_loss / valid_range from older configs: ignored
#     ):
#         super().__init__()
#         if gradient_loss_fn not in (None, "grad"):
#             raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")
#         self.weight = weight
#         self.use_gradient_loss = gradient_loss_fn is not None
#         self.grad_weight = grad_weight
#         self.gradient_scales = gradient_scales
#         self.layer_weights = layer_weights
#         self.log_space = log_space
#         self.order_weight = order_weight
#         self.order_margin = order_margin
#         self.exist_weight = exist_weight
#         self.exist_pos_weight = exist_pos_weight

#     @staticmethod
#     def _median_ratio(num, den, m):
#         """Per-image median of num/den over mask m; inputs (B*S, P). Images with no valid pixel -> 1."""
#         r = torch.where(m, num / den.clamp_min(1e-6), torch.full_like(num, float("nan")))
#         med = torch.nanmedian(r, dim=1).values
#         return torch.where(torch.isfinite(med), med, torch.ones_like(med))

#     def forward(self, predictions, gt_layers, gt_layer_masks):
#         pred = predictions["layered_depth"].float()
#         exist = predictions.get("layered_depth_exist")
#         B, S, N, H, W = pred.shape
#         if gt_layers.shape[2] < N:
#             raise ValueError(f"head predicts {N} layers but GT has only {gt_layers.shape[2]}")
#         gt = gt_layers[:, :, :N].float()
#         mask = gt_layer_masks[:, :, :N].bool() & (gt > 0)
#         lw = self.layer_weights or ([3.0] + [1.0] * (N - 1))
#         zero = (0.0 * pred).mean()

#         # per-image scale from layer 0 (logging only)
#         with torch.no_grad():
#             d0f, g0f, m0f = (t[:, :, 0].reshape(B * S, -1) for t in (pred, gt, mask))
#             scale = self._median_ratio(g0f, d0f, m0f).view(B, S, 1, 1)

#         out = {}
#         total, wsum = zero, 0.0
#         for k in range(N):
#             m = mask[:, :, k]
#             d, g = pred[:, :, k], gt[:, :, k]

#             # ---- per-layer logging stats (no grad) ----
#             with torch.no_grad():
#                 cnt = m.reshape(B * S, -1).sum(dim=1).float()
#                 has = cnt > 0
#                 out[f"valid_pct_L{k}"] = m.float().mean() * 100.0
#                 out[f"n_imgs_valid_L{k}"] = has.sum().float()
#                 out[f"avg_valid_px_L{k}"] = cnt.mean()
#                 if m.any():
#                     err = (d[m] - g[m]).abs()
#                     out[f"mae_L{k}"] = err.mean()
#                     out[f"absrel_L{k}"] = (err / g[m]).mean()
#                     out[f"absrel_aligned_L{k}"] = ((d * scale)[m] - g[m]).abs().div(g[m]).mean()
#                     if k > 0:
#                         both = m & mask[:, :, 0]
#                         if both.any():
#                             gap = (g[both] - gt[:, :, 0][both]).median()
#                             base = gt[:, :, 0] + gap
#                             out[f"absrel_gapbase_L{k}"] = (base[both] - g[both]).abs().div(g[both]).mean()
#                 else:
#                     for name in ("mae", "absrel", "absrel_aligned"):
#                         out[f"{name}_L{k}"] = zero.detach()

#             if not m.any():
#                 out[f"loss_depth_L{k}"] = zero.detach()
#                 continue
#             if self.log_space:
#                 l1 = (d[m].clamp_min(1e-3).log() - g[m].log()).abs().mean()
#             else:
#                 l1 = (d[m] - g[m]).abs().mean()
#             if self.use_gradient_loss:
#                 grad = depth_gradient_loss_multi_scale(
#                     d.reshape(B * S, H, W, 1), g.reshape(B * S, H, W, 1),
#                     m.reshape(B * S, H, W), scales=self.gradient_scales)
#             else:
#                 grad = zero
#             layer_loss = l1 + self.grad_weight * grad
#             out[f"loss_depth_L{k}"] = layer_loss.detach()
#             total = total + lw[k] * layer_loss
#             wsum += lw[k]
#         loss_layers = total / wsum if wsum > 0 else zero

#         both = mask[:, :, :-1] & mask[:, :, 1:]
#         if N > 1 and both.any():
#             loss_order = F.relu(pred[:, :, :-1] - pred[:, :, 1:] + self.order_margin)[both].mean()
#         else:
#             loss_order = zero

#         valid_px = mask[:, :, :1].expand_as(mask)
#         if exist is not None and valid_px.any():
#             pw = None
#             if self.exist_pos_weight is not None:
#                 pw = pred.new_tensor(self.exist_pos_weight[:N]).view(1, 1, N, 1, 1).expand_as(mask)[valid_px]
#             loss_exist = F.binary_cross_entropy_with_logits(
#                 exist[valid_px].float(), mask[valid_px].float(), pos_weight=pw)
#         else:
#             loss_exist = zero

#         loss = (loss_layers + self.order_weight * loss_order + self.exist_weight * loss_exist) * self.weight
#         out.update({
#             "objective": loss,
#             "loss_layers": loss_layers.detach(),
#             "loss_order": loss_order.detach(),
#             "loss_exist": loss_exist.detach(),
#         })
#         return out


# -----------------------------------------------------------------------------

class LayeredDepthLoss(nn.Module):
    """
    Loss for LayeredDPTHead (no confidence): N depths per pixel, sorted near -> far (layer 0 = visible surface),
    plus an existence logit per layer. Head unchanged: every fix is in the loss (parameter-free, probe-fair).

    Valid GT: gt_layer_masks & min_depth < gt < max_depth (default 0.1 - 16 m, as the LMDB agreement check).

    Per layer k, on the pixels where GT layer k exists, log residual  r_k = log d_k - log gt_k :
        scale_invariant=True: r_k <- r_k - s,  s = per-image median of r_0 over layer-0 pixels (detached),
                              the SAME s for every layer -> hidden layers are judged relative to the visible
                              surface, and the probe is not asked for metric scale (which SSL features need not
                              know). anchor_weight * mean|r_0| (unshifted) keeps the absolute scale from drifting.
        l1_k    = mean |r_k|                     (log_space=True)   or   mean |d_k - gt_k|  (metres)
        grad_k  = multi-scale gradient loss, ONLY on the layers in grad_layers (default layer 0: on the sparse
                  back layers there are almost no valid neighbours, so it is mostly noise)
        layer_k = l1_k + grad_weight * grad_k
    Layers are combined with layer_weights (default [3, 1, ..., 1] = loss_L0 + mean(L1..L(N-1)), normalised).

    gap (k >= 1)  = mean | (d_k - d_{k-1}) - (gt_k - gt_{k-1}) | / gt_k   on pixels where GT layers k-1, k exist
                    (pred scaled by exp(-s) in scale-invariant mode): supervises "how far behind" directly, so a
                    hidden layer is not penalised twice for an error it inherits from the layer in front.
    order         = mean relu(d_k + order_margin - d_{k+1})   (order_margin default 16/256 m = the LDI's minimum
                    layer separation)
    exist         = BCE(exist_logit_k, [GT layer k exists]) on pixels where GT layer 0 exists (pos_weight optional)

    loss = weight * ( sum_k w_k layer_k / sum_k w_k + anchor_weight * anchor + gap_weight * gap
                      + order_weight * order + exist_weight * exist )

    Extra logging (no grad):
        absrel_aligned_L{k}      AbsRel after scaling each image by the median gt/pred ratio of its layer 0
        absrel_gapbase_L{k}      AbsRel of gt_0 + median(gt_k - gt_0)                (oracle: GT layer 0)
        absrel_gapbase_pred_L{k} AbsRel of aligned pred_0 + median(gt_k - gt_0)
                                 -> compare absrel_aligned_L{k} to this: the floor a hidden layer must beat
        gap_q25/q50/q75_L{k}     quartiles of gt_k - gt_0 (m); ~0.1-0.3 m = mostly back faces of objects
    """

    def __init__(
        self,
        weight=1.0,
        gradient_loss_fn="grad",        # "grad" or None
        grad_weight=0.10,
        gradient_scales=4,
        grad_layers=(0,),             # layers that get the gradient loss
        layer_weights=None,           # list of N floats; default [3, 1, ..., 1]
        log_space=True,               # L1 on log-depth (scale-balanced, matches the exp activation)
        scale_invariant=True,         # per-image log shift from layer 0, shared by all layers (needs log_space)
        anchor_weight=0.1,            # unshifted log L1 on layer 0, keeps the absolute scale (scale_invariant only)
        gap_weight=1.0,               # loss on the predicted vs GT gap between consecutive layers
        min_depth=0.1,                # metres
        max_depth=16.0,               # metres
        order_weight=0.05,
        order_margin=16 / 256,        # metres; the LDI's minimum layer separation
        exist_weight=0.0,             # 0 = existence head gets no gradient (BCE still logged)
        exist_pos_weight=None,        # list of N floats, e.g. [0.07, 3.2, 9.2, 39]
        **kwargs,                     # gamma / alpha / depth_loss / valid_range from older configs: ignored
    ):
        super().__init__()
        if gradient_loss_fn not in (None, "grad"):
            raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")
        if scale_invariant and not log_space:
            raise ValueError("scale_invariant needs log_space=True")
        self.weight = weight
        self.use_gradient_loss = gradient_loss_fn is not None
        self.grad_weight = grad_weight
        self.gradient_scales = gradient_scales
        self.grad_layers = set(grad_layers)
        self.layer_weights = layer_weights
        self.log_space = log_space
        self.scale_invariant = scale_invariant
        self.anchor_weight = anchor_weight
        self.gap_weight = gap_weight
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.order_weight = order_weight
        self.order_margin = order_margin
        self.exist_weight = exist_weight
        self.exist_pos_weight = exist_pos_weight

    @staticmethod
    def _masked_median(x, m):
        """Per-image median of x over mask m; inputs (B*S, P). Images with no valid pixel -> nan."""
        return torch.nanmedian(torch.where(m, x, torch.full_like(x, float("nan"))), dim=1).values

    def forward(self, predictions, gt_layers, gt_layer_masks):
        pred = predictions["layered_depth"].float()
        exist = predictions.get("layered_depth_exist")
        B, S, N, H, W = pred.shape
        if gt_layers.shape[2] < N:
            raise ValueError(f"head predicts {N} layers but GT has only {gt_layers.shape[2]}")
        gt = gt_layers[:, :, :N].float()
        mask = gt_layer_masks[:, :, :N].bool() & (gt > self.min_depth) & (gt < self.max_depth)
        lw = self.layer_weights or ([3.0] + [1.0] * (N - 1))
        zero = (0.0 * pred).mean()

        log_pred = pred.clamp_min(1e-3).log()
        log_gt = gt.clamp_min(1e-3).log()
        m0f = mask[:, :, 0].reshape(B * S, -1)

        with torch.no_grad():
            # per-image log shift s from layer 0: pred * exp(-s) has the GT's median scale
            s = self._masked_median((log_pred - log_gt)[:, :, 0].reshape(B * S, -1), m0f)
            s = torch.where(torch.isfinite(s), s, torch.zeros_like(s)).view(B, S, 1, 1)
            scale = torch.exp(-s)                                     # for logging: aligned pred = pred * scale

        shift = s if self.scale_invariant else torch.zeros_like(s)    # (B, S, 1, 1), detached
        out = {}
        total, wsum = zero, 0.0
        for k in range(N):
            m = mask[:, :, k]
            d, g = pred[:, :, k], gt[:, :, k]

            # ---- per-layer logging stats (no grad) ----
            with torch.no_grad():
                cnt = m.reshape(B * S, -1).sum(dim=1).float()
                has = cnt > 0
                out[f"valid_pct_L{k}"] = m.float().mean() * 100.0
                out[f"n_imgs_valid_L{k}"] = has.sum().float()
                out[f"avg_valid_px_L{k}"] = cnt.mean()
                if m.any():
                    err = (d[m] - g[m]).abs()
                    out[f"mae_L{k}"] = err.mean()
                    out[f"absrel_L{k}"] = (err / g[m]).mean()
                    out[f"absrel_aligned_L{k}"] = ((d * scale)[m] - g[m]).abs().div(g[m]).mean()
                    if k > 0:
                        both = m & mask[:, :, 0]
                        if both.any():
                            gaps = g[both] - gt[:, :, 0][both]
                            gap = gaps.median()
                            base = gt[:, :, 0] + gap
                            out[f"absrel_gapbase_L{k}"] = (base[both] - g[both]).abs().div(g[both]).mean()
                            base_p = pred[:, :, 0] * scale + gap
                            out[f"absrel_gapbase_pred_L{k}"] = (base_p[both] - g[both]).abs().div(g[both]).mean()
                            q = torch.quantile(gaps[:100000], gaps.new_tensor([0.25, 0.5, 0.75]))
                            out[f"gap_q25_L{k}"], out[f"gap_q50_L{k}"], out[f"gap_q75_L{k}"] = q[0], q[1], q[2]
                else:
                    for name in ("mae", "absrel", "absrel_aligned"):
                        out[f"{name}_L{k}"] = zero.detach()

            if not m.any():
                out[f"loss_depth_L{k}"] = zero.detach()
                continue
            if self.log_space:
                r = log_pred[:, :, k] - log_gt[:, :, k] - shift              # shift (B, S, 1, 1)
                l1 = r[m].abs().mean()
            else:
                l1 = (d[m] - g[m]).abs().mean()
            if self.use_gradient_loss and k in self.grad_layers:
                grad = depth_gradient_loss_multi_scale(
                    d.reshape(B * S, H, W, 1), g.reshape(B * S, H, W, 1),
                    m.reshape(B * S, H, W), scales=self.gradient_scales)
            else:
                grad = zero
            layer_loss = l1 + self.grad_weight * grad
            out[f"loss_depth_L{k}"] = layer_loss.detach()
            total = total + lw[k] * layer_loss
            wsum += lw[k]
        loss_layers = total / wsum if wsum > 0 else zero

        # anchor: unshifted log L1 on layer 0 (only matters in scale-invariant mode)
        m0 = mask[:, :, 0]
        if self.scale_invariant and m0.any():
            loss_anchor = (log_pred[:, :, 0] - log_gt[:, :, 0])[m0].abs().mean()
        else:
            loss_anchor = zero

        # gap between consecutive layers, relative to the GT depth of the back layer
        both = mask[:, :, :-1] & mask[:, :, 1:]
        if N > 1 and both.any():
            p_al = pred * torch.exp(-shift).unsqueeze(2)                  # (B, S, N, H, W)
            gap_p = p_al[:, :, 1:] - p_al[:, :, :-1]
            gap_g = gt[:, :, 1:] - gt[:, :, :-1]
            loss_gap = ((gap_p - gap_g).abs() / gt[:, :, 1:].clamp_min(self.min_depth))[both].mean()
            loss_order = F.relu(pred[:, :, :-1] + self.order_margin - pred[:, :, 1:])[both].mean()
        else:
            loss_gap = loss_order = zero

        valid_px = mask[:, :, :1].expand_as(mask)
        if exist is not None and valid_px.any():
            pw = None
            if self.exist_pos_weight is not None:
                pw = pred.new_tensor(self.exist_pos_weight[:N]).view(1, 1, N, 1, 1).expand_as(mask)[valid_px]
            loss_exist = F.binary_cross_entropy_with_logits(
                exist[valid_px].float(), mask[valid_px].float(), pos_weight=pw)
        else:
            loss_exist = zero

        loss = (loss_layers + self.anchor_weight * loss_anchor + self.gap_weight * loss_gap
                + self.order_weight * loss_order + self.exist_weight * loss_exist) * self.weight
        out.update({
            "objective": loss,
            "loss_layers": loss_layers.detach(),
            "loss_anchor": loss_anchor.detach(),
            "loss_gap": loss_gap.detach(),
            "loss_order": loss_order.detach(),
            "loss_exist": loss_exist.detach(),
        })
        return out

def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """L1 loss on spatial gradients (only pairs of valid pixels count)."""
    mask = mask.bool()
    diff = torch.where(mask, prediction - target, torch.zeros_like(prediction))

    mask_x = mask[:, :, 1:] & mask[:, :, :-1]
    mask_y = mask[:, 1:, :] & mask[:, :-1, :]

    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs().clamp(max=100)
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs().clamp(max=100)

    if conf is not None:
        conf_x = torch.where(mask_x, conf[:, :, 1:], torch.ones_like(conf[:, :, 1:]))
        conf_y = torch.where(mask_y, conf[:, 1:, :], torch.ones_like(conf[:, 1:, :]))
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    grad_x = torch.where(mask_x, grad_x, torch.zeros_like(grad_x))
    grad_y = torch.where(mask_y, grad_y, torch.zeros_like(grad_y))

    loss = (grad_x.sum() + grad_y.sum()) / mask.sum().clamp(min=1)
    return check_and_fix_inf_nan(loss, "loss_grad")


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=3, gradient_loss_fn=gradient_loss, conf=None):
    """Apply a spatial loss at multiple resolutions."""
    total = prediction.new_zeros(())
    for scale in range(scales):
        step = 2 ** scale
        conf_s = conf[:, ::step, ::step] if conf is not None else None
        total = total + gradient_loss_fn(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                         mask[:, ::step, ::step], conf=conf_s)
    return total / scales


# -----------------------------------------------------------------------------
# Pointmap loss
# -----------------------------------------------------------------------------


class PointmapLoss(nn.Module):
    """
    3D pointmap regression loss with confidence and optional normal/gradient loss.

    Expected predictions:
        world_points:      (B, S, H, W, 3)
        world_points_conf: (B, S, H, W)

    Expected batch:
        world_points: (B, S, H, W, 3)
        point_masks:  (B, S, H, W)
    """

    def __init__(
        self,
        weight=1.0,
        gamma=1.0,
        alpha=0.2,
        gradient_loss_fn=None,
        valid_range=-1,
        min_valid_points=100,
        gradient_scales=3,
        **kwargs,
    ):
        super().__init__()

        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
        self.min_valid_points = min_valid_points
        self.gradient_scales = gradient_scales

    def forward(self, predictions, batch):
        pred = predictions["world_points"]
        conf = predictions["world_points_conf"]

        gt = check_and_fix_inf_nan(
            batch["world_points"],
            "gt_points",
        )

        mask = batch["point_masks"].bool()

        if mask.sum() < self.min_valid_points:
            zero = pred.mean() * 0.0

            return {
                "loss_pointmap": zero,
                "loss_conf_point": zero,
                "loss_reg_point": zero,
                "loss_grad_point": zero,
            }

        loss_reg = self._regression_loss(pred, gt, mask)
        loss_conf = self._confidence_loss(pred, gt, mask, conf)
        loss_grad = self._gradient_loss(pred, gt, mask, conf)

        loss_pointmap = (
            loss_conf
            + loss_reg
            + loss_grad
        ) * self.weight

        return {
            "loss_pointmap": loss_pointmap,
            "loss_conf_point": loss_conf,
            "loss_reg_point": loss_reg,
            "loss_grad_point": loss_grad,
        }

    def _regression_loss(self, pred, gt, mask):
        error = torch.norm(gt[mask] - pred[mask], dim=-1)
        error = check_and_fix_inf_nan(error, "loss_reg_point")

        if self.valid_range > 0:
            error = self._filter_outliers(error)

        if error.numel() == 0:
            return pred.mean() * 0.0

        return error.mean()

    def _confidence_loss(self, pred, gt, mask, conf):
        error = torch.norm(gt[mask] - pred[mask], dim=-1)
        confidence = conf[mask]

        loss = (
            self.gamma * error * confidence
            - self.alpha * torch.log(confidence)
        )

        loss = check_and_fix_inf_nan(loss, "loss_conf_point")

        if self.valid_range > 0:
            loss = self._filter_outliers(loss)

        if loss.numel() == 0:
            return pred.mean() * 0.0

        return loss.mean()

    def _gradient_loss(self, pred, gt, mask, conf):
        if self.gradient_loss_fn is None:
            return pred.mean() * 0.0

        B, S, H, W, C = pred.shape

        pred = pred.reshape(B * S, H, W, C)
        gt = gt.reshape(B * S, H, W, C)
        mask = mask.reshape(B * S, H, W)
        conf = conf.reshape(B * S, H, W)

        if "conf" in self.gradient_loss_fn:
            gradient_conf = conf
        else:
            gradient_conf = None

        if "normal" in self.gradient_loss_fn:
            fn = normal_loss
        elif "grad" in self.gradient_loss_fn:
            fn = gradient_loss
        else:
            raise ValueError(
                f"Unknown gradient loss: {self.gradient_loss_fn}"
            )

        return gradient_loss_multi_scale_wrapper(
            pred,
            gt,
            mask,
            scales=self.gradient_scales,
            gradient_loss_fn=fn,
            conf=gradient_conf,
        )

    @staticmethod
    def _filter_outliers(loss, valid_range):
        if loss.numel() <= 1000:
            return loss

        loss = loss.clamp(max=100)

        threshold = torch_quantile(
            loss.detach(),
            valid_range,
        )

        threshold = min(threshold, 100)
        keep = loss < threshold

        if keep.sum() > 1000:
            return loss[keep]

        return loss


# -----------------------------------------------------------------------------
# Shared spatial losses
# -----------------------------------------------------------------------------



def normal_loss(
    prediction,
    target,
    mask,
    cos_eps=1e-8,
    conf=None,
    gamma=1.0,
    alpha=0.2,
):
    """Surface-normal consistency loss for point maps."""

    pred_normals, pred_valid = point_map_to_normal(
        prediction,
        mask,
        eps=cos_eps,
    )

    gt_normals, gt_valid = point_map_to_normal(
        target,
        mask,
        eps=cos_eps,
    )

    valid = pred_valid & gt_valid

    if valid.sum() < 10:
        return prediction.mean() * 0.0

    pred_normals = pred_normals[valid]
    gt_normals = gt_normals[valid]

    dot = torch.sum(pred_normals * gt_normals, dim=-1)
    dot = dot.clamp(-1 + cos_eps, 1 - cos_eps)

    loss = 1.0 - dot

    if conf is not None:
        conf = conf[None].expand(4, -1, -1, -1)
        conf = conf[valid]

        loss = (
            gamma * loss * conf
            - alpha * torch.log(conf)
        )

    return check_and_fix_inf_nan(
        loss,
        "normal_loss",
    ).mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """Compute four local surface normals from a point map."""

    with torch.cuda.amp.autocast(enabled=False):
        padded_mask = F.pad(
            mask,
            (1, 1, 1, 1),
            value=False,
        )

        points = point_map.permute(0, 3, 1, 2)
        points = F.pad(
            points,
            (1, 1, 1, 1),
            value=0,
        )
        points = points.permute(0, 2, 3, 1)

        center = points[:, 1:-1, 1:-1]
        up = points[:, :-2, 1:-1]
        left = points[:, 1:-1, :-2]
        down = points[:, 2:, 1:-1]
        right = points[:, 1:-1, 2:]

        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        normals = torch.stack(
            [
                torch.cross(up_dir, left_dir, dim=-1),
                torch.cross(left_dir, down_dir, dim=-1),
                torch.cross(down_dir, right_dir, dim=-1),
                torch.cross(right_dir, up_dir, dim=-1),
            ],
            dim=0,
        )

        valid = torch.stack(
            [
                padded_mask[:, :-2, 1:-1]
                & padded_mask[:, 1:-1, 1:-1]
                & padded_mask[:, 1:-1, :-2],

                padded_mask[:, 1:-1, :-2]
                & padded_mask[:, 1:-1, 1:-1]
                & padded_mask[:, 2:, 1:-1],

                padded_mask[:, 2:, 1:-1]
                & padded_mask[:, 1:-1, 1:-1]
                & padded_mask[:, 1:-1, 2:],

                padded_mask[:, 1:-1, 2:]
                & padded_mask[:, 1:-1, 1:-1]
                & padded_mask[:, :-2, 1:-1],
            ],
            dim=0,
        )

        normals = F.normalize(
            normals,
            p=2,
            dim=-1,
            eps=eps,
        )

    return normals, valid


# -----------------------------------------------------------------------------
# Quantile filtering
# -----------------------------------------------------------------------------


def torch_quantile(
    input,
    q,
    dim=None,
    keepdim=False,
):
    """Efficient scalar quantile using kthvalue."""

    q = float(q)

    if not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")

    if dim is None:
        input = input.reshape(-1)
        dim = 0
        squeeze = True
    else:
        squeeze = False

    k = round(q * (input.shape[dim] - 1)) + 1

    output = torch.kthvalue(
        input,
        k,
        dim=dim,
        keepdim=True,
    ).values

    if keepdim:
        return output

    return output.squeeze(dim)