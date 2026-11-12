import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging



from models.utils.pose_enc import extri_intri_to_pose_encoding

def normalize_scene_by_points(
    extrinsics: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor | None = None,   # unused, kept for call-site compatibility
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ONE scale per sample for depth AND camera translations.

      1. recentre the extrinsics on frame 0 (frame 0 = identity, world = camera-0 frame),
      2. scale = mean of the valid GT depth values over all views of the sample,
      3. divide the translations and the depths by that scale.

    A sample without any valid depth falls back to the mean camera-centre distance from
    camera 0 (or 1 for a single camera), so translations are never divided by ~0.

    Args:
        extrinsics: (B, S, 3, 4) or (B, S, 4, 4) cam-from-world.
        depths:     (B, S, H, W) GT z-depth (0 = invalid).
        masks:      (B, S, H, W) valid-depth mask.
    Returns:
        extrinsics_norm (B, S, 4, 4), depths_norm (B, S, H, W), scale (B,)
    """
    B, S, H, W = depths.shape
    w2c = extrinsics.float()
    if w2c.shape[-2] == 3:
        bottom = torch.zeros_like(w2c[..., :1, :])
        bottom[..., 0, 3] = 1.0
        w2c = torch.cat([w2c, bottom], dim=-2)
    w2c = w2c @ torch.linalg.inv(w2c[:, :1])                       # frame 0 = identity

    valid = masks.bool() & (depths > 0) & torch.isfinite(depths)
    d = depths.float()

    n_valid = valid.sum(dim=(1, 2, 3)).float()
    depth_sum = torch.where(valid, d, torch.zeros_like(d)).sum(dim=(1, 2, 3))
    scale = depth_sum / n_valid.clamp_min(1.0)

    # fallback for samples without valid depth: mean camera-centre distance from camera 0, else 1
    centers = -(w2c[..., :3, :3].transpose(-1, -2) @ w2c[..., :3, 3:]).squeeze(-1)   # (B, S, 3), c_0 = 0
    cam_scale = centers.norm(dim=-1).sum(dim=1) / max(S - 1, 1)
    fallback = torch.where(cam_scale > 1e-6, cam_scale, torch.ones_like(cam_scale))
    scale = torch.where(n_valid > 0, scale, fallback).clamp(min=1e-6, max=1e6)        # (B,)

    w2c_norm = w2c.clone()
    # w2c_norm[..., :3, 3] = w2c[..., :3, 3] / scale.view(B, 1)
    w2c_norm[..., :3, 3] = w2c[..., :3, 3] / scale.view(B, 1, 1)
    depths_norm = torch.where(valid, d / scale.view(B, 1, 1, 1), torch.zeros_like(d))

    return w2c_norm, depths_norm, scale

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
    """Camera + depth losses. GT depth and GT poses are normalised once, with one scale per sample
    (normalize_scene_by_points), so both losses see targets in the same units."""

    def __init__(self, camera=None, depth=None):
        super().__init__()
        self.camera_loss = CameraLoss(**camera) if camera is not None else None
        self.depth_loss = DepthLoss(**depth) if depth is not None else None

    def forward(self, predictions, gt_depths, masks, gt_intrinsics, gt_extrinsics, images):
        with torch.autocast(device_type=gt_depths.device.type, enabled=False):
            gt_extrinsics, gt_depths, scene_scale = normalize_scene_by_points(gt_extrinsics, gt_depths, masks, gt_intrinsics)

        losses = {"scene_scale": scene_scale.mean().detach()}
        objective = 0.0

        if self.camera_loss is not None and "pose_enc_list" in predictions:
            losses.update(self.camera_loss(predictions, gt_extrinsics, gt_intrinsics, images))
            objective = objective + losses["loss_camera"]

        if self.depth_loss is not None and "depth" in predictions:
            losses.update(self.depth_loss(predictions, gt_depths, masks))
            objective = objective + losses["loss_depth"]

        losses["objective"] = objective
        return losses

# -----------------------------------------------------------------------------
# Camera loss
# -----------------------------------------------------------------------------
class CameraLoss(nn.Module):
    def __init__(self, pose_encoding_type="absT_quaR_FoV", **kwargs):   # other old args accepted, unused
        super().__init__()
        self.pose_encoding_type = pose_encoding_type

    def forward(self, predictions, gt_extrinsics, gt_intrinsics, images, gt_depth_masks=None):
        pred_pose_encoding = predictions["pose_enc_list"][-1]
        gt_pose_encoding = extri_intri_to_pose_encoding(
            gt_extrinsics[..., :3, :], gt_intrinsics, images.shape[-2:], pose_encoding_type=self.pose_encoding_type)
        loss = F.l1_loss(pred_pose_encoding, gt_pose_encoding)
        return {"loss_camera": loss}


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
        predictions["depth_conf"]: (B, S, H, W), or None / missing -> no confidence term (loss_conf = 0)
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
        conf = predictions.get("depth_conf")          # None -> no confidence term
        gt = gt_depths
        mask = masks.bool()

        if mask.sum() < self.min_valid_points:
            zero = (0.0 * pred).mean()
            return {"loss_depth": zero, "loss_conf_depth": zero, "loss_reg_depth": zero, "loss_grad_depth": zero}

        err = (pred[mask] - gt[mask]).abs()

        if conf is not None:
            confidence = conf[mask]
            loss_conf = self.gamma * err * confidence - self.alpha * torch.log(confidence)
            loss_conf = filter_by_quantile(loss_conf, self.valid_range).mean()
        else:
            loss_conf = (0.0 * pred).mean()

        loss_reg = filter_by_quantile(err, self.valid_range).mean()

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

class SILogLoss(nn.Module):
    """
    Scale-invariant log loss (Eigen et al.), per image, plus your multi-scale gradient loss
    (depth_gradient_loss_multi_scale, as in DepthLoss) computed on log depth.

        d          = log(pred) - log(gt)                         (valid pixels)
        loss_silog = sqrt( mean(d^2) - lam * mean(d)^2 )         per image, then averaged
        loss_grad  = depth_gradient_loss_multi_scale(log pred, log gt, mask)
        loss_depth = (loss_silog + grad_weight * loss_grad) * weight

    lam = 0   -> plain RMSE in log space (scale matters)
    lam = 1   -> fully scale-invariant (global scale is free)
    lam = 0.5 -> in between (common choice); 0.85 as in AdaBins / BTS

    Same interface as DepthLoss:
        predictions["depth"]:  (B, S, H, W) or (B, S, H, W, 1)   ("depth_conf" ignored)
        gt_depths:             (B, S, H, W)
        masks:                 (B, S, H, W)
    Returns a dict with loss_depth (the objective), loss_silog, loss_grad_depth.
    """

    def __init__(self, lam=0.85, weight=1.0, gradient_loss_fn="grad", grad_weight=0.5, gradient_scales=4,
                 eps=1e-6, **kwargs):
        super().__init__()
        if gradient_loss_fn not in (None, "grad"):
            raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")
        self.lam = lam
        self.weight = weight
        self.use_gradient_loss = gradient_loss_fn is not None and grad_weight > 0
        self.grad_weight = grad_weight
        self.gradient_scales = gradient_scales
        self.eps = eps

    def _silog(self, d, mask):
        """d, mask: [N, H, W] -> scalar (mean over images that have valid pixels)."""
        n = mask.flatten(1).sum(1)
        valid = n > 0
        if not valid.any():
            return (0.0 * d).sum()
        d = torch.where(mask, d, torch.zeros_like(d))
        mean = d.flatten(1).sum(1) / n.clamp_min(1)
        mean_sq = (d ** 2).flatten(1).sum(1) / n.clamp_min(1)
        per_img = torch.sqrt((mean_sq - self.lam * mean ** 2).clamp_min(self.eps))
        return per_img[valid].mean()

    def forward(self, predictions, gt_depths, masks):
        pred = predictions["depth"]
        if pred.ndim == 5:
            pred = pred.squeeze(-1)
        B, S, H, W = pred.shape
        pred = pred.reshape(B * S, H, W).float()
        gt = gt_depths.reshape(B * S, H, W).float()
        mask = masks.reshape(B * S, H, W).bool() & (gt > self.eps) & torch.isfinite(gt)

        if not mask.any():
            zero = (0.0 * pred).mean()
            return {"loss_depth": zero, "loss_silog": zero, "loss_grad_depth": zero}

        log_pred = torch.log(pred.clamp_min(self.eps))
        log_gt = torch.log(gt.clamp_min(self.eps))
        loss_silog = self._silog(log_pred - log_gt, mask)

        if self.use_gradient_loss:
            loss_grad = depth_gradient_loss_multi_scale(
                log_pred.reshape(B * S, H, W, 1),
                log_gt.reshape(B * S, H, W, 1),
                mask,
                scales=self.gradient_scales,
            )
        else:
            loss_grad = (0.0 * pred).mean()

        loss_depth = (loss_silog + self.grad_weight * loss_grad) * self.weight
        return {"loss_depth": loss_depth, "loss_silog": loss_silog, "loss_grad_depth": loss_grad}
# -----------------------------------------------------------------------------
# Layered depth loss  (DepthLoss per layer + ordering + layer existence)
class LayeredDepthLoss(nn.Module):
    """
    Simple loss for LayeredDPTHead: N depths per pixel, sorted near -> far (layer 0 = visible surface), plus ray
    stopping logits (N + 1 classes per pixel, as LaRI). Depth, gradient (layer 0) and stopping terms:

        depth = sum_K 1/K  mean over the valid pixels of layer K of |d_K - gt_K|   (depth_norm="l1", metres)
                                                                or (d_K - gt_K)^2  (depth_norm="l2")
                K = 1..N (1-based layer index: layer 1 = visible surface / 1, layer 2 / 2, ...);
                a layer with no valid pixel in the batch adds 0
                depth_norm="silog": the loss that gave good LDIs with the VGGT backbone (SiLogLoss), per layer k
                    silog_k = sqrt( mean r^2 - lambda * (mean r)^2 ),  r = log gt_k - log d_k over the layer's valid
                              pixels of the WHOLE batch (one scale per layer and batch), lambda = 0.85
                    depth   = mean_k silog_k + 0.2 * mean_k |d_k - gt_k| + 0.05 * mean relu(d_k - d_{k+1})
                              (ordering on pixels where layers k and k+1 are both valid)
        grad  = depth_gradient_loss_multi_scale on LAYER 0 only (multi-scale L1 on the spatial gradients of
                d_0 - gt_0, gradient_scales scales; gradient_loss_fn=None turns it off)
        stop  = stop_loss="bce" (default): BCEWithLogits(stop_logits, one-hot(C))  over the N + 1 classes
                stop_loss="ce":            CrossEntropy(stop_logits, C)             (LaRI's RegrStopPoint)
                on pixels where GT layer 0 is valid
        loss  = weight * (depth + grad_weight * grad + exist_weight * stop)

    GT stopping index exactly as LaRI (RegrStopPoint.mask_to_stop_index): C = mask.sum over the layers, in {0..N}
    (0 = no intersection). Inference (as LaRI): C = argmax_c stop_logits, layer k exists iff k < C.
    Valid GT for layer k: gt_layer_masks[k] & min_depth < gt_k < max_depth.

    Logged (no gradient): objective, loss_layers, loss_grad, loss_exist, absrel_L{k}, stop_acc, stop_acc_base,
        mask_iou
        stop_acc      accuracy of argmax_c on the supervised pixels
        stop_acc_base accuracy of always predicting index 1 ("only the visible surface") on the same pixels
        mask_iou      IoU of the hidden-layer masks (layers 1 .. N-1) from the predicted vs GT stopping index

    Predictions:
        layered_depth:       (B, S, N, H, W)      > 0
        layered_depth_stop:  (B, S, N + 1, H, W)  ray stopping logits (None / missing: no stopping term)
    Ground truth:
        gt_layers:      (B, S, L, H, W), L >= N, sorted near -> far (z-depth, metres)
        gt_layer_masks: (B, S, L, H, W) bool, True where layer k exists
    """

    def __init__(
        self,
        weight=1.0,
        depth_norm="l1",              # "l1", "l2" or "silog" (the VGGT LDI loss)
        silog_lambda=0.85,            # silog only
        silog_l1_weight=0.2,          # silog only: + L1 "for sharpness"
        silog_order_weight=0.05,      # silog only: + ordering
        gradient_loss_fn="grad",      # "grad" or None: gradient loss on layer 0
        grad_weight=0.10,
        gradient_scales=4,
        exist_weight=0.40,             # weight of the stopping loss (0 = depth only)
        stop_loss="bce",              # "bce": BCE on one-hot(C) over N + 1 classes; "ce": cross-entropy (LaRI)
        min_depth=0.1,                # metres
        max_depth=16.0,               # metres
        **kwargs,                     # options of older configs (order_weight, stop_weight, ...): ignored
    ):
        super().__init__()
        if stop_loss not in ("bce", "ce"):
            raise ValueError(f"stop_loss must be 'bce' or 'ce', got {stop_loss!r}")
        self.stop_loss = stop_loss
        if depth_norm not in ("l1", "l2", "silog"):
            raise ValueError(f"depth_norm must be 'l1', 'l2' or 'silog', got {depth_norm!r}")
        self.silog_lambda = silog_lambda
        self.silog_l1_weight = silog_l1_weight
        self.silog_order_weight = silog_order_weight
        self.weight = weight
        self.depth_norm = depth_norm
        self.exist_weight = exist_weight
        if gradient_loss_fn not in (None, "grad"):
            raise ValueError(f"Unknown depth gradient loss: {gradient_loss_fn}")
        self.use_gradient_loss = gradient_loss_fn is not None
        self.grad_weight = grad_weight
        self.gradient_scales = gradient_scales
        self.min_depth = min_depth
        self.max_depth = max_depth

    @staticmethod
    def stop_index_from_masks(mask):
        """(..., N, H, W) bool layer masks -> (..., H, W) long stopping index, as LaRI: mask.sum over the layers
        (0 = no intersection). Same as counting from the front when the valid layers have no gap."""
        return mask.long().sum(dim=-3)

    @staticmethod
    def masks_from_stop_index(C, N):
        """(..., H, W) stopping index -> (..., N, H, W) bool: layer k exists iff k < C."""
        return torch.arange(N, device=C.device).view(N, 1, 1) < C.unsqueeze(-3)

    def forward(self, predictions, gt_layers, gt_layer_masks):
        pred = predictions["layered_depth"].float()                  # (B, S, N, H, W)
        stop = predictions.get("layered_depth_stop", predictions.get("layered_depth_exist"))   # N + 1 logits
        B, S, N, H, W = pred.shape
        if gt_layers.shape[2] < N:
            raise ValueError(f"head predicts {N} layers but GT has only {gt_layers.shape[2]}")
        gt = gt_layers[:, :, :N].float()
        mask = gt_layer_masks[:, :, :N].bool() & (gt > self.min_depth) & (gt < self.max_depth)
        zero = (0.0 * pred).mean()
        out = {}

        # ---------------- depth: L1 / L2 of layer K (1-based) divided by K, summed over the layers ----------------
        layer_losses, silog_terms, l1_terms = [], [], []
        for k in range(N):
            m = mask[:, :, k]
            if not m.any():
                out[f"absrel_L{k}"] = zero.detach()
                continue
            d, g = pred[:, :, k][m], gt[:, :, k][m]
            diff = d - g
            out[f"absrel_L{k}"] = (diff.abs() / g).mean().detach()
            if self.depth_norm == "silog":                            # as SiLogLoss: whole batch, one scale per layer
                r = torch.log(g + 1e-6) - torch.log(d + 1e-6)
                silog_terms.append(torch.sqrt(r.pow(2).mean() - self.silog_lambda * r.mean().pow(2) + 1e-6))
                l1_terms.append(diff.abs().mean())
            else:
                l_k = diff.abs().mean() if self.depth_norm == "l1" else diff.pow(2).mean()
                layer_losses.append(l_k / (k + 1))                    # layer k (1-based K = k + 1): / K
        if self.depth_norm == "silog":
            both = mask[:, :, :-1] & mask[:, :, 1:]
            order = F.relu(pred[:, :, :-1] - pred[:, :, 1:])[both].mean() if (N > 1 and both.any()) else zero
            # SiLogLoss divides each layer's term by K (all layers), also when a layer has no pixels
            loss_layers = (torch.stack(silog_terms).sum() / N + self.silog_l1_weight * torch.stack(l1_terms).sum() / N
                           + self.silog_order_weight * order) if silog_terms else zero
            out["loss_order"] = order.detach()
        else:
            loss_layers = torch.stack(layer_losses).sum() if layer_losses else zero   # sum_K loss_K / K

        # ---------------- gradient loss: layer 0 only ----------------
        loss_grad = zero
        if self.use_gradient_loss and mask[:, :, 0].any():
            loss_grad = depth_gradient_loss_multi_scale(
                pred[:, :, 0].reshape(B * S, H, W, 1), gt[:, :, 0].reshape(B * S, H, W, 1),
                mask[:, :, 0].reshape(B * S, H, W), scales=self.gradient_scales)

        # ---------------- ray stopping index (LaRI) ----------------
        loss_exist = zero
        if stop is not None:
            if stop.shape[2] != N + 1:
                raise ValueError(f"stopping logits have {stop.shape[2]} channels, expected N + 1 = {N + 1}")
            C = self.stop_index_from_masks(mask)                      # (B, S, H, W) in {0..N}, LaRI: mask.sum
            valid = mask[:, :, 0]                                     # pixels with a visible surface
            logits = stop.float().permute(0, 1, 3, 4, 2)[valid]       # (n, N + 1)
            target = C[valid]                                         # (n,)
            if target.numel() > 0:
                if self.stop_loss == "bce":
                    loss_exist = F.binary_cross_entropy_with_logits(logits, F.one_hot(target, N + 1).float())
                else:
                    loss_exist = F.cross_entropy(logits, target)
            with torch.no_grad():
                C_pred = stop.argmax(dim=2)                           # (B, S, H, W)
                if valid.any():
                    out["stop_acc"] = (C_pred == C)[valid].float().mean()
                    out["stop_acc_base"] = (C == 1)[valid].float().mean()
                else:
                    out["stop_acc"] = out["stop_acc_base"] = zero.detach()
                pm = self.masks_from_stop_index(C_pred, N) & valid[:, :, None]
                gm = self.masks_from_stop_index(C, N) & valid[:, :, None]
                a, b = pm[:, :, 1:], gm[:, :, 1:]
                out["mask_iou"] = (a & b).sum().float() / (a | b).sum().float() if (a | b).any() \
                    else pred.new_tensor(1.0)

        loss = (loss_layers + self.grad_weight * loss_grad + self.exist_weight * loss_exist) * self.weight
        out.update({
            "objective": loss,
            "loss_layers": loss_layers.detach(),
            "loss_grad": loss_grad.detach(),
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