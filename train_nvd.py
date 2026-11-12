"""
Novel-view depth (NVD) probe training on top of a frozen SSL encoder.

    --model      factory string, e.g. "vista_nvd_base_downstream()" (evaled)
    --ckpt_path  SSL run folder; encoder.* weights are loaded from <ckpt_path>/checkpoint-final.pth
    encoder      model.encoder is always frozen (no grads, eval mode); only the probe is trained
    resume       automatic from <output_dir>/checkpoint-last.pth
    batch        S >= 1 source views (encoder input) + T >= 1 target views (camera only, depth = GT)
                 -> datasets built with n_target_views=T, max_views=S+T; S may vary between batches

Adapted from the depth / camera fine-tuning script: same optimisation, checkpointing, resume,
logging and VGGT DepthLoss; the model and batch are the novel-view-depth ones.
"""

# Standard library
import argparse
import datetime
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized

# Third party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Project
from datasets import get_data_loader  # noqa: E402
from models import (  # noqa: E402,F401  (used through eval(args.model))
vista_nvd_base_downstream, 
mae_nvd_base_downstream, 
mum_nvd_base_downstream,
muskie_nvd_base_downstream,
poincar3_nvd_large_downstream
)
from models.losses import normalize_scene_by_points
from models import DepthLoss,SILogLoss  # noqa: E402  (VGGT depth loss; adjust the import to where DepthLoss lives)
import utils.misc as misc  # noqa: E402
from utils.inference import vis_depth_maps  # noqa: E402

logger = logging.getLogger("train_nvd")
logging.basicConfig(level=logging.INFO)

TENSOR_KEYS = [
    "s_images", "t_images",
    "t_depths", "t_depth_masks",
    "s_extrinsics", "t_intrinsics", "t_extrinsics",
]


# ======================================================================
# Arguments
# ======================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args_parser():
    parser = argparse.ArgumentParser("Novel-view depth probe", add_help=False)

    # model
    parser.add_argument("--model", default="vista_nvd_base_downstream()", type=str, help="model constructor string (evaled)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="SSL pretraining run folder (encoder weights, frozen)")
    parser.add_argument("--ckpt_name", type=str, default="checkpoint-final.pth",
                        help="weights file inside --ckpt_path")

    # data
    parser.add_argument("--train_dataset", required=True, type=str, help="training set (get_data_loader string)")
    parser.add_argument("--test_dataset", default="[None]", type=str, help="test set(s) separated by +")

    # optimisation
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="batch size per GPU")
    parser.add_argument("--accum_iter", default=1, type=int, help="gradient accumulation iterations")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr", type=float, default=2e-4, help="absolute learning rate")
    parser.add_argument("--warmup_percent_steps", type=float, default=0.10, help="fraction of steps for LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="<= 0 disables clipping")

    # system / distributed
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--gpu", default=None, type=int)

    # logging / checkpoints
    parser.add_argument("--eval_freq", type=int, default=1, help="epochs between evaluations")
    parser.add_argument("--save_freq", default=1, type=int, help="epochs between 'last' checkpoints")
    parser.add_argument("--keep_freq", default=20, type=int, help="epochs between kept checkpoints")
    parser.add_argument("--print_freq", default=20, type=int, help="iterations between prints")
    parser.add_argument("--output_dir", default="./output/", type=str)
    return parser


# ======================================================================
# Helpers
# ======================================================================
def move_selected_keys_to_device(batch: dict, keys, device="cuda"):
    """Move selected keys to `device` as float32 (list elements are stacked on the GPU)."""
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, list) and isinstance(value[0], torch.Tensor):
            tensor = torch.stack([v.to(device, non_blocking=True) for v in value])
        elif isinstance(value, list):
            tensor = torch.from_numpy(np.stack(value)).to(device, non_blocking=True)
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(device, non_blocking=True)
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device, non_blocking=True)
        else:
            continue
        batch[key] = tensor.float()
    return batch


def prepare_batch(batch: dict, device) -> dict:
    """All source views (encoder input + cameras) and all target views (camera + GT depth only),
    exactly as the data loader returns them."""
    batch = move_selected_keys_to_device(batch, TENSOR_KEYS, device=device)
    return dict(
        src_images=batch["s_images"],                    # (B, S, 3, H, W)
        src_w2c=batch["s_extrinsics"],                   # (B, S, 4, 4) world2cam
        tgt_K=batch["t_intrinsics"],                     # (B, T, 3, 3)
        tgt_w2c=batch["t_extrinsics"],                   # (B, T, 4, 4) world2cam
        tgt_depth=batch["t_depths"],                     # (B, T, H, W)
        tgt_mask=batch["t_depth_masks"],                  # (B, T, H, W)
        tgt_image=batch["t_images"],                     # (B, T, 3, H, W) logging only, never fed to the model
    )


@torch.no_grad()
def median_aligned_metrics(pred, gt, mask):
    """AbsRel and delta1 after per-image median scaling (same protocol as the depth evaluation).
    pred, gt, mask: (B, T, H, W) or (N, H, W); every target view is scored as its own image."""
    pred, gt, mask = (x.reshape(-1, *x.shape[-2:]) for x in (pred, gt, mask))
    abs_rel, delta1 = [], []
    for b in range(pred.shape[0]):
        m = mask[b] & (gt[b] > 0) & torch.isfinite(pred[b]) & (pred[b] > 0)
        if m.sum() < 10:
            continue
        p, g = pred[b][m], gt[b][m]
        p = p * (g.median() / p.median())
        abs_rel.append(((p - g).abs() / g).mean())
        delta1.append((torch.maximum(p / g, g / p) < 1.25).float().mean())
    if not abs_rel:
        nan = torch.tensor(float("nan"), device=pred.device)
        return {"abs_rel": nan, "delta1": nan}
    return {"abs_rel": torch.stack(abs_rel).mean(), "delta1": torch.stack(delta1).mean()}


def forward_and_loss(model, criterion, inputs: dict):
    src_w2c, tgt_w2c = inputs["src_w2c"], inputs["tgt_w2c"]
    gt = inputs["tgt_depth"]                                     # (B, T, H, W), metric
    mask = inputs["tgt_mask"].bool()

    if not mask.any():   # otherwise DepthLoss silently returns 0 and the metrics NaN
        d, m = inputs["tgt_depth"], inputs["tgt_mask"]
        raise RuntimeError(
            f"no valid target pixel in the batch: depth {tuple(d.shape)} min {d.min().item():.4g} "
            f"max {d.max().item():.4g} finite {torch.isfinite(d).float().mean().item():.3f}; "
            f"mask {tuple(m.shape)} valid {m.float().mean().item():.3f}")

    # ---- scene normalisation: ONE frame + ONE scale per sample, shared by the model input and the GT ----
    # views = [sources..., targets...]; frame 0 = first source (the view the model sees first).
    # Only the target depth is available here, so sources get an all-invalid mask and the scale is the
    # mean valid target depth.
    B, S = src_w2c.shape[:2]
    all_w2c = torch.cat([src_w2c, tgt_w2c], dim=1)                          # (B, S+T, 4, 4) or (.., 3, 4)
    all_depth = torch.cat([gt.new_zeros(B, S, *gt.shape[-2:]), gt], dim=1)  # (B, S+T, H, W)
    all_mask = torch.cat([torch.zeros(B, S, *gt.shape[-2:], dtype=torch.bool, device=gt.device), mask], dim=1)
    with torch.autocast(device_type=gt.device.type, enabled=False):
        w2c_n, depth_n, scale = normalize_scene_by_points(all_w2c, all_depth, all_mask)
    src_w2c_n, tgt_w2c_n = w2c_n[:, :S], w2c_n[:, S:]
    if src_w2c.shape[-2] == 3:                                   # model expects (3, 4)? keep the input format
        src_w2c_n, tgt_w2c_n = src_w2c_n[..., :3, :], tgt_w2c_n[..., :3, :]
    gt = depth_n[:, S:]                                          # (B, T, H, W), normalised units

    # the encoder runs in bf16 inside encode_sources; the probe and the geometry run in fp32
    out = model(inputs["src_images"], src_w2c_n, inputs["tgt_K"], tgt_w2c_n)
    pred = out["depth"].float()   
    pred_conf = out["depth_conf"].float()                               # (B, T, H, W), normalised units

    # DepthLoss takes (B, S, H, W): the T targets play the role of the views
    loss_dict = criterion(
        predictions={"depth": pred, "depth_conf": pred_conf},
        gt_depths=gt,
        masks=mask,
    )
    loss_dict = {"objective": loss_dict["loss_depth"], **loss_dict, "scene_scale": scale.detach().mean()}
    return {"depth": pred, "gt": gt, "mask": mask, "scale": scale}, loss_dict

# def forward_and_loss(model, criterion, inputs: dict):
#     # the encoder runs in bf16 inside encode_sources; the probe and the geometry run in fp32
#     # out = model(inputs["src_images"], inputs["src_w2c"], inputs["tgt_w2c"])
#     out = model(inputs["src_images"], inputs["src_w2c"], inputs["tgt_K"], inputs["tgt_w2c"])
#     pred = out["depth"].float()                                  # (B, T, H, W), metric
#     # conf = out["depth_conf"].float()
#     gt = inputs["tgt_depth"]                                     # (B, T, H, W), metric
#     # mask = inputs["tgt_mask"] & (gt > 0) & torch.isfinite(gt)
#     mask = inputs["tgt_mask"].bool()
#     if not mask.any():   # otherwise DepthLoss silently returns 0 and the metrics NaN
#         d, m = inputs["tgt_depth"], inputs["tgt_mask"]
#         raise RuntimeError(
#             f"no valid target pixel in the batch: depth {tuple(d.shape)} min {d.min().item():.4g} "
#             f"max {d.max().item():.4g} finite {torch.isfinite(d).float().mean().item():.3f}; "
#             f"mask {tuple(m.shape)} valid {m.float().mean().item():.3f}")

#     # DepthLoss takes (B, S, H, W): the T targets play the role of the views
#     loss_dict = criterion(
#         predictions={"depth": pred, "depth_conf": None},
#         gt_depths=gt,
#         masks=mask,
        
#     )
#     loss_dict = {"objective": loss_dict["loss_depth"], **loss_dict}
#     return {"depth": pred, "gt": gt, "mask": mask}, loss_dict



def to_scalars(loss_dict: dict) -> dict:
    """Loss terms other than 'objective', as python floats."""
    return {k: float(v.detach().mean()) if torch.is_tensor(v) else float(v)
            for k, v in loss_dict.items() if k != "objective"}


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def cosine_warmup_lr(total_steps, warmup_ratio=0.10, min_lr_ratio=0.10):
    """Linear warmup 0 -> 1, then half-cycle cosine 1 -> min_lr_ratio."""
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return lr_lambda

def is_poincar3(model) -> bool:
    return type(model).__name__ == "Poincar3NVD"        # no import needed; matches poincar3_nvd.py's class



def setup_encoder(model, ckpt_path, ckpt_name="checkpoint-final.pth"):
    """Load the pretrained backbone into model.encoder and freeze it.

    Poincar3: the model loads its own backbone when it is built (released checkpoint, or
        run_path='...' in --model), so nothing is loaded here; --ckpt_path is ignored.
    Others: encoder.* weights from <ckpt_path>/<ckpt_name> (strict; ckpt_path required).
    """
    if is_poincar3(model):
        if ckpt_path is not None:
            logger.warning("Poincar3: --ckpt_path %s ignored; the backbone is loaded by the model "
                           "(use --model \"poincar3_nvd_large_downstream(run_path='...')\" for another run)",
                           ckpt_path)
        logger.info("Poincar3 backbone: %s", model.run_path or "released checkpoint")
    else:
        if ckpt_path is None:
            raise ValueError(f"--ckpt_path is required for {type(model).__name__} (SSL run folder)")
        load_encoder(model, ckpt_path, fname=ckpt_name)

    freeze_encoder(model)                             # always frozen; only the probe is trained
    return model

def load_encoder(model, checkpoint_path, fname="checkpoint-final.pth"):
    """Load encoder.* weights of an SSL checkpoint into model.encoder (strict)."""
    checkpoint = torch.load(os.path.join(checkpoint_path, fname), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    if any(k.startswith("encoder.") for k in state_dict):     # MuM / Muskie / Vista: encoder is a submodule
        encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    else:                                                     # MAE: the checkpoint IS the MaskedAutoencoderViT
        encoder_state = state_dict
    msg = model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info("Loaded encoder from %s: %s", checkpoint_path, msg)
    return model


def freeze_encoder(model):
    """Freeze model.encoder: no gradients, eval mode (no dropout / drop-path)."""
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    assert not any(p.requires_grad for p in model.encoder.parameters())


# ======================================================================
# Checkpoints
# ======================================================================
def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, lr_scheduler, fname=None, best_so_far=None):
    checkpoint_path = Path(args.output_dir) / f"checkpoint-{fname if fname is not None else epoch}.pth"
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_scaler": loss_scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": args,
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_scheduler):
    """Resume from args.resume if set. Returns best_so_far (or None)."""
    args.start_epoch = 0
    if args.resume is None:
        return None
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "loss_scaler" in checkpoint:
        loss_scaler.load_state_dict(checkpoint["loss_scaler"])
    if "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    args.start_epoch = checkpoint["epoch"] + 1
    best_so_far = checkpoint.get("best_so_far")
    print(f"Resumed {args.resume} with optim & sched, start_epoch={args.start_epoch}"
          + (f", best_so_far={best_so_far:g}" if best_so_far is not None else ""))
    return best_so_far


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    checkpoint_path = Path(args.output_dir) / "checkpoint-final.pth"
    to_save = {"args": args, "model": model_without_ddp.state_dict(), "epoch": epoch}
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(to_save, checkpoint_path)


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def test_one_epoch(model, criterion, data_loader: Sized, device, epoch: int, args,
                   log_writer: SummaryWriter = None, prefix: str = "test", write_freq: int = 50):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, f"Test Epoch: [{epoch}]")):
        inputs = prepare_batch(batch, device)
        outputs, loss_dict = forward_and_loss(model, criterion, inputs)

        # ratio = [(outputs["depth"][b][outputs["mask"][b]].median() / outputs["gt"][b][outputs["mask"][b]].median()).item() for b in range(outputs["depth"].shape[0])]
        # print("pred/gt median ratio:", [round(r, 3) for r in ratio])
        # breakpoint()

        loss_value = float(loss_dict["objective"])
        # eval: loss terms + median-aligned depth metrics
        terms = to_scalars(loss_dict)
        terms.update({k: float(v) for k, v in
                      median_aligned_metrics(outputs["depth"], outputs["gt"], outputs["mask"]).items()})
        metric_logger.update(loss=loss_value, **terms)

        if log_writer is not None and batch_idx % write_freq == 0:
            step_1000x = int((epoch + batch_idx / max(1, len(data_loader))) * 1000)
            log_writer.add_scalar(f"{prefix}/val_loss", loss_value, step_1000x)
            for name, val in terms.items():
                log_writer.add_scalar(f"{prefix}/val_{name}", val, step_1000x)

            n_vis = min(2, outputs["gt"].shape[0])
            gt_vis = vis_depth_maps(outputs["gt"][:n_vis])                       # (n, T, H, W, 3)
            pred_vis = vis_depth_maps(outputs["depth"][:n_vis])
            for b in range(n_vis):
                # RGB: the source views the model was given, and each target (never seen by the model)
                for s in range(inputs["src_images"].shape[1]):
                    log_writer.add_image(f"{prefix}_val/source_rgb_{b}_{s}",
                                         inputs["src_images"][b, s].clamp(0, 1), epoch, dataformats="CHW")
                for t in range(outputs["gt"].shape[1]):
                    log_writer.add_image(f"{prefix}_val/target_rgb_{b}_{t}",
                                         inputs["tgt_image"][b, t].clamp(0, 1), epoch, dataformats="CHW")
                    log_writer.add_image(f"{prefix}_val/gt_target_depth_{b}_{t}", gt_vis[b, t], epoch,
                                         dataformats="HWC")
                    log_writer.add_image(f"{prefix}_val/pred_target_depth_{b}_{t}", pred_vis[b, t], epoch,
                                         dataformats="HWC")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    results = {f"{k}_{tag}": getattr(meter, attr)
               for k, meter in metric_logger.meters.items()
               for tag, attr in (("avg", "global_avg"), ("med", "median"))}
    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f"{prefix}/val_{name}", val, 1000 * epoch)
    return results


# ======================================================================
# Training
# ======================================================================
def train_one_epoch(model, criterion, loss_scaler, data_loader: Sized, optimizer, device, epoch: int, args,
                    lr_scheduler=None, log_writer: SummaryWriter = None):
    model.train()
    getattr(model, "module", model).encoder.eval()    # frozen: model.train() would re-enable dropout / drop-path

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    accum_iter = args.accum_iter
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    skipped_steps = 0

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, f"Epoch: [{epoch}]")):
        epoch_f = epoch + step / len(data_loader)
        inputs = prepare_batch(batch, device)
        _, loss_dict = forward_and_loss(model, criterion, inputs)

        total_loss = loss_dict["objective"]
        loss_scaler.scale(total_loss / accum_iter).backward()

        update_grads = (step + 1) % accum_iter == 0
        if update_grads:
            if torch.isfinite(total_loss):
                loss_scaler.unscale_(optimizer)
                with torch.no_grad():            # NaN -> 0, +-inf -> +-1e-6
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                skipped_steps += 1
                print(f"[SKIP] NaN/Inf loss at step {step}, scaler {loss_scaler.get_scale():.2f}")
                new_scale = max(loss_scaler._init_scale, loss_scaler._scale * 0.5)
                loss_scaler._scale = torch.tensor(new_scale).to(loss_scaler._scale.device)
                loss_scaler._growth_tracker.fill_(0)
                total_loss = torch.zeros_like(total_loss)

            if lr_scheduler is not None:         # stepped either way to keep the schedule
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # # train: loss only (no depth metrics)
        # lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(epoch=epoch_f, lr=lr, loss=float(total_loss.detach()))

        # if update_grads and (step + 1) % (accum_iter * args.print_freq) == 0:
        #     loss_reduced = misc.all_reduce_mean(total_loss.detach())    # must run on all ranks
        #     if log_writer is not None:
        #         step_1000x = int(epoch_f * 1000)
        #         log_writer.add_scalar("train_loss", loss_reduced, step_1000x)
        #         log_writer.add_scalar("train_lr", lr, step_1000x)
        #         log_writer.add_scalar("train_skipped_steps", skipped_steps, step_1000x)

                # train: every loss term + scene scale (no depth metrics such as abs_rel / delta1)
        lr = optimizer.param_groups[0]["lr"]
        terms = {k: v for k, v in to_scalars(loss_dict).items() if math.isfinite(v)}   # skip NaN/Inf terms
        metric_logger.update(epoch=epoch_f, lr=lr, loss=float(total_loss.detach()), **terms)

        if update_grads and (step + 1) % (accum_iter * args.print_freq) == 0:
            loss_reduced = misc.all_reduce_mean(total_loss.detach())    # must run on all ranks
            terms_reduced = {k: misc.all_reduce_mean(torch.tensor(v, device=device)) for k, v in terms.items()}
            if log_writer is not None:
                step_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_reduced, step_1000x)
                for k, v in terms_reduced.items():
                    log_writer.add_scalar(f"train_{k}", v, step_1000x)
                log_writer.add_scalar("train_lr", lr, step_1000x)
                log_writer.add_scalar("train_skipped_steps", skipped_steps, step_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(f"Skipped optimizer steps (NaN/Inf loss): {skipped_steps}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_dataset(dataset: str, batch_size: int, num_workers: int, test: bool = False):
    split = "Test" if test else "Train"
    logger.info("Building %s data loader: %s", split, dataset)
    loader = get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_mem=True, shuffle=not test, drop_last=not test)
    logger.info("%s dataset length: %d", split, len(loader))
    return loader


# ======================================================================
# Main
# ======================================================================
def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    args.resume = last_ckpt if os.path.isfile(last_ckpt) else None
    logger.info("Args:\n%s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # ---------------- data ----------------
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    test_datasets = {}
    for dataset in args.test_dataset.split("+"):
        if dataset and dataset.strip() != "[None]":
            test_datasets[dataset.split("(")[0].strip()] = build_dataset(dataset, args.batch_size,
                                                                         args.num_workers, test=True)

    # ---------------- model ----------------
    logger.info("Instantiating model: %s", args.model)
    model = eval(args.model)
    setup_encoder(model, args.ckpt_path, args.ckpt_name)
    model.to(device)
    logger.info("Parameters: %s total, %s trainable",
                f"{count_parameters(model, False):,}", f"{count_parameters(model):,}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    criterion = DepthLoss()                   # VGGT defaults: gamma 1, alpha 0.2, gradient loss, 0.98 quantile

    # ---------------- optimisation ----------------
    trainable = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = torch.cuda.amp.GradScaler()
    effective_steps = len(data_loader_train) * args.epochs // args.accum_iter
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_warmup_lr(effective_steps, warmup_ratio=args.warmup_percent_steps))

    best_so_far = load_model(args, model_without_ddp, optimizer, loss_scaler, lr_scheduler)
    if best_so_far is None:
        best_so_far = float("inf")

    def save(epoch, fname):
        save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, lr_scheduler,
                   fname=fname, best_so_far=best_so_far)

    log_writer = SummaryWriter(log_dir=args.output_dir) if (global_rank == 0 and args.output_dir) else None

    # ---------------- loop ----------------
    logger.info("Start training for %d epochs", args.epochs)
    start_time = time.time()
    train_stats, test_stats = {}, {}

    for epoch in range(args.start_epoch, args.epochs + 1):
        if epoch > args.start_epoch and args.save_freq and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save(epoch - 1, "last")
            if dist.is_initialized():
                dist.barrier()

        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for name, loader in test_datasets.items():
                stats = test_one_epoch(model, criterion, loader, device, epoch, args,
                                       log_writer=log_writer, prefix=name)
                test_stats[name] = stats
                if stats.get("loss_med", float("inf")) < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_stats = dict(epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()})
            for name, stats in test_stats.items():
                log_stats.update({f"{name}_{k}": v for k, v in stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch > args.start_epoch and args.keep_freq and epoch % args.keep_freq == 0:
            save(epoch - 1, str(epoch))
        if new_best:
            save(epoch - 1, "best")
        if epoch >= args.epochs:
            break

        train_stats = train_one_epoch(model, criterion, loss_scaler, data_loader_train, optimizer, device, epoch,
                                      args, lr_scheduler=lr_scheduler, log_writer=log_writer)

    logger.info("Training time %s", str(datetime.timedelta(seconds=int(time.time() - start_time))))
    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


if __name__ == "__main__":
    train(get_args_parser().parse_args())