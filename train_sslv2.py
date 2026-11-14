# Standard library imports
import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sized, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# Enable TF32 matmul for Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Project imports
from datasets import get_data_loader  # noqa: E402
from models import vista_base, vista_large, vista_v2_base, vista_v2_large


import utils.misc as misc  # noqa: E402
from utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402
from utils.inference import vis_depth_maps, vis_images  # noqa: E402
from evals import compute_reconstruction_metrics

# Logger setup
logger = logging.getLogger("vista_trainer")
logging.basicConfig(level=logging.INFO)

# Per-term losses returned by VistaV2 in its loss_dict
LOSS_KEYS = ("loss_b", "loss_c")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def requires_grad(model: torch.nn.Module, flag: bool = True):
    """
    Set requires_grad for all parameters of a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_args_parser():
    parser = argparse.ArgumentParser('3D SSL training', add_help=False)

    # model / training
    parser.add_argument('--model', default="vista_base()",
                        type=str, help="string containing the model to build (evaled)")
    parser.add_argument('--warmup', default=False, type=str2bool, help="enable warmup")

    # dataset
    parser.add_argument('--train_dataset', required=True, type=str, help="training set (string expected by get_data_loader)")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set(s) separated by +")

    # training hyperparams
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size per GPU")
    parser.add_argument('--accum_iter', default=1, type=int, help="Gradient accumulation iterations")
    parser.add_argument('--epochs', default=500, type=int, help="Maximum number of epochs")
    parser.add_argument('--start_epoch', default=0, type=int, help="Starting epoch")
    parser.add_argument('--base_lr', type=float, default=2e-4, help='base learning rate (absolute lr)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (absolute lr)')
    parser.add_argument('--warmup_percent_steps', type=float, default=0.10, help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--mask_ratio', default=0.75, help='Mask ratio for masking images')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Pretrained weights from stage1')

    # system / distributed
    parser.add_argument('--num_workers', default=13, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id (if using single-GPU)')

    # misc
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency (epochs)')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency (epochs) to save checkpoint')
    parser.add_argument('--keep_freq', default=20, type=int, help='frequency (epochs) to keep checkpoints')
    parser.add_argument('--print_freq', default=20, type=int, help='iterations between prints')
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping. Set to <= 0 to disable."
    )
    return parser


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model (strip DDP)."""
    return model.module if hasattr(model, "module") else model


def unpack_outputs(outputs):
    """
    Normalise model outputs to 6 values:
        loss, masked_rgb, reconstructed_rgb, loss_dict, b_masked_rgb, b_recon_rgb

    Vista   returns 3: (loss, masked_rgb, reconstructed_rgb)
    VistaV2 returns 6: (loss, masked_rgb, reconstructed_rgb, loss_dict, b_masked_rgb, b_recon_rgb)
    """
    total_loss, masked_rgb, reconstructed_rgb = outputs[:3]
    loss_dict = outputs[3] if len(outputs) > 3 else {}
    b_masked_rgb = outputs[4] if len(outputs) > 4 else None
    b_recon_rgb = outputs[5] if len(outputs) > 5 else None
    return total_loss, masked_rgb, reconstructed_rgb, loss_dict, b_masked_rgb, b_recon_rgb


def move_selected_keys_to_device(batch: dict, keys, device="cuda"):
    """Move selected keys to `device` as float32. List elements arrive pinned from the DataLoader:
    copy each one asynchronously and stack ON THE GPU (stacking on the CPU makes an unpinned copy)."""
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
        batch[key] = tensor.float()              # dtype conversion on the GPU (no-op if already float32)
    return batch


def to_uint8_hwc(images: torch.Tensor) -> np.ndarray:
    """[B, S, 3, H, W] in [0, 1] -> numpy uint8 [B, S, H, W, 3]."""
    return (
        images.float().clamp(0, 1).mul(255)
        .permute(0, 1, 3, 4, 2)
        .cpu().numpy().astype("uint8")
    )


def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, lr_scheduler, fname=None, best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_scheduler):
    args.start_epoch = 0
    best_so_far = None
    if args.resume is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("Resume checkpoint %s" % args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'loss_scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'best_so_far' in checkpoint:
            best_so_far = checkpoint['best_so_far']
            print(" & best_so_far={:g}".format(best_so_far))
        else:
            print("")
        print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
    return best_so_far


@torch.no_grad()
def test_ss_3d_model(model: torch.nn.Module,
                     data_loader: Sized,
                     device: torch.device,
                     epoch: int,
                     args,
                     log_writer: SummaryWriter = None,
                     prefix: str = "test",
                     total_steps: int = 0,
                     write_freq: int = 100,
                     ):
    """
    Test loop. Returns aggregated metrics dict.

    For VistaV2 also logs:
        - loss_b / loss_c (from the returned loss_dict)
        - PSNR / SSIM of the encoder's reconstruction of the masked reference views (B)
        - individual images:
            <prefix>_val_C/  gt, masked, recon of the target views (decoder)
            <prefix>_val_B/  gt, masked, recon of the masked reference views (encoder)
    """

    model.eval()
    num_clean = getattr(unwrap(model), "num_clean", None)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        # Move inputs to device
        tensor_keys = ["s_images", "t_images"]
        batch = move_selected_keys_to_device(batch, tensor_keys, device=device)
        source_images = batch["s_images"]
        target_images = batch["t_images"]

        B, S, C, H, W = target_images.shape

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Model forward under autocast
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            (total_loss, masked_rgb, reconstructed_rgb,
             loss_dict, b_masked_rgb, b_recon_rgb) = unpack_outputs(model(source_images, target_images))

        has_b = b_recon_rgb is not None and num_clean is not None
        num_clean = (source_images.shape[1] - b_recon_rgb.shape[1] if b_recon_rgb is not None else None)
        
        has_b = b_recon_rgb is not None

        # ---------------------------------------------------
        # Target (C) reconstruction metrics
        # ---------------------------------------------------
        metrics_model = compute_reconstruction_metrics(
            to_uint8_hwc(target_images).reshape(B * S, H, W, C),
            to_uint8_hwc(reconstructed_rgb).reshape(B * S, H, W, C),
            device,
            B * S,
            metrics_to_compute=("psnr", "ssim"),
            disable_bar=True,
        )

        metrics = {f"val_c_{k}": v for k, v in metrics_model.items()}
        metrics["loss"] = total_loss.mean()
        metrics["recon_loss"] = total_loss.mean()

        # ---------------------------------------------------
        # VistaV2: separate loss terms from the returned loss_dict
        # ---------------------------------------------------
        for key in LOSS_KEYS:
            if key in loss_dict:
                metrics[key] = float(loss_dict[key])

        # ---------------------------------------------------
        # VistaV2: encoder reconstruction of B (masked reference views)
        # ---------------------------------------------------
        if has_b:
            b_gt = source_images[:, num_clean:]                        # [B, S_b, 3, H, W]
            S_b = b_gt.shape[1]

            metrics_b = compute_reconstruction_metrics(
                to_uint8_hwc(b_gt).reshape(B * S_b, H, W, C),
                to_uint8_hwc(b_recon_rgb).reshape(B * S_b, H, W, C),
                device,
                B * S_b,
                metrics_to_compute=("psnr", "ssim"),
                disable_bar=True,
            )
            for k, v in metrics_b.items():
                metrics[f"val_b_{k}"] = v

        loss_value = float(metrics["loss"])
        filtered_loss_dict = {k: v for k, v in metrics.items() if k != "loss"}

        # Update metric logger
        metric_logger.update(loss=loss_value, **filtered_loss_dict)

        if log_writer is not None and batch_idx % write_freq == 0:
            # SPEED: visualize only 2 samples
            n_vis = min(2, B)

            # -----------------------------------------------
            # C (decoder): individual images
            # -----------------------------------------------
            c_gt_vis = vis_images(target_images[:n_vis])
            c_masked_vis = vis_images(masked_rgb[:n_vis])
            c_recon_vis = vis_images(reconstructed_rgb[:n_vis])

            for b in range(c_gt_vis.shape[0]):
                for s in range(c_gt_vis.shape[1]):
                    log_writer.add_image(f"{prefix}_val_C/gt_image_view_{b}_{s}", c_gt_vis[b, s], epoch, dataformats='HWC')
                    log_writer.add_image(f"{prefix}_val_C/mask_image_view_{b}_{s}", c_masked_vis[b, s], epoch, dataformats='HWC')
                    log_writer.add_image(f"{prefix}_val_C/recon_image_view_{b}_{s}", c_recon_vis[b, s], epoch, dataformats='HWC')

            # -----------------------------------------------
            # B (encoder): individual images
            # -----------------------------------------------
            if has_b:
                b_gt_vis = vis_images(b_gt[:n_vis])
                b_masked_vis = vis_images(b_masked_rgb[:n_vis])
                b_recon_vis = vis_images(b_recon_rgb[:n_vis])

                for b in range(b_gt_vis.shape[0]):
                    for s in range(b_gt_vis.shape[1]):
                        log_writer.add_image(f"{prefix}_val_B/gt_image_view_{b}_{s}", b_gt_vis[b, s], epoch, dataformats='HWC')
                        log_writer.add_image(f"{prefix}_val_B/mask_image_view_{b}_{s}", b_masked_vis[b, s], epoch, dataformats='HWC')
                        log_writer.add_image(f"{prefix}_val_B/recon_image_view_{b}_{s}", b_recon_vis[b, s], epoch, dataformats='HWC')

            # Write scalars
            epoch_1000x = int((epoch + batch_idx / max(1, len(data_loader))) * 1000)
            log_writer.add_scalar(f'{prefix}/val_loss', loss_value, epoch_1000x)
            for name, val in filtered_loss_dict.items():
                val_scalar = float(val) if not torch.is_tensor(val) else float(val.mean())
                log_writer.add_scalar(f'{prefix}/val_{name}', val_scalar, epoch_1000x)

    # gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr)
               for k, meter in metric_logger.meters.items()
               for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f"{prefix}/val_{name}", val, 1000 * epoch)

    return results


def train_ss_3d_model(
    model: torch.nn.Module,
    loss_scaler,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: float,
    clip_grad=1.0,
    lr_scheduler=None,
    args=None,
    log_writer: SummaryWriter = None
):
    """
    Train for one epoch. For VistaV2 also logs loss_b / loss_c from the returned loss_dict.
    """
    assert torch.backends.cuda.matmul.allow_tf32 is True

    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    # Set epoch for distributed samplers if present
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)

    grad_clip_norm = float(getattr(args, "grad_clip_norm", clip_grad))
    cur_param_update_step = 0
    skipped_steps = 0

    # SPEED: losses are accumulated on the GPU and read back only at print steps
    running_loss = torch.zeros((), device=device)
    running_parts = {}                       # VistaV2: {"loss_b": tensor, "loss_c": tensor}
    n_running = 0

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + step / len(data_loader)

        # Move inputs to device
        tensor_keys = ["s_images", "t_images"]
        batch = move_selected_keys_to_device(batch, tensor_keys, device=device)
        source_images = batch["s_images"]
        target_images = batch["t_images"]

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Model forward under autocast
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            total_loss, _, _, loss_dict, _, _ = unpack_outputs(model(source_images, target_images))

        loss_scaler.scale(total_loss / accum_iter).backward()

        # Gradient accumulation: only perform optimizer operations every accum_iter
        update_grads = (step + 1) % accum_iter == 0

        if update_grads:
            # Unscales the gradients (no-op when the scaler is disabled for bf16)
            loss_scaler.unscale_(optimizer)

            # Clip gradients if specified
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            # Perform optimizer step
            loss_scaler.step(optimizer)
            cur_param_update_step += 1
            loss_scaler.update()

            # Update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Always zero gradients for next accumulation
            optimizer.zero_grad(set_to_none=True)

        lr = optimizer.param_groups[0]["lr"]
        # SPEED: only Python floats here (no sync); losses stay on the GPU
        metric_logger.update(epoch=epoch_f, lr=lr)
        running_loss += total_loss.detach().mean()
        for key in LOSS_KEYS:
            if key in loss_dict:
                running_parts[key] = running_parts.get(key, torch.zeros((), device=device)) + loss_dict[key].float()
        n_running += 1

        if update_grads and ((step + 1) % (accum_iter * args.print_freq)) == 0:
            # SPEED: single sync point, once per print_freq steps
            denom = max(1, n_running)
            loss_avg_t = running_loss / denom
            parts_avg_t = {k: v / denom for k, v in sorted(running_parts.items())}

            # all_reduce MUST be executed by all ranks, in the same order
            loss_value_reduce = float(misc.all_reduce_mean(loss_avg_t))
            parts_reduce = {k: float(misc.all_reduce_mean(v)) for k, v in parts_avg_t.items()}

            loss_avg = float(loss_avg_t)                                   # this rank's value
            running_loss.zero_()
            running_parts = {}
            n_running = 0

            if not math.isfinite(loss_avg):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step + 1}")
            metric_logger.update(loss=loss_avg, recon_loss=loss_avg, **parts_reduce)

            if log_writer is None:
                continue
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            log_writer.add_scalar('train_skipped_steps', skipped_steps, epoch_1000x)
            for name, val in parts_reduce.items():
                log_writer.add_scalar(f'train_{name}', val, epoch_1000x)

    # Gather metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(f"Total skipped steps due to excessive loss: {skipped_steps}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_dataset(dataset: str, batch_size: int, num_workers: int, test: bool = False):
    split = 'Test' if test else 'Train'
    logger.info("Building %s Data loader for dataset: %s", split, dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not test,
                             drop_last=not test)
    logger.info("%s dataset length: %d", split, len(loader))
    return loader


def clear_all_gpus_cache():
    # Only call after ensuring your tensors are freed
    dist.barrier()  # synchronize all processes
    torch.cuda.empty_cache()


def cosine_warmup_lr(total_steps, warmup_ratio=0.10, min_lr_ratio=0.0):
    """
    MAE-style schedule (multiplier for LambdaLR):
        [0, warmup]           : linear warmup 0 -> 1
        [warmup, total_steps] : half-cycle cosine 1 -> min_lr_ratio, reaching it at the last step
    min_lr_ratio = min_lr / base_lr  (MAE default min_lr = 0)
    """
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return lr_lambda


def train(args):

    # initialize distributed / misc
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    eff_batch_size = args.batch_size * args.accum_iter * world_size
    args.lr = args.base_lr * eff_batch_size / 256

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume if exists
    last_ckpt_path_fname = os.path.join(args.output_dir, 'checkpoint-last.pth')
    args.resume = last_ckpt_path_fname if os.path.isfile(last_ckpt_path_fname) else None
    logger.info('job dir: %s', os.path.dirname(os.path.realpath(__file__)))
    logger.info("Args:\n%s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random seeds
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    def save_ss_model(epoch, fname, best_so_far):
        save_model(args=args, model_without_ddp=model_without_ddp, loss_scaler=loss_scaler,
                   optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, fname=fname,
                   best_so_far=best_so_far)

    # datasets
    logger.info('Building train dataset: %s', args.train_dataset)

    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    test_datasets = {}

    logger.info('Building test datasets: %s', args.test_dataset)

    for dataset in args.test_dataset.split('+'):
        if not dataset or dataset == '[None]':
            continue
        key = dataset.split('(')[0]
        test_datasets[key] = build_dataset(dataset, args.batch_size, args.num_workers, test=True)

    # ----------------------------------------
    # Instantiate Model
    # ----------------------------------------
    logger.info("Instantiating model: %s", args.model)

    model = eval(args.model)
    model.to(device)

    model_without_ddp = model
    if args.ckpt_path is not None:
        checkpoint_path = f"{args.ckpt_path}/checkpoint-best.pth"
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        logger.info("Loaded model weights from %s", checkpoint_path)
        logger.info("Initializing model from stage1 training checkpoint")

    logger.info("Model parameter count: %s", sum(p.numel() for p in model.parameters()))

    if args.distributed:
        # VistaV2 with lambda_b=0 leaves the B head (encoder_norm / encoder_pred) without
        # gradients; DDP then needs find_unused_parameters=True.
        find_unused = getattr(model, "lambda_b", None) == 0
        if find_unused:
            logger.info("lambda_b=0: enabling find_unused_parameters for DDP")

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=find_unused,
        )

        model_without_ddp = model.module

    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay, skip_list=[f"pos_embed", "decoder_pos_embed"])
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # SPEED: loss scaling is only needed for fp16. Disabled for bf16, which removes its per-step inf-check sync.
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    loss_scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    steps_per_epoch = len(data_loader_train)
    total_steps = steps_per_epoch * args.epochs
    effective_steps = total_steps // args.accum_iter

    # 10 % warmup
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_lr(total_steps=effective_steps))

    # resume / checkpoint helper
    best_so_far = load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                             lr_scheduler=lr_scheduler, loss_scaler=loss_scaler)

    if best_so_far is None:
        best_so_far = float('inf')

    log_writer = SummaryWriter(log_dir=args.output_dir) if (global_rank == 0 and args.output_dir) else None

    logger.info("Start training for %d epochs", args.epochs)
    start_time = time.time()
    train_stats = {}
    test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):
        # Save immediate checkpoint at epoch boundaries if required
        if epoch > args.start_epoch and args.save_freq and (epoch % args.save_freq == 0 or epoch == args.epochs):
            # save a 'last' checkpoint
            save_ss_model(epoch - 1, 'last', best_so_far)
            if dist.is_initialized():
                dist.barrier()

        # Evaluate
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in test_datasets.items():
                stats = test_ss_3d_model(model=model, data_loader=testset, total_steps=total_steps,
                                         device=device, epoch=epoch, args=args,
                                         log_writer=log_writer, prefix=test_name)
                test_stats[test_name] = stats

            # best checkpoint = lowest MEAN (over test datasets) of the per-dataset median val loss
            if test_stats:
                mean_loss = float(np.mean([s.get('loss_med', float('inf')) for s in test_stats.values()]))
                if log_writer is not None:
                    log_writer.add_scalar('test/mean_loss_med', mean_loss, 1000 * epoch)
                if mean_loss < best_so_far:
                    best_so_far = mean_loss
                    new_best = True

        # Write logs for this epoch
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in (train_stats or {}).items()})
            for test_name, stats in (test_stats or {}).items():
                log_stats.update({f'{test_name}_{k}': v for k, v in stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Keep extra checkpoints if requested
        if epoch > args.start_epoch and args.keep_freq and epoch % args.keep_freq == 0:
            save_ss_model(epoch - 1, str(epoch), best_so_far)

        if new_best:
            save_ss_model(epoch - 1, 'best', best_so_far)

        # Final termination condition (safety)
        if epoch >= args.epochs:
            break

        # Train for one epoch
        train_stats = train_ss_3d_model(model=model, loss_scaler=loss_scaler,
                                        data_loader=data_loader_train, optimizer=optimizer,
                                        device=device, epoch=epoch,
                                        lr_scheduler=lr_scheduler,
                                        total_steps=total_steps,
                                        args=args,
                                        log_writer=log_writer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time %s', total_time_str)

    # final checkpoint
    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    # Ensure start_epoch exists
    if not hasattr(args, "start_epoch"):
        args.start_epoch = 0
    train(args)