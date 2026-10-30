# Standard library imports
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

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# Enable TF32 matmul for Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Project imports
from datasets import get_data_loader  # noqa: E402
from models import vista_jepa_base  # noqa: E402,F401  (adjust to where VistaJEPA lives)

import utils.misc as misc  # noqa: E402

# Logger setup
logger = logging.getLogger("vista_jepa_trainer")
logging.basicConfig(level=logging.INFO)


# ============================================================
# Args
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser("Vista JEPA training", add_help=False)

    # model
    parser.add_argument("--model", default="vista_jepa_base()", type=str,
                        help="string containing the model to build (evaled)")

    # dataset
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", default="[None]", type=str)

    # optimisation
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr", type=float, default=1.5e-4, help="peak learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="final lr = lr * ratio")
    parser.add_argument("--warmup_percent_steps", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.04, help="initial weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0.4, help="final weight decay (cosine)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="<= 0 disables clipping")

    # EMA (target encoder)
    parser.add_argument("--ema_start", type=float, default=0.996)
    parser.add_argument("--ema_end", type=float, default=1.0)

    # init
    parser.add_argument("--ckpt_path", type=str, default=None, help="dir with checkpoint-best.pth to init from")

    # system / distributed
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--gpu", default=None, type=int)

    # misc
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--diag_freq", type=int, default=500, help="steps between source-shuffle diagnostics")
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--keep_freq", default=20, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--output_dir", default="./output/", type=str)
    return parser


# ============================================================
# Helpers
# ============================================================

def move_selected_keys_to_device(batch: dict, keys, device="cuda"):
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            first = value[0]
            tensor = torch.stack(value) if isinstance(first, torch.Tensor) else torch.from_numpy(np.stack(value))
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            continue
        batch[key] = tensor.to(dtype=torch.float32, device=device, non_blocking=True)
    return batch


def autocast_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def build_optimizer(model, lr, weight_decay):
    """AdamW over trainable params only (EMA target encoder is excluded).
    No weight decay on biases, norms, mask tokens and position embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "mask_token" in name or "pos_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    logger.info("Optimizer: %d decay tensors, %d no-decay tensors", len(decay), len(no_decay))
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay, "apply_wd": True},
         {"params": no_decay, "weight_decay": 0.0, "apply_wd": False}],
        lr=lr, betas=(0.9, 0.95),
    )


def cosine_warmup_lr(total_steps, warmup_ratio=0.05, final_ratio=0.01):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return lr_lambda


def cosine_value(step, total_steps, start, end):
    t = min(1.0, step / max(1, total_steps))
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def feature_std(x, mask=None):
    """Mean per-dimension std across patches. Falling toward 0 = collapse."""
    x = x.float().reshape(-1, x.shape[-1])
    if mask is not None:
        m = mask.reshape(-1).bool()
        if m.any():
            x = x[m]
    return x.std(dim=0).mean().item()


@torch.no_grad()
def source_shuffle_gap(model, source_images, target_images, mask):
    """Loss with source views shuffled across the batch minus normal loss.
    ~0 means the predictor ignores the source views."""
    if source_images.shape[0] < 2:
        return float("nan")
    perm = torch.roll(torch.arange(source_images.shape[0], device=source_images.device), 1)
    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype()):
        loss, *_ = model(source_images, target_images, mask=mask)
        loss_shuf, *_ = model(source_images[perm], target_images, mask=mask)
    return (loss_shuf - loss).item()


# ============================================================
# Checkpointing
# ============================================================

def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, lr_scheduler, fname=None, best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / f"checkpoint-{fname}.pth"
    to_save = {
        "model": model_without_ddp.state_dict(),   # includes the EMA target encoder
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
    args.start_epoch = 0
    best_so_far = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        print(f"Resume checkpoint {args.resume}")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        args.start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_so_far = checkpoint.get("best_so_far")
        print(f"With optim & sched! start_epoch={args.start_epoch}")
    return best_so_far


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    checkpoint_path = Path(args.output_dir) / "checkpoint-final.pth"
    to_save = {"args": args, "model": model_without_ddp.state_dict(), "epoch": epoch}
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(to_save, checkpoint_path)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def test_jepa(model, data_loader: Sized, device, epoch, args,
              log_writer: SummaryWriter = None, prefix="test"):
    """
    Validation JEPA loss with a fixed RNG seed (same masks every epoch),
    plus collapse and source-usage diagnostics.

    Note: val loss is measured against the *current* EMA targets, so it is
    not strictly comparable across epochs. Use it as a sanity signal and
    judge representations with downstream probes.
    """
    model.eval()
    core = model.module if hasattr(model, "module") else model

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))
    header = f"Test Epoch: [{epoch}]"

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(0)

    # Fixed masks across epochs
    rng_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1234 + misc.get_rank())

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        batch = move_selected_keys_to_device(batch, ["s_images", "t_images"], device=device)
        source_images = batch["s_images"]
        target_images = batch["t_images"]

        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype()):
            loss, pred, target, mask = core(source_images, target_images)

        stats = {
            "std_target": feature_std(target),
            "std_pred": feature_std(pred, mask),
            "mask_frac": mask.float().mean().item(),
        }
        if batch_idx < 10:   # shuffle gap on a few batches only (doubles cost)
            stats["source_shuffle_gap"] = source_shuffle_gap(
                core, source_images, target_images, mask.reshape(-1, mask.shape[-1])
            )

        metric_logger.update(loss=float(loss), **stats)

    torch.cuda.set_rng_state(rng_state)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    results = {f"{k}_{tag}": getattr(meter, attr)
               for k, meter in metric_logger.meters.items()
               for tag, attr in [("avg", "global_avg"), ("med", "median")]}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f"{prefix}/val_{name}", val, 1000 * epoch)

    return results


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, model_without_ddp, loss_scaler, data_loader: Sized, optimizer,
                    device, epoch, effective_steps, lr_scheduler, args,
                    log_writer: SummaryWriter = None):
    model.train()   # VistaJEPA.train() keeps the EMA target encoder in eval mode

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    trainable = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    skipped_steps = 0
    grad_norm = torch.tensor(0.0)

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + step / len(data_loader)
        global_step = (epoch * len(data_loader) + step) // accum_iter

        batch = move_selected_keys_to_device(batch, ["s_images", "t_images"], device=device)
        source_images = batch["s_images"]
        target_images = batch["t_images"]

        # ---- forward
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype()):
            loss, pred, target, mask = model(source_images, target_images)

        loss_scaler.scale(loss / accum_iter).backward()

        update_grads = (step + 1) % accum_iter == 0
        if update_grads:

            if not torch.isfinite(loss):
                print(f"[SKIP] non-finite loss at step {step}")
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                continue

            # ---- weight decay schedule (cosine, rising)
            wd = cosine_value(global_step, effective_steps, args.weight_decay, args.weight_decay_end)
            for g in optimizer.param_groups:
                if g.get("apply_wd", False):
                    g["weight_decay"] = wd

            # ---- unscale, check, clip
            loss_scaler.unscale_(optimizer)

            n_bad = sum((~torch.isfinite(p.grad)).any().item()
                        for p in trainable if p.grad is not None)
            if n_bad:
                print(f"[WARN] non-finite grads in {n_bad} tensors at step {step}")

            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip_norm)

            # ---- optimizer step
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if lr_scheduler is not None:
                lr_scheduler.step()

            # ---- EMA: target encoder <- online target branch
            model_without_ddp.update_target_encoder(global_step, effective_steps)

        # ---- logging
        lr = optimizer.param_groups[0]["lr"]
        std_tgt = feature_std(target)
        std_pred = feature_std(pred, mask)

        metric_logger.update(epoch=epoch_f, lr=lr, loss=loss.item(),
                             std_target=std_tgt, std_pred=std_pred,
                             mask_frac=mask.float().mean().item())

        if update_grads and ((step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss.item())   # MUST BE EXECUTED BY ALL NODES
            if log_writer is not None:
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train/loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("train/lr", lr, epoch_1000x)
                log_writer.add_scalar("train/weight_decay", optimizer.param_groups[0]["weight_decay"], epoch_1000x)
                log_writer.add_scalar("train/grad_norm", float(grad_norm), epoch_1000x)
                log_writer.add_scalar("train/std_target", std_tgt, epoch_1000x)
                log_writer.add_scalar("train/std_pred", std_pred, epoch_1000x)
                log_writer.add_scalar("train/mask_frac", mask.float().mean().item(), epoch_1000x)
                log_writer.add_scalar("train/skipped_steps", skipped_steps, epoch_1000x)

        # ---- source-usage diagnostic (rank 0 only, no grad)
        if (args.diag_freq > 0 and update_grads and global_step % args.diag_freq == 0
                and misc.is_main_process()):
            model_without_ddp.eval()
            gap = source_shuffle_gap(model_without_ddp, source_images, target_images,
                                     mask.reshape(-1, mask.shape[-1]))
            model_without_ddp.train()
            print(f"    [diag] step {global_step}: source-shuffle gap = {gap:.4f}")
            if log_writer is not None:
                log_writer.add_scalar("train/source_shuffle_gap", gap, int(epoch_f * 1000))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(f"Skipped steps this epoch: {skipped_steps}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_dataset(dataset: str, batch_size: int, num_workers: int, test: bool = False):
    split = "Test" if test else "Train"
    logger.info("Building %s Data loader for dataset: %s", split, dataset)
    loader = get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_mem=True, shuffle=not test, drop_last=not test)
    logger.info("%s dataset length: %d", split, len(loader))
    return loader


# ============================================================
# Main
# ============================================================

def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    args.resume = last_ckpt if os.path.isfile(last_ckpt) else None
    logger.info("Args:\n%s", vars(args))

    device = torch.device("cuda")

    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # ---- data
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    test_datasets = {}
    for dataset in args.test_dataset.split("+"):
        dataset = dataset.strip()
        if not dataset or dataset == "[None]":
            continue
        key = dataset.split("(")[0].split("@")[-1].strip()
        test_datasets[key] = build_dataset(dataset, args.batch_size, args.num_workers, test=True)

    # ---- model
    logger.info("Instantiating model: %s", args.model)
    model = eval(args.model)
    model.ema_start, model.ema_end = args.ema_start, args.ema_end
    model.to(device)
    model_without_ddp = model

    if args.ckpt_path is not None:
        checkpoint = torch.load(f"{args.ckpt_path}/checkpoint-best.pth", weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        logger.info("Initialized from %s", args.ckpt_path)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %s total, %s trainable (EMA target encoder frozen)", f"{n_total:,}", f"{n_train:,}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False,
        )
        model_without_ddp = model.module

    # ---- optimisation
    optimizer = build_optimizer(model_without_ddp, args.lr, args.weight_decay)
    loss_scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype() == torch.float16))

    steps_per_epoch = len(data_loader_train)
    effective_steps = (steps_per_epoch * args.epochs) // args.accum_iter

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=cosine_warmup_lr(effective_steps, args.warmup_percent_steps, args.min_lr_ratio),
    )

    best_so_far = load_model(args, model_without_ddp, optimizer, loss_scaler, lr_scheduler)
    if best_so_far is None:
        best_so_far = float("inf")

    log_writer = SummaryWriter(log_dir=args.output_dir) if (global_rank == 0 and args.output_dir) else None

    def save(epoch, fname):
        save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, lr_scheduler,
                   fname=fname, best_so_far=best_so_far)

    logger.info("Start training for %d epochs (%d optimizer steps)", args.epochs, effective_steps)
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
            for test_name, testset in test_datasets.items():
                stats = test_jepa(model, testset, device, epoch, args, log_writer, prefix=test_name)
                test_stats[test_name] = stats
                if stats.get("loss_med", float("inf")) < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_stats = dict(epoch=epoch, **{f"train_{k}": v for k, v in (train_stats or {}).items()})
            for test_name, stats in (test_stats or {}).items():
                log_stats.update({f"{test_name}_{k}": v for k, v in stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch > args.start_epoch and args.keep_freq and epoch % args.keep_freq == 0:
            save(epoch - 1, str(epoch))

        if new_best:
            save(epoch - 1, "best")

        if epoch >= args.epochs:
            break

        train_stats = train_one_epoch(
            model=model, model_without_ddp=model_without_ddp, loss_scaler=loss_scaler,
            data_loader=data_loader_train, optimizer=optimizer, device=device, epoch=epoch,
            effective_steps=effective_steps, lr_scheduler=lr_scheduler, args=args,
            log_writer=log_writer,
        )

    logger.info("Training time %s", str(datetime.timedelta(seconds=int(time.time() - start_time))))
    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)