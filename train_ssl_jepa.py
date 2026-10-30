"""
Vista JEPA training.

Codebase side (your conventions):
    argparse CLI, get_data_loader(...) datasets, batch keys "s_images" / "t_images",
    utils.misc (distributed, MetricLogger / SmoothedValue logging format),
    save_model / load_model with checkpoint-last / checkpoint-best / checkpoint-<epoch>,
    auto-resume, log.txt (json per epoch), TensorBoard

I-JEPA side (training logic, facebookresearch/ijepa main.py):
    * per-iteration schedules: WarmupCosine LR, Cosine WD, linear EMA momentum generator
    * train_step():  forward_target()  -> EMA encoder, no grad, layer_norm
                     forward_context() -> source encoder + masked target branch + predictor
                     loss_fn()         -> smooth-L1 on the target-block patches
                     backward -> step -> zero_grad -> momentum update of the target encoder
    * grad_logger, resume fast-forwards all schedules by start_epoch * ipe
"""

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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import get_data_loader  # noqa: E402
from models import vista_jepa_base, vista_jepa_large  # noqa: E402,F401
import utils.misc as misc  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vista_jepa_trainer")


# ====================================================================
# Args
# ====================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("Vista JEPA training", add_help=False)

    # -- model / data
    parser.add_argument("--model", default="vista_jepa_base()", type=str,
                        help="string containing the model to build (evaled)")
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", default="[None]", type=str)
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--accum_iter", default=1, type=int, help="1 = I-JEPA behaviour")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)

    # -- optimization (I-JEPA names)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--warmup", default=10, type=float, help="warmup epochs")
    parser.add_argument("--start_lr", default=1.5e-5, type=float)
    parser.add_argument("--lr", default=1.5e-4, type=float, help="reference (peak) lr")
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--weight_decay", default=0.04, type=float)
    parser.add_argument("--final_weight_decay", default=0.4, type=float)
    parser.add_argument("--ema", default=[0.996, 1.0], type=float, nargs=2)
    parser.add_argument("--ipe_scale", default=1.0, type=float)
    parser.add_argument("--clip_grad", default=0.0, type=float, help="0 = off (I-JEPA)")

    # -- init
    parser.add_argument("--ckpt_path", type=str, default=None, help="dir with checkpoint-best.pth to init from")

    # -- logging / checkpointing
    parser.add_argument("--print_freq", default=20, type=int, help="iterations between prints")
    parser.add_argument("--diag_freq", default=200, type=int, help="optimizer steps between diagnostics")
    parser.add_argument("--eval_freq", default=1, type=int, help="epochs between validation")
    parser.add_argument("--save_freq", default=1, type=int, help="epochs between checkpoint-last")
    parser.add_argument("--keep_freq", default=20, type=int, help="epochs between numbered checkpoints")

    # -- distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--gpu", default=None, type=int)
    return parser


# ====================================================================
# I-JEPA schedulers (src/utils/schedulers.py)
# ====================================================================

class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))
        new_wd = max(self.final_wd, new_wd) if self.final_wd <= self.ref_wd else min(self.final_wd, new_wd)
        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd


# ====================================================================
# I-JEPA helpers (src/helper.py, src/utils/logging.py)
# ====================================================================

def init_opt(context_model, wd, final_wd, start_lr, ref_lr, final_lr,
             iterations_per_epoch, warmup, num_epochs, ipe_scale, use_scaler):
    param_groups = [
        {"params": (p for n, p in context_model.named_parameters()
                    if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
                    and "mask_token" not in n and "pos_embed" not in n)},
        {"params": (p for n, p in context_model.named_parameters()
                    if p.requires_grad and (("bias" in n) or (len(p.shape) == 1)
                                            or "mask_token" in n or "pos_embed" in n)),
         "WD_exclude": True, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    total = int(ipe_scale * num_epochs * iterations_per_epoch)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr, T_max=total)
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=wd, final_wd=final_wd, T_max=total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    return optimizer, scaler, scheduler, wd_scheduler


def grad_logger(named_params):
    """First/last qkv grad norm and min/max (I-JEPA)."""
    norms = [p.grad.norm().item() for n, p in named_params
             if p.grad is not None and "qkv" in n and n.endswith("weight")]
    if not norms:
        return None
    return dict(first_layer=norms[0], last_layer=norms[-1], min=min(norms), max=max(norms))


def gpu_timer(closure):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = closure()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def tb_log(writer, prefix, scalars, step):
    """Write a dict of scalars to TensorBoard under '<prefix>/<name>' (rank 0 only)."""
    if writer is None:
        return
    for k, v in scalars.items():
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            writer.add_scalar(f"{prefix}/{k}", v, step)


# ====================================================================
# Vista pieces
# ====================================================================

class ContextModel(nn.Module):
    """
    Trainable path (I-JEPA 'encoder + predictor'):
        source views -> source encoder ------------------------------+
        target view  -> mask -> online target branch -> predictor <--+ -> predictor_norm
    Shares parameters with the VistaJEPA instance; wrapped in DDP on its own.
    """

    def __init__(self, core):
        super().__init__()
        self.encoder = core.encoder
        self.target_branch = core.target_branch
        self.predictor = core.predictor
        self.predictor_norm = core.predictor_norm
        self.reference_projector = getattr(core, "reference_projector", nn.Identity())

    def forward(self, source_imgs, target_imgs, mask):
        ref, psi = self.encoder(source_imgs)
        ref = self.reference_projector(ref[:, :, psi:, :])            # [B,S_ref,P,*]
        q = self.target_branch(target_imgs, mask=mask)                # [B,S_tgt,P,D]
        B, S, P, D = q.shape
        z = self.predictor(target_tokens=q, reference_tokens=ref)
        return self.predictor_norm(z.reshape(B, S, P, D))


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


def amp_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def masked_smooth_l1(z, h, mask):
    """I-JEPA loss restricted to target-block patches. z,h: [B,S,P,D], mask: [B,S,P]."""
    per_patch = F.smooth_l1_loss(z.float(), h.float(), reduction="none").mean(-1)
    return (per_patch * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def feature_std(x, mask=None):
    x = x.float().reshape(-1, x.shape[-1])
    if mask is not None:
        m = mask.reshape(-1).bool()
        if m.any():
            x = x[m]
    return x.std(0).mean().item()


@torch.no_grad()
def effective_rank(x, max_n=4096):
    x = x.float().reshape(-1, x.shape[-1])
    x = x[torch.randperm(x.shape[0], device=x.device)[:max_n]]
    s = torch.linalg.svdvals(x - x.mean(0))
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


def sample_masks(core, tgt, device):
    """masks: 1 = hidden from the target branch; pred_masks: 1 = target block (scored)."""
    B, S, _, H, W = tgt.shape
    return core.sample_mask(B * S, H // core.patch_size, W // core.patch_size, device)


# ====================================================================
# Checkpointing (your save_model / load_model)
# ====================================================================

def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, fname=None, best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / f"checkpoint-{fname}.pth"
    to_save = {
        "model": model_without_ddp.state_dict(),   # encoder, target_branch, predictor, target_encoder
        "optimizer": optimizer.state_dict(),
        "loss_scaler": loss_scaler.state_dict(),
        "args": args,
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
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
        best_so_far = checkpoint.get("best_so_far")
        if best_so_far is not None:
            print(f" & best_so_far={best_so_far:g}")
        print(f"With optim & sched! start_epoch={args.start_epoch}")
    return best_so_far


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    checkpoint_path = Path(args.output_dir) / "checkpoint-final.pth"
    to_save = {"args": args, "model": model_without_ddp.state_dict(), "epoch": epoch}
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(to_save, checkpoint_path)


# ====================================================================
# Evaluation (your MetricLogger format, I-JEPA forward split, fixed masks)
# ====================================================================

@torch.no_grad()
def test_jepa(core, context_model, data_loader: Sized, device, epoch, args,
              log_writer: SummaryWriter = None, prefix="test"):
    core.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))
    header = f"Test Epoch: [{epoch}]"

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(0)

    # same masks every epoch
    rng_state = torch.cuda.get_rng_state()
    cpu_rng_state = torch.get_rng_state()
    torch.cuda.manual_seed(1234 + misc.get_rank())
    torch.manual_seed(1234 + misc.get_rank())
    mask_itr = core._mask_itr

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        batch = move_selected_keys_to_device(batch, ["s_images", "t_images"], device=device)
        src, tgt = batch["s_images"], batch["t_images"]
        B, S = tgt.shape[:2]

        masks, pred_masks = sample_masks(core, tgt, device)
        mask_b = pred_masks.reshape(B, S, -1)

        with torch.amp.autocast("cuda", dtype=amp_dtype()):
            h = F.layer_norm(core.target_encoder(tgt), (core.target_embed_dim,))
            z = context_model(src, tgt, masks)
        loss = masked_smooth_l1(z, h, mask_b).item()
        std_t = feature_std(h)

        stats = {
            "norm_loss": loss / (std_t ** 2 + 1e-8),
            "std_target": std_t,
            "std_pred": feature_std(z, mask_b),
            "context_frac": 1.0 - masks.float().mean().item(),
            "target_frac": pred_masks.float().mean().item(),
        }
        if batch_idx < 10:   # source ablation on a few batches (zeroed source views)
            with torch.amp.autocast("cuda", dtype=amp_dtype()):
                z0 = context_model(torch.zeros_like(src), tgt, masks)
            stats["source_ablation_gap"] = masked_smooth_l1(z0, h, mask_b).item() - loss

        metric_logger.update(loss=loss, **stats)

    torch.cuda.set_rng_state(rng_state)
    torch.set_rng_state(cpu_rng_state)
    core._mask_itr = mask_itr

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    results = {f"{k}_{tag}": getattr(meter, attr)
               for k, meter in metric_logger.meters.items()
               for tag, attr in [("avg", "global_avg"), ("med", "median")]}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(f"{prefix}/val_{name}", val, 1000 * epoch)

    return results


# ====================================================================
# Training (your MetricLogger format, I-JEPA train_step)
# ====================================================================

def train_one_epoch(core, context_model, context_ddp, data_loader: Sized, optimizer, loss_scaler,
                    scheduler, wd_scheduler, momentum_scheduler, device, epoch, ipe, args,
                    log_writer: SummaryWriter = None):

    core.train()                                     # keeps target_encoder in eval mode

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    trainable = [p for p in context_model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    new_lr = optimizer.param_groups[0].get("lr", args.start_lr)
    new_wd = optimizer.param_groups[0].get("weight_decay", args.weight_decay)

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + step / len(data_loader)
        last_micro = (step + 1) % accum_iter == 0
        global_step = epoch * ipe + step // accum_iter

        # -- load images + masks (masks sampled here instead of in a collator)
        batch = move_selected_keys_to_device(batch, ["s_images", "t_images"], device=device)
        src, tgt = batch["s_images"], batch["t_images"]
        masks, pred_masks = sample_masks(core, tgt, device)

        def train_step():
            nonlocal new_lr, new_wd
            if last_micro:
                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()

            def forward_target():
                with torch.no_grad():
                    h = core.target_encoder(tgt)                       # full target view
                    h = F.layer_norm(h, (h.size(-1),))                 # normalize over feature-dim
                    return h

            def forward_context():
                return context_ddp(src, tgt, masks)                    # sources + context block

            def loss_fn(z, h):
                B, S, P, _ = h.shape
                return masked_smooth_l1(z, h, pred_masks.reshape(B, S, P))

            # Step 1. Forward
            with torch.amp.autocast("cuda", dtype=amp_dtype()):
                h = forward_target()
                z = forward_context()
                loss = loss_fn(z, h)

            # Step 2. Backward & step
            loss_scaler.scale(loss / accum_iter).backward()
            grad_stats, grad_norm, m = None, None, None
            if last_micro:
                loss_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, args.clip_grad if args.clip_grad > 0 else float("inf")).item()
                grad_stats = grad_logger(context_model.encoder.named_parameters())
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(core.target_branch.parameters(),
                                                core.target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

            return loss, grad_stats, grad_norm, m, h, z

        (loss, grad_stats, grad_norm, m, h, z), etime = gpu_timer(train_step)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value} at epoch {epoch} step {step}")

        # -- metrics (MetricLogger format)
        B, S, P, _ = h.shape
        mask_b = pred_masks.reshape(B, S, P)
        std_t, std_p = feature_std(h), feature_std(z, mask_b)
        stats = dict(
            norm_loss=loss_value / (std_t ** 2 + 1e-8),
            std_target=std_t,
            std_pred=std_p,
            context_frac=1.0 - masks.float().mean().item(),
            target_frac=pred_masks.float().mean().item(),
        )
        metric_logger.update(epoch=epoch_f, lr=new_lr, loss=loss_value, **stats)
        if last_micro:
            metric_logger.update(wd=new_wd, ema=m, grad_norm=grad_norm)

        # -- TensorBoard (every optimizer step)
        if last_micro:
            loss_value_reduce = misc.all_reduce_mean(loss_value)   # MUST BE EXECUTED BY ALL NODES
            if log_writer is not None:
                epoch_1000x = int(epoch_f * 1000)
                tb_log(log_writer, "train", dict(
                    loss=loss_value_reduce, lr=new_lr, wd=new_wd, ema=m, grad_norm=grad_norm,
                    iter_time_ms=etime, mem_mb=torch.cuda.max_memory_allocated() / 1024. ** 2,
                    **stats), epoch_1000x)
                if grad_stats is not None:
                    tb_log(log_writer, "grad_stats", grad_stats, epoch_1000x)

        # -- diagnostics: collapse + source usage (rank 0)
        if (args.diag_freq > 0 and last_micro and global_step % args.diag_freq == 0
                and misc.is_main_process()):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype()):
                core.eval()
                z0 = context_model(torch.zeros_like(src), tgt, masks)
                core.train()
            diag = dict(
                source_ablation_gap=masked_smooth_l1(z0, h, mask_b).item() - loss_value,
                eff_rank_target=effective_rank(h),
                within_view_std_target=h.float().std(dim=2).mean().item(),
            )
            if grad_stats is not None:
                diag.update({f"grad_{k}": v for k, v in grad_stats.items()})
            print(f"    [diag] step {global_step}: " +
                  "  ".join(f"{k} = {v:.4g}" for k, v in diag.items()))
            tb_log(log_writer, "diag", diag, int(epoch_f * 1000))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_dataset(dataset: str, batch_size: int, num_workers: int, test: bool = False):
    split = "Test" if test else "Train"
    logger.info("Building %s Data loader for dataset: %s", split, dataset)
    loader = get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_mem=True, shuffle=not test, drop_last=not test)
    logger.info("%s dataset length: %d", split, len(loader))
    return loader


# ====================================================================
# Main
# ====================================================================

def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume if exists
    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    args.resume = last_ckpt if os.path.isfile(last_ckpt) else None
    logger.info("job dir: %s", os.path.dirname(os.path.realpath(__file__)))
    logger.info("Args:\n%s", vars(args))

    device = torch.device("cuda")

    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
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
    core = eval(args.model).to(device)                # VistaJEPA (holds target_encoder)
    for p in core.target_encoder.parameters():
        p.requires_grad = False
    model_without_ddp = core

    if args.ckpt_path is not None:
        checkpoint = torch.load(f"{args.ckpt_path}/checkpoint-best.pth", weights_only=False, map_location="cpu")
        core.load_state_dict(checkpoint["model"], strict=True)
        logger.info("Initialized model weights from %s", args.ckpt_path)

    context_model = ContextModel(core)
    logger.info("Params: %s total, %s trainable (EMA target encoder frozen)",
                f"{sum(p.numel() for p in core.parameters()):,}",
                f"{sum(p.numel() for p in context_model.parameters() if p.requires_grad):,}")

    # ---- optimisation (I-JEPA)
    ipe = len(data_loader_train) // args.accum_iter          # optimizer steps per epoch
    optimizer, loss_scaler, scheduler, wd_scheduler = init_opt(
        context_model, wd=args.weight_decay, final_wd=args.final_weight_decay,
        start_lr=args.start_lr, ref_lr=args.lr, final_lr=args.final_lr,
        iterations_per_epoch=ipe, warmup=args.warmup, num_epochs=args.epochs,
        ipe_scale=args.ipe_scale, use_scaler=(amp_dtype() == torch.float16))

    context_ddp = context_model
    if args.distributed:
        context_ddp = DistributedDataParallel(context_model, device_ids=[args.gpu], static_graph=True)

    total_iters = int(ipe * args.epochs * args.ipe_scale)
    momentum_scheduler = (args.ema[0] + i * (args.ema[1] - args.ema[0]) / total_iters
                          for i in range(total_iters + 1))

    # ---- resume, then fast-forward all schedules (I-JEPA)
    best_so_far = load_model(args, model_without_ddp, optimizer, loss_scaler)
    if best_so_far is None:
        best_so_far = float("inf")
    for _ in range(args.start_epoch * ipe):
        scheduler.step()
        wd_scheduler.step()
        next(momentum_scheduler)

    log_writer = SummaryWriter(log_dir=args.output_dir) if (global_rank == 0 and args.output_dir) else None

    def save(epoch, fname):
        save_model(args, epoch, model_without_ddp, optimizer, loss_scaler,
                   fname=fname, best_so_far=best_so_far)

    logger.info("Start training for %d epochs (%d optimizer steps / epoch)", args.epochs, ipe)
    start_time = time.time()
    train_stats, test_stats = {}, {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # -- save 'last' checkpoint at epoch boundaries
        if epoch > args.start_epoch and args.save_freq and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save(epoch - 1, "last")
            if dist.is_initialized():
                dist.barrier()

        # -- evaluate
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in test_datasets.items():
                stats = test_jepa(core, context_model, testset, device, epoch, args,
                                  log_writer=log_writer, prefix=test_name)
                test_stats[test_name] = stats
                if stats.get("loss_med", float("inf")) < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True
            core.train()

        # -- write logs for this epoch
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_stats = dict(epoch=epoch, **{f"train_{k}": v for k, v in (train_stats or {}).items()})
            for test_name, stats in (test_stats or {}).items():
                log_stats.update({f"{test_name}_{k}": v for k, v in stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # -- keep extra checkpoints
        if epoch > args.start_epoch and args.keep_freq and epoch % args.keep_freq == 0:
            save(epoch - 1, str(epoch))

        # -- always save the best checkpoint
        if new_best:
            save(epoch - 1, "best")

        if epoch >= args.epochs:
            break

        # -- train for one epoch
        train_stats = train_one_epoch(
            core=core, context_model=context_model, context_ddp=context_ddp,
            data_loader=data_loader_train, optimizer=optimizer, loss_scaler=loss_scaler,
            scheduler=scheduler, wd_scheduler=wd_scheduler, momentum_scheduler=momentum_scheduler,
            device=device, epoch=epoch, ipe=ipe, args=args, log_writer=log_writer,
        )

    total_time = time.time() - start_time
    logger.info("Training time %s", str(datetime.timedelta(seconds=int(total_time))))
    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)