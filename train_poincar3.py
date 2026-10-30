#!/usr/bin/env python
"""
Poincar3 SSL pretraining on OUR data and code base.

What comes from Poincar3 (unchanged): the model (student + EMA teacher), Poincar3Loss, and the optimisation
recipe of their trainer:
    AdamW, param groups with layer-wise LR decay + DINOv2 no-weight-decay rule, linear warmup + cosine LR,
    cosine weight-decay schedule, EMA momentum schedule (linear warmup, then cosine), last-layer freezing of the
    prototype heads, grad clipping, their GradScaler settings.
What comes from our code base: the data (our *_NVS datasets through get_data_loader, 6 views per sample: the
4 masked views of our method = Poincar3's student views, all 6 = teacher views, see Poincar3Views),
torchrun / utils.misc distributed setup, epochs + per-GPU batch size, auto-resume from <output_dir>/checkpoint-last.pth,
TensorBoard + log.txt, checkpoint-final.pth that the downstream probes load (finetune_*.py: encoder.* keys).

Poincar3's step-based schedules (500k steps, 10k LR warmup, 30k EMA warmup) are kept as FRACTIONS of the run,
because our runs are defined in epochs: total_steps = epochs x steps per epoch.

    torchrun --nproc_per_node=2 train_poincar3.py \
        --train_dataset "Hypersim_NVS(split='train', ...) + ..." --epochs 110 --batch_size 64 --output_dir runs/poincar3/

The only places that touch Poincar3 internals (check them against the poincar3 package):
    build_model_and_loss()   -> how the student/teacher model and the loss are constructed
    Poincar3Views            -> our 6-view batch -> their Batch (student / teacher views, masks, augmentation)
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

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
from datasets import get_data_loader

logger = logging.getLogger("poincar3")
logging.basicConfig(level=logging.INFO)


# ======================================================================
# Arguments (our launch-script style; Poincar3 defaults where they exist)
# ======================================================================
def get_args_parser():
    p = argparse.ArgumentParser("Poincar3 SSL pretraining on our data", add_help=False)
    # data
    p.add_argument("--train_dataset", required=True, type=str, help="get_data_loader string ('+'-separated)")
    p.add_argument("--batch_size", default=16, type=int, help="sequences (of 6 views) per GPU")
    p.add_argument("--num_workers", default=12, type=int)
    p.add_argument("--img_size", default=224, type=int, help="our crops (patch grid = img_size / 16)")
    p.add_argument("--masked_views", default="s_", choices=["s_", "t_"],
                   help="which of our views are masked = Poincar3 student views (the others = extra teacher views)")
    # run length
    p.add_argument("--epochs", default=110, type=int)
    p.add_argument("--seed", default=0, type=int)
    # optimisation (Poincar3 defaults)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--warmup_frac", default=10_000 / 500_000, type=float, help="LR warmup (Poincar3: 10k of 500k steps)")
    p.add_argument("--layerwise_decay", default=1.0, type=float, help="per-layer LR discount of the encoder (1 = off)")
    p.add_argument("--patch_embed_lr_mult", default=1.0, type=float)
    p.add_argument("--weight_decay", default=0.04, type=float)
    p.add_argument("--weight_decay_end", default=None, type=float, help="cosine ramp target (None = flat)")
    p.add_argument("--grad_clip_norm", default=1.0, type=float)
    p.add_argument("--ema_decay", default=0.999, type=float)
    p.add_argument("--ema_decay_end", default=None, type=float, help="cosine target after warmup (None = flat)")
    p.add_argument("--ema_decay_warmup_start", default=0.994, type=float)
    p.add_argument("--ema_warmup_frac", default=30_000 / 500_000, type=float, help="Poincar3: 30k of 500k steps")
    p.add_argument("--freeze_last_layer_steps", default=2000, type=int, help="prototype last layers frozen (steps)")
    # logging / checkpoints (our style)
    p.add_argument("--print_freq", default=50, type=int)
    p.add_argument("--save_freq", default=1, type=int, help="epochs between checkpoint-last.pth")
    p.add_argument("--keep_freq", default=10, type=int, help="epochs between kept checkpoint-<epoch>.pth")
    p.add_argument("--output_dir", default="./output/", type=str)
    # distributed (filled by misc.init_distributed_mode)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--gpu", default=None, type=int)
    return p


# ======================================================================
# Poincar3 model + loss  (CHECK against the poincar3 package)
# ======================================================================
# def build_model_and_loss():
#     """Student + EMA teacher in one module (has .student, .teacher, .update_teacher(decay)) and Poincar3Loss,
#     both with Poincar3's default configs."""
#     from models.poincar3.loss import Poincar3Loss
#     from models.poincar3.model import Poincar3

#     model = Poincar3(Poincar3.Cfg())
#     loss = Poincar3Loss(Poincar3Loss.Cfg())
#     return model, loss

def build_model_and_loss(size="vitb"):
    """Student + EMA teacher in one module (has .student, .teacher, .update_teacher(decay)) and Poincar3Loss,
    both with Poincar3's  configs."""
    from models.poincar3.loss import Poincar3Loss
    from models.poincar3.model import Poincar3, SSLModel

    if size == "vitb":
        cfg = Poincar3.Cfg(
            encoder_size="vitb",      # per-frame encoder: ViT-B/16 (768-d, 12 blocks, 12 heads)
            embed_dim=768,            # multi-view decoder width = encoder width
            num_heads=12,             # 768 / 12 = 64-d heads, as in ViT-B
            depth=6,                  # 6 frame blocks + 6 inter-frame blocks (as Vista base: 6 frame + 6 global)
            register_attention_block_indices=[1, 3, 4],   # all 6 inter-frame blocks use global attention
        )
    else:
        cfg = Poincar3.Cfg()          # their ViT-L default
    return SSLModel(cfg), Poincar3Loss(Poincar3Loss.Cfg())    # SSLModel = student Poincar3 + EMA teacher copy


class Poincar3Views:
    """Our 6-view NVS sample -> Poincar3's student / teacher views (what their SequenceDataset builds).

    Poincar3: a sequence of M + T frames; the STUDENT sees only the first M frames, masked (iBOT block mask), the
    TEACHER sees all M + T frames unmasked; the losses are computed on the M student frames (the T extra frames only
    give the teacher more context). Each branch gets its own photometric augmentation draw.

    Our data: the masked views are the student views, the unmasked views are the extra teacher views:
        student (M) = 's_images' (masked views of our method, 4)
        teacher     = 's_images' + 't_images' (all 6, unmasked), student frames FIRST (their loss reads P_t[:, :M])
    -> M = 4, T = 2 for every sample. Swap student_key / teacher_extra_key if your masked views are 't_'.

    Our datasets must return UN-augmented images here (their photometric aug is applied twice, independently;
    augmenting an already augmented image would chain the two).
    """

    def __init__(self, img_size=224, patch_size=16, student_key="s_", teacher_extra_key="t_"):
        from models.poincar3.data.masking import MaskingCfg
        from models.poincar3.data.photometric import PhotometricAug

        self.aug = PhotometricAug(PhotometricAug.Cfg())        # their default colour jitter / grayscale / blur
        self.masking = MaskingCfg()                            # their default iBOT masking
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.student_key, self.extra_key = student_key, teacher_extra_key

    @staticmethod
    def _views(x):
        """Collated per-view list [S x (B, ...)] -> (B, S, ...) on the CPU."""
        return torch.stack(list(x), dim=1) if isinstance(x, (list, tuple)) else torch.as_tensor(x)

    def __call__(self, batch: dict, device):
        from models.poincar3.data.masking import build_student_mask
        from models.poincar3.types import Batch

        s, t = self.student_key, self.extra_key
        frames = torch.cat([self._views(batch[f"{s}images"]), self._views(batch[f"{t}images"])], 1).float()  # (B, M+T, 3, H, W)
        K = torch.cat([self._views(batch[f"{s}intrinsics"]), self._views(batch[f"{t}intrinsics"])], 1).float()
        poses = torch.cat([self._views(batch[f"{s}extrinsics"]), self._views(batch[f"{t}extrinsics"])], 1).float()
        M = self._views(batch[f"{s}images"]).shape[1]

        imgs, student_imgs, masks = [], [], []
        for seq in frames:                                     # per sequence, exactly as their dataset does
            views = list(seq)
            student_imgs.append(torch.stack(self.aug(views[:M])))   # student: own augmentation draw
            imgs.append(torch.stack(self.aug(views)))               # teacher: independent draw on the same raw frames
            masks.append(build_student_mask(M, self.grid, self.masking))
        return Batch(
            imgs=torch.stack(imgs),                            # (B, M+T, 3, H, W) teacher views, student frames first
            student_imgs=torch.stack(student_imgs),            # (B, M, 3, H, W)
            mask=torch.stack([torch.as_tensor(m) for m in masks]),   # (B, M, h*w) student patch mask
            poses=poses,                                       # (B, M+T, 4, 4) world-to-camera (not used by the loss)
            K=K,                                               # (B, M+T, 3, 3) (not used by the loss)
            image_paths=[[] for _ in range(len(imgs))],
        ).to(device)


# ======================================================================
# Poincar3 optimisation recipe (from their trainer)
# ======================================================================
class CosineSchedule:
    """Linear warmup from start_warmup_value to base_value, then cosine to final_value (None = flat)."""

    def __init__(self, base_value, total_steps, final_value=None, warmup_steps=0, start_warmup_value=None):
        self.base, self.total, self.final = base_value, total_steps, final_value
        self.warmup, self.start = warmup_steps, start_warmup_value

    def __call__(self, step):
        if step < self.warmup:
            start = self.base if self.start is None else self.start
            return start + (self.base - start) * step / self.warmup
        final = self.base if self.final is None else self.final
        progress = min(step - self.warmup, self.total - self.warmup) / max(self.total - self.warmup, 1)
        return final + 0.5 * (self.base - final) * (1 + math.cos(math.pi * progress))


def no_weight_decay(name):
    """DINOv2: biases, norm parameters and LayerScale gammas are not decayed."""
    return name.endswith(".bias") or "norm" in name or "gamma" in name


def vit_lr_decay_rate(name, layerwise_decay, num_layers):
    """DINOv2-style layer-wise LR decay: stem = heaviest discount, block i -> layer i + 1, rest undiscounted."""
    layer_id = num_layers + 1
    if any(t in name for t in (".patch_embed", ".cls_token", ".mask_token", ".storage_tokens")):
        layer_id = 0
    elif ".blocks." in name:
        layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
    return layerwise_decay ** (num_layers + 1 - layer_id)


def create_optimizer_and_scheduler(student, args, total_steps, warmup_steps):
    """Param groups as Poincar3: per-layer LR multipliers for the encoder, 'other' at lr, each split by
    no_weight_decay and tagged with wd_multiplier (the WD schedule is applied every step)."""
    num_layers = student.encoder.n_blocks
    groups = defaultdict(list)
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            group, lr_mult = "encoder", vit_lr_decay_rate(name, args.layerwise_decay, num_layers)
            if ".patch_embed" in name:
                lr_mult *= args.patch_embed_lr_mult
        else:
            group, lr_mult = "other", 1.0
        groups[(group, lr_mult, not no_weight_decay(name))].append(param)

    optimizer = AdamW([
        {"params": params, "lr": args.lr * lr_mult, "weight_decay": args.weight_decay * float(decay),
         "wd_multiplier": float(decay), "lr_multiplier": lr_mult, "group": group}
        for (group, lr_mult, decay), params in groups.items() if params
    ])
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps),
                    CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


def create_grad_scaler():
    """Poincar3's GradScaler: start at 2^16 and practically never change the scale."""
    return torch.amp.GradScaler("cuda", init_scale=65536, growth_interval=2000,
                                backoff_factor=1 - 1e-8, growth_factor=1 + 1e-8)


def cancel_last_layer_gradients(ssl_model, step, freeze_steps):
    """DINO's freeze_last_layer: no update of the prototype heads' last layer during the first steps."""
    if step >= freeze_steps:
        return
    student = getattr(ssl_model, "module", ssl_model).student
    for head in (student.head, student.global_head):
        for p in head.last_layer.parameters():
            p.grad = None


# ======================================================================
# Checkpoints (our layout)
# ======================================================================
def save_checkpoint(path, args, epoch, step, model, optimizer, scheduler, scaler, loss):
    misc.save_on_master({
        "model": model.state_dict(),                  # student + teacher
        "loss": loss.state_dict() if hasattr(loss, "state_dict") else None,   # e.g. DINO centers
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "step": step, "args": args,
    }, path)


def save_final(path, args, epoch, step, model):
    """checkpoint-final.pth for the downstream probes: 'model' = the TEACHER (EMA) network, whose encoder.* keys
    finetune_*.py load_encoder() reads; the full student/teacher state is kept under 'ssl_model'."""
    misc.save_on_master({"model": model.teacher.state_dict(), "ssl_model": model.state_dict(),
                         "epoch": epoch, "step": step, "args": args}, path)


def maybe_resume(args, model, optimizer, scheduler, scaler, loss):
    ckpt_path = Path(args.output_dir) / "checkpoint-last.pth"
    if not ckpt_path.is_file():
        return 0, 0
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("loss") is not None and hasattr(loss, "load_state_dict"):
        loss.load_state_dict(ckpt["loss"])
    logger.info(f"Resumed {ckpt_path}: epoch {ckpt['epoch'] + 1}, step {ckpt['step']}")
    return ckpt["epoch"] + 1, ckpt["step"]


# ======================================================================
# Training
# ======================================================================
def train_one_epoch(*, model, d_model, loss, to_poincar3_batch, loader, optimizer, scheduler, scaler, lr_wd_ema, step, epoch,
                    args, device, log_writer):
    wd_schedule, ema_schedule = lr_wd_ema
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    for it, batch in enumerate(metric_logger.log_every(loader, args.print_freq, f"Epoch: [{epoch}]")):
        t0 = time.perf_counter()
        poincar3_batch = to_poincar3_batch(batch, device)

        # weight-decay schedule (Poincar3 updates it every step, per group multiplier)
        weight_decay = wd_schedule(step)
        for g in optimizer.param_groups:
            g["weight_decay"] = weight_decay * g["wd_multiplier"]

        # ---- Poincar3 train step ----
        loss_value, stats = loss(batch=poincar3_batch, model=d_model, step=step)
        scaler.scale(loss_value).backward()
        scaler.unscale_(optimizer)
        cancel_last_layer_gradients(d_model, step, args.freeze_last_layer_steps)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=args.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            logger.warning(f"non-finite grad norm at step {step}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # ---- EMA teacher ----
        ema_decay = ema_schedule(step)
        model.update_teacher(ema_decay)

        # ---- logging ----
        lr = next(g["lr"] for g in optimizer.param_groups if g["group"] == "other")
        scalars = {k: float(v) for k, v in stats.items()
                   if k not in ("loss", "lr", "grad_norm")              # already logged explicitly below
                   and ((torch.is_tensor(v) and v.numel() == 1) or isinstance(v, (int, float)))}
        metric_logger.update(loss=float(loss_value.detach()), lr=lr, grad_norm=float(grad_norm), **scalars)
        if log_writer is not None and step % args.print_freq == 0:
            log_writer.add_scalar("train/loss", float(loss_value.detach()), step)
            log_writer.add_scalar("train/lr", lr, step)
            log_writer.add_scalar("train/weight_decay", weight_decay, step)
            log_writer.add_scalar("train/ema_decay", ema_decay, step)
            log_writer.add_scalar("train/grad_norm", float(grad_norm), step)
            log_writer.add_scalar("train/scale", scaler.get_scale(), step)
            log_writer.add_scalar("train/step_time", time.perf_counter() - t0, step)
            for k, v in scalars.items():
                log_writer.add_scalar(f"train/{k}", v, step)
        step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return step, {k: m.global_avg for k, m in metric_logger.meters.items()}


def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Poincar3 torch settings
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("highest")

    # ---------------- data (ours) ----------------
    loader = get_data_loader(args.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_mem=True, shuffle=True, drop_last=True)
    to_poincar3_batch = Poincar3Views(args.img_size, 16, student_key=args.masked_views,
                                      teacher_extra_key="t_" if args.masked_views == "s_" else "s_")
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    ema_warmup_steps = int(args.ema_warmup_frac * total_steps)
    logger.info(f"{steps_per_epoch} steps / epoch x {args.epochs} epochs = {total_steps} steps "
                f"(LR warmup {warmup_steps}, EMA warmup {ema_warmup_steps})")

    # ---------------- model / loss / optimiser (Poincar3) ----------------
    model, loss = build_model_and_loss()
    model.teacher.requires_grad_(False)                  # EMA only: no gradients (also keeps DDP happy)
    model.to(device)
    loss.to(device)
    optimizer, scheduler = create_optimizer_and_scheduler(model.student, args, total_steps, warmup_steps)
    scaler = create_grad_scaler()
    wd_schedule = CosineSchedule(args.weight_decay, total_steps, final_value=args.weight_decay_end)
    ema_schedule = CosineSchedule(args.ema_decay, total_steps, final_value=args.ema_decay_end,
                                  warmup_steps=ema_warmup_steps, start_warmup_value=args.ema_decay_warmup_start)
    n_student = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    logger.info(f"Poincar3: {sum(p.numel() for p in model.parameters()):,} parameters, {n_student:,} trainable (student)")
    breakpoint()

    start_epoch, step = maybe_resume(args, model, optimizer, scheduler, scaler, loss)
    d_model = model
    if args.distributed:
        d_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], gradient_as_bucket_view=False)
    log_writer = SummaryWriter(log_dir=args.output_dir, purge_step=step) if rank == 0 else None
    if rank == 0:
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    # ---------------- loop ----------------
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        for obj in (getattr(loader, "dataset", None), getattr(loader, "sampler", None)):
            if hasattr(obj, "set_epoch"):
                obj.set_epoch(epoch)
        step, stats = train_one_epoch(model=model, d_model=d_model, loss=loss, to_poincar3_batch=to_poincar3_batch,
                                      loader=loader, optimizer=optimizer,
                                      scheduler=scheduler, scaler=scaler, lr_wd_ema=(wd_schedule, ema_schedule),
                                      step=step, epoch=epoch, args=args, device=device, log_writer=log_writer)

        last = epoch + 1 == args.epochs
        if (epoch + 1) % args.save_freq == 0 or last:
            save_checkpoint(Path(args.output_dir) / "checkpoint-last.pth", args, epoch, step, model, optimizer,
                            scheduler, scaler, loss)
        if args.keep_freq and (epoch + 1) % args.keep_freq == 0:
            save_checkpoint(Path(args.output_dir) / f"checkpoint-{epoch + 1}.pth", args, epoch, step, model,
                            optimizer, scheduler, scaler, loss)
        if misc.is_main_process():
            with open(Path(args.output_dir) / "log.txt", "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": step, **{f"train_{k}": v for k, v in stats.items()}}) + "\n")
            if log_writer is not None:
                log_writer.flush()
        if dist.is_initialized():
            dist.barrier()

    save_final(Path(args.output_dir) / "checkpoint-final.pth", args, args.epochs - 1, step, model)
    logger.info(f"Training time {datetime.timedelta(seconds=int(time.time() - t_start))}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())