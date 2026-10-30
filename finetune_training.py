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
from models import mae_base_downstream, mae_large_downstream, mum_base_downstream, mum_large_downstream, muskie_base_downstream, muskie_large_downstream, vista_base_downstream, vista_large_downstream, DepthLoss


import utils.misc as misc  # noqa: E402
from utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402
from utils.inference import vis_depth_maps, vis_images  # noqa: E402
from evals import compute_reconstruction_metrics

# Logger setup
logger = logging.getLogger("vista_trainer")
logging.basicConfig(level=logging.INFO)

criterion = DepthLoss()


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
    parser.add_argument('--model', default="MuMCrossViewAutoEncoder_DPT()",
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
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (absolute lr)')
    parser.add_argument('--warmup_percent_steps', type=float, default=0.10, help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    # parser.add_argument('--freeze_dpt', type=bool, default=False, help='Freeze the DPT head')
    parser.add_argument('--ckpt_path', type=str, default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/checkpoints/mum_model_ssl__large_scale_training/", help='Pretrained weights from stage1')






    # system / distributed
    parser.add_argument('--num_workers', default=8, type=int)
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




def move_selected_keys_to_device(batch: dict, keys, device="cuda"):
    """
    Move selected keys in a batch  to device as float32 tensors.
    """
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        # convert lists/ndarrays to tensors
        if isinstance(value, list):
            first = value[0]
            if isinstance(first, torch.Tensor):
                tensor = torch.stack(value)
            else:
                tensor = torch.from_numpy(np.stack(value))
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            # unknown type, skip
            continue

        batch[key] = tensor.to(dtype=torch.float32, device=device, non_blocking=True)
    return batch

# 
def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler,lr_scheduler, fname=None,  best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None: fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    if best_so_far is not None: to_save['best_so_far'] = best_so_far
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
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only = False)
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
                             criterion,
                             epoch: int,
                             args,
                             log_writer: SummaryWriter = None,
                             prefix: str = "test",
                             total_steps : int = 0,
                             write_freq: int = 50,
                             max_depth=8.0):
    """
    Test loop for MUM model 
    Returns aggregated metrics dict.
    """

    model.eval()
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
        tensor_keys = ["s_images", "t_images", "s_depths", "t_depths", "s_depth_masks", "t_depth_masks"]
        batch = move_selected_keys_to_device(batch, tensor_keys, device=device)
        source_images = batch["s_images"]
        target_images = batch["t_images"]
        source_depths = batch["s_depths"]
        target_depths = batch["t_depths"]
        target_masks = batch["t_depth_masks"]
        source_masks = batch["s_depth_masks"]

        images = torch.concat([source_images, target_images], dim = 1)
        depths = torch.concat([source_depths, target_depths], dim = 1)
        masks = torch.concat([source_masks, target_masks], dim = 1)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Model forward under autocast
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            predictions  = model(images)

        loss_dict = criterion(predictions = predictions, gt_depths = depths, masks = masks)

        loss_value = float(loss_dict["loss_depth"]) 
        filtered_loss_dict = {k: v for k, v in loss_dict.items() if k != "loss"}

        # Update metric logger
        metric_logger.update(loss=loss_value, **filtered_loss_dict)

        
        if log_writer is not None and batch_idx % write_freq == 0:        
            gt_depths = vis_depth_maps(depths)
            pred_depths = vis_depth_maps(predictions["depth"])

            for b in range(gt_depths.shape[0]):
                for s in range(gt_depths.shape[1]):
                    log_writer.add_image(f"{prefix}_val/gt_depth_view_{b}_{s}", gt_depths[b, s], epoch, dataformats='HWC')
                    log_writer.add_image(f"{prefix}_val/pred_depth_view_{b}_{s}", pred_depths[b, s], epoch, dataformats='HWC')

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
    criterion,
    loss_scaler,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: float,
    clip_grad = 1.0, 
    max_depth = 8.0,
    lr_scheduler=None,
    args=None,
    log_writer: SummaryWriter = None
):
    """
    Train the diffusion model
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

    # helper counters / thresholds from args (with safe defaults)
    grad_clip_norm = float(getattr(args, "grad_clip_norm", clip_grad))
    cur_param_update_step = 0
    
    # Track skipped steps for monitoring
    skipped_steps = 0

    for step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + step / len(data_loader)

        global_step = (epoch * len(data_loader) + step) // accum_iter

        # Move inputs to device
        tensor_keys = ["s_images", "t_images", "s_depths", "t_depths", "s_depth_masks", "t_depth_masks"]
        batch = move_selected_keys_to_device(batch, tensor_keys, device=device)
        source_images = batch["s_images"]
        target_images = batch["t_images"]
        source_depths = batch["s_depths"]
        target_depths = batch["t_depths"]
        target_masks = batch["t_depth_masks"]
        source_masks = batch["s_depth_masks"]

        images = torch.concat([source_images, target_images], dim = 1)
        depths = torch.concat([source_depths, target_depths], dim = 1)
        masks = torch.concat([source_masks, target_masks], dim = 1)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Model forward under autocast
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            predictions  = model(images)

        loss_dict = criterion(predictions = predictions, gt_depths = depths, masks = masks)
        total_loss = loss_dict["loss_depth"]

        loss_scaler.scale(total_loss / accum_iter).backward()

        # Gradient accumulation: only perform optimizer operations every accum_iter
        update_grads = (step + 1) % accum_iter == 0
        
        if update_grads:
            skip_optimizer_step = False
            skip_reason = None
            
            # Check 1: NaN or Inf loss
            if not torch.isfinite(total_loss):
                print(f"[WARNING] NaN or Inf loss detected at step {step}, skipping optimizer step")
                skip_optimizer_step = True
                skip_reason = "NaN/Inf"
                
            # Process gradients only if not skipping
            if not skip_optimizer_step:
                # Unscales the gradients
                loss_scaler.unscale_(optimizer)
                
                # For all gradients, safely change NaN -> 0., inf -> 1e-6, -inf -> -1e-6
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
                
                # Clip gradients if specified
                if grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    
                    # # Monitor gradient norm
                    # if grad_norm > grad_clip_norm * 10:
                    #     print(f"[WARNING] Very large gradient norm: {grad_norm:.2f} (clipped to {grad_clip_norm})")
                
                # Perform optimizer step
                loss_scaler.step(optimizer)
                cur_param_update_step += 1
                
            
                # Update scaler ONLY after successful step
                loss_scaler.update()
                
                # Update learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()
            else:
                # Skip detected - handle gracefully
                print(f"[SKIP] Reason: {skip_reason}, Scaler: {loss_scaler.get_scale():.2f}")

                # Reset scaler to prevent runaway scaling - DIRECT MODIFICATION
                if skip_reason == "NaN/Inf":
                    # More aggressive reset for NaN/Inf
                    new_scale = max(loss_scaler._init_scale, loss_scaler._scale * 0.5)
                    loss_scaler._scale = torch.tensor(new_scale).to(loss_scaler._scale.device)
                    loss_scaler._growth_tracker.fill_(0)
                else:
                    # Gentler reset for excessive loss
                    new_scale = max(loss_scaler._init_scale, loss_scaler._scale * 0.9)
                    loss_scaler._scale = torch.tensor(new_scale).to(loss_scaler._scale.device)
                    loss_scaler._growth_tracker.fill_(0)
                
                print(f"[SKIP] Scaler reset to: {loss_scaler.get_scale():.2f}")
                
                # Still update LR scheduler to maintain training schedule
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # Set loss to zero for logging purposes
                total_loss = torch.zeros_like(total_loss)
            
            # Always zero gradients for next accumulation
            optimizer.zero_grad(set_to_none=True)

        # Store for logging
        filtered_loss_dict = {k: v for k, v in loss_dict.items() if k != "loss"}

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f, lr=lr, loss=total_loss, **filtered_loss_dict)

        if update_grads and ((step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(total_loss)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            log_writer.add_scalar('train_skipped_steps', skipped_steps, epoch_1000x)
            for name, val in filtered_loss_dict.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

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


def warmup_then_linear_decay_lr(total_steps, warmup_ratio=0.1, hold_ratio=0.7):
    warmup_steps = int(warmup_ratio * total_steps)
    hold_steps = int(hold_ratio * total_steps)

    def lr_lambda(step):
        # 1. Warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # 2. Hold base LR
        elif step < hold_steps:
            return 1.0

        # 3. Linear decay to zero
        else:
            decay_steps = total_steps - hold_steps
            remaining = total_steps - step
            return max(0.0, remaining / max(1, decay_steps))

    return lr_lambda


def cosine_warmup_hold_lr(
    total_steps,
    warmup_ratio=0.10,
    decay_ratio=0.80,
    final_ratio=0.10,
):
    """
    Schedule:
        [0, warmup]              : linear warmup 0 -> 1
        [warmup, warmup+decay]   : cosine decay 1 -> final_ratio
        [end_decay, total_steps] : constant final_ratio
    """

    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = int(total_steps * decay_ratio)

    decay_start = warmup_steps
    decay_end = decay_start + decay_steps

    assert decay_end <= total_steps

    def lr_lambda(step):

        # warmup
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        # cosine decay
        elif step < decay_end:

            progress = (
                step - decay_start
            ) / max(1, decay_steps)

            cosine = 0.5 * (
                1 + math.cos(math.pi * progress)
            )

            return final_ratio + (1 - final_ratio) * cosine

        # hold final lr
        else:
            return final_ratio

    return lr_lambda




def load_and_freeze_vista_encoder(model, checkpoint_path):

    checkpoint = torch.load(
        f"{checkpoint_path}/checkpoint-best.pth",
        map_location="cpu",
        weights_only=False
    )
   
    # Handle common checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Extract encoder weights
    encoder_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key[len("encoder."):]] = value

    # Load vista encoder weights
    msg = model.encoder.load_state_dict(
        encoder_state_dict,
        strict=True,
    )

    print("Loaded vista encoder:")
    print(msg)

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.encoder.eval()

    return model
    
    
def train(args):

    # initialize distributed / misc
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume if exists
    last_ckpt_path_fname = os.path.join(args.output_dir, 'checkpoint-best.pth')
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
        # save_model(args=args, model_without_ddp = model_without_ddp,  loss_scaler = loss_scaler,  optimizer = optimizer, epoch = epoch, fname = fname, best_so_far = best_so_far)
        save_model(args=args, model_without_ddp = model_without_ddp,  loss_scaler = loss_scaler,  optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    
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
    # print("Model = %s" % str(model_without_ddp))    
    if args.ckpt_path is not None:
        model = load_and_freeze_vista_encoder(model, args.ckpt_path)

    logger.info("Model parameter count: %s", sum(p.numel() for p in model.parameters()))
   

    if args.distributed:
        # Wrap with DDP (keep wrapper!)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
            static_graph=True,
        )

        model_without_ddp = model.module


    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay = args.weight_decay)    
    loss_scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch = len(data_loader_train)
    total_steps = steps_per_epoch * args.epochs
    effective_steps= total_steps//args.accum_iter

    # 10 % warmup
    num_warmup_steps = int(args.warmup_percent_steps * effective_steps)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=cosine_warmup_hold_lr(total_steps = effective_steps))

    # resume / checkpoint helper
    best_so_far = load_model(args=args, model_without_ddp = model_without_ddp, optimizer=optimizer, lr_scheduler = lr_scheduler, loss_scaler = loss_scaler)

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
                stats = test_ss_3d_model(model=model,data_loader=testset, criterion = criterion, total_steps = total_steps, device=device,epoch=epoch, args=args, log_writer=log_writer, prefix=test_name)
                test_stats[test_name] = stats

                # if a 'loss_med' summary exists, check best
                if stats.get('loss_med', float('inf')) < best_so_far:
                    best_so_far = stats['loss_med']
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
        train_stats = train_ss_3d_model(model=model, loss_scaler = loss_scaler, criterion = criterion,
                                               data_loader=data_loader_train, optimizer=optimizer,
                                               device=device, epoch=epoch,
                                               lr_scheduler=lr_scheduler, 
                                               total_steps = total_steps,
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
