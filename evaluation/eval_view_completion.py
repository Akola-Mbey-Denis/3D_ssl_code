#!/usr/bin/env python
"""
Target-view completion under extreme masking (PSNR / LPIPS) for PRETRAINED SSL backbones.

Each sample: N clean source views + 1 target view masked at ratio r. Every model gets the same
sources, the same target and the same mask (seeded per sequence / mask type / ratio, identical
across jobs). Metrics are computed on the masked pixels only.

Modes (--mode)
  single  : target alone, no sources                          (MAE lower bound)
  encoder : target goes through the encoder with the sources  (Muskie, MuM, Vista encoder-side head)
  query   : target is never encoded; reconstructed from the source scene tokens (Vista decoder)

Reuses the checkpoint resolution and the locked, append-only results store of eval_ssl.py.

Usage
  python eval_completion_ssl.py --model models.muskie_pretrain:muskie_base --model_name muskie_v2 \
      --ckpt <muskie_v2_pretrain_run> --mode encoder \
      --dataset "NRGBD_NVS(split='test', config_path='...', img_size=224, max_views=3, n_target_views=1)"
  python eval_completion_ssl.py --model models.vista_pretrain:vista_base --model_name vista \
      --ckpt <vista_pretrain_run> --mode query --dataset "..."
  python eval_completion_ssl.py --show                   # PSNR table of everything stored

The one place to adapt: ssl_reconstruct() -> how each pretrained model is called.
"""

import argparse
import datetime
import hashlib
import importlib
import json
import logging
import os
import os.path as osp
import re
import sys
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from datasets import get_data_loader
from eval_ssl import (CKPT_ROOT, _atomic_pickle, _tensor, ckpt_fingerprint, ckpt_run_name,
                      locked_store, resolve_ckpt)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("eval_completion")

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OVERLAP_BINS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.01)]

try:
    CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]
except Exception:  # noqa: BLE001
    CODE_SHA = "unknown"


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser("SSL target-view completion: PSNR / LPIPS under extreme masking")
    p.add_argument("--model", type=str, help="Pretraining constructor as 'package.module:function'")
    p.add_argument("--model_kwargs", type=str, default="{}")
    p.add_argument("--model_name", type=str, help="Key in the store (default: run folder name)")
    p.add_argument("--mode", choices=["single", "encoder", "query"], default="encoder")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--ckpt_root", type=str, default=CKPT_ROOT)
    p.add_argument("--amp", choices=list(AMP), default="bf16")
    # data
    p.add_argument("--dataset", type=str, default="", help="'+'-separated dataset strings (1 target view each)")
    p.add_argument("--n_sources", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=-1, help="Debug: stop after N batches per dataset")
    # masking
    p.add_argument("--mask_ratios", type=str, default="0.75,0.9,0.95,0.98")
    p.add_argument("--mask_types", type=str, default="random,block")
    p.add_argument("--patch_size", type=int, default=16)
    # model input / output spaces
    p.add_argument("--input_norm", choices=["none", "imagenet"], default="none",
                   help="Space the model expects its input images in (and predicts pixels in)")
    p.add_argument("--pred_space", choices=["pixel", "normpix"], default="pixel",
                   help="normpix: model predicts per-patch normalized pixels (MAE norm_pix_loss); "
                        "de-normalized with the GT patch mean/std, identically for every model")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--save_vis", type=int, default=8, help="Save image grids for the first N sequences")
    p.add_argument("--seed", type=int, default=0)
    # output
    p.add_argument("--results_dir", type=str, default="results/completion")
    p.add_argument("--store", type=str, default="completion.pkl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


# =============================================================================
# Model loading + reconstruction  (ADAPT HERE)
# =============================================================================
def build_model(spec, kwargs):
    if ":" not in spec:
        raise SystemExit(f"--model must be 'package.module:function', got '{spec}'")
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)(**kwargs)


def load_model(args, device):
    model = build_model(args.model, json.loads(args.model_kwargs))
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for k in ("model", "state_dict", "model_state_dict"):
                if k in ckpt:
                    state = ckpt[k]
                    break
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded {args.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            logger.warning(f"Missing (first 10): {missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected (first 10): {unexpected[:10]}")
    return model.to(device).eval()


def ssl_reconstruct(model, images, view_masks, mode):
    """
    images     : (B, V, 3, H, W) in the model's input space. View 0 = TARGET, views 1.. = clean sources
                 (mode 'single': V = 1, target only).
    view_masks : (B, V, N) bool, True = masked. Sources are all False; N = (H/p) * (W/p).
    mode       : 'single' | 'encoder' | 'query' (see module docstring).
    Returns    : (B, N, p*p*3) predicted patches of the TARGET view, in the space set by --pred_space.
                 Patch layout must match patchify() below (row-major patches, (p, p, 3) inside a patch).

    ---- ADAPT HERE ----
    Simplest: add a `reconstruct(images, view_masks, mode)` method to each pretraining model that
      * Muskie / MuM / MAE : runs the encoder on all views with these masks and returns the head's
                             predictions for view 0
      * Vista, encoder     : view 0 as a "B" view (masked, inside the encoder) -> encoder-side linear head
      * Vista, query       : view 0 as a "C" view (never encoded) -> decoder cross-attending to the
                             source tokens -> decoder linear head
    """
    if hasattr(model, "reconstruct"):
        return model.reconstruct(images, view_masks, mode=mode)
    raise NotImplementedError(
        f"{type(model).__name__} has no reconstruct(images, view_masks, mode); implement it or edit "
        "ssl_reconstruct() in eval_completion_ssl.py")


# =============================================================================
# Masks, patches, metrics
# =============================================================================
def make_mask(gh, gw, ratio, kind, seed):
    """(gh*gw,) bool, True = masked. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    n = gh * gw
    n_mask = int(round(ratio * n))
    if kind == "random":
        m = np.zeros(n, bool)
        m[rng.permutation(n)[:n_mask]] = True
    elif kind == "block":  # one contiguous rectangle, topped up with random patches to reach the ratio
        sh, sw = max(1, int(np.sqrt(ratio) * gh)), max(1, int(np.sqrt(ratio) * gw))
        y0, x0 = rng.integers(0, gh - sh + 1), rng.integers(0, gw - sw + 1)
        m2 = np.zeros((gh, gw), bool)
        m2[y0:y0 + sh, x0:x0 + sw] = True
        m = m2.reshape(-1)
        missing = n_mask - m.sum()
        if missing > 0:
            free = np.flatnonzero(~m)
            m[rng.permutation(free)[:missing]] = True
    else:
        raise ValueError(kind)
    return torch.from_numpy(m)


def patchify(img, p):
    """(B, 3, H, W) -> (B, N, p*p*3)."""
    B, C, H, W = img.shape
    x = img.reshape(B, C, H // p, p, W // p, p)
    return torch.einsum("bchpwq->bhwpqc", x).reshape(B, (H // p) * (W // p), p * p * C)


def unpatchify(x, p, H, W):
    """(B, N, p*p*3) -> (B, 3, H, W)."""
    B = x.shape[0]
    x = x.reshape(B, H // p, W // p, p, p, 3)
    return torch.einsum("bhwpqc->bchpwq", x).reshape(B, 3, H, W)


def to_model_space(img, norm):
    if norm == "imagenet":
        return (img - IMAGENET_MEAN.to(img)) / IMAGENET_STD.to(img)
    return img


def from_model_space(img, norm):
    if norm == "imagenet":
        return img * IMAGENET_STD.to(img) + IMAGENET_MEAN.to(img)
    return img


def pred_to_image(pred_patches, target_model, mask, args):
    """Predicted target patches -> (B, 3, H, W) in [0, 1], visible patches pasted from GT."""
    p = args.patch_size
    H, W = target_model.shape[-2:]
    gt_patches = patchify(target_model, p)
    pred = pred_patches.float()
    if args.pred_space == "normpix":
        mean = gt_patches.mean(-1, keepdim=True)
        var = gt_patches.var(-1, keepdim=True)
        pred = pred * (var + 1e-6).sqrt() + mean
    pred = torch.where(mask[..., None], pred, gt_patches)
    return from_model_space(unpatchify(pred, p, H, W), args.input_norm).clamp(0, 1)


def pixel_mask(mask, p, H, W):
    """(B, N) patch mask -> (B, H, W) pixel mask."""
    B = mask.shape[0]
    m = mask.view(B, H // p, W // p).float()
    return m.repeat_interleave(p, 1).repeat_interleave(p, 2).bool()


def masked_psnr(pred, gt, pix):
    """pred/gt (B,3,H,W) in [0,1]; pix (B,H,W). Per-sample PSNR on masked pixels."""
    out = []
    for b in range(pred.shape[0]):
        mse = (pred[b] - gt[b])[:, pix[b]].pow(2).mean().clamp_min(1e-10)
        out.append(float(10 * torch.log10(1.0 / mse)))
    return out


# =============================================================================
# Evaluation loop
# =============================================================================
@torch.no_grad()
def evaluate_dataset(model, loader, name, args, device, lpips_fn, vis_dir):
    ratios = [float(r) for r in args.mask_ratios.split(",")]
    kinds = [k.strip() for k in args.mask_types.split(",")]
    p = args.patch_size
    amp_dtype = AMP[args.amp]
    records, seq_idx = [], 0

    for it, batch in enumerate(loader):
        if 0 <= args.max_batches <= it:
            break
        target = _tensor(batch["t_images"]).float()[:, 0].to(device)                 # (B, 3, H, W)
        sources = _tensor(batch["s_images"]).float()[:, :args.n_sources].to(device)   # (B, S, 3, H, W)
        B, _, H, W = target.shape
        gh, gw = H // p, W // p

        overlap = [float("nan")] * B
        if "s_overlaps" in batch and len(batch["s_overlaps"]):
            ov = _tensor(batch["s_overlaps"]).float()
            ov = ov.reshape(B, -1)[:, :args.n_sources] if ov.dim() > 1 else ov.reshape(-1, B).T[:, :args.n_sources]
            overlap = ov.mean(1).tolist()

        target_m = to_model_space(target, args.input_norm)
        sources_m = to_model_space(sources.flatten(0, 1), args.input_norm).view_as(sources)

        for ki, kind in enumerate(kinds):
            for ri, r in enumerate(ratios):
                mask = torch.stack([make_mask(gh, gw, r, kind, seed=(seq_idx + b) * 1000 + ki * 100 + ri)
                                    for b in range(B)]).to(device)                     # (B, N)
                if args.mode == "single":
                    images, view_masks = target_m[:, None], mask[:, None]
                else:
                    images = torch.cat([target_m[:, None], sources_m], 1)
                    src_masks = torch.zeros(B, sources.shape[1], gh * gw, dtype=torch.bool, device=device)
                    view_masks = torch.cat([mask[:, None], src_masks], 1)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    out = ssl_reconstruct(model, images, view_masks, args.mode)
                pix = pixel_mask(mask, p, H, W)
                if out.dim() == 4:   # model returned the RGB image (B, 3, H, W) in [0, 1]
                    pred = torch.where(pix[:, None], out.float().clamp(0, 1), target)
                else:                # model returned patches (B, P, p*p*3)
                    pred = pred_to_image(out, target_m, mask, args)
                psnrs = masked_psnr(pred, target, pix)
                lp = [float("nan")] * B
                if lpips_fn is not None:
                    lp = lpips_fn(pred * 2 - 1, target * 2 - 1).flatten().tolist()

                for b in range(B):
                    records.append({"seq": seq_idx + b, "mask_type": kind, "mask_ratio": r,
                                    "psnr": psnrs[b], "lpips": lp[b], "overlap": overlap[b]})

                if vis_dir is not None:
                    save_vis(vis_dir, seq_idx, sources, target, pix, pred, kind, r, args.save_vis)

        seq_idx += B
        if it % 20 == 0:
            logger.info(f"[{name}] batch {it}/{len(loader)}")
    return records


def save_vis(vis_dir, seq_idx, sources, target, pix, pred, kind, r, limit):
    from torchvision.utils import save_image
    for b in range(sources.shape[0]):
        if seq_idx + b >= limit:
            return
        masked = target[b] * (~pix[b]).float()[None] + 0.5 * pix[b].float()[None]
        row = list(sources[b]) + [masked, pred[b], target[b]]
        save_image(torch.stack(row), osp.join(vis_dir, f"seq{seq_idx + b:04d}_{kind}_{r:.2f}.png"), nrow=len(row))


def summarize(records):
    """Mean PSNR / LPIPS per (mask_type, ratio), overall and per overlap bin."""
    out = {}
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["mask_type"], rec["mask_ratio"])].append(rec)
    for (kind, r), recs in sorted(groups.items()):
        key = f"{kind}@{r:.2f}"
        out[key] = {"psnr": float(np.mean([x["psnr"] for x in recs])),
                    "lpips": float(np.nanmean([x["lpips"] for x in recs])) if any(
                        np.isfinite(x["lpips"]) for x in recs) else float("nan"),
                    "n": len(recs)}
        for lo, hi in OVERLAP_BINS:
            sub = [x for x in recs if lo <= x["overlap"] < hi]
            if sub:
                out[key][f"psnr_overlap_{lo:.1f}-{min(hi, 1.0):.1f}"] = float(np.mean([x["psnr"] for x in sub]))
    return out


# =============================================================================
# Store
# =============================================================================
def settings_of(args):
    return {"mode": args.mode, "n_sources": args.n_sources, "mask_ratios": args.mask_ratios,
            "mask_types": args.mask_types, "patch_size": args.patch_size, "input_norm": args.input_norm,
            "pred_space": args.pred_space, "model_class": args.model, "model_kwargs": args.model_kwargs,
            "max_batches": args.max_batches}


def store_key(args):
    return f"{args.model_name}[{args.mode}]"


def show_store(args):
    path = osp.join(args.results_dir, args.store)
    if not osp.exists(path):
        print(f"No store at {path}")
        return
    with locked_store(path, write=False) as store:
        runs = store["runs"]
    by_ds = defaultdict(dict)
    for model, per_ds in runs.items():
        for ds, hist in per_ds.items():
            rec = next((r for r in reversed(hist) if r["complete"]), None)
            if rec is not None:
                by_ds[ds][model] = rec["metrics"]
    for ds, models in sorted(by_ds.items()):
        cols = sorted({k for m in models.values() for k in m})
        print(f"\n{ds}  (PSNR on masked pixels, dB)")
        print(f"{'model':<26}" + "".join(f"{c:>14}" for c in cols))
        for model, m in sorted(models.items()):
            print(f"{model:<26}" + "".join(f"{m[c]['psnr']:>14.2f}" if c in m else f"{'--':>14}" for c in cols))


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args.show:
        show_store(args)
        return
    if not (args.model and args.dataset):
        raise SystemExit("--model and --dataset are required unless --show")
    if args.mode == "single" and args.n_sources:
        logger.info("mode 'single': sources are ignored")
    if not args.model_name:
        args.model_name = ckpt_run_name(args.ckpt, args.ckpt_root) if args.ckpt else args.model.split(":")[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", args.model_name):
        raise SystemExit("--model_name may only use letters, digits and . _ + -")
    args.ckpt = resolve_ckpt(args.ckpt, args.ckpt_root) if args.ckpt else None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store_file = osp.join(args.results_dir, args.store)
    ckpt_fp = ckpt_fingerprint(args.ckpt)
    settings = settings_of(args)
    key = store_key(args)

    datasets = [d.strip() for d in args.dataset.split("+") if d.strip()]
    with locked_store(store_file, write=False) as store:
        todo = []
        for ds in datasets:
            name = ds.split("(")[0].strip()
            done = [r for r in store["runs"].get(key, {}).get(name, [])
                    if r["complete"] and r["ckpt"] == ckpt_fp and r["settings"] == settings]
            if done and not args.overwrite:
                logger.info(f"[{key} / {name}] already evaluated -> skipping (use --overwrite)")
                continue
            todo.append((ds, name))
    if not todo:
        logger.info("Nothing to evaluate.")
        return

    model = load_model(args, device)
    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    failures = []
    for ds, name in todo:
        try:
            logger.info(f"Evaluating {key} on {name}")
            loader = get_data_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_mem=True, shuffle=False, drop_last=False)
            vis_dir = None
            if args.save_vis > 0:
                vis_dir = osp.join(args.results_dir, "vis", key, name)
                os.makedirs(vis_dir, exist_ok=True)
            records = evaluate_dataset(model, loader, name, args, device, lpips_fn, vis_dir)

            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            raw_file = osp.join("raw", key, f"{name}__{stamp}__{os.getpid()}.pkl")
            _atomic_pickle(records, osp.join(args.results_dir, raw_file))
            errors = []
            if not records:
                errors.append("no samples evaluated")
            if args.max_batches >= 0:
                errors.append(f"debug run (--max_batches {args.max_batches})")
            rec = {"model": key, "dataset": name, "time": datetime.datetime.now().isoformat(timespec="seconds"),
                   "ckpt": ckpt_fp, "settings": settings, "code_sha": CODE_SHA,
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "num_samples": len({r['seq'] for r in records}),
                   "complete": not errors, "errors": errors, "metrics": summarize(records), "raw_file": raw_file}
            with locked_store(store_file) as store:
                store["runs"].setdefault(key, {}).setdefault(name, []).append(rec)
            for k, v in rec["metrics"].items():
                logger.info(f"[{key} / {name}] {k}: PSNR {v['psnr']:.2f}  LPIPS {v['lpips']:.4f}  (n={v['n']})")
            for e in errors:
                logger.error(f"[{key} / {name}] {e}")
        except Exception:  # noqa: BLE001
            failures.append(name)
            logger.error(f"[{key} / {name}] FAILED:\n{traceback.format_exc()}")
        finally:
            torch.cuda.empty_cache()

    logger.info(f"Results stored in {store_file}. Table: python eval_completion_ssl.py --show "
                f"--results_dir {args.results_dir}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()