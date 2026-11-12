"""
Sanity check for the NVD probe: memorise ONE fixed batch, then test whether the output depends on its inputs.

    python overfit_one_batch.py --train_dataset "Hypersim_NVS(...)" --ckpt_path <ssl run> [--model ...]

Expected for a working probe:
  - loss and abs_rel fall to ~0 within ~2k steps, and the saved prediction is sharp;
  - swapping the sources or the target pose changes the prediction a lot (relative change >> 0.01).
If it cannot memorise a single batch, the bug is in the model / loss path, not in the data.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train_nvd import (prepare_batch, forward_and_loss, median_aligned_metrics, load_encoder, freeze_encoder,
                       build_dataset, DepthLoss)
from models import vista_nvd_base_downstream  # noqa: F401  (used through eval)

p = argparse.ArgumentParser()
p.add_argument("--train_dataset", required=True)
p.add_argument("--model", default="vista_nvd_base_downstream()")
p.add_argument("--ckpt_path", default=None)
p.add_argument("--ckpt_name", default="checkpoint-final.pth")
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--out", default="overfit_one_batch.png")
args = p.parse_args()

device = torch.device("cuda")
loader = build_dataset(args.train_dataset, args.batch_size, num_workers=2, test=True)
x = prepare_batch(next(iter(loader)), device)                      # the ONE batch, reused every step

model = eval(args.model)
if args.ckpt_path:
    load_encoder(model, args.ckpt_path, fname=args.ckpt_name)
freeze_encoder(model)
model.to(device)
criterion = DepthLoss()
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)

for step in range(args.steps + 1):
    model.train()
    out, loss_dict = forward_and_loss(model, criterion, x)
    opt.zero_grad(set_to_none=True)
    loss_dict["objective"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        m = median_aligned_metrics(out["depth"].detach(), out["gt"], out["mask"])
        print(f"step {step:5d}  loss {loss_dict['objective'].item():.4f}  "
              f"abs_rel {m['abs_rel'].item():.4f}  delta1 {m['delta1'].item():.4f}")

# ---- does the output depend on its inputs? ----
model.eval()
with torch.no_grad():
    base = model(x["src_images"], x["src_w2c"], x["tgt_w2c"])["depth"]
    swap_src = model(x["src_images"].roll(1, 0), x["src_w2c"].roll(1, 0), x["tgt_w2c"])["depth"]
    swap_pose = model(x["src_images"], x["src_w2c"], x["tgt_w2c"].roll(1, 0))["depth"]
rel = lambda y: ((y - base).abs().mean() / base.abs().mean()).item()
print(f"relative change  sources swapped: {rel(swap_src):.4f}   target pose swapped: {rel(swap_pose):.4f}")

# ---- save GT vs prediction for the first targets ----
n = min(4, base.shape[0])
fig, ax = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
for b in range(n):
    gt, pr = x["tgt_depth"][b, 0].cpu(), base[b, 0].cpu()
    vmin, vmax = gt[gt > 0].min().item(), gt.max().item()
    ax[0, b].imshow(gt, cmap="Spectral", vmin=vmin, vmax=vmax); ax[0, b].set_title(f"GT {b}")
    ax[1, b].imshow(pr, cmap="Spectral", vmin=vmin, vmax=vmax); ax[1, b].set_title(f"pred {b}")
    for a in ax[:, b]:
        a.axis("off")
plt.tight_layout(); plt.savefig(args.out, dpi=100)
print("saved", args.out)