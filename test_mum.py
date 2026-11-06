import torch
# from models.downstreams.mum import mum_base, mum_large
# from models.downstreams.vista import vista_base, vista_large
# from models.downstreams.muskie import muskie_base, muskie_large
# from models.downstreams.mae import mae_base, mae_large
from models import poincar3_large_ldi

import torch

B, S, H, W = 2, 4, 224, 224
images = torch.rand(B, S, 3, H, W, device="cuda")                 # [0, 1], like the loader

model = poincar3_large_ldi(num_layers=4).cuda()
# or: model = poincar3_large_ldi(num_layers=4); model.load_pretrained(run_path); model.cuda()

model.eval()
with torch.no_grad():
    pred = model(images)
breakpoint()

for k, v in pred.items():
    print(k, tuple(v.shape), v.dtype, f"{v.min().item():.3f} .. {v.max().item():.3f}")
# expected:
#   layered_depth       (2, 4, 4, 224, 224)   > 0
#   layered_depth_stop  (2, 4, 5, 224, 224)   logits
#   depth               (2, 4, 224, 224)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.encoder.parameters())
print(f"trainable {trainable / 1e6:.1f}M (head only), frozen backbone {frozen / 1e6:.1f}M")
assert not any(p.requires_grad for p in model.encoder.parameters())
breakpoint()
