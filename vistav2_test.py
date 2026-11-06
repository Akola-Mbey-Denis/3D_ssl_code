import torch

from models import vista_basev2_downstream   # adjust path to where Vistav2 lives

B = 2
S = 6            # all views go through the encoder together (no clean/masked split downstream)
H = W = 224

images = torch.rand(B, S, 3, H, W).cuda()     # [0, 1] like real inputs

model = vista_basev2_downstream().cuda()

# parameter counts: encoder is frozen (run under no_grad), only heads should train
for p in model.encoder.parameters():
    p.requires_grad_(False)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params: {n_total:,} total, {n_train:,} trainable")

predictions = model(images)

for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"{k:14s} {tuple(v.shape)}  {v.dtype}  finite={torch.isfinite(v).all().item()}")
    else:
        print(f"{k:14s} list[{len(v)}] of {tuple(v[0].shape)}")

# expected:
#   depth       (B, S, H, W)
#   depth_conf  (B, S, H, W)
#   pose_enc    (B, S, 9)
#   pose_enc_list list[4] of (B, S, 9)

# backward check: gradients should reach the heads, not the encoder
loss = predictions["depth"].mean() + predictions["pose_enc"].pow(2).mean()
loss.backward()
print("camera head grad:", model.camera_head.pose_branch.fc2.weight.grad is not None)
print("dpt head grad:   ", any(p.grad is not None for p in model.dpt_head.parameters()))
print("encoder grad:    ", any(p.grad is not None for p in model.encoder.parameters()))   # should be False

breakpoint()