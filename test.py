import torch

from models import vista_baseline_base

B = 2
S_ref = 4
S_t = 2
H = W = 224

ref_images = torch.randn(B, S_ref, 3, H, W).cuda()
ref_w2c = torch.randn(B, S_ref, 4, 4).cuda()
target_images = torch.randn(B, S_t, 3, H, W).cuda()
target_w2c = torch.randn(B, S_t, 4, 4).cuda()
target_K = torch.randn(B, S_t, 3, 3).cuda()



model = vista_baseline_base().cuda()


# predictions = model(src_images = ref_images, src_w2c = ref_w2c, tgt_K = target_K, tgt_w2c = target_w2c)

# breakpoint()

loss, pred, pred_rgb,  mask = model(ref_images, target_images)
breakpoint()
