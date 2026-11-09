import torch

from models import vista_basev3_downstream
# MuMCrossViewAutoEncoder_DPT, MuMCrossViewDecoders

B = 2
S_ref = 4
S_t = 2
H = W = 224

reference_images = torch.randn(B, S_ref, 3, H, W).cuda()
source_images = torch.randn(B, S_t, 3, H, W).cuda()

model = vista_basev3_downstream().cuda()

predictions = model(reference_images)

breakpoint()

# loss, pred, pred_rgb,  mask = model(reference_images, source_images, mask_ratio = 0.75)
# breakpoint()
