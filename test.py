import torch

from models import mum_vit_base, MuMCrossViewAutoEncoder_DPT, MuMCrossViewDecoders

B = 2
S_ref = 4
S_t = 2
H = W = 518

reference_images = torch.randn(B, S_ref, 3, H, W).cuda()
source_images = torch.randn(B, S_t, 3, H, W).cuda()

model = MuMCrossViewDecoders().cuda()

predictions = model(reference_images)
breakpoint()

# loss, pred, pred_rgb,  mask = model(reference_images, source_images, mask_ratio = 0.75)
# breakpoint()
