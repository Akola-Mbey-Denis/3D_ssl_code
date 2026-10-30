import torch
# from models.downstreams.mum import mum_base, mum_large
# from models.downstreams.vista import vista_base, vista_large
# from models.downstreams.muskie import muskie_base, muskie_large
# from models.downstreams.mae import mae_base, mae_large
from models import mum_base_downstream


B = 2
S_ref = 4
S_t = 2
H = W = 224

reference_images = torch.randn(B, S_ref, 3, H, W).cuda()
source_images = torch.randn(B, S_t, 3, H, W).cuda()

model = mum_base_downstream().cuda()

predictions = model(reference_images)
breakpoint()
