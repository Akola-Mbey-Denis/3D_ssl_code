from .model import vista_base, vista_small, vista_large
from .modelv2 import vista_v2_base, vista_v2_large, vista_v2_small
from .modelv3 import vista_v3_base, vista_v3_large
# from .model_depth import MuMCrossViewAutoEncoder_DPT
from .vista_decoders import  VistaDecoders
from .mum import mum_vit_base as mum_base
from .mum import mum_vit_large as mum_large
from .mae import mae_vit_base_patch16 as mae_base
from .mae import  mae_vit_large_patch16 as mae_large
from .muskie import muskie_base, muskie_large
# from .jepa import vista_jepa_base, vista_jepa_large
from .losses import MultitaskLoss, DepthLoss, LayeredDepthLoss
from .downstreams import mae_base_downstream, mae_large_downstream, mum_base_downstream, mum_large_downstream, muskie_base_downstream, muskie_large_downstream, vista_base_downstream, vista_large_downstream
from .downstreams import mae_base_ldi, mum_base_ldi, muskie_base_ldi, vista_base_ldi, poincar3_large_ldi, vista_basev2_downstream, vista_largev2_downstream, vista_basev3_downstream, vista_largev3_downstream

