from typing import Optional, Sequence

import torch
from torch import nn

from models.autoencoder import Encoder
from models.heads.dpt_head import DPTHead
from models.heads.camera_head import CameraHead


def even_layers(depth: int, n: int = 4):
    """n encoder blocks spread evenly from the first to the last (depth 6 -> [0, 2, 3, 5])."""
    return [int(round(i * (depth - 1) / (n - 1))) for i in range(n)]


class Vista(nn.Module):
    """Encoder + DPT depth head (+ optional camera head)."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        output_channels: int = 2,               # depth + confidence
        embed_dim: int = 768,
        encoder_depth: int = 6,
        encoder_num_heads: int = 12,
        patch_start_idx: int = 1,
        patch_size: int = 16,
        n_storage_tokens: int = 0,
        intermediate_layer_idx: Optional[Sequence[int]] = None,   # encoder blocks fed to the DPT head
        enable_camera: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.patch_size = patch_size
        self.patch_start_idx = patch_start_idx

        if intermediate_layer_idx is None:
            intermediate_layer_idx = even_layers(encoder_depth)
        assert max(intermediate_layer_idx) < encoder_depth, \
            f"{intermediate_layer_idx} out of range for encoder_depth={encoder_depth}"

        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            n_storage_tokens=n_storage_tokens,
        )

        self.dpt_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=output_channels,
            patch_size=patch_size,
            activation="exp",
            conf_activation="expp1",
            intermediate_layer_idx=list(intermediate_layer_idx),
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim, global_feat = True, model_type = "vista") if enable_camera else None

    def forward(self, images, return_intermediate_outputs: bool = True):

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
           aggregated_tokens_list, patch_start_idx = self.encoder(images=images,return_intermediate_outputs=return_intermediate_outputs)
        
        aggregated_tokens_list = [t.float() for t in aggregated_tokens_list]
       
        predictions = {}

        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]          # last refinement iteration
                predictions["pose_enc_list"] = pose_enc_list

            depth, depth_conf = self.dpt_head(
                images=images,
                aggregated_tokens_list=aggregated_tokens_list,
                patch_start_idx=patch_start_idx,                     # as reported by the encoder
            )
            predictions["depth"] = depth.squeeze(-1)
            predictions["depth_conf"] = depth_conf

        return predictions




VISTA_CONFIGS = {
    # your current defaults
    "base": dict(embed_dim=768, encoder_depth=6, encoder_num_heads=12, intermediate_layer_idx=[0, 2, 3, 5]),
    # wider (ViT-L width / heads) and twice as deep
    "large": dict(embed_dim=1024, encoder_depth=12, encoder_num_heads=16, intermediate_layer_idx=[2, 5, 8, 11]),
}


def _vista(size: str, **kwargs) -> Vista:
    return Vista(**{**VISTA_CONFIGS[size], **kwargs})        # any argument can be overridden


def vista_base(**kwargs) -> Vista:
    """768-d, 6 encoder blocks, 12 heads; DPT reads blocks 0, 2, 3, 5."""
    return _vista("base", **kwargs)


def vista_large(**kwargs) -> Vista:
    """1024-d, 12 encoder blocks, 16 heads; DPT reads blocks 0, 4, 7, 11."""
    return _vista("large", **kwargs)