from typing import Optional, Sequence

import torch
from torch import nn

from models.mum.model import MuMAutoEncoder
from models.heads.dpt_head import LayeredDPTHead


# DINOv2 / Depth Anything DPT depth-head layers, per encoder depth
DINO_DPT_LAYERS = {
    12: [2, 5, 8, 11],      # ViT-B
    24: [4, 11, 17, 23],    # ViT-L
}


def dpt_layers(depth: int) -> list[int]:
    """DINOv2 depth-head layers for ViT-B / ViT-L depths; otherwise the last block of each quarter."""
    return DINO_DPT_LAYERS.get(depth, [depth * (q + 1) // 4 - 1 for q in range(4)])


class MuM_LDI(nn.Module):
    """
    MuM encoder + layered DPT depth head (LDI only, no camera head).

    The layered DPT head reads `intermediate_layer_idx` of the encoder blocks and predicts `num_layers` depths
    per pixel (near -> far), each with a confidence and an existence logit.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_layers: int = 4,                    # depth layers per pixel (layer 0 = visible surface)
        predict_exist: bool = True,             # also predict P(layer k exists) per pixel
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        patch_start_idx: int = 1,
        patch_size: int = 16,
        n_storage_tokens: int = 0,
        intermediate_layer_idx: Optional[Sequence[int]] = None,   # encoder blocks fed to the DPT head
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.patch_size = patch_size
        self.patch_start_idx = patch_start_idx
        self.num_layers = num_layers

        if intermediate_layer_idx is None:
            intermediate_layer_idx = dpt_layers(encoder_depth)
        assert max(intermediate_layer_idx) < encoder_depth, \
            f"{intermediate_layer_idx} out of range for encoder_depth={encoder_depth}"

        self.encoder = MuMAutoEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            n_storage_tokens=n_storage_tokens,
        )

        self.dpt_head = LayeredDPTHead(
            dim_in=embed_dim,
            num_layers=num_layers,
            predict_exist=predict_exist,
            patch_size=patch_size,
            activation="exp",
            conf_activation="expp1",
            intermediate_layer_idx=list(intermediate_layer_idx),
        )

    def forward(self, images, return_intermediate_outputs: bool = True):
        """
        images: (B, S, 3, H, W), H and W multiples of patch_size.
        Returns dict with
            layered_depth, layered_depth_exist (logits or None): (B, S, N, H, W)
            depth layer 0, (B, S, H, W)  -- the ordinary depth
        """
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            aggregated_tokens_list = self.encoder.forward_encoder(x=images, return_all_blocks=return_intermediate_outputs)

        aggregated_tokens_list = [t.float() for t in aggregated_tokens_list]
        predictions = {}
        with torch.autocast(device_type="cuda", enabled=False):
            layered_depth, layered_exist = self.dpt_head(
                images=images,
                aggregated_tokens_list=aggregated_tokens_list,
                patch_start_idx=self.patch_start_idx,
            )
            predictions["layered_depth"] = layered_depth                # (B, S, N, H, W)
            predictions["layered_depth_exist"] = layered_exist
            predictions["depth"] = layered_depth[:, :, 0]               # layer 0 = visible surface
        return predictions


MUM_CONFIGS = {
    "base":  dict(embed_dim=768,  encoder_depth=12, encoder_num_heads=12),   # ViT-B/16, DPT layers 2, 5, 8, 11
    "large": dict(embed_dim=1024, encoder_depth=24, encoder_num_heads=16),   # ViT-L/16, DPT layers 4, 11, 17, 23
}


def _mum_ldi(size: str, **kwargs) -> MuM_LDI:
    return MuM_LDI(**{**MUM_CONFIGS[size], **kwargs})        # any argument can be overridden


def mum_base_ldi(**kwargs) -> MuM_LDI:
    """ViT-B: 768-d, 12 blocks, 12 heads; DPT reads blocks 2, 5, 8, 11 (as DINOv2 ViT-B)."""
    return _mum_ldi("base", **kwargs)


def mum_large_ldi(**kwargs) -> MuM_LDI:
    """ViT-L: 1024-d, 24 blocks, 16 heads; DPT reads blocks 4, 11, 17, 23 (as DINOv2 ViT-L)."""
    return _mum_ldi("large", **kwargs)