from typing import Optional, Sequence

import torch
from torch import nn

from models.autoencoder import Encoder
from models.heads.dpt_head import LayeredDPTHead


def even_layers(depth: int, n: int = 4):
    """n encoder blocks spread evenly from the first to the last (depth 6 -> [0, 2, 3, 5])."""
    return [int(round(i * (depth - 1) / (n - 1))) for i in range(n)]


class Vista_LDI(nn.Module):
    """
    Vista encoder + layered DPT depth head (LDI only, no camera head).

    The layered DPT head reads `intermediate_layer_idx` of the encoder blocks (2 * embed_dim tokens) and
    predicts `num_layers` depths per pixel (near -> far), each with a confidence and an existence logit.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_layers: int = 4,                    # depth layers per pixel (layer 0 = visible surface)
        predict_exist: bool = True,             # also predict P(layer k exists) per pixel
        embed_dim: int = 768,
        encoder_depth: int = 6,
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

        self.dpt_head = LayeredDPTHead(
            dim_in=2 * embed_dim,
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
            layered_depth, layered_depth_conf, layered_depth_exist (logits or None): (B, S, N, H, W)
            depth, depth_conf: layer 0, (B, S, H, W)  -- the ordinary depth, for eval_ssl.py
        """
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            aggregated_tokens_list, patch_start_idx = self.encoder(images=images, return_intermediate_outputs=return_intermediate_outputs)

        aggregated_tokens_list = [t.float() for t in aggregated_tokens_list]

        predictions = {}
        with torch.autocast(device_type="cuda", enabled=False):
            layered_depth, layered_exist = self.dpt_head(
                images=images,
                aggregated_tokens_list=aggregated_tokens_list,
                patch_start_idx=patch_start_idx,                     # as reported by the encoder
            )
            predictions["layered_depth"] = layered_depth                # (B, S, N, H, W)
            predictions["layered_depth_exist"] = layered_exist
            predictions["depth"] = layered_depth[:, :, 0]               # layer 0 = visible surface

        return predictions


VISTA_CONFIGS = {
    # your current defaults
    "base": dict(embed_dim=768, encoder_depth=6, encoder_num_heads=12, intermediate_layer_idx=[0, 2, 3, 5]),
    # wider (ViT-L width / heads) and twice as deep
    "large": dict(embed_dim=1024, encoder_depth=12, encoder_num_heads=16, intermediate_layer_idx=[2, 5, 8, 11]),
}


def _vista_ldi(size: str, **kwargs) -> Vista_LDI:
    return Vista_LDI(**{**VISTA_CONFIGS[size], **kwargs})    # any argument can be overridden


def vista_base_ldi(**kwargs) -> Vista_LDI:
    """768-d, 6 encoder blocks, 12 heads; DPT reads blocks 0, 2, 3, 5."""
    return _vista_ldi("base", **kwargs)


def vista_large_ldi(**kwargs) -> Vista_LDI:
    """1024-d, 12 encoder blocks, 16 heads; DPT reads blocks 0, 4, 7, 11."""
    return _vista_ldi("large", **kwargs)