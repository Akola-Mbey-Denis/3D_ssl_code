from typing import Optional, Sequence

import torch
from torch import nn

from models.muskie.models import MultiViewCroco, AABlock      # AABlock: import from wherever it is defined
from models.heads.dpt_head import LayeredDPTHead


class Muskie_LDI(nn.Module):
    """
    MultiViewCroco encoder + layered DPT depth head (LDI only, no camera head).

    The layered DPT head reads `intermediate_layer_idx` of the encoder blocks and predicts `num_layers` depths
    per pixel (near -> far), each with a confidence and an existence logit.
    """

    def __init__(
        self,
        block_type=AABlock,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        rope_freq: float = 100,
        enable_checkpoint: bool = False,
        num_ref: int = -1,                      # -1: Muskie default (random count); n >= 0: first n views are references
        num_layers: int = 4,                    # depth layers per pixel (layer 0 = visible surface)
        predict_exist: bool = True,             # also predict P(layer k exists) per pixel
        intermediate_layer_idx: Optional[Sequence[int]] = None,   # encoder blocks fed to the DPT head
    ):
        super().__init__()
        self.patch_start_idx = num_register_tokens
        self.num_layers = num_layers

        if intermediate_layer_idx is None:      # 4 evenly spaced blocks ending at the last one
            intermediate_layer_idx = [depth // 4 * (i + 1) - 1 for i in range(4)]
        assert max(intermediate_layer_idx) < depth, f"{intermediate_layer_idx} out of range for depth={depth}"

        self.encoder = MultiViewCroco(
            block_type=block_type,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            rope_freq=rope_freq,
            enable_checkpoint=enable_checkpoint,
            num_ref=num_ref,
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
            layered_depth, layered_depth_conf, layered_depth_stop (logits or None): (B, S, N, H, W)
            depth, depth_conf: layer 0, (B, S, H, W)  -- the ordinary depth, for eval_ssl.py
        """
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            aggregated_tokens_list = self.encoder.get_latent_embeddings(images=images)

        aggregated_tokens_list = [t.float() for t in aggregated_tokens_list]

        predictions = {}
        with torch.autocast(device_type="cuda", enabled=False):
            layered_depth, layered_depth_stop = self.dpt_head(
                images=images,
                aggregated_tokens_list=aggregated_tokens_list,
                patch_start_idx=self.patch_start_idx,
            )
            predictions["layered_depth"] = layered_depth                # (B, S, N, H, W)
            predictions["layered_depth_stop"] = layered_depth_stop
            predictions["depth"] = layered_depth[:, :, 0]               # layer 0 = visible surface

        return predictions


MUSKIE_CONFIGS = {
    # ViT-B/16
    "base": dict(embed_dim=768, num_ref = -1,  depth=6, num_heads=12, intermediate_layer_idx=[0, 2, 3, 5]),
    # ViT-L/16
    "large": dict(embed_dim=1024, depth=12, num_ref =-1, num_heads=16, intermediate_layer_idx=[2, 5, 8, 11]),
}



def _muskie_ldi(size: str, **kwargs) -> Muskie_LDI:
    cfg = {**MUSKIE_CONFIGS[size], **kwargs}        # any argument can be overridden
    return Muskie_LDI(**cfg)


def muskie_base_ldi(**kwargs) -> Muskie_LDI:
    """ViT-B encoder: 768-d, 12 blocks, 12 heads; DPT reads blocks 2, 5, 8, 11."""
    return _muskie_ldi("base", **kwargs)


def muskie_large_ldi(**kwargs) -> Muskie_LDI:
    """ViT-L encoder: 1024-d, 24 blocks, 16 heads; DPT reads blocks 5, 11, 17, 23."""
    return _muskie_ldi("large", **kwargs)