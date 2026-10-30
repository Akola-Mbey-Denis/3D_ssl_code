from functools import partial
from typing import Any, Sequence, Tuple, Union, Optional
import torch
import torch.nn.init
from torch import nn
import torch.nn.functional as F
from .utils import named_apply
from .autoencoder import Encoder
from models.heads.dpt_head import DPTHead
from models.heads.camera_head import CameraHead


class VistaDecoders(nn.Module):
    def __init__(
        self,
        img_size=518,
        in_channels=3,
        output_channels=2,
        embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        patch_start_idx=1,
        patch_size=14,
        n_storage_tokens=0,
        enable_camera = True
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.patch_size = patch_size
        self.patch_start_idx = patch_start_idx

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
            dim_in = 2 * embed_dim,
            output_dim = output_channels,
            patch_size = patch_size,
            activation = "exp",
            conf_activation="expp1",
        )

        self.camera_head = CameraHead(dim_in = 2 * embed_dim)  if enable_camera else None


    def forward(self, images, return_intermediate_outputs=True):
        aggregated_tokens_list, patch_start_idx = self.encoder(
            images=images,
            return_intermediate_outputs=return_intermediate_outputs,
        )
        predictions = {}

        with torch.cuda.amp.autocast(enabled =False):
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            predictions["pose_enc_list"] = pose_enc_list


        with torch.cuda.amp.autocast(enabled=False):
            depth, depth_conf = self.dpt_head(
                images=images,
                aggregated_tokens_list = aggregated_tokens_list,
                patch_start_idx = self.patch_start_idx,
            )

            predictions["depth"] = depth.squeeze(-1)
            predictions["depth_conf"] = depth_conf

        return predictions



