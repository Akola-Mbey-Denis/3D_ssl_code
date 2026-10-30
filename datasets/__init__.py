# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# from vggt.utils.transforms import *
from .batched_sampler import BatchedRandomSampler  # noqa
from .habitat_hm3d_NVS import Habitat_HM3D_NVS
from .hypersim_NVS import Hypersim_NVS
from .taskonomy_NVS import Taskonomy_NVS
from .replica_NVS import Replica_NVS
from .matterport_NVS import Matterport3D_NVS
from .megadepth_NVS import MegaDepth_NVS

from .dl3dv_NVS import DL3DV_NVS
from .map_free_NVS import Mapfree_NVS
from .scannet_NVS import ScanNet_NVS
from .arkitscenes_NVS import Arkitscenes_NVS
from .arkitscenes_LDI import ARKitScenes_LDI
from .matterport3D_LDI import Matterport3D_LDI
from .scannet_LDI import ScanNet_LDI
from .scannetpp_LDI import ScannetPP_LDI

#evaluation datasets
from .dtu_NVS import DTU_NVS
from .eth3d_NVS import ETH3D_NVS
from .sevenscenes_NVS import Sevenscenes_NVS
from .nrgbd_NVS import NRGBD_NVS





import os.path as osp
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

# def seed_worker(worker_id):
#     # Create a unique seed for each worker
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    import cv2
    cv2.setNumThreads(0)          # stop OpenCV thread pools oversubscribing the cores
    torch.set_num_threads(1)



def safe_collate(batch):
    try:
        return default_collate(batch)
    except RuntimeError as e:
        print("Batch items:")
        for i, item in enumerate(batch):
            print(f"Item {i}: {[ (k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in item.items() ]}")
        
        # breakpoint()
        # breakpoint()
        raise e


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)


    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=safe_collate,
        prefetch_factor=2,
        persistent_workers=(num_workers > 0),
        worker_init_fn = seed_worker
        )

    return data_loader
