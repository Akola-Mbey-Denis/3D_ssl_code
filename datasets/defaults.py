import os.path as osp

from fvcore.common.config import CfgNode

_C = CfgNode()
_C.MATTERPORT_PATH = ""
_C.TASKONOMY_PATH = ""
_C.TASKONOMY_COMPRESSED_PATH = ""
_C.TRI_ROBOT_DATA_PATH = ""
_C.START_IND = 0
_C.END_IND = 1000
_C.MIN_NUM_NEIGHBORS = 5
_C.DATALOADER = CfgNode()
_C.DATALOADER.IMG_SIZE = 256
_C.DATALOADER.SPLIT = "medium/train"
_C.DATALOADER.MODE = "train"

_C.SCANNET_PATH = "/nfs/turbo/justincj-turbo/nileshk/SceneTSDF/datasets/ScanNet/"

_C.TDF_AUXILIARY_DATA_PATH = ""
_C.TDF_REFERENCE_DATA_PATH = ""
_C.TDF_MESH_PATH = ""
_C.VIS = False


_C.SCANNET = CfgNode()
_C.SCANNET.VIEWS_PER_SCENE = 50


def get_cfg_defaults() -> CfgNode:
    return _C.clone()
