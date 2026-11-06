"""
Preprocess one split of a low-res LDI dump (ScanNet, ScanNet++, Matterport3D, ARKitScenes or NRGBD) into ONE LMDB of
clean, cropped frames and 6-view sequences. Reads the dump folders directly (no dataset classes).

For every frame of the split that has an LDI (dump_ldis*.py), read ONCE:
  1. load the RGB, depth, camera and LDI; crop / resize RGB + depth to 224 x 224 exactly as the LDI was rendered
     (crop around the principal point, then resize; utils.cropping, aug_crop=-1)
  2. layer-1 check: fraction of pixels (with a depth and a layer 1, depth <= 16 m) where LDI layer 1 is within
     5 cm / 2% of the depth map; frames with < --min_agreement (default 0.8) are dropped
  3. good frames are written to the LMDB right away (already cropped), every frame's agreement is recorded
Then the sequences: one per anchor (every good frame for train; a fixed subset of --n_val / --n_test anchors,
seed 0 / 1, for val / test, test anchors disjoint from the val anchors when both come from the same scenes) =
the anchor + num_views - 1 good frames of the same scene, chosen by --sequence_mode:
  overlap (default): views that SEE WHAT IS HIDDEN in the anchor. The anchor's hidden LDI points (layers >= 1) and
           visible points (layer 0) are lifted to the world and projected into every other good frame of the scene;
           a point is seen by view j if it lands on view j's layer 0 (the rendered visible surface) within
           5 cm / 2%. Views need --min_overlap <= visible overlap <= --max_overlap (shared context, no
           near-duplicates); among those, views are picked greedily to cover the most hidden points not yet covered,
           the rest at random among the valid views (fixed seed per sequence); if fewer than num_views - 1 views are
           valid, the most-overlapping remaining views fill in. Per sequence, the hidden-point coverage is stored in
           __meta__["sequence_info"] (report / bin probe results by it).
  matrix:  from PRECOMPUTED overlap matrices (scannet_overlaps.py / scannetpp_overlaps.py):
           <overlap_basedir>/<scene>/camera_overlap.npy, camera_occlusion.npy  ([target, source]), rows <->
           frame_names.pkl in that folder if present, else --overlap_scene_file[scene] (the rgb list the matrices
           were computed from). For anchor t, sources need --min_overlap <= overlap[t, s] <= --max_overlap and are
           ranked by revealed = overlap[t, s] * occlusion[t, s] (the share of the source's surfaces that lie in t's
           view BEHIND t's visible surface = what the source sees of t's hidden geometry); the top num_views - 1
           are taken (ties / zero scores: random among the valid views, fixed seed), then the most overlapping
           remaining frames if too few are valid. ONLY the frames listed in the matrices (e.g. the <= 1000 frames
           selected per scene) are read, checked and used -- as anchors and as sources; scenes without matrices are
           dropped. The scene file entry may be [rgb paths], (rgb_paths, ...) or {"img_paths": [...], ...}.
           --matrix_rank overlap: sources ranked by overlap[t, s] (ties: occlusion) instead of revealed.
           --strict_overlap: anchors with fewer than num_views - 1 valid sources are DROPPED (no fallback), so every
           sequence has num_views - 1 sources within [--min_overlap, --max_overlap].
           __meta__["sequence_info"]: [{overlap, occlusion, revealed: [per other view], fallback}, ...].
  window:  the old rule: random good frames within +-window of the anchor in the sorted frame list.
Scenes with fewer than num_views good frames get no sequences.
Re-running on a finished LMDB only rebuilds the sequences (frames are already checked); the previous "__meta__" is
kept as "__meta_prev__".

Dumps:
    ScanNet / ScanNet++ / Matterport : <dump_root>/<scene>/rgb/<frame>.jpg, depth/<frame>.npy, cameras.pkl
                           ({rgb name: {K, RT_c2w, depth}})
                           <dump_root>/LDIs/<scene>/<frame>.npz            (ldi (224, 224, L) float16 z-depth)
    ARKitScenes          : <dump_root>/<Training|Validation>/<video_id>/rgb/<frame>.png, depth/<frame>.npy,
                           cameras.npz (frame_ids, K, c2w);   <dump_root>/LDIs/<train|val>/<video_id>/<frame>.npz
    NRGBD (raw dataset)  : <dump_root>/<scene>/images/img<i>.png, depth/depth<i>.png (uint16 mm), poses.txt
                           (4 lines per frame, OpenGL cam2world, 'nan' = invalid), fixed K (fx = fy = 554.256, 320, 240)
                           <dump_root>/LDIs/<scene>/img<i>.npz             (dump_ldis_nrgbd.py)
Splits: Matterport3D houses / ScanNet scenes / ScanNet++ scenes from the lists below (ScanNet test = the val scenes;
ScanNet++ has its own disjoint 240 / 5 / 5 train / val / test scenes), ARKitScenes folds (train = Training,
val / test = Validation); NRGBD: every scene (a zero-shot benchmark: build --split test);
--split_file (one scene per line) overrides.

LMDB (default <dump_root>/LDIs/<dataset>_<split>_ldi_seq<num_views>.lmdb), read by datasets/ldi_lmdb.py:
    "frame:<scene>/<frame>"     pickle {rgb: JPEG bytes 224 x 224 (q95), depth: float16 (224, 224) m, 0 = invalid,
                                        ldi: float16 (L, 224, 224) z-depth near -> far, 0 = none
                                        (raw float16 bytes, not compressed),
                                        K: (3, 3) float32, w2c: (4, 4) float32 world-to-camera}   (good frames only)
    "agreement:<scene>/<frame>" float layer-1 agreement of every checked frame (-1 = unreadable)
    "__meta__"                  pickle {dataset, split, img_size, num_layers, num_views, window, min_agreement,
                                        scene_frames: {scene: [good frames, sorted]},
                                        sequences: [(scene, [frame x num_views]), ...],  anchor first
                                        sequence_mode, sequence_info (overlap mode, aligned with sequences):
                                        [{hidden_cov, n_hidden, overlap: [per other view], n_greedy, fallback}, ...]}
"__meta__" is written last (its presence = complete). Re-running resumes: frames already checked are skipped.

Usage (one job per dataset / split, give it CPUs: the frames are read in --workers processes):
    python preprocess_ldi_lmdb.py --dataset matterport --split train --dump_root /path/to/low_res_matterport
    python preprocess_ldi_lmdb.py --dataset scannetpp --split train --dump_root /path/to/low_res_scannetpp
    python preprocess_ldi_lmdb.py --dataset nrgbd --split test --dump_root /scratch/dma9300/nrgbd/NRGBD/
    python preprocess_ldi_lmdb.py --dataset scannet --split train --dump_root /path/to/low_res_scannet \
        --sequence_mode matrix --overlap_basedir /path/to/low_res_scannet_overlaps \
        --overlap_scene_file /path/to/scannet_low_res_overlap_scene_files.pkl
"""
import argparse
import io
import re
import os
import os.path as osp
import pickle as pkl
import random
from multiprocessing import Pool

import lmdb
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio

import utils.cropping as cropping

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# ======================================================================
# Splits
# ======================================================================
# Matterport3D house splits (61 train / 11 val / 18 test)
MATTERPORT_SPLITS = {
    "train": [
        "17DRP5sb8fy", "1LXtFkjw3qL", "1pXnuDYAj8r", "29hnd4uzFmX", "5LpN3gDmAk7", "5q7pvUzZiYa", "759xd9YjKW5",
        "7y3sRwLe3Va", "82sE5b5pLXE", "8WUmhLawc2A", "aayBHfsNo7d", "ac26ZMwG7aT", "B6ByNegPMKs", "b8cTxDM8gDG",
        "cV4RVeZvu5T", "D7N2EKCX4Sj", "e9zR4mvMWw7", "EDJbREhghzL", "GdvgFV5R1Z5", "gTV8FGcVJC9", "HxpKQynjfin",
        "i5noydFURQK", "JeFG25nYj2p", "JF19kD82Mey", "jh4fc5c5qoQ", "kEZ7cmS4wCh", "mJXqzFtmKg4", "p5wJjkQkbXX",
        "Pm6F8kyY3z2", "pRbA3pwrgk9", "PuKPg4mmafe", "PX4nDJXEHrG", "qoiz87JEwZ2", "rPc6DW4iMge", "s8pcmisQ38h",
        "S9hNv5qa7GM", "sKLMLpTHeUy", "SN83YJsR3w2", "sT4fr6TAbpF", "ULsKaCPVFJR", "uNb9QFRL6hY", "Uxmj2M2itWa",
        "V2XKFyX4ASd", "VFuaQ6m2Qom", "VVfe2KiqLaN", "Vvot9Ly1tCj", "vyrNrziPKCB", "VzqfbhrpDEA", "XcA2TqTSSAj",
        "2n8kARJN3HM", "D7G3Y4RVNrH", "dhjEzFoUFzH", "E9uDoFAP3SH", "gZ6f7yhEvPG", "JmbYfDe2QKZ", "r1Q1Z4BcV1o",
        "r47D5H71a5s", "ur6pFq6Qu1A", "VLzqgDo317F", "YmJkqBEsHnH", "ZMojNkEp431",
    ],
    "val": [
        "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "oLBMNvg9in8", "QUCTc6BB5sX", "TbHJrupSAjP", "X7HyMhZNoso",
        "pLe4wQe7qrG", "x8F5xyUWy9e", "Z6MFQCViBuw", "zsNo4HB9uLZ",
    ],
    "test": [
        "2t7WUuJeko7", "5ZKStnWn8Zo", "ARNzJeq3xxb", "fzynW3qQPVF", "jtcxE69GiFV", "pa4otMbVnkk", "q9vSo1VnCiC",
        "rqfALeAoiTq", "UwV83HsGsw3", "wc2JMjhGNzB", "WYY7iVyf5p8", "YFuZgdQ5vWj", "yqstnuAEVhm", "YVUC4YcDtcY",
        "gxdoqLR6rwA", "gYvKGZ5eRqb", "RPmz2sHmrrY", "Vt2qJdWjCF2",
    ],
}

# ScanNet scene splits (no test split: test = the val scenes)
SCANNET_SPLITS = {
    "train": [
        "scene0001_00", "scene0002_01", "scene0013_01", "scene0016_02", "scene0017_00", "scene0018_00", "scene0020_01", "scene0021_00",
        "scene0026_00", "scene0031_01", "scene0040_00", "scene0041_01", "scene0047_00", "scene0052_02", "scene0053_00", "scene0062_01",
        "scene0066_00", "scene0068_01", "scene0070_00", "scene0072_00", "scene0073_00", "scene0079_00", "scene0089_00", "scene0092_04",
        "scene0097_00", "scene0109_00", "scene0110_02", "scene0111_00", "scene0112_02", "scene0113_01", "scene0117_00", "scene0119_00",
        "scene0120_01", "scene0126_01", "scene0127_01", "scene0133_00", "scene0135_00", "scene0141_02", "scene0142_00", "scene0148_00",
        "scene0155_02", "scene0161_01", "scene0165_01", "scene0166_02", "scene0167_00", "scene0168_00", "scene0171_01", "scene0172_00",
        "scene0173_02", "scene0174_00", "scene0183_00", "scene0184_00", "scene0186_01", "scene0188_00", "scene0191_02", "scene0197_01",
        "scene0199_00", "scene0218_01", "scene0226_01", "scene0228_00", "scene0230_00", "scene0239_02", "scene0243_00", "scene0252_00",
        "scene0262_01", "scene0266_00", "scene0269_02", "scene0272_01", "scene0274_00", "scene0276_01", "scene0279_00", "scene0280_01",
        "scene0284_00", "scene0285_00", "scene0286_00", "scene0290_00", "scene0291_02", "scene0297_00", "scene0299_01", "scene0303_00",
        "scene0309_00", "scene0310_00", "scene0315_00", "scene0318_00", "scene0320_00", "scene0322_00", "scene0325_00", "scene0326_00",
        "scene0330_00", "scene0333_00", "scene0335_01", "scene0337_01", "scene0346_01", "scene0347_02", "scene0348_00", "scene0349_00",
        "scene0358_02", "scene0359_01", "scene0363_00", "scene0366_00", "scene0367_00", "scene0368_01", "scene0370_02", "scene0371_00",
        "scene0375_01", "scene0384_00", "scene0388_00", "scene0390_00", "scene0395_02", "scene0396_02", "scene0398_01", "scene0400_00",
        "scene0403_00", "scene0405_00", "scene0415_00", "scene0416_00", "scene0417_00", "scene0419_00", "scene0429_00", "scene0431_00",
        "scene0433_00", "scene0434_00", "scene0437_00", "scene0445_01", "scene0446_00", "scene0457_02", "scene0466_01", "scene0469_02",
        "scene0470_01", "scene0471_00", "scene0484_00", "scene0485_00", "scene0492_01", "scene0497_00", "scene0501_00", "scene0503_00",
        "scene0505_04", "scene0511_00", "scene0520_00", "scene0523_00", "scene0526_00", "scene0528_00", "scene0531_00", "scene0537_00",
        "scene0542_00", "scene0545_01", "scene0546_00", "scene0547_00", "scene0548_02", "scene0562_00", "scene0567_00", "scene0570_01",
        "scene0577_00", "scene0579_02", "scene0585_00", "scene0590_01", "scene0597_02", "scene0601_00", "scene0603_00", "scene0610_02",
        "scene0611_00", "scene0612_00", "scene0613_02", "scene0619_00", "scene0623_01", "scene0625_01", "scene0630_00", "scene0634_00",
        "scene0639_00", "scene0640_00", "scene0642_03", "scene0646_00", "scene0649_00", "scene0657_00", "scene0662_02", "scene0672_01",
        "scene0673_04", "scene0677_02", "scene0687_00", "scene0688_00",
    ],
    "val": [
        "scene0081_00", "scene0086_00", "scene0144_00", "scene0222_00", "scene0249_00", "scene0277_00", "scene0355_00", "scene0426_00",
        "scene0430_00", "scene0458_00", "scene0490_00", "scene0496_00", "scene0565_00", "scene0568_00", "scene0575_00", "scene0583_00",
        "scene0593_00", "scene0606_00", "scene0607_00", "scene0618_00", "scene0648_00", "scene0651_00", "scene0663_00", "scene0665_00",
        "scene0670_00", "scene0685_00", "scene0686_00", "scene0695_00", "scene0696_00", "scene0699_00",
    ],
}

# ScanNet++ scene splits (250 scenes: 240 train / 5 val / 5 test, scene-level). val / test are the original seeded
# picks (seed 0); every other scene of the dump is train. Disjoint scenes for every split, so test does NOT reuse val.
SCANNETPP_SPLITS = {
    "train": [
        "036bce3393", "076c822ecc", "079a326597", "07f5b601ee", "07ff1c45bb", "08bbbdcc3d", "09bced689e",
        "09c1414f1b", "0a184cf634", "0a5c013435", "0a7cc12c0e", "0b031f3119", "0c5385e84b", "0cf2e9402d",
        "0d2ee665be", "0d41f3a279", "0e75f3c4d9", "104acbf7d2", "108ec0b806", "116456116b", "1204e08f17",
        "1366d5ae89", "16c9bd2e1e", "1841a0b525", "1a130d092a", "1a8e0d78c0", "1ada7a0617", "1ae9e5d2a6",
        "1b75758486", "1b9692f0c7", "1c4b893630", "1f7cbbdde1", "210f741378", "21d970d8de", "251443268c",
        "25927bb04c", "25aa952aa3", "25f3b7a318", "260db9cf5a", "260fa55d50", "27dd4da69e", "281bc17764",
        "286b55a2bf", "28a9ee4557", "290ef3f2c9", "2970e95b65", "2a496183e1", "2b1dc6d6a5", "2e74812d00",
        "302a7f6b67", "303745abc7", "30966f4c6e", "30f4a2b44d", "31a2c91c43", "320c3af000", "32280ecbca",
        "324d07a5b3", "37ea1c52f0", "3864514494", "38d58a7a31", "3928249b53", "394a542a19", "39e6ee46df",
        "39f36da05b", "3db0a1c8f3", "3e8bba0176", "3e928dc2f6", "3f15a9266d", "3f1e1610de", "40aec5fffa",
        "40b56bf310", "410c470782", "419cbe7c11", "41b00feddb", "4422722c49", "45b0dac5e3", "45d2e33be1",
        "47b37eb6f9", "480ddaadc0", "49a82360aa", "4a1a3a7dc5", "4ba22fa7e4", "4c5c60fa76", "4ea827f5a1",
        "50809ea0d8", "52599ae063", "546292a9db", "55b2bf8036", "5654092cc2", "56a0ec536c", "578511c8a9",
        "5942004064", "59e3f1ea37", "5a269ba6fe", "5d152fab1b", "5ee7c22ba0", "5f99900f09", "6115eddb86",
        "61adeff7d5", "646af5e14b", "66c98f4a9b", "67d702f2e8", "6855e1ac32", "689fec23d7", "69e5939669",
        "6b40d1a939", "6cc2231b9c", "6d89a7320d", "6ee2fc1070", "6f1848d1e3", "7079b59642", "709ab5bffe",
        "712dc47104", "75d29d69b8", "7831862f02", "785e7504b9", "7977624358", "7e09430da7", "7e7cd69a59",
        "7eac902fd5", "7f77abce34", "80ffca8a48", "8133208cb6", "824d9cfa6e", "85251de7d1", "87f6d7d564",
        "88627b561e", "88cf747085", "893fb90e89", "8a20d62ac0", "8a35ef3cfe", "8b2c0938d6", "8b5caf3398",
        "8be0cd3817", "8d563fc2cc", "8e00ac7f59", "8e6ff28354", "9071e139d9", "9471b8d485", "94ee15e8ba",
        "9859de300f", "98b4ec142f", "98fe276aa8", "99fa5c25e1", "9f139a318d", "9f21bdec45", "9f79564dbf",
        "a003a6585e", "a05ee63164", "a08dda47a8", "a1d9da703c", "a24f64f7fb", "a29cccc784", "a4e227f506",
        "aaa11940d3", "ab046f8faf", "ab11145646", "ab6983ae6c", "acd69a1746", "acd95847c5", "ad2d07fd11",
        "ada5304e41", "b08a908f0f", "b09431c547", "b0a08200c9", "b1d75ecd55", "b20a261fdf", "b5918e4637",
        "b73f5cdc41", "b97261909e", "bb87c292ad", "bc03d88fc3", "bc2fce1d81", "bc400d86e1", "bcd2436daf",
        "bd7375297e", "bd9305480d", "bde1e479ad", "be0ed6b33c", "bf6e439e38", "bfd3fd54d2", "c06a983e63",
        "c0c863b72d", "c0f5742640", "c173f62b15", "c24f94007b", "c413b34238", "c49a8c6cff", "c4c04e6d6c",
        "c50d2d1d42", "c5439f4607", "c545851c4f", "c856c41c99", "c8f2218ee2", "c9abde4c4b", "cc5237fd77",
        "ccfd3ed9c7", "cf1ffd871d", "d2f44bf242", "d415cc449b", "d6702c681d", "d6d9ddb03f", "d755b3d9d8",
        "d7abfc4b17", "daffc70503", "dc263dfbf0", "dfac5b38df", "e01b287af5", "e050c15a8d", "e0abd740ba",
        "e0de253456", "e1b1d9de55", "e398684d27", "e3ecd49e2b", "e7ac609391", "e7af285f7d", "e898c76c1f",
        "e8e81396b6", "e8ea9b4da8", "e9e16b6043", "ebc200e928", "ed2216380b", "ef18cf0708", "ef25276c25",
        "ef69d58016", "f07340dfea", "f248c2bcdc", "f25f5e6f63", "f2dc06b1d2", "f34d532901", "f3685d06a9",
        "f3d64c30f8", "f5401524e5", "f8062cb7ce", "f8f12e4e6b", "fb05e13ad1", "fb564c935d", "fb5a96b1a2",
        "fd361ab85f", "fe1733741f",
    ],
    "val": ["5eb31827b7", "7bc286c1b6", "a8bf42d646", "a980334473", "ac48a9b736"],
    "test": ["13c3e046d7", "5748ce6f01", "7b6477cb95", "825d228aec", "f9f95681fd"],
}

NRGBD_K = np.array([[554.2562584220408, 0.0, 320.0],
                    [0.0, 554.2562584220408, 240.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)                  # same for every NRGBD scene
ARKIT_FOLDS = {"train": "Training", "val": "Validation", "test": "Validation"}   # split -> dump fold
LDI_FOLDS = {"Training": "train", "Validation": "val"}                           # dump fold -> LDIs/ folder
DATASETS = ["scannet", "scannetpp", "matterport", "arkitscenes", "nrgbd"]


def list_scenes(dataset, dump_root):
    if dataset == "arkitscenes":
        scenes = []
        for fold in ("Training", "Validation"):
            fold_dir = osp.join(dump_root, fold)
            if osp.isdir(fold_dir):
                scenes += [f"{fold}/{v}" for v in sorted(os.listdir(fold_dir))
                           if osp.exists(osp.join(fold_dir, v, "cameras.npz"))]
        return scenes
    if dataset == "nrgbd":                                           # raw NRGBD: images/ + poses.txt per scene
        return [s for s in sorted(os.listdir(dump_root))
                if osp.isdir(osp.join(dump_root, s, "images")) and osp.exists(osp.join(dump_root, s, "poses.txt"))]
    # scannet / scannetpp / matterport: <scene>/cameras.pkl
    return [s for s in sorted(os.listdir(dump_root))
            if s != "LDIs" and osp.exists(osp.join(dump_root, s, "cameras.pkl"))]


def split_scenes(dataset, dump_root, split, split_file=None):
    scenes = list_scenes(dataset, dump_root)
    if dataset == "arkitscenes" and split_file is None:
        return [s for s in scenes if s.startswith(ARKIT_FOLDS[split] + "/")]
    if split_file is not None:
        with open(split_file) as f:
            wanted = {line.strip() for line in f if line.strip()}
    elif dataset == "matterport":
        wanted = set(MATTERPORT_SPLITS[split])
    elif dataset == "scannetpp":                                     # own disjoint train / val / test scenes
        wanted = set(SCANNETPP_SPLITS[split])
    elif dataset == "nrgbd":                                         # zero-shot benchmark: every scene
        wanted = set(scenes)
    else:                                                            # ScanNet: test = the val scenes
        wanted = set(SCANNET_SPLITS["val" if split == "test" else split])
    missing = sorted(wanted - set(scenes))
    if missing:
        print(f"[{dataset}] {len(missing)} {split} scenes not in the dump"
              f" (e.g. {', '.join(missing[:5])}{', ...' if len(missing) > 5 else ''})")
    return [s for s in scenes if s in wanted]


def scene_dirs(dataset, dump_root, scene):
    """(LDI folder, dump folder) of a scene."""
    if dataset == "arkitscenes":
        fold, video_id = scene.split("/")
        return osp.join(dump_root, "LDIs", LDI_FOLDS[fold], video_id), osp.join(dump_root, scene)
    return osp.join(dump_root, "LDIs", scene), osp.join(dump_root, scene)


# ======================================================================
# Cameras
# ======================================================================
def strip_image_ext(name):
    """Base name without an image extension; other dots stay ("41069021_305.244.png" -> "41069021_305.244")."""
    name = osp.basename(str(name))
    stem, ext = osp.splitext(name)
    return stem if ext.lower() in IMAGE_EXTS else name


def arkit_frame_key(name):
    """Same key for "<video_id>_<timestamp>", "<timestamp>" and the timestamp as a number."""
    tail = name.rsplit("_", 1)[-1]
    try:
        return f"{float(tail):.4f}"
    except ValueError:
        return name


def nrgbd_cameras(scene_dir):
    """NRGBD poses.txt -> {"img<i>": {"K", "RT_c2w" (OpenCV cam2world), "depth": "depth<i>.png"}}, nan frames left out."""
    with open(osp.join(scene_dir, "poses.txt")) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    cams = {}
    for i in range(0, len(lines), 4):
        block = lines[i:i + 4]
        if any("nan" in line.lower() for line in block):
            continue
        c2w = np.array([[float(x) for x in line.split()] for line in block], dtype=np.float64)
        c2w[:3, 1:3] *= -1.0                                          # OpenGL -> OpenCV, still cam2world
        cams[f"img{i // 4}"] = {"K": NRGBD_K.copy(), "RT_c2w": c2w, "depth": f"depth{i // 4}.png"}
    return cams


def load_cameras(dataset, scene_dir):
    """{frame key: {"K", "RT_c2w", ("depth")}} of a scene."""
    if dataset == "nrgbd":
        return nrgbd_cameras(scene_dir)
    if dataset == "arkitscenes":
        data = np.load(osp.join(scene_dir, "cameras.npz"), allow_pickle=True)
        Ks, Ps, ids = np.asarray(data["K"], np.float64), np.asarray(data["c2w"], np.float64), data["frame_ids"]
        if Ks.ndim == 2:
            Ks = np.broadcast_to(Ks, (len(ids), 3, 3))
        return {arkit_frame_key(strip_image_ext(n)): {"K": Ks[i], "RT_c2w": Ps[i]} for i, n in enumerate(ids)}
    # scannet / scannetpp / matterport
    with open(osp.join(scene_dir, "cameras.pkl"), "rb") as f:
        return {strip_image_ext(name): cam for name, cam in pkl.load(f).items()}


# ======================================================================
# One frame: read, crop, check, encode
# ======================================================================
def rotate_inverse(se3):
    se3 = np.asarray(se3, np.float64)
    if se3.shape == (3, 4):
        se3 = np.vstack([se3, [0, 0, 0, 1]])
    inv = np.eye(4)
    inv[:3, :3] = se3[:3, :3].T
    inv[:3, 3] = -se3[:3, :3].T @ se3[:3, 3]
    return inv


def crop_resize(image, depthmap, intrinsics, resolution, mask=None):
    """Same crop as the LDI dump scripts / loaders (crop around the principal point, then resize, aug_crop=-1)."""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x, min_margin_y = min(cx, W - cx), min(cy, H - cy)
    bbox = (cx - min_margin_x, cy - min_margin_y, cx + min_margin_x, cy + min_margin_y)
    image, depthmap, intrinsics, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, bbox, mask=mask)

    W, H = image.size
    if H > 1.1 * W:
        resolution = resolution[::-1]
    image, depthmap, intrinsics, mask = cropping.rescale_image_depthmap(
        image, depthmap, intrinsics, np.array(resolution), mask=mask)
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2, mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, bbox, mask=mask)
    return image, depthmap, intrinsics2, mask


def layer1_agreement(ldi, depth, max_depth=16.0, tol=0.05, rel_tol=0.02):
    """Fraction of pixels (with both a depth and a layer 1) where layer 1 is within 5 cm / 2% of the depth."""
    depth = np.where(depth > max_depth, 0.0, depth)
    both = (ldi[0] > 0) & (depth > 0)
    if not both.any():
        return 0.0
    err = np.abs(ldi[0][both] - depth[both])
    return float((err < np.maximum(tol, rel_tol * depth[both])).mean())


def load_depth(dataset, path):
    """Depth in metres, 0 = invalid: .npy dumps (ScanNet / ScanNet++ / Matterport / ARKit) or NRGBD uint16 mm PNGs."""
    if dataset == "nrgbd":
        depth = imageio.imread(path).astype(np.float32) / 1000.0
        depth[(depth > 10) | (depth < 1e-3)] = 0.0                    # as nrgbd_overlaps.py / dump_ldis_nrgbd.py
    else:
        depth = np.load(path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def pack_array(x, dtype=np.float16):
    """Raw float16 bytes, not compressed: reading is a plain np.frombuffer."""
    x = np.ascontiguousarray(np.asarray(x, dtype))
    return {"shape": x.shape, "dtype": str(x.dtype), "data": x.tobytes()}


_CAMERAS = {}                                                        # per worker process: scene -> (cameras, rgb names)


def process_frame(job):
    """(scene, frame, agreement, record bytes or None) for one frame; runs in a worker process."""
    dataset, dump_root, scene, stem, img_size, num_layers, min_agreement = job
    try:
        ldi_dir, scene_dir = scene_dirs(dataset, dump_root, scene)
        rgb_folder = "images" if dataset == "nrgbd" else "rgb"
        if scene not in _CAMERAS:
            rgb_dir = osp.join(scene_dir, rgb_folder)
            rgb = {strip_image_ext(f): f for f in os.listdir(rgb_dir) if f.lower().endswith(IMAGE_EXTS)}
            _CAMERAS.clear()                                         # frames come grouped by scene
            _CAMERAS[scene] = (load_cameras(dataset, scene_dir), rgb)
        cams, rgb = _CAMERAS[scene]
        cam = cams[arkit_frame_key(stem) if dataset == "arkitscenes" else stem]

        image = np.array(Image.open(osp.join(scene_dir, rgb_folder, rgb[stem])).convert("RGB"))
        depth = load_depth(dataset, osp.join(scene_dir, "depth", cam.get("depth", f"{stem}.npy")))
        K = np.asarray(cam["K"], np.float64)
        image, depth, K, mask = crop_resize(image, depth, K, (img_size, img_size), mask=(depth > 0).astype(np.int32))
        depth = np.where(np.asarray(mask) > 0, np.asarray(depth, np.float32), 0.0)

        ldi = np.asarray(np.load(osp.join(ldi_dir, f"{stem}.npz"))["ldi"], np.float32)[..., :num_layers]
        if ldi.shape[:2] != depth.shape:
            raise ValueError(f"LDI {ldi.shape[:2]} vs view {depth.shape}: recompute the LDI")
        ldi = ldi.transpose(2, 0, 1)                                 # (H, W, L) -> (L, H, W)

        agreement = layer1_agreement(ldi, depth)
        if agreement < min_agreement:
            return scene, stem, agreement, None
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        record = pkl.dumps({"rgb": buf.getvalue(), "depth": pack_array(depth), "ldi": pack_array(ldi),
                            "K": K.astype(np.float32), "w2c": rotate_inverse(cam["RT_c2w"]).astype(np.float32)},
                           protocol=pkl.HIGHEST_PROTOCOL)
        return scene, stem, agreement, record
    except Exception as e:                                           # unreadable / inconsistent frame: dropped
        print(f"[skip] {scene}/{stem}: {type(e).__name__}: {e}")
        return scene, stem, -1.0, None


# ======================================================================
# Sequences
# ======================================================================
def frame_sort_key(stem):
    """Numeric order for names with a number (img2 < img10); plain string order otherwise."""
    nums = re.findall(r"\d+", stem)
    return (0, int(nums[-1]), stem) if nums else (1, 0, stem)


def pick_views(pool, anchor, num_views, window, rng):
    """The anchor + num_views - 1 frames of its scene within +-window (whole scene if too few)."""
    i = pool.index(anchor)
    near = [j for j in range(max(0, i - window), min(len(pool), i + window + 1)) if j != i]
    if len(near) < num_views - 1:
        near = [j for j in range(len(pool)) if j != i]
    others = rng.choice(near, size=num_views - 1, replace=False)
    return [anchor] + [pool[j] for j in sorted(others)]


def fixed_anchors(frames, n, seed):
    pool = sorted(frames)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def select_anchors(good, split, num_views, n_val, n_test, test_shares_val_scenes, dataset=None):
    """{scene: sorted good frames} (scenes with >= num_views) and the anchors [(scene, frame), ...]."""
    pools = {}
    for scene, stem in good:
        pools.setdefault(scene, []).append(stem)
    pools = {s: sorted(v, key=frame_sort_key) if dataset == "nrgbd" else sorted(v)
             for s, v in pools.items() if len(v) >= num_views}
    candidates = [(s, n) for s, v in pools.items() for n in v]
    if split == "train":
        anchors = sorted(candidates)
    elif split == "val":
        anchors = fixed_anchors(candidates, n_val, seed=0)
    else:                                                            # test: disjoint from the val anchors
        exclude = set(fixed_anchors(candidates, n_val, seed=0)) if test_shares_val_scenes else set()
        anchors = fixed_anchors([c for c in candidates if c not in exclude], n_test, seed=1)
    return pools, anchors


def window_sequences(pools, anchors, num_views, window):
    return [(scene, pick_views(pools[scene], anchor, num_views, window, np.random.default_rng(i)))
            for i, (scene, anchor) in enumerate(anchors)]


# ======================================================================
# Overlap-based sequences
# ======================================================================
def unpack_array(d):
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])


def backproject(depth, K, c2w, ys, xs):
    """World points of pixels (ys, xs) with depth (z) values; OpenCV pixel centres at integer coordinates."""
    z = depth[ys, xs].astype(np.float64)
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def sample_points(ldi, K, c2w, rng, n_vis, n_hid, min_depth, max_depth, vis_stride=4, hid_stride=2):
    """(visible world points from layer 0, hidden world points from layers >= 1), each subsampled."""
    ok = lambda d: (d > min_depth) & (d < max_depth)
    d0 = ldi[0, ::vis_stride, ::vis_stride]
    ys, xs = np.nonzero(ok(d0))
    if len(ys) > n_vis:
        keep = rng.choice(len(ys), n_vis, replace=False)
        ys, xs = ys[keep], xs[keep]
    vis = backproject(ldi[0], K, c2w, ys * vis_stride, xs * vis_stride)

    hid = []
    for k in range(1, ldi.shape[0]):
        dk = ldi[k, ::hid_stride, ::hid_stride]
        ys, xs = np.nonzero(ok(dk))
        hid.append(backproject(ldi[k], K, c2w, ys * hid_stride, xs * hid_stride))
    hid = np.concatenate(hid) if hid else np.zeros((0, 3))
    if len(hid) > n_hid:
        hid = hid[rng.choice(len(hid), n_hid, replace=False)]
    return vis, hid


def seen_by(pts, w2c, K, depth_lookup, stride, tol=0.05, rel_tol=0.02):
    """(C, n) bool: world point i lands on view c's visible surface (layer 0 at 224 / stride) within tol / rel_tol."""
    C, h, w = depth_lookup.shape
    if len(pts) == 0:
        return np.zeros((C, 0), bool)
    cam = np.einsum("cij,nj->cni", w2c[:, :3, :3], pts) + w2c[:, None, :3, 3]      # (C, n, 3)
    z = cam[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = K[:, None, 0, 0] * cam[..., 0] / zs + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * cam[..., 1] / zs + K[:, None, 1, 2]
    ui = np.round((u + 0.5) / stride - 0.5).astype(np.int64)                        # 224 px -> lookup grid
    vi = np.round((v + 0.5) / stride - 0.5).astype(np.int64)
    inside = (z > 1e-6) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    d = depth_lookup[np.arange(C)[:, None], np.clip(vi, 0, h - 1), np.clip(ui, 0, w - 1)].astype(np.float64)
    return inside & (d > 0) & (np.abs(z - d) < np.maximum(tol, rel_tol * z))


_SEQ = {}                                                            # per worker process: the read-only LMDB


def _seq_worker_init(lmdb_path):
    _SEQ["env"] = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)


def overlap_scene(job):
    """Sequences of one scene: [(global anchor index, [frames], info), ...]; runs in a worker process."""
    (scene, frames, anchor_items, num_views, min_overlap, max_overlap, n_vis, n_hid,
     lookup_stride, min_depth, max_depth) = job
    env = _SEQ["env"]
    Ks, w2cs, lookups, vis_pts, hid_pts = [], [], [], [], []
    with env.begin() as txn:
        for fi, f in enumerate(frames):
            rec = pkl.loads(txn.get(f"frame:{scene}/{f}".encode()))
            ldi = unpack_array(rec["ldi"]).astype(np.float32)                      # (L, 224, 224)
            K = np.asarray(rec["K"], np.float64)
            w2c = np.asarray(rec["w2c"], np.float64)
            c2w = rotate_inverse(w2c)
            vis, hid = sample_points(ldi, K, c2w, np.random.default_rng(fi), n_vis, n_hid, min_depth, max_depth)
            lookup = ldi[0, lookup_stride // 2::lookup_stride, lookup_stride // 2::lookup_stride]
            lookup = np.where((lookup > min_depth) & (lookup < max_depth), lookup, 0).astype(np.float16)
            Ks.append(K); w2cs.append(w2c); lookups.append(lookup); vis_pts.append(vis); hid_pts.append(hid)
    Ks, w2cs, lookups = np.stack(Ks), np.stack(w2cs), np.stack(lookups)
    index = {f: i for i, f in enumerate(frames)}

    results = []
    for gi, anchor in anchor_items:
        rng = np.random.default_rng(gi)
        a = index[anchor]
        overlap = seen_by(vis_pts[a], w2cs, Ks, lookups, lookup_stride).mean(axis=1) if len(vis_pts[a]) \
            else np.zeros(len(frames))
        hidden = seen_by(hid_pts[a], w2cs, Ks, lookups, lookup_stride)                # (C, n_hidden)
        overlap[a], hidden[a] = -1.0, False                                             # never pick the anchor
        valid = (overlap >= min_overlap) & (overlap <= max_overlap)

        chosen, covered = [], np.zeros(hidden.shape[1], bool)
        while len(chosen) < num_views - 1:                                              # greedy: new hidden points
            gain = (hidden & ~covered).sum(axis=1).astype(np.float64)
            gain[~valid] = -1
            gain[chosen] = -1
            j = int(np.argmax(gain + 1e-3 * overlap))                                   # ties -> more overlap
            if gain[j] <= 0:
                break
            chosen.append(j)
            covered |= hidden[j]
        n_greedy = len(chosen)
        rest = [j for j in np.flatnonzero(valid) if j not in chosen]
        if rest and len(chosen) < num_views - 1:                                        # random among valid views
            chosen += [int(j) for j in rng.choice(rest, min(len(rest), num_views - 1 - len(chosen)), replace=False)]
        fallback = len(chosen) < num_views - 1
        if fallback:                                                                    # most overlapping remaining
            for j in np.argsort(-overlap):
                if len(chosen) == num_views - 1:
                    break
                if j != a and j not in chosen:
                    chosen.append(int(j))
        others = sorted(chosen)
        info = {"hidden_cov": float(covered.mean()) if covered.size else float("nan"),
                "n_hidden": int(hidden.shape[1]),
                "overlap": [float(overlap[j]) for j in others],
                "n_greedy": n_greedy, "fallback": bool(fallback)}
        results.append((gi, [anchor] + [frames[j] for j in others], info))
    return results


def _matrix_frame_names(entry):
    """Frame list of one scene from an overlap scene file entry. Accepts
    [paths], (rgb_paths, depth_paths, ...), {"rgb": [...], "depth": [...], ...}."""
    def looks_like_images(x):
        return (isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
                and isinstance(x[0], (str, bytes, os.PathLike))
                and strip_image_ext(x[0]) != osp.basename(str(x[0])))     # has an image extension
    if isinstance(entry, dict):
        for k in ("img_paths", "rgb", "rgbs", "rgb_files", "rgb_paths", "images", "image", "img", "imgs", "frames"):
            if k in entry:
                return list(entry[k])
        cands = [v for v in entry.values() if looks_like_images(v)]
        if cands:
            return list(cands[0])
        raise ValueError(f"no frame list among keys {list(entry)}")
    entry = list(entry)
    if entry and isinstance(entry[0], (list, tuple, np.ndarray)):      # (rgb_list, depth_list, ...)
        rgb = [v for v in entry if looks_like_images(v) and "depth" not in str(v[0]).lower()]
        return list(rgb[0] if rgb else entry[0])
    return entry


def _matrix_stem(name):
    """Matrix row name -> frame stem used in the LMDB (NRGBD: depth<i> -> img<i>)."""
    stem = strip_image_ext(name)
    m = re.fullmatch(r"depth(\d+)", stem)
    return f"img{m.group(1)}" if m else stem


def load_overlap_matrices(overlap_basedir, scene, scene_files):
    """(frame stems of the rows, overlap [t, s], occlusion [t, s]) of a scene, or None if not available /
    inconsistent (the scene then falls back to the window rule, with a warning)."""
    d = osp.join(overlap_basedir, scene)
    if not osp.exists(osp.join(d, "camera_overlap.npy")):
        return None
    overlap = np.load(osp.join(d, "camera_overlap.npy"))
    occlusion = np.load(osp.join(d, "camera_occlusion.npy"))
    try:
        if osp.exists(osp.join(d, "frame_names.pkl")):
            with open(osp.join(d, "frame_names.pkl"), "rb") as f:
                names = _matrix_frame_names(pkl.load(f))
        elif scene_files is not None and scene in scene_files:
            names = _matrix_frame_names(scene_files[scene])
        else:
            print(f"[matrix] {scene}: no frame_names.pkl and not in --overlap_scene_file -> window rule")
            return None
    except Exception as e:                                                  # noqa: BLE001
        print(f"[matrix] {scene}: cannot read the frame list ({e}) -> window rule")
        return None
    if overlap.shape != (len(names), len(names)) or occlusion.shape != overlap.shape:
        print(f"[matrix] {scene}: overlap {overlap.shape} / occlusion {occlusion.shape} vs {len(names)} frame "
              f"names -> window rule")
        return None
    return [_matrix_stem(osp.basename(str(n))) for n in names], overlap, occlusion


def load_scene_files(path):
    if not path:
        return None
    with open(path, "rb") as f:
        return pkl.load(f)


def matrix_frame_sets(args, scenes):
    """{scene: set of frame stems in the scene's overlap matrix (the selected frames)} or None for a scene
    without usable matrices."""
    scene_files = load_scene_files(args.overlap_scene_file)
    out = {}
    for scene in scenes:
        m = load_overlap_matrices(args.overlap_basedir, scene, scene_files)
        out[scene] = set(m[0]) if m is not None else None
    return out


def matrix_sequences(pools, anchors, args):
    """Sequences from precomputed overlap / occlusion matrices (see the module docstring, --sequence_mode matrix)."""
    scene_files = load_scene_files(args.overlap_scene_file)
    V = args.num_views
    sequences, infos, cache = [], [], {}
    n_no_matrix = n_anchor_missing = n_dropped = 0
    for gi, (scene, anchor) in enumerate(tqdm(anchors, desc=f"{args.dataset} [{args.split}] matrix sequences")):
        if scene not in cache:
            cache.clear()                                            # anchors come grouped by scene
            m = load_overlap_matrices(args.overlap_basedir, scene, scene_files)
            if m is None:
                cache[scene] = None
            else:
                stems, ov, occ = m
                row = {s: i for i, s in enumerate(stems)}
                pool = pools[scene]
                idx = np.array([row.get(f, -1) for f in pool])       # good frame -> matrix row (-1 = not in matrix)
                cache[scene] = (pool, idx, ov, occ)
        rng = np.random.default_rng(gi)
        entry = cache[scene]
        pool = pools[scene]
        a = pool.index(anchor)
        if entry is None or entry[1][a] < 0:
            n_no_matrix += entry is None
            n_anchor_missing += entry is not None
            if args.strict_overlap:
                n_dropped += 1
                continue
            sequences.append((scene, pick_views(pool, anchor, V, args.window, rng)))
            infos.append({"overlap": [], "occlusion": [], "revealed": [], "fallback": True, "no_matrix": True})
            continue
        _, idx, ov, occ = entry
        in_m = idx >= 0
        o = np.where(in_m, ov[idx[a], np.clip(idx, 0, None)], -1.0)  # overlap[t = anchor, s] per good frame
        c = np.where(in_m, occ[idx[a], np.clip(idx, 0, None)], 0.0)
        o[a] = -1.0
        valid = in_m & (o >= args.min_overlap) & (o <= args.max_overlap)
        revealed = np.where(valid, o * c, -1.0)

        if args.strict_overlap and valid.sum() < V - 1:               # not enough well-overlapping sources: drop
            n_dropped += 1
            continue
        if args.matrix_rank == "overlap":                            # highest overlap first, ties: occlusion
            cand = np.flatnonzero(valid)
            cand = cand[np.lexsort((rng.random(len(cand)), -c[cand], -o[cand]))]
        else:                                                        # most revealed hidden geometry first
            cand = np.flatnonzero(valid & (revealed > 0))
            cand = cand[np.lexsort((rng.random(len(cand)), -revealed[cand]))]   # best first, random ties
        chosen = [int(j) for j in cand[:V - 1]]
        rest = [j for j in np.flatnonzero(valid) if j not in chosen]
        if len(chosen) < V - 1 and rest:
            chosen += [int(j) for j in rng.choice(rest, min(len(rest), V - 1 - len(chosen)), replace=False)]
        fallback = len(chosen) < V - 1
        if fallback:
            for j in np.argsort(-o, kind="stable"):
                if len(chosen) == V - 1:
                    break
                if j != a and j not in chosen:
                    chosen.append(int(j))
        others = sorted(chosen)
        sequences.append((scene, [anchor] + [pool[j] for j in others]))
        infos.append({"overlap": [float(o[j]) for j in others], "occlusion": [float(c[j]) for j in others],
                      "revealed": [float(max(revealed[j], 0.0)) for j in others], "fallback": bool(fallback),
                      "no_matrix": False})
    if (n_no_matrix or n_anchor_missing) and not args.strict_overlap:
        print(f"[{args.dataset} {args.split}] window-rule fallback: {n_no_matrix:,} anchors in scenes without "
              f"matrices, {n_anchor_missing:,} anchors missing from their scene's matrix")
    if args.strict_overlap:
        print(f"[{args.dataset} {args.split}] strict overlap: dropped {n_dropped:,} of {len(anchors):,} anchors "
              f"without {V - 1} sources with {args.min_overlap:g} <= overlap <= {args.max_overlap:g}; "
              f"kept {len(sequences):,}")
    return sequences, infos


def overlap_sequences(lmdb_path, pools, anchors, args):
    by_scene = {}
    for gi, (scene, anchor) in enumerate(anchors):
        by_scene.setdefault(scene, []).append((gi, anchor))
    jobs = [(scene, pools[scene], items, args.num_views, args.min_overlap, args.max_overlap, args.n_vis_points,
             args.n_hidden_points, args.lookup_stride, 0.1, 16.0) for scene, items in by_scene.items()]
    jobs.sort(key=lambda j: -len(j[1]) * len(j[2]))                                    # big scenes first
    sequences, infos = [None] * len(anchors), [None] * len(anchors)
    with Pool(args.workers, initializer=_seq_worker_init, initargs=(lmdb_path,)) as pool:
        for res in tqdm(pool.imap_unordered(overlap_scene, jobs), total=len(jobs),
                        desc=f"{args.dataset} [{args.split}] overlap sequences (scenes)"):
            for gi, frames, info in res:
                sequences[gi] = (anchors[gi][0], frames)
                infos[gi] = info
    return sequences, infos


# ======================================================================
# Main
# ======================================================================
def main(args):
    if args.sequence_mode == "matrix" and not args.overlap_basedir:
        raise SystemExit("--sequence_mode matrix needs --overlap_basedir (and --overlap_scene_file for ScanNet)")
    scenes = split_scenes(args.dataset, args.dump_root, args.split, args.split_file)
    frames = []
    for scene in scenes:
        ldi_dir, _ = scene_dirs(args.dataset, args.dump_root, scene)
        if osp.isdir(ldi_dir):
            frames += [(scene, f[:-4]) for f in sorted(os.listdir(ldi_dir))
                       if f.endswith(".npz") and not f.endswith(".tmp.npz")]
    if not frames:
        raise SystemExit(f"no LDIs for {args.dataset} [{args.split}] under {osp.join(args.dump_root, 'LDIs')}")

    # matrix mode: ONLY the frames selected for the overlap matrices (anchors and sources)
    if args.sequence_mode == "matrix":
        selected = matrix_frame_sets(args, scenes)
        n_before = len(frames)
        frames = [(s, n) for s, n in frames if selected.get(s) is not None and n in selected[s]]
        no_matrix = sorted(s for s, v in selected.items() if v is None)
        print(f"[{args.dataset} {args.split}] matrix frames only: {len(frames):,} of {n_before:,} frames with an LDI"
              + (f"; scenes without usable matrices dropped: {no_matrix}" if no_matrix else ""))
        if not frames:
            raise SystemExit("no frame of the split is in an overlap matrix -- check --overlap_basedir / "
                             "--overlap_scene_file")

    out = args.out or osp.join(args.dump_root, "LDIs", f"{args.dataset}_{args.split}_ldi_seq{args.num_views}.lmdb")
    if args.overwrite and osp.exists(out):
        os.remove(out)
    env = lmdb.open(out, subdir=False, map_size=int(args.map_size_gb * 1024 ** 3), readahead=False, meminit=False)

    # resume: every checked frame has an "agreement:" key
    agreements = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            if key.startswith(b"agreement:"):
                scene, stem = key[len(b"agreement:"):].decode().rsplit("/", 1)
                agreements[(scene, stem)] = pkl.loads(value)
    todo = [f for f in frames if f not in agreements]
    print(f"[{args.dataset} {args.split}] {len(scenes)} scenes, {len(frames):,} frames with an LDI, "
          f"{len(frames) - len(todo):,} already checked, {len(todo):,} to read with {args.workers} workers -> {out}")

    jobs = [(args.dataset, args.dump_root, s, n, args.img_size, args.num_layers, args.min_agreement) for s, n in todo]
    txn, n_put = env.begin(write=True), 0
    with Pool(args.workers) as pool:
        for scene, stem, agreement, record in tqdm(pool.imap(process_frame, jobs, chunksize=16), total=len(jobs),
                                                   desc=f"{args.dataset} [{args.split}] read / check / write"):
            txn.put(f"agreement:{scene}/{stem}".encode(), pkl.dumps(agreement))
            if record is not None:
                txn.put(f"frame:{scene}/{stem}".encode(), record)
            agreements[(scene, stem)] = agreement
            n_put += 1
            if n_put >= args.commit_every:
                txn.commit()
                txn, n_put = env.begin(write=True), 0
    txn.commit()

    # sequences from the good frames (only frames written as records)
    good = [f for f in frames if agreements.get(f, -1.0) >= args.min_agreement]
    # val / test anchors share scenes only for ScanNet / ARKitScenes (test = val scenes there). ScanNet++ has disjoint
    # test scenes and NRGBD is a test-only benchmark (no val LMDB), so those draw test anchors from every frame.
    test_shares_val_scenes = args.split == "test" and args.dataset in ("scannet", "arkitscenes") and args.split_file is None
    pools, anchors = select_anchors(good, args.split, args.num_views, args.n_val, args.n_test,
                                    test_shares_val_scenes, dataset=args.dataset)
    env.sync()
    env.close()                                                      # workers open their own read-only handles
    infos = None
    if args.sequence_mode == "overlap":
        sequences, infos = overlap_sequences(out, pools, anchors, args)
    elif args.sequence_mode == "matrix":
        sequences, infos = matrix_sequences(pools, anchors, args)
    else:
        sequences = window_sequences(pools, anchors, args.num_views, args.window)

    meta = {"dataset": args.dataset, "split": args.split, "img_size": args.img_size, "num_layers": args.num_layers,
            "num_views": args.num_views, "window": args.window, "min_agreement": args.min_agreement,
            "scene_frames": pools, "sequences": sequences, "sequence_mode": args.sequence_mode,
            "sequence_info": infos,
            "overlap_params": {"min_overlap": args.min_overlap, "max_overlap": args.max_overlap,
                               "overlap_basedir": args.overlap_basedir,
                               "n_vis_points": args.n_vis_points, "n_hidden_points": args.n_hidden_points,
                               "lookup_stride": args.lookup_stride}}
    env = lmdb.open(out, subdir=False, map_size=int(args.map_size_gb * 1024 ** 3), readahead=False, meminit=False)
    with env.begin(write=True) as txn:
        prev = txn.get(b"__meta__")
        if prev is not None:
            txn.put(b"__meta_prev__", prev)                          # keep the previous sequences
        txn.put(b"__meta__", pkl.dumps(meta, protocol=pkl.HIGHEST_PROTOCOL))   # last: marks the LMDB complete
    env.sync()
    env.close()

    a = np.array([agreements.get(f, -1.0) for f in frames])
    print(f"[{args.dataset} {args.split}] layer-1 agreement median {np.median(a):.0%}; good frames {len(good):,} of "
          f"{len(frames):,} ({len(good) / len(frames):.0%}); {len(sequences):,} sequences of {args.num_views} views "
          f"from {len(pools)} scenes; {osp.getsize(out) / 1024 ** 3:.2f} GB")
    if infos and args.sequence_mode == "matrix":
        rv = np.array([r for i in infos for r in i["revealed"]])
        ov = np.array([o for i in infos for o in i["overlap"]])
        if len(ov):
            print(f"[{args.dataset} {args.split}] matrix sequences: per view overlap median {np.median(ov):.0%}, "
                  f"revealed (overlap x occlusion) median {np.median(rv):.3f}; "
                  f"{np.mean([i['fallback'] for i in infos]):.1%} sequences needed a fallback "
                  f"({np.mean([i['no_matrix'] for i in infos]):.1%} without a matrix row)")
        else:
            print(f"[{args.dataset} {args.split}] matrix sequences: NO anchor had a matrix row -- all "
                  f"{len(infos):,} sequences use the window rule (check --overlap_basedir / --overlap_scene_file)")
    elif infos:
        cov = np.array([i["hidden_cov"] for i in infos if i["n_hidden"] > 0])
        ov = np.array([o for i in infos for o in i["overlap"]])
        print(f"[{args.dataset} {args.split}] overlap sequences: hidden-point coverage median {np.median(cov):.0%}"
              f" (q25 {np.quantile(cov, .25):.0%}, q75 {np.quantile(cov, .75):.0%}) over {len(cov):,} anchors with"
              f" hidden points; view overlap median {np.median(ov):.0%};"
              f" {np.mean([i['fallback'] for i in infos]):.1%} sequences needed the fallback")


def get_args():
    p = argparse.ArgumentParser(description="Clean + crop + 6-view sequences of a low-res LDI dump -> one LMDB")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--dump_root", required=True)
    p.add_argument("--out", default=None, help="LMDB file (default <dump_root>/LDIs/<dataset>_<split>_ldi_seq<V>.lmdb)")
    p.add_argument("--split_file", default=None, help="one scene per line; overrides the built-in split")
    p.add_argument("--min_agreement", type=float, default=0.8, help="layer-1 vs depth agreement to keep a frame")
    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--sequence_mode", choices=["overlap", "matrix", "window"], default="overlap",
                   help="overlap: views that see the anchor's hidden LDI surfaces; matrix: from precomputed "
                        "overlap / occlusion matrices; window: random within +-window")
    p.add_argument("--overlap_basedir", default=None, help="matrix mode: <dir>/<scene>/camera_overlap.npy, ...")
    p.add_argument("--overlap_scene_file", default=None, help="matrix mode: {scene: [rgb paths]} the matrices "
                   "were computed from (used when a scene folder has no frame_names.pkl)")
    p.add_argument("--matrix_rank", choices=["revealed", "overlap"], default="revealed",
                   help="matrix mode: rank valid sources by overlap x occlusion (revealed) or by overlap")
    p.add_argument("--strict_overlap", action="store_true",
                   help="matrix mode: drop anchors without num_views - 1 sources in [min_overlap, max_overlap] "
                        "(no fallback views)")
    p.add_argument("--window", type=int, default=10, help="window mode: views within +-window frames of the anchor")
    p.add_argument("--min_overlap", type=float, default=0.1, help="overlap mode: min fraction of the anchor's "
                   "visible points seen by a view")
    p.add_argument("--max_overlap", type=float, default=0.9, help="overlap mode: max (drops near-duplicate views)")
    p.add_argument("--n_vis_points", type=int, default=256, help="overlap mode: visible points sampled per frame")
    p.add_argument("--n_hidden_points", type=int, default=1024, help="overlap mode: hidden points sampled per frame")
    p.add_argument("--lookup_stride", type=int, default=2, help="overlap mode: layer-0 lookup at 224 / stride")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--n_val", type=int, default=2000, help="val sequences (fixed, seed 0)")
    p.add_argument("--n_test", type=int, default=2000, help="test sequences (fixed, seed 1)")
    p.add_argument("--workers", type=int,
                   default=min(32, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()))
    p.add_argument("--commit_every", type=int, default=500)
    p.add_argument("--map_size_gb", type=float, default=300, help="LMDB map size (upper bound, not allocated)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    # os.umask(0o022)      
    main(get_args())