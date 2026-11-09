"""
Create ScanNet++ target/source sequences from precomputed camera overlap matrices.

Same as nrgbd_sequences.py, but cameras follow scannetpp_overlaps.py (the dumped ScanNet++ layout):

    <dump_root>/<scene>/rgb/<frame>.jpg
    <dump_root>/<scene>/depth/<cameras[frame]["depth"]>     .npy, meters, 0 = invalid
    <dump_root>/<scene>/cameras.pkl                         frame (rgb file name) -> {"K", "RT_c2w", "depth"}
                                                            K at the dumped resolution, RT_c2w OpenCV c2w

scene_file (written by scannetpp_overlaps.py / collect_scene_files):
    scene_name -> [rgb file paths, sorted]
    Row/column i of the overlap matrices <-> scene_file[scene_name][i].

overlap_basedir/<scene_name>/:
    camera_overlap.npy      camera_overlap[target, src]
    camera_occlusion.npy    camera_occlusion[target, src]

Output
------
<save_dir>/preprocessed_scannetpp_3d_<split>_set.pkl

{
    "<scene>__<frame>": {
        "target":  {"image", "depth", "K", "extrinsics"},
        "sources": [{"image", "depth", "K", "extrinsics", "overlap", "occlusion"}],
    }
}

K and depth correspond to the dumped images (not the 518 x 518 crop used for the overlaps;
the dataset crops itself). extrinsics is the OpenCV camera-to-world matrix from cameras.pkl.

Splits (scene level): SCANNETPP_SPLITS below (240 train / 5 val / 5 test, disjoint)
    all: every scene in the scene file

Usage:
    python scannetpp_sequences.py --split train
    python scannetpp_sequences.py --split val
    python scannetpp_sequences.py --split test
"""

import argparse
import os
import pickle as pkl

import numpy as np
from tqdm import tqdm


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


def frame_name(path):
    """<frame>.jpg -> <frame>."""
    return os.path.splitext(os.path.basename(path))[0]


def load_scene_file(scene_file):
    """scene_name -> [rgb file paths], in overlap-matrix order."""
    with open(scene_file, "rb") as f:
        return pkl.load(f)


def load_cameras(scene_dir):
    """cameras.pkl: rgb file name -> {"K", "RT_c2w", "depth"}."""
    with open(os.path.join(scene_dir, "cameras.pkl"), "rb") as f:
        return pkl.load(f)


def get_scene_names(scene_dict, split):
    """Scene names for the requested split (SCANNETPP_SPLITS); 'all' keeps every scene."""
    if split == "all":
        return set(scene_dict.keys())

    names = set(SCANNETPP_SPLITS[split])

    missing = names - set(scene_dict.keys())
    if missing:
        print(f"[WARN] {len(missing)} {split} scenes are absent from scene_file: {sorted(missing)}")

    return names & set(scene_dict.keys())


def load_matrix(path, scene_name, num_frames):
    """Load and validate an overlap or occlusion matrix."""
    matrix = np.load(path)

    if matrix.shape != (num_frames, num_frames):
        raise ValueError(
            f"{scene_name}: {os.path.basename(path)} has shape "
            f"{matrix.shape}, expected {(num_frames, num_frames)}"
        )

    return matrix.astype(np.float32, copy=False)


def build_overlap_sequences(
    scene_dict,
    dump_root,
    overlap_basedir,
    overlap_thres=0.010,
    max_sources=100,
    split="all",
):
    """
    One sequence per valid ScanNet++ target frame.

    Matrix row/column i corresponds directly to scene_dict[scene_name][i].

    Sources are other valid frames whose target-to-source overlap meets
    overlap_thres, sorted by descending overlap and then occlusion.
    """
    results = {}
    selected_scenes = get_scene_names(scene_dict, split)

    counts = {
        "scenes": 0,
        "no_overlaps": 0,
        "no_cameras": 0,
        "targets": 0,
        "no_sources": 0,
        "invalid_frames": 0,
    }

    for scene_name in tqdm(sorted(selected_scenes), desc=f"ScanNet++ {split}"):
        scene_dir = os.path.join(dump_root, scene_name)
        overlaps_dir = os.path.join(overlap_basedir, scene_name)

        overlap_path = os.path.join(overlaps_dir, "camera_overlap.npy")
        occlusion_path = os.path.join(overlaps_dir, "camera_occlusion.npy")

        if not (os.path.isfile(overlap_path) and os.path.isfile(occlusion_path)):
            print(f"[SKIP] {scene_name}: overlap matrices missing")
            counts["no_overlaps"] += 1
            continue

        if not os.path.isfile(os.path.join(scene_dir, "cameras.pkl")):
            print(f"[SKIP] {scene_name}: cameras.pkl missing")
            counts["no_cameras"] += 1
            continue

        # Preserve the exact ordering used to compute the matrices.
        img_paths = list(scene_dict[scene_name])
        num_frames = len(img_paths)
        cameras = load_cameras(scene_dir)

        camera_overlap = load_matrix(overlap_path, scene_name, num_frames)
        camera_occlusion = load_matrix(occlusion_path, scene_name, num_frames)

        # Cameras as in scannetpp_overlaps.load_dumped_scene: keyed by rgb file name,
        # K at the dumped resolution, RT_c2w already OpenCV camera-to-world.
        K_list, RT_list, depth_paths = [], [], []
        valid_camera = np.zeros(num_frames, dtype=bool)

        for i, rgb_path in enumerate(img_paths):
            cam = cameras.get(os.path.basename(rgb_path))

            if cam is None or "depth" not in cam:
                K_list.append(None)
                RT_list.append(None)
                depth_paths.append("")
                continue

            K = np.asarray(cam["K"], dtype=np.float32)
            RT_c2w = np.asarray(cam["RT_c2w"], dtype=np.float32)

            valid_camera[i] = (
                K.shape == (3, 3)
                and RT_c2w.shape == (4, 4)
                and np.isfinite(K).all()
                and np.isfinite(RT_c2w).all()
            )
            K_list.append(K)
            RT_list.append(RT_c2w)
            depth_paths.append(os.path.join(scene_dir, "depth", cam["depth"]))

        # Reject frames with invalid overlap rows/columns.
        # The diagonal must indicate a valid frame.
        valid_overlap = (
            np.isfinite(np.diag(camera_overlap))
            & (np.diag(camera_overlap) > 0)
            & np.all(np.isfinite(camera_overlap), axis=1)
            & np.all(camera_overlap >= 0, axis=1)
            & np.all(camera_overlap >= 0, axis=0)
        )

        view_names = [frame_name(path) for path in img_paths]

        file_valid = np.asarray(
            [
                os.path.isfile(img_paths[i]) and os.path.isfile(depth_paths[i])
                for i in range(num_frames)
            ],
            dtype=bool,
        )

        valid_frames = valid_camera & valid_overlap & file_valid

        counts["scenes"] += 1
        counts["invalid_frames"] += int((~valid_frames).sum())

        def view(idx):
            return {
                "image": img_paths[idx],
                "depth": depth_paths[idx],
                "K": K_list[idx].copy(),
                "extrinsics": RT_list[idx].copy(),     # OpenCV c2w
            }

        for target_idx in np.flatnonzero(valid_frames):
            counts["targets"] += 1

            overlap_scores = camera_overlap[target_idx]
            occlusion_scores = camera_occlusion[target_idx]

            # Exclude the target itself and every invalid source.
            valid_indices = np.flatnonzero(
                valid_frames
                & np.isfinite(overlap_scores)
                & np.isfinite(occlusion_scores)
                & (overlap_scores >= overlap_thres)
            )
            valid_indices = valid_indices[valid_indices != target_idx]

            if len(valid_indices) == 0:
                counts["no_sources"] += 1
                continue

            # Descending overlap, with descending occlusion as tie-breaker.
            sorted_indices = valid_indices[
                np.lexsort(
                    (
                        -occlusion_scores[valid_indices],
                        -overlap_scores[valid_indices],
                    )
                )
            ]

            sources = [
                {
                    **view(src_idx),
                    "overlap": float(overlap_scores[src_idx]),
                    "occlusion": float(occlusion_scores[src_idx]),
                }
                for src_idx in sorted_indices[:max_sources]
            ]

            key = f"{scene_name}__{view_names[target_idx]}"
            results[key] = {
                "target": view(target_idx),
                "sources": sources,
            }

    print(
        f"[SUMMARY] {split}: {len(results)} target sequences "
        f"from {counts['scenes']} scenes "
        f"({counts['targets']} valid targets, "
        f"{counts['no_sources']} without sources, "
        f"{counts['invalid_frames']} invalid frames, "
        f"{counts['no_overlaps']} scenes without overlap matrices, "
        f"{counts['no_cameras']} scenes without cameras.pkl)"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Scene-level split, or all scenes",
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        default=(
            "/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data/"
            "scannetpp_low_res_overlap_scene_files.pkl"
        ),
        help="Scene mapping written by scannetpp_overlaps.py (scene -> rgb paths)",
    )
    parser.add_argument(
        "--dump_root",
        type=str,
        default="/scratch/dma9300/scannetpp_preprocessed/",
        help="Dumped ScanNet++ data: <scene>/{rgb, depth, cameras.pkl}",
    )
    parser.add_argument(
        "--overlap_basedir",
        type=str,
        default=(
            "/projects/rps/dff6142/fouheylab/shared_datasets/"
            "dma9300/dataset_overlaps/low_res_scannetpp_overlaps/"
        ),
        help="Directory containing precomputed overlap matrices",
    )
    parser.add_argument(
        "--overlap_thres",
        type=float,
        default=0.010,
        help="Minimum directional source overlap",
    )
    parser.add_argument(
        "--max_sources",
        type=int,
        default=100,
        help="Maximum sources per target",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/projects/rps/dff6142/fouheylab/dma9300/3d_ssl_project/preprocessed_data",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.max_sources < 1:
        parser.error("--max_sources must be >= 1")

    if args.overlap_thres < 0:
        parser.error("--overlap_thres must be >= 0")

    all_scene_dict = load_scene_file(args.scene_file)

    results = build_overlap_sequences(
        scene_dict=all_scene_dict,
        dump_root=args.dump_root,
        overlap_basedir=args.overlap_basedir,
        overlap_thres=args.overlap_thres,
        max_sources=args.max_sources,
        split=args.split,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(
        args.save_dir,
        f"preprocessed_scannetpp_3d_{args.split}_set.pkl",
    )

    with open(save_path, "wb") as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(results)} sequences to {save_path}")