import os
import json
import numpy as np
from pathlib import Path

def colmap_raw_pose_to_RT(image_pose):
    """Convert COLMAP [qw,qx,qy,qz,tx,ty,tz] to 4x4 RT matrix (world-to-camera)."""
    qvec = image_pose[:4] / np.linalg.norm(image_pose[:4])
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    
    t = image_pose[4:7]
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = t
    # breakpoint()
    return RT

def parse_cameras_txt_custom(cameras_file):
    """
    Parse cameras.txt with format:
    camera_id, model, width, height, focal, cx, cy, k0
    Returns: camera_id -> ((width,height), K, distortion)
    """
    camera_intrinsics = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            camera = line.split()
            cam_id = int(camera[0])
            # breakpoint()
            width, height, focal, cx, cy, k0 = [float(elem) for elem in camera[2:]]
            K = np.eye(3)
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = cx
            K[1, 2] = cy
            camera_intrinsics[cam_id] = (
                (int(width), int(height)),
                K,
                (k0, 0, 0, 0)
            )
    return camera_intrinsics

def parse_images_txt(images_file):
    """
    Parse images.txt to get filename -> {"RT":..., "camera_id":...}
    """
    poses = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        filename = parts[9]
        cam_id = int(parts[8])
        image_pose = list(map(float, parts[1:8]))
        RT = colmap_raw_pose_to_RT(image_pose)
        poses[filename] = {"RT": RT.tolist(), "camera_id": cam_id}
    return poses



def generate_json_per_subfolder(root_dir, output_dir):
    """
    Recursively generate JSON camera matrices for all valid subfolders under sparse/
    Each JSON is saved in output_dir preserving the sequence and subfolder structure.
    """
    root_path = Path(root_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = sorted([p for p in root_path.glob('*/') if p.is_dir()])

    for seq_path in sequences:
        sparse_root = seq_path / "sparse"
        if not sparse_root.exists():
            print(f"[SKIP] No sparse folder in {seq_path}, skipping sequence.")
            continue

        # recursively iterate all subfolders under sparse
        for subfolder in sparse_root.rglob('*'):
            if not subfolder.is_dir():
                continue

            cameras_file = subfolder / "cameras.txt"
            images_file = subfolder / "images.txt"

            if not cameras_file.exists() or not images_file.exists():
                continue  # skip subfolder without required files

            try:
                # parse intrinsics and poses
                intrinsics = parse_cameras_txt_custom(cameras_file)
                poses = parse_images_txt(images_file)

                if not poses or not intrinsics:
                    continue  # skip if parsing failed

                data = {}
                for filename, pose_info in poses.items():
                    cam_id = pose_info["camera_id"]
                    width_height, K, distortion = intrinsics.get(cam_id, (None, np.eye(3), (0,0,0,0)))
                    data[filename] = {
                        "K": K.tolist(),
                        "RT": pose_info["RT"],
                        "distortion": distortion,
                        "size": width_height
                    }

                # save JSON preserving sequence + relative subfolder structure
                rel_path = subfolder.relative_to(root_path)  # e.g., 0001/sparse/manhattan/0
                json_out_dir = out_root / rel_path
                json_out_dir.mkdir(parents=True, exist_ok=True)
                json_file = json_out_dir / "camera_matrices.json"
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"[SAVED] {json_file}")

            except Exception as e:
                print(f"[ERROR] Failed to process {subfolder}: {e}")
                continue  # safely skip folder

if __name__ == "__main__":
    root_dir = "/megadepth_sfm/MegaDepth_v1_SfM/"       # your sequences folder
    output_dir = "/projects/rps/dff6142/fouheylab/shared_datasets/MegaDepth/pose_data"
    generate_json_per_subfolder(root_dir, output_dir)
