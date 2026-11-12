#!/usr/bin/env python
"""
GT-only check of a dataset's cameras: builds the combined GT point cloud of a few samples under four pose conventions,
prints how well the views agree (median relative depth error, ~0.01 = consistent) and writes one .ply per convention,
so the right one can also be checked by eye in MeshLab (one room vs scattered slabs).

  python debug_gt_poses.py --dataset "Habitat_HM3D_NVS(split='vis', config_path='...', img_size=224, max_views=6, n_target_views=2, load_depth=True)" --n 3
"""
import argparse
import os

import numpy as np
import torch
import trimesh

from check_poses import FLIP, consistency, to_4x4, to_tensor


def rgb_uint8(img):
    img = img.float()
    if img.min() < -0.01:
        img = (img + 1) / 2
    elif img.max() > 1.5:
        img = img / 255
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def cloud(depth, mask, K, w2c, rgb):
    pts, cols = [], []
    for v in range(depth.shape[0]):
        H, W = depth[v].shape
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).double()
        cam = (pix @ torch.linalg.inv(K[v]).T) * depth[v].reshape(-1, 1)
        c2w = torch.linalg.inv(w2c[v])
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        m = mask[v].reshape(-1)
        pts.append(world[m].numpy())
        cols.append(rgb[v].reshape(-1, 3)[m.numpy()])
    return np.concatenate(pts), np.concatenate(cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out_dir", default="debug_gt_poses")
    args = p.parse_args()
    from datasets import get_data_loader
    loader = get_data_loader(args.dataset, batch_size=1, num_workers=1, pin_mem=False, shuffle=False, drop_last=False)
    os.makedirs(args.out_dir, exist_ok=True)

    for it, batch in enumerate(loader):
        if it >= args.n:
            break
        g = lambda k: torch.cat([to_tensor(batch["t_" + k]), to_tensor(batch["s_" + k])], 1)[0]  # noqa: E731
        depth, K, ext = g("depths").double(), g("intrinsics").double(), to_4x4(g("extrinsics").double())
        V = depth.shape[0]
        depth = depth.reshape(V, *depth.shape[-2:])
        mask = (g("depth_masks").reshape(depth.shape) > 0) & (depth > 0)
        rgb = [rgb_uint8(im) for im in g("images")]
        print(f"sample {it}: depth min/median/max {depth[mask].min():.3f} / {depth[mask].median():.3f} / "
              f"{depth[mask].max():.3f}   |t| per view {ext[:, :3, 3].norm(dim=-1).numpy().round(2)}")
        cands = {
            "as_given_w2c": ext,
            "inverted_c2w": torch.linalg.inv(ext),
            "opengl_flip": torch.linalg.inv(torch.linalg.inv(ext) @ FLIP),
            "inverted_opengl_flip": torch.linalg.inv(ext @ FLIP),
        }
        for name, w2c in cands.items():
            e = consistency(depth, mask, K, w2c)
            pts, cols = cloud(depth, mask, K, w2c, rgb)
            path = os.path.join(args.out_dir, f"sample{it}_{name}.ply")
            trimesh.PointCloud(pts.astype(np.float32), colors=cols).export(path)
            print(f"   {name:<22} view agreement {e:.4f}   -> {path}")


if __name__ == "__main__":
    main()