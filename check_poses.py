#!/usr/bin/env python
"""
Which camera convention makes the GT views of a dataset agree?

Projects the GT depth of every view into every other view and compares with the GT depth there (median relative
error). The right convention gives ~0.01-0.05; a wrong one gives >> 0.1 and GT point clouds that fan apart.

  python check_poses.py --dataset "NRGBD_NVS(split='vis', config_path=..., img_size=224, max_views=6, n_target_views=2, load_depth=True)" --n 5
"""
import argparse

import numpy as np
import torch

FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))   # OpenGL <-> OpenCV camera axes


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.stack([to_tensor(v) for v in x])


def to_4x4(T):
    if T.shape[-2] == 3:
        T = torch.cat([T, torch.tensor([[0.0, 0, 0, 1]], dtype=T.dtype).expand(*T.shape[:-2], 1, 4)], -2)
    return T


def consistency(depth, mask, K, w2c, max_pts=4000):
    V, H, W = depth.shape
    c2w = torch.linalg.inv(w2c)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).double()
    errs = []
    for i in range(V):
        cam = (pix @ torch.linalg.inv(K[i]).T) * depth[i].reshape(-1, 1)
        P = (cam @ c2w[i, :3, :3].T + c2w[i, :3, 3])[mask[i].reshape(-1)]
        P = P[:: max(1, len(P) // max_pts)]
        for j in range(V):
            if i == j:
                continue
            q = P @ w2c[j, :3, :3].T + w2c[j, :3, 3]
            z = q[:, 2]
            uv = q @ K[j].T
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            ok = (z > 1e-3) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            if ok.sum() < 50:
                continue
            ui, vi = u[ok].round().long(), v[ok].round().long()
            d, m = depth[j][vi, ui], mask[j][vi, ui]
            if m.sum() >= 50:
                errs.append(((z[ok][m] - d[m]).abs() / d[m]))
    return float(torch.cat(errs).median()) if errs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=5, help="samples to check")
    args = p.parse_args()
    from datasets import get_data_loader
    loader = get_data_loader(args.dataset, batch_size=1, num_workers=1, pin_mem=False, shuffle=False, drop_last=False)

    results = {"as given (w2c)": [], "inverted (c2w)": [], "OpenGL flip": [], "inverted + OpenGL flip": []}
    for it, batch in enumerate(loader):
        if it >= args.n:
            break
        g = lambda k: torch.cat([to_tensor(batch["t_" + k]), to_tensor(batch["s_" + k])], 1)[0].double()  # noqa: E731
        depth, K, ext = g("depths"), g("intrinsics"), to_4x4(g("extrinsics"))
        V = depth.shape[0]
        depth = depth.reshape(V, *depth.shape[-2:])
        mask = (g("depth_masks").reshape(depth.shape) > 0) & (depth > 0)
        inv = torch.linalg.inv(ext)
        cands = {
            "as given (w2c)": ext,
            "inverted (c2w)": inv,
            "OpenGL flip": torch.linalg.inv(torch.linalg.inv(ext) @ FLIP),   # w2c of c2w_gl @ flip
            "inverted + OpenGL flip": torch.linalg.inv(ext @ FLIP),
        }
        for k, w2c in cands.items():
            results[k].append(consistency(depth, mask, K, w2c))
    print("median reprojection depth error (lower = views agree):")
    for k, v in results.items():
        print(f"  {k:<24} {np.nanmedian(v):.4f}   per sample: {[round(x, 3) for x in v]}")


if __name__ == "__main__":
    main()