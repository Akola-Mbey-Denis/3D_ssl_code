#!/usr/bin/env python
"""
Find the convention of the poses STORED in a *_NVS LMDB, by reprojecting GT depth across views.

The loaders assume the stored `extrinsics` are camera-to-world in OpenCV axes (x right, y down,
z forward), invert them to world-to-camera, then build relative poses. If that assumption is wrong,
the relative poses are wrong too, and a camera head trained on them cannot learn.

This script turns the relative-pose step off, so it sees the ABSOLUTE poses, and tests:
    stored c2w, OpenCV   (what the loaders assume)
    stored w2c, OpenCV
    stored c2w, OpenGL   (x right, y up, z backward; e.g. Habitat / some Matterport dumps)
    stored w2c, OpenGL
plus the relative poses the loader actually returns. Only the right one makes the views agree.

    python check_dataset_poses.py \
        --dataset "Matterport3D_NVS(split='test', config_path='...', img_size=224, max_views=6, n_target_views=2, load_depth=True)" \
        --n 20 --ply fused.ply
"""

import argparse

import numpy as np

F = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenGL <-> OpenCV camera axes


def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def to_4x4(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape == (3, 4):
        T = np.vstack([T, [0, 0, 0, 1]])
    return T


def unproject(depth, K, c2w, mask):
    H, W = depth.shape
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    pts = np.stack([u, v, np.ones_like(u)], 1).astype(np.float64) @ np.linalg.inv(K).T * z[:, None]
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


TOL = 0.05   # relative depth tolerance


def reprojection_error(depths, masks, Ks, w2cs, max_pts=20000, rng=None):
    """Reproject view i's GT points into every other view j and compare with view j's GT depth.
    Occlusion-robust: a point behind view j's surface may simply be occluded, but a point clearly IN FRONT
    of it (free-space violation) is impossible with correct poses.
    Returns (inlier ratio |z - d_j| / d_j < TOL, free-space violation ratio z < (1 - TOL) d_j, in-view fraction),
    the ratios over points that land inside view j on valid depth."""
    rng = rng or np.random.default_rng(0)
    n_in, n_inlier, n_violation, inview = 0, 0, 0, []
    for i in range(len(depths)):
        P = unproject(depths[i], Ks[i], np.linalg.inv(w2cs[i]), masks[i])
        if len(P) == 0:
            continue
        P = P[rng.choice(len(P), min(max_pts, len(P)), replace=False)]
        for j in range(len(depths)):
            if i == j:
                continue
            H, W = depths[j].shape
            Xc = P @ w2cs[j][:3, :3].T + w2cs[j][:3, 3]
            uvw = Xc @ Ks[j].T
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]
            ok = (Xc[:, 2] > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            inview.append(ok.mean())
            if not ok.any():
                continue
            ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
            dj = depths[j][vi, ui]
            good = (dj > 0) & masks[j][vi, ui]
            if good.any():
                z, d = Xc[ok, 2][good], dj[good]
                n_in += len(z)
                n_inlier += int((np.abs(z - d) / d < TOL).sum())
                n_violation += int((z < (1 - TOL) * d).sum())
    if n_in == 0:
        return float("nan"), float("nan"), 0.0
    return n_inlier / n_in, n_violation / n_in, float(np.mean(inview)) if inview else 0.0


def sample_views(ds, idx):
    d = ds.get_data(seq_index=idx)
    out = {k: [] for k in ("depth", "mask", "K", "pose")}
    for p in ("t_", "s_"):
        for dep, m, K, E in zip(d[p + "depths"], d[p + "depth_masks"], d[p + "intrinsics"], d[p + "extrinsics"]):
            out["depth"].append(to_np(dep).astype(np.float64).reshape(to_np(dep).shape[-2:]))
            out["mask"].append(to_np(m).astype(bool).reshape(to_np(m).shape[-2:]))
            out["K"].append(to_np(K).astype(np.float64))
            out["pose"].append(to_4x4(to_np(E)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset string, as for eval_ssl.py (needs load_depth=True)")
    ap.add_argument("--n", type=int, default=20, help="Number of samples to check")
    ap.add_argument("--ply", default=None, help="Write the fused GT point cloud of sample 0 for the best hypothesis")
    args = ap.parse_args()

    import importlib
    import pkgutil
    import datasets
    namespace = dict(vars(datasets))
    if args.dataset.split("(")[0].strip() not in namespace:        # class lives in a submodule
        for m in pkgutil.walk_packages(getattr(datasets, "__path__", []), datasets.__name__ + "."):
            try:
                namespace.update(vars(importlib.import_module(m.name)))
            except Exception:  # noqa: BLE001 -- unrelated submodule failing to import
                pass
    ds = eval(args.dataset, namespace)
    if not getattr(ds, "load_depth", True):
        raise SystemExit("pass load_depth=True in --dataset")

    relative = ds.compute_all_relative_poses                     # the loader's own relative-pose step
    ds.compute_all_relative_poses = lambda pose_list, reference_camera_index=0: list(pose_list)

    # absolute pose returned by the loader: E = inverse(stored)  ("world2cam" if stored is c2w)
    hypotheses = {
        "stored c2w, OpenCV  (loader assumption)": lambda E: E,
        "stored w2c, OpenCV":                      lambda E: np.linalg.inv(E),
        "stored c2w, OpenGL":                      lambda E: F @ E,
        "stored w2c, OpenGL":                      lambda E: F @ np.linalg.inv(E),
    }
    scores = {h: [] for h in hypotheses}
    loader_rel = []
    first = None
    for idx in range(min(args.n, len(ds))):
        v = sample_views(ds, idx)
        if first is None:
            first = v
        for h, fn in hypotheses.items():
            scores[h].append(reprojection_error(v["depth"], v["mask"], v["K"], [fn(E) for E in v["pose"]]))
        # what the loader returns (camera -> reference), read as c2w
        T = ds.n_target_views
        rel = [to_4x4(r) for r in relative(pose_list=[E.astype(np.float32) for E in v["pose"]],
                                           reference_camera_index=T)]
        loader_rel.append(reprojection_error(v["depth"], v["mask"], v["K"], [np.linalg.inv(r) for r in rel]))

    print(f"\n{type(ds).__name__}: {min(args.n, len(ds))} samples, {len(first['depth'])} views each")
    print(f"  {'hypothesis':42s} {'inliers (<5% depth err)':>24s} {'free-space violations':>22s} "
          f"{'points in other views':>22s}")

    def agg(rows):
        return (float(np.nanmean([r[0] for r in rows])), float(np.nanmean([r[1] for r in rows])),
                float(np.mean([r[2] for r in rows])))

    summary, violations = {}, {}
    for h in hypotheses:
        inl, vio, vis = agg(scores[h])
        summary[h], violations[h] = inl, vio
        print(f"  {h:42s} {inl:24.1%} {vio:22.1%} {vis:22.1%}")
    rel_inl, rel_vio, rel_vis = agg(loader_rel)
    print(f"  {'loader relative poses, read as c2w':42s} {rel_inl:24.1%} {rel_vio:22.1%} {rel_vis:22.1%}")
    print("  (right convention: many inliers, few violations; occluded points count as neither)")

    best = max(summary, key=summary.get)
    print()
    if summary[best] < 0.3 or violations[best] >= 0.10:   # consistent poses leave (almost) no free-space violations
        print("No hypothesis clearly makes the views agree: the problem is not the pose convention.\n"
              "Check depth units / validity, the intrinsics after cropping, and that each view's pose belongs\n"
              "to that image (view ordering in the LMDB).")
    elif best.startswith("stored c2w, OpenCV"):
        if rel_inl < 0.5 * summary[best] or rel_vio >= 0.10:
            print("Stored poses are fine (c2w, OpenCV), but the RELATIVE poses the loader builds are not:\n"
                  "check compute_all_relative_poses / reference_camera_index.")
        else:
            print("Stored poses match the loader's assumption (c2w, OpenCV). Poses are consistent; if the camera\n"
                  "head still fails, the problem is in training, not the data.")
    else:
        fix = {
            "stored w2c, OpenCV": "the stored poses are world-to-camera: drop the rotate_inverse() in get_data\n"
                                  "  (use extri = np.array(view['extrinsics']) directly).",
            "stored c2w, OpenGL": "the stored poses are camera-to-world in OpenGL axes: convert before inverting,\n"
                                  "  c2w = np.array(view['extrinsics']) @ np.diag([1, -1, -1, 1]).",
            "stored w2c, OpenGL": "the stored poses are world-to-camera in OpenGL axes: use\n"
                                  "  extri = np.diag([1, -1, -1, 1]) @ np.array(view['extrinsics']) (no rotate_inverse).",
        }[best]
        print(f"MISMATCH: best fit is '{best}' -> {fix}\n"
              "Fix the loader, check again, then RETRAIN the camera probes (they were trained on the wrong poses).")

    if args.ply and first is not None:
        import open3d as o3d
        cols_per_view = np.random.default_rng(0).random((len(first["depth"]), 3))
        to_write = [("stored c2w, OpenCV  (loader assumption)", args.ply)]
        if not best.startswith("stored c2w, OpenCV"):
            stem, ext = args.ply.rsplit(".", 1) if "." in args.ply else (args.ply, "ply")
            to_write.append((best, f"{stem}_best_fit.{ext}"))
        for h, path in to_write:
            fn = hypotheses[h]
            pts = np.concatenate([unproject(d, K, np.linalg.inv(fn(E)), m)
                                  for d, m, K, E in zip(first["depth"], first["mask"], first["K"], first["pose"])])
            cols = np.concatenate([np.tile(c, (int((m & (d > 0)).sum()), 1))
                                   for c, d, m in zip(cols_per_view, first["depth"], first["mask"])])
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            pcd.colors = o3d.utility.Vector3dVector(cols)
            o3d.io.write_point_cloud(path, pcd)
            print(f"wrote {path} (sample 0, '{h.strip()}', one colour per view -- surfaces should overlap)")


if __name__ == "__main__":
    main()