"""Back-project DA3 depth to a coloured 3D point cloud, save .ply, view in Open3D."""
from __future__ import annotations

import argparse
import os

import numpy as np

from da3_common import add_common_args, resolve_out_dir, run_da3


def unproject(depth: np.ndarray, image: np.ndarray, K: np.ndarray,
              conf: np.ndarray | None = None,
              max_depth: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """depth [H,W], image [H,W,3] uint8, K [3,3] -> (points [N,3], colors [N,3] uint8)."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32)

    valid = np.isfinite(z) & (z > 0)
    if conf is not None:
        valid &= conf > np.quantile(conf[np.isfinite(conf)], 0.05)
    if max_depth is not None:
        valid &= z < max_depth

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)[valid]
    cols = image[valid]
    return pts.astype(np.float32), cols.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--max-depth", type=float, default=80.0,
                    help="drop points farther than this (metres for metric models, "
                         "or relative units for non-metric models — set 0 to disable)")
    ap.add_argument("--no-viewer", action="store_true", help="skip the Open3D window")
    args = ap.parse_args()
    out = resolve_out_dir(args.image, args.out_dir)

    res = run_da3(args.image, args.model, args.device)
    max_d = None if args.max_depth <= 0 else args.max_depth
    pts, cols = unproject(res.depth, res.image, res.K, conf=res.conf, max_depth=max_d)
    print(f"[pcd] {pts.shape[0]:,} points")

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

    ply_path = out / "pointcloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"[pcd] wrote {ply_path}")

    if not args.no_viewer and os.environ.get("DISPLAY"):
        # Camera looks down +Z; flip Y/Z so it's roughly upright in the viewer.
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d.visualization.draw_geometries([pcd, cam_frame])
    elif not args.no_viewer:
        print("[pcd] no $DISPLAY — skipping Open3D window")


if __name__ == "__main__":
    main()
