from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d


def voxel_down_sample(
    original_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    down_pcd, _, traced_indices = original_pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=original_pcd.get_min_bound(),
        max_bound=original_pcd.get_max_bound(),
    )

    new_normals = np.asarray(down_pcd.normals)
    norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_normals = new_normals / norms

    for i in range(len(new_normals)):
        original_normals = np.asarray(original_pcd.normals)[traced_indices[i]]
        if original_normals.size == 0:
            continue
        closest_z_normal = original_normals[np.abs(original_normals[:, 2]).argmin(), 2]
        x, y, _ = new_normals[i]
        xy_squared = x ** 2 + y ** 2
        if xy_squared > 1e-8:
            alpha = np.sqrt(max(1.0 - closest_z_normal ** 2, 0.0) / xy_squared)
            down_pcd_normal_xy = np.array([x * alpha, y * alpha])
        else:
            down_pcd_normal_xy = np.array([0.0, 0.0])
        new_normal = np.append(down_pcd_normal_xy, closest_z_normal)
        new_normals[i] = new_normal

    down_pcd.normals = o3d.utility.Vector3dVector(new_normals)
    return down_pcd


def estimate_local_region_stats(
    points_np: np.ndarray,
    sphere_radius: float,
    num_samples: int = 64,
) -> Dict[str, float]:
    if points_np.shape[0] == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "num_samples": 0}

    n = points_np.shape[0]
    num_samples = min(num_samples, n)
    idx = np.random.choice(n, size=num_samples, replace=False)
    centers = points_np[idx]

    counts = []
    r2 = sphere_radius ** 2
    for c in centers:
        d2 = np.sum((points_np - c) ** 2, axis=1)
        counts.append(int((d2 <= r2).sum()))

    counts = np.asarray(counts, dtype=np.int32)
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "num_samples": int(num_samples),
    }


class PointCloudPreprocessor:
    def __init__(self, voxel_size: float) -> None:
        self.voxel_size = voxel_size

    def preprocess(self, pcd_path: Path):
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if pcd.is_empty():
            raise RuntimeError(f"Loaded empty point cloud from {pcd_path}")

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=4.0)
        pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.03)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)

        down_pcd = voxel_down_sample(pcd, voxel_size=self.voxel_size)

        down_points_np = np.asarray(down_pcd.points)
        down_normals_np = np.asarray(down_pcd.normals)

        if down_points_np.shape[0] == 0:
            raise RuntimeError("Downsampled point cloud is empty after preprocessing.")

        return pcd, down_pcd, down_points_np, down_normals_np
