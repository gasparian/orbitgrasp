from __future__ import annotations

import copy
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d


def save_visualized_grasp_pcd(
    pcd_path: Path,
    poses: List[np.ndarray],
    out_path: Path,
    sphere_radius: float = 0.03,
    sphere_density: int = 40,
):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise RuntimeError(f"Loaded empty point cloud from {pcd_path}")

    spheres = []
    for pose in poses:
        x, y, z = pose[0], pose[1], pose[2]
        grasp_pos = np.array([x, y, z], dtype=np.float64)

        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius,
            resolution=sphere_density,
        )
        sphere_mesh.translate(grasp_pos)
        sphere_mesh.paint_uniform_color([0.0, 1.0, 0.0])

        sphere_pcd = sphere_mesh.sample_points_uniformly(number_of_points=5000)
        sphere_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 0.0], (len(sphere_pcd.points), 1))
        )
        spheres.append(copy.deepcopy(sphere_pcd))

    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 1.0], (len(pcd.points), 1))
    )

    merged = pcd
    for sphere_pcd in spheres:
        merged += sphere_pcd

    if out_path.suffix.lower() == ".pcd":
        o3d.io.write_point_cloud(str(out_path), merged)
    else:
        o3d.io.write_point_cloud(str(out_path.with_suffix(".ply")), merged)

    print(f"Saved merged grasp visualization: {out_path}")
