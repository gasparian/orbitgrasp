from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from e3nn.o3 import spherical_harmonics_alpha_beta
from torch_geometric.nn import radius

from se3_grasper_bce import OrbitGrasper  # type: ignore
from utils.FeaturedPoints import FeaturedPoints  # type: ignore
from utils.utils_3d import FarthestSampler  # type: ignore
from utils.torch_utils import set_seed  # type: ignore

from scripts.orbitgrasp_inference.config import OrbitGraspConfig
from scripts.orbitgrasp_inference.geometry import extract_finger_tips_from_pcd, quaternion_from_two_vectors
from scripts.orbitgrasp_inference.pointcloud import (
    PointCloudPreprocessor,
    estimate_local_region_stats,
)


def vector_to_spherical(vectors: torch.Tensor):
    r = torch.sqrt(torch.sum(vectors ** 2, dim=-1))
    theta = torch.acos(vectors[..., 1] / r)
    phi = torch.atan2(vectors[..., 0], vectors[..., 2])
    return r, theta, phi


def compute_spherical_harmonics(vectors: torch.Tensor, lmax: int = 3):
    _, theta, phi = vector_to_spherical(vectors)
    harmonics_list = []
    for l in range(lmax + 1):
        harmonics = spherical_harmonics_alpha_beta(
            l, phi, theta, normalization="component"
        )
        harmonics_list.append(harmonics)
    harmonics = torch.cat(harmonics_list, dim=-1)
    return harmonics.detach()


def normalize_feature_points(data: FeaturedPoints) -> FeaturedPoints:
    pos = data.x
    center = pos.mean(dim=0, keepdim=True)
    pos = pos - center
    return FeaturedPoints(x=pos, n=data.n, b=data.b)


def build_grasp_candidates_for_sphere(
    sphere_points: torch.Tensor,
    sphere_normals: torch.Tensor,
    num_intervals: int,
):
    n_points = sphere_points.shape[0]

    z_dirs = -sphere_normals
    z_dirs_np = z_dirs.cpu().numpy()
    positions_np = sphere_points.cpu().numpy()

    grasp_poses = np.zeros((n_points, num_intervals, 7), dtype=np.float32)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(n_points):
        target = z_dirs_np[i]
        if np.linalg.norm(target) < 1e-6:
            target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        quat = quaternion_from_two_vectors(z_axis, target)
        pose = np.concatenate([positions_np[i], quat], axis=0)
        grasp_poses[i, :, :] = pose[None, :]

    z_dirs_norm = z_dirs / (torch.norm(z_dirs, dim=-1, keepdim=True) + 1e-8)
    z_dirs_tiled = z_dirs_norm.unsqueeze(1).repeat(1, num_intervals, 1)

    return grasp_poses, z_dirs_tiled


class OrbitGraspPipeline:
    def __init__(self, config_path: Path) -> None:
        self.config = OrbitGraspConfig(config_path)
        cfg = self.config.config

        self.device = self.config.device
        self.num_intervals = self.config.num_intervals
        self.harmonics_lmax = self.config.harmonics_lmax

        ckpt_root = self.config.ckpt_root
        ckpt_name = self.config.ckpt_name

        self.orbit_grasper = OrbitGrasper(
            device=self.device,
            load=self.config.load_epoch,
            param_dir=ckpt_root,
            num_channel=self.num_intervals,
            lmax=self.config.orbit_lmax,
            mmax=self.config.orbit_mmax,
            load_name=ckpt_name,
            training_config=cfg,
        )

        set_seed(self.config.seed)

    def grasp_from_pcd(
        self,
        pcd_path: Path,
        voxel_size: float = 0.0055,
        sphere_radius: float = 0.05,
        min_allowed_points: Optional[int] = None,
        max_allowed_points: Optional[int] = None,
        num_centers: int = 10,
        gripper_width: float = 0.04,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preprocessor = PointCloudPreprocessor(voxel_size=voxel_size)
        pcd, down_pcd, down_points_np, down_normals_np = preprocessor.preprocess(pcd_path)

        if min_allowed_points is None or max_allowed_points is None:
            stats = estimate_local_region_stats(down_points_np, sphere_radius)
            print(
                f"[density] radius={sphere_radius:.3f} m -> "
                f"min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']:.1f}, median={stats['median']:.1f} "
                f"(over {stats['num_samples']} samples)"
            )

            if min_allowed_points is None:
                min_allowed_points = max(20, int(stats["median"] * 0.3))

            if max_allowed_points is None:
                max_allowed_points = int(stats["max"] * 1.1)

            print(
                f"[density] using min_allowed_points={min_allowed_points}, "
                f"max_allowed_points={max_allowed_points}"
            )

        down_points = torch.from_numpy(down_points_np).float().to(self.device)
        down_normals = torch.from_numpy(down_normals_np).float().to(self.device)

        point_sampler = FarthestSampler()
        center_points, _ = point_sampler(
            down_points.cpu().numpy(), num_centers, sphere_radius + 0.002
        )

        down_points_tensor = down_points.clone().detach()

        all_data_list: List[FeaturedPoints] = []
        all_sphere_indices_list: List[torch.Tensor] = []
        all_harmonics_list: List[torch.Tensor] = []
        all_grasp_poses_list: List[np.ndarray] = []
        region_sizes: List[int] = []
        valid_mask: List[bool] = []

        for center in center_points:
            center_tensor = torch.tensor(
                center, dtype=torch.float32, device=self.device
            ).view(1, 3)

            edge_index = radius(
                down_points_tensor,
                center_tensor,
                r=sphere_radius,
                max_num_neighbors=900,
            )

            if edge_index.numel() == 0:
                region_sizes.append(0)
                valid_mask.append(False)
                continue

            local_idx = edge_index[1]
            sphere_points = down_points[local_idx]
            sphere_normals = down_normals[local_idx]
            region_size = int(sphere_points.shape[0])
            region_sizes.append(region_size)

            feature_points = FeaturedPoints(
                x=sphere_points,
                n=sphere_normals,
                b=torch.ones(
                    sphere_points.shape[0], dtype=torch.long, device=self.device
                ),
            )
            feature_points = normalize_feature_points(feature_points)
            all_data_list.append(feature_points)

            sphere_indices = torch.arange(
                sphere_points.shape[0], device=self.device
            )
            all_sphere_indices_list.append(sphere_indices)

            grasp_poses, z_dirs_tiled = build_grasp_candidates_for_sphere(
                sphere_points, sphere_normals, num_intervals=self.num_intervals
            )
            all_grasp_poses_list.append(grasp_poses)

            harmonics = compute_spherical_harmonics(
                z_dirs_tiled, lmax=self.harmonics_lmax
            )
            all_harmonics_list.append(harmonics.to(self.device))

            is_valid = True
            if min_allowed_points is not None and region_size < min_allowed_points:
                is_valid = False
            if max_allowed_points is not None and region_size > max_allowed_points:
                is_valid = False
            valid_mask.append(is_valid)

        if region_sizes:
            print(
                f"[regions] built {len(region_sizes)} regions with sphere_radius={sphere_radius:.3f}: "
                f"min={min(region_sizes)}, max={max(region_sizes)}, "
                f"mean={np.mean(region_sizes):.1f}"
            )

        if any(valid_mask):
            data_list = [d for d, v in zip(all_data_list, valid_mask) if v]
            sphere_indices_list = [
                s for s, v in zip(all_sphere_indices_list, valid_mask) if v
            ]
            harmonics_list = [
                h for h, v in zip(all_harmonics_list, valid_mask) if v
            ]
            grasp_poses_list = [
                g for g, v in zip(all_grasp_poses_list, valid_mask) if v
            ]
        else:
            print(
                "[regions] WARNING: no regions matched min/max thresholds; "
                "falling back to using all regions."
            )
            data_list = all_data_list
            sphere_indices_list = all_sphere_indices_list
            harmonics_list = all_harmonics_list
            grasp_poses_list = all_grasp_poses_list

        if not data_list:
            raise RuntimeError(
                "No valid local regions found at all. "
                "Point cloud might be extremely sparse or degenerate."
            )

        score_list, _ = self.orbit_grasper.predict(
            data_list, sphere_indices_list, harmonics_list
        )
        score_list_arr = torch.cat(score_list, dim=0).cpu().numpy()
        grasp_poses_array = np.concatenate(
            grasp_poses_list, axis=0
        )

        flat_idx = np.argmax(score_list_arr)
        best_i, best_k = np.unravel_index(flat_idx, score_list_arr.shape)
        best_pose = grasp_poses_array[best_i, best_k]

        tip1, tip2 = extract_finger_tips_from_pcd(
            pcd, best_pose, gripper_width=gripper_width
        )

        return best_pose, tip1, tip2


def grasp_from_pcd(
    pcd_path: Path,
    config_path: Path,
    voxel_size: float = 0.0055,
    sphere_radius: float = 0.05,
    min_allowed_points: Optional[int] = None,
    max_allowed_points: Optional[int] = None,
    num_centers: int = 10,
    gripper_width: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipeline = OrbitGraspPipeline(config_path=config_path)
    return pipeline.grasp_from_pcd(
        pcd_path=pcd_path,
        voxel_size=voxel_size,
        sphere_radius=sphere_radius,
        min_allowed_points=min_allowed_points,
        max_allowed_points=max_allowed_points,
        num_centers=num_centers,
        gripper_width=gripper_width,
    )
