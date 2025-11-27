import copy
import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import open3d as o3d
import yaml

from e3nn.o3 import spherical_harmonics_alpha_beta
from torch_geometric.nn import radius

# Make sure we can import from this repo when running as a standalone script
import sys
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(REPO_ROOT))

from se3_grasper_bce import OrbitGrasper  # type: ignore
from utils.FeaturedPoints import FeaturedPoints  # type: ignore
from utils.utils_3d import FarthestSampler  # type: ignore
from utils.torch_utils import set_seed  # type: ignore


# ---------- Small helpers copied / adapted from test_grasp_single.py ----------

def voxel_down_sample(original_pcd: o3d.geometry.PointCloud,
                      voxel_size: float):
    """Voxel downsample that also refines normals similar to test_grasp_single.py."""
    down_pcd, _, traced_indices = original_pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=original_pcd.get_min_bound(),
        max_bound=original_pcd.get_max_bound(),
    )

    new_normals = np.asarray(down_pcd.normals)
    norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_normals = new_normals / norms

    # Copy z-component from the closest original normal in the voxel
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
            l, phi, theta, normalization='component'
        )
        harmonics_list.append(harmonics)
    harmonics = torch.cat(harmonics_list, dim=-1)
    return harmonics.detach()


def normalize_feature_points(data: FeaturedPoints) -> FeaturedPoints:
    pos = data.x
    center = pos.mean(dim=0, keepdim=True)
    pos = pos - center
    return FeaturedPoints(x=pos, n=data.n, b=data.b)


def quaternion_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Quaternion that rotates v0 onto v1. Returns [x, y, z, w]."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    if dot < -0.999999:
        # 180 deg rotation around any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v0[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - v0 * np.dot(v0, axis)
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)

    cross = np.cross(v0, v1)
    s = np.sqrt((1.0 + dot) * 2.0)
    invs = 1.0 / s
    qx = cross[0] * invs
    qy = cross[1] * invs
    qz = cross[2] * invs
    qw = 0.5 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def load_config(config_path: Path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_checkpoint(root_dir: Path, load_epoch: int,
                    ckpt_prefix: str = 'orbitgrasp-ckpt-'):
    files = os.listdir(root_dir)
    matching_files = [
        f for f in files
        if f'{ckpt_prefix}{load_epoch}-' in f and f.endswith('.pt')
    ]
    if not matching_files:
        raise ValueError(
            f"No checkpoints found with prefix '{ckpt_prefix}{load_epoch}-' in '{root_dir}'"
        )
    matching_files.sort(reverse=True)
    return matching_files[0]


def build_grasp_candidates_for_sphere(
    sphere_points: torch.Tensor,
    sphere_normals: torch.Tensor,
    num_intervals: int,
):
    """
    Build simple SE(3) grasp candidates for a local sphere.

    For each point, we create `num_intervals` candidates that all share the
    same approach direction (opposite normal). OrbitGrasper scores these;
    we just use the best-scoring pose.
    """
    n_points = sphere_points.shape[0]

    # Use opposite normal as approach direction (point camera/gripper -Z along surface normal)
    z_dirs = -sphere_normals  # [N, 3]
    z_dirs_np = z_dirs.cpu().numpy()
    positions_np = sphere_points.cpu().numpy()

    grasp_poses = np.zeros((n_points, num_intervals, 7), dtype=np.float32)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(n_points):
        target = z_dirs_np[i]
        if np.linalg.norm(target) < 1e-6:
            target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        quat = quaternion_from_two_vectors(z_axis, target)  # [x, y, z, w]
        pose = np.concatenate([positions_np[i], quat], axis=0)
        # Same pose repeated over all intervals: quality is a function over S^2,
        # yaw around the normal is irrelevant.
        grasp_poses[i, :, :] = pose[None, :]

    # Torch tensor for harmonic computation: [N, num_intervals, 3]
    z_dirs_norm = z_dirs / (torch.norm(z_dirs, dim=-1, keepdim=True) + 1e-8)
    z_dirs_tiled = z_dirs_norm.unsqueeze(1).repeat(1, num_intervals, 1)

    return grasp_poses, z_dirs_tiled


def estimate_local_region_stats(points_np: np.ndarray, sphere_radius: float, num_samples: int = 64):
    """
    Quick density probe on the downsampled cloud.

    Samples `num_samples` random centers and counts how many neighbors they
    have inside a ball of radius `sphere_radius`.
    """
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


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("Quaternion must be shape (4,) as [qx, qy, qz, qw].")

    qx, qy, qz, qw = q
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    qx /= n
    qy /= n
    qz /= n
    qw /= n

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def compute_local_gripper_frame(
    best_pose: np.ndarray,
    points: np.ndarray,
    local_radius: float = 0.06,
):
    """
    From the grasp pose and local neighborhood geometry,
    compute:
      pos: grasp center (3,)
      a  : approach axis (unit, world frame)
      u  : opening axis (finger-to-finger direction, unit)
      v  : a x u (unit)

    Returns (pos, a, u, v).
    """
    pos = np.asarray(best_pose[:3], dtype=np.float64)
    quat = np.asarray(best_pose[3:], dtype=np.float64)

    # Approach axis: in our script we mapped gripper +Z to surface normal / approach
    R = quat_to_rot(quat)
    a = R[:, 2]           # gripper +Z in world frame
    a = a / (np.linalg.norm(a) + 1e-9)

    points = np.asarray(points, dtype=np.float64)
    rel = points - pos
    dist = np.linalg.norm(rel, axis=1)

    # Use neighbors around the grasp point
    mask = dist < local_radius
    if np.sum(mask) < 10:
        # Not enough neighbors; fall back to arbitrary basis in plane ⟂ a
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-9)
        v = np.cross(a, u)
        return pos, a, u, v

    rel_local = rel[mask]

    # Project into plane ⟂ a
    proj_along_a = np.dot(rel_local, a)[:, None] * a[None, :]
    rel_plane = rel_local - proj_along_a

    plane_norm = np.linalg.norm(rel_plane, axis=1)
    if np.all(plane_norm < 1e-6):
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-9)
        v = np.cross(a, u)
        return pos, a, u, v

    # 2D basis in the plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, a)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u0 = np.cross(a, tmp)
    u0 = u0 / (np.linalg.norm(u0) + 1e-9)
    v0 = np.cross(a, u0)

    coords_2d = np.stack(
        [np.dot(rel_plane, u0), np.dot(rel_plane, v0)],
        axis=1,
    )

    # PCA in 2D
    cov = coords_2d.T @ coords_2d / max(coords_2d.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    e1 = eigvecs[:, np.argmax(eigvals)]

    # Opening axis in 3D
    u = e1[0] * u0 + e1[1] * v0
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(a, u)

    return pos, a, u, v

def extract_finger_tips(
    best_pose: np.ndarray,
    points: np.ndarray,
    gripper_width: float = 0.08,
    plane_thickness: float = 0.015,
    max_other_offset: float = 0.02,
    min_separation: float = 0.01,
    local_radius: float = 0.06,
    approach_offset: float = 0.0,
):
    """
    Given a grasp pose and a point cloud, estimate two fingertip
    contact points for a parallel-jaw gripper.

    Returns (tip1, tip2) as 3D points in world frame.
    """
    points = np.asarray(points, dtype=np.float64)
    pos, a, u, v = compute_local_gripper_frame(
        best_pose, points, local_radius=local_radius
    )

    rel = points - pos
    d_a = rel @ a  # along approach
    d_u = rel @ u  # along opening (finger-to-finger)
    d_v = rel @ v  # sideways

    # Thin slab around grasp plane & near midline
    mask_plane = np.abs(d_a) < plane_thickness
    mask_band = np.abs(d_v) < max_other_offset
    mask = mask_plane & mask_band

    if np.sum(mask) < 2:
        # Relax constraints if too strict
        mask = np.abs(d_v) < max_other_offset
    if np.sum(mask) < 2:
        # Final fallback: synthetic symmetric tips
        center = pos - approach_offset * a
        return center - 0.5 * gripper_width * u, center + 0.5 * gripper_width * u

    d_u_sel = d_u[mask]
    pts_sel = points[mask]

    # Split into two sides along u
    left_mask = d_u_sel < -min_separation
    right_mask = d_u_sel > +min_separation

    left_pts = pts_sel[left_mask]
    right_pts = pts_sel[right_mask]
    left_d = d_u_sel[left_mask]
    right_d = d_u_sel[right_mask]

    if left_pts.shape[0] == 0 or right_pts.shape[0] == 0:
        # Not nicely split; just pick extreme points along u
        idx_min = np.argmin(d_u_sel)
        idx_max = np.argmax(d_u_sel)
        p_left = pts_sel[idx_min]
        p_right = pts_sel[idx_max]
    else:
        half = gripper_width / 2.0

        # Restrict to within gripper width if possible
        left_valid = left_d >= -half
        right_valid = right_d <= half
        if np.any(left_valid):
            left_d = left_d[left_valid]
            left_pts = left_pts[left_valid]
        if np.any(right_valid):
            right_d = right_d[right_valid]
            right_pts = right_pts[right_valid]

        # Farthest on each side
        idx_left = np.argmin(left_d)   # most negative
        idx_right = np.argmax(right_d) # most positive
        p_left = left_pts[idx_left]
        p_right = right_pts[idx_right]

    if approach_offset != 0.0:
        p_left = p_left - approach_offset * a
        p_right = p_right - approach_offset * a

    return p_left, p_right


def extract_finger_tips_from_pcd(
    pcd,
    best_pose: np.ndarray,
    gripper_width: float = 0.08,
):
    if pcd.is_empty():
        raise RuntimeError(f"empty point cloud")
    pts = np.asarray(pcd.points)
    tip1, tip2 = extract_finger_tips(
        best_pose,
        pts,
        gripper_width=gripper_width,
    )
    return tip1, tip2

# ---------- Main PCD -> OrbitGrasp pipeline ----------

def grasp_from_pcd(
    pcd_path: Path,
    config_path: Path,
    voxel_size: float = 0.0055,
    sphere_radius: float = 0.05,
    min_allowed_points: Optional[int] = None,
    max_allowed_points: Optional[int] = None,
    num_centers: int = 10,
    gripper_width: float = 0.04,
):
    """
    Run OrbitGrasp on a single point cloud and return one grasp pose:
    [x, y, z, qx, qy, qz, qw].
    """
    config = load_config(config_path)

    base_path = config_path.parent
    device = config['orbit_grasper']['device']
    num_intervals = config['orbit_grasper']['num_channel']
    lmax = (
        config['train_dataset']['lmax']
        if 'train_dataset' in config
        else config.get('lmax', 3)
    )

    ckpt_root = base_path / config['test']['root_dir']
    ckpt_name = find_checkpoint(
        ckpt_root,
        config['test']['load_epoch'],
        config['test'].get('ckpt_prefix', 'orbitgrasp-ckpt-')
    )

    orbit_grasper = OrbitGrasper(
        device=device,
        load=config['test']['load_epoch'],
        param_dir=ckpt_root,
        num_channel=num_intervals,
        lmax=config['orbit_grasper'].get('lmax', 3),
        mmax=config['orbit_grasper'].get('mmax', 2),
        load_name=ckpt_name,
        training_config=config,
    )

    set_seed(config.get('seed', 12345))

    # --- Load and preprocess PCD (mostly same as test_grasp_single) ---
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise RuntimeError(f"Loaded empty point cloud from {pcd_path}")

    # Denoising
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=4.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.03)

    # Normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    down_pcd = voxel_down_sample(pcd, voxel_size=voxel_size)

    down_points_np = np.asarray(down_pcd.points)
    down_normals_np = np.asarray(down_pcd.normals)

    if down_points_np.shape[0] == 0:
        raise RuntimeError("Downsampled point cloud is empty after preprocessing.")

    # --- Auto-tune region size thresholds if user did not override them ---
    if min_allowed_points is None or max_allowed_points is None:
        stats = estimate_local_region_stats(down_points_np, sphere_radius)
        print(
            f"[density] radius={sphere_radius:.3f} m -> "
            f"min={stats['min']}, max={stats['max']}, "
            f"mean={stats['mean']:.1f}, median={stats['median']:.1f} "
            f"(over {stats['num_samples']} samples)"
        )

        if min_allowed_points is None:
            # allow smaller regions, but avoid extremely tiny ones
            min_allowed_points = max(20, int(stats["median"] * 0.3))

        if max_allowed_points is None:
            max_allowed_points = int(stats["max"] * 1.1)

        print(
            f"[density] using min_allowed_points={min_allowed_points}, "
            f"max_allowed_points={max_allowed_points}"
        )

    down_points = torch.from_numpy(down_points_np).float().to(device)
    down_normals = torch.from_numpy(down_normals_np).float().to(device)

    point_sampler = FarthestSampler()
    center_points, _ = point_sampler(
        down_points.cpu().numpy(), num_centers, sphere_radius + 0.002
    )

    down_points_tensor = down_points.clone().detach()

    all_data_list = []
    all_sphere_indices_list = []
    all_harmonics_list = []
    all_grasp_poses_list = []
    region_sizes = []
    valid_mask = []

    for center in center_points:
        center_tensor = torch.tensor(
            center, dtype=torch.float32, device=device
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

        # Build features for OrbitGrasp (for *all* regions)
        feature_points = FeaturedPoints(
            x=sphere_points,
            n=sphere_normals,
            b=torch.ones(
                sphere_points.shape[0], dtype=torch.long, device=device
            ),
        )
        feature_points = normalize_feature_points(feature_points)
        all_data_list.append(feature_points)

        sphere_indices = torch.arange(
            sphere_points.shape[0], device=device
        )
        all_sphere_indices_list.append(sphere_indices)

        grasp_poses, z_dirs_tiled = build_grasp_candidates_for_sphere(
            sphere_points, sphere_normals, num_intervals=num_intervals
        )
        all_grasp_poses_list.append(grasp_poses)

        harmonics = compute_spherical_harmonics(
            z_dirs_tiled, lmax=lmax
        )
        all_harmonics_list.append(harmonics.to(device))

        # Decide if this region passes thresholds (if thresholds are set)
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

    # Prefer only regions that pass thresholds; otherwise fall back to all of them
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

    # --- Run OrbitGrasp and pick the best-scoring grasp ---
    score_list, _ = orbit_grasper.predict(
        data_list, sphere_indices_list, harmonics_list
    )
    score_list = torch.cat(score_list, dim=0).cpu().numpy()
    grasp_poses_array = np.concatenate(
        grasp_poses_list, axis=0
    )  # [num_grasps, num_intervals, 7]

    flat_idx = np.argmax(score_list)
    best_i, best_k = np.unravel_index(flat_idx, score_list.shape)
    best_pose = grasp_poses_array[best_i, best_k]  # [x, y, z, qx, qy, qz, qw]

    tip1, tip2 = extract_finger_tips_from_pcd(pcd, best_pose, gripper_width=gripper_width)

    return best_pose, tip1, tip2


def save_visualized_grasp_pcd(
    pcd_path: Path,
    poses: list,
    out_path: Path,
    sphere_radius: float = 0.03,
    sphere_density: int = 40,
):
    """
    Creates a NEW point cloud containing:
      - original PCD
      - a dense green sphere at the grasp contact point
    and saves it as a .pcd or .ply file.
    """

    # Load original cloud
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise RuntimeError(f"Loaded empty point cloud from {pcd_path}")

    spheres = []
    for pose in poses:
        # Extract grasp point
        x, y, z = pose[0], pose[1], pose[2]
        grasp_pos = np.array([x, y, z], dtype=np.float64)

        # Create mesh sphere
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius,
            resolution=sphere_density,
        )
        sphere_mesh.translate(grasp_pos)
        sphere_mesh.paint_uniform_color([0.0, 1.0, 0.0])  # green

        # Convert sphere to point cloud
        sphere_pcd = sphere_mesh.sample_points_uniformly(
            number_of_points=5000
        )
        sphere_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 0.0], (len(sphere_pcd.points), 1))
        )
        spheres.append(copy.deepcopy(sphere_pcd))

    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 1.0], (len(pcd.points), 1))
    )

    # Merge clouds
    merged = pcd
    for sphere_pcd in spheres:
        merged += sphere_pcd

    # Save
    if out_path.suffix.lower() == ".pcd":
        o3d.io.write_point_cloud(str(out_path), merged)
    else:
        o3d.io.write_point_cloud(str(out_path.with_suffix(".ply")), merged)

    print(f"Saved merged grasp visualization: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict a single grasp pose from a .pcd file using OrbitGrasp."
    )
    parser.add_argument(
        "pcd_path",
        type=str,
        help="Path to input .pcd file (point cloud in camera/world frame).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(THIS_DIR / "single_config.yaml"),
        help="Path to OrbitGrasp YAML config (defaults to scripts/single_config.yaml).",
    )
    parser.add_argument(
        "--save-pcd",
        type=str,
        default=None,
        help="Save a new PCD/PLY with a green sphere at the grasp point."
    )

    args = parser.parse_args()

    pcd_path = Path(args.pcd_path)
    config_path = Path(args.config)

    best_pose, tip1, tip2 = grasp_from_pcd(
        pcd_path=pcd_path,
        config_path=config_path,
        voxel_size=0.0055,          # denser cloud
        sphere_radius=0.05,        # bigger region
        num_centers=10,            # increase sampling
        min_allowed_points=None,    # ↓↓↓ VERY IMPORTANT ↓↓↓
        max_allowed_points=None,
    )

    # Print as space-separated line for easy parsing
    x, y, z, qx, qy, qz, qw = best_pose
    print(f"{x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    if args.save_pcd is not None:
        save_visualized_grasp_pcd(
            pcd_path,
            [best_pose, tip1, tip2],
            Path(args.save_pcd),
            sphere_radius=0.005,
        )


"""
python scripts/grasp_from_pcd.py masked.pcd --save-pcd masked_grasp.pcd
"""

if __name__ == "__main__":
    main()
