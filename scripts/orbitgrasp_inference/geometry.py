from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d


def quaternion_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    if dot < -0.999999:
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


def quat_to_rot(q: np.ndarray) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(best_pose[:3], dtype=np.float64)
    quat = np.asarray(best_pose[3:], dtype=np.float64)

    R = quat_to_rot(quat)
    a = R[:, 2]
    a = a / (np.linalg.norm(a) + 1e-9)

    points = np.asarray(points, dtype=np.float64)
    rel = points - pos
    dist = np.linalg.norm(rel, axis=1)

    mask = dist < local_radius
    if np.sum(mask) < 10:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-9)
        v = np.cross(a, u)
        return pos, a, u, v

    rel_local = rel[mask]

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

    cov = coords_2d.T @ coords_2d / max(coords_2d.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    e1 = eigvecs[:, np.argmax(eigvals)]

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
    points = np.asarray(points, dtype=np.float64)
    pos, a, u, v = compute_local_gripper_frame(
        best_pose, points, local_radius=local_radius
    )

    rel = points - pos
    d_a = rel @ a
    d_u = rel @ u
    d_v = rel @ v

    mask_plane = np.abs(d_a) < plane_thickness
    mask_band = np.abs(d_v) < max_other_offset
    mask = mask_plane & mask_band

    if np.sum(mask) < 2:
        mask = np.abs(d_v) < max_other_offset
    if np.sum(mask) < 2:
        center = pos - approach_offset * a
        return center - 0.5 * gripper_width * u, center + 0.5 * gripper_width * u

    d_u_sel = d_u[mask]
    pts_sel = points[mask]

    left_mask = d_u_sel < -min_separation
    right_mask = d_u_sel > +min_separation

    left_pts = pts_sel[left_mask]
    right_pts = pts_sel[right_mask]
    left_d = d_u_sel[left_mask]
    right_d = d_u_sel[right_mask]

    if left_pts.shape[0] == 0 or right_pts.shape[0] == 0:
        idx_min = np.argmin(d_u_sel)
        idx_max = np.argmax(d_u_sel)
        p_left = pts_sel[idx_min]
        p_right = pts_sel[idx_max]
    else:
        half = gripper_width / 2.0

        left_valid = left_d >= -half
        right_valid = right_d <= half
        if np.any(left_valid):
            left_d = left_d[left_valid]
            left_pts = left_pts[left_valid]
        if np.any(right_valid):
            right_d = right_d[right_valid]
            right_pts = right_pts[right_valid]

        idx_left = np.argmin(left_d)
        idx_right = np.argmax(right_d)
        p_left = left_pts[idx_left]
        p_right = right_pts[idx_right]

    if approach_offset != 0.0:
        p_left = p_left - approach_offset * a
        p_right = p_right - approach_offset * a

    return p_left, p_right


def extract_finger_tips_from_pcd(
    pcd: o3d.geometry.PointCloud,
    best_pose: np.ndarray,
    gripper_width: float = 0.08,
):
    if pcd.is_empty():
        raise RuntimeError("empty point cloud")
    pts = np.asarray(pcd.points)
    tip1, tip2 = extract_finger_tips(
        best_pose,
        pts,
        gripper_width=gripper_width,
    )
    return tip1, tip2
