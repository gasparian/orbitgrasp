from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.orbitgrasp_inference.pipeline import OrbitGraspPipeline  # type: ignore


class DetectRequest(BaseModel):
    points: List[List[float]] = Field(
        ...,
        description="Point cloud as a list of [x, y, z] in the camera/world frame.",
        min_items=1,
    )


class DetectResponse(BaseModel):
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class OrbitGraspService:
    def __init__(self, config_path: Path) -> None:
        self.pipeline = OrbitGraspPipeline(config_path=config_path)

    def detect_from_points(self, points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points must have shape (N, 3).")
        if points.shape[0] == 0:
            raise ValueError("Point cloud contains no points.")

        tmp_dir = THIS_DIR / "tmp_pcd"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}.pcd"

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            o3d.io.write_point_cloud(str(tmp_path), pcd)

            best_pose, _, _ = self.pipeline.grasp_from_pcd(
                pcd_path=tmp_path,
                voxel_size=0.0055,
                sphere_radius=0.05,
                min_allowed_points=None,
                max_allowed_points=None,
                num_centers=10,
                gripper_width=0.04,
            )
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return best_pose


def _default_config_path() -> Path:
    env_path = os.environ.get("ORBITGRASP_CONFIG")
    if env_path:
        return Path(env_path).resolve()
    return (THIS_DIR / "single_config.yaml").resolve()


def get_service() -> OrbitGraspService:
    config_path = _default_config_path()
    if not config_path.is_file():
        raise RuntimeError(f"Config file not found: {config_path}")
    if not hasattr(get_service, "_instance"):
        get_service._instance = OrbitGraspService(config_path=config_path)  # type: ignore[attr-defined]
    return get_service._instance  # type: ignore[attr-defined]


app = FastAPI(
    title="OrbitGrasp Inference API",
    version="0.1.0",
    description=(
        "FastAPI server that runs OrbitGrasp on a single point cloud and returns "
        "one SE(3) grasp pose."
    ),
)


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect a single grasp pose from a point cloud",
    tags=["inference"],
)
def detect_grasp(request: DetectRequest, service: OrbitGraspService = Depends(get_service)):
    try:
        points = np.asarray(request.points, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid points format: {exc}")

    try:
        best_pose = service.detect_from_points(points)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    x, y, z, qx, qy, qz, qw = best_pose
    return DetectResponse(
        x=float(x),
        y=float(y),
        z=float(z),
        qx=float(qx),
        qy=float(qy),
        qz=float(qz),
        qw=float(qw),
    )
