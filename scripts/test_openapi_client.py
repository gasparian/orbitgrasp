from typing import List

from orbit_grasp_inference_api_client import Client
from orbit_grasp_inference_api_client.models import DetectRequest, DetectResponse
from orbit_grasp_inference_api_client.api.inference import detect_grasp_detect_post as detect_grasps
from orbit_grasp_inference_api_client.types import Response

import numpy as np
import open3d as o3d


def load_points_from_pcd(path: str) -> List[List[float]]:
    pcd = o3d.io.read_point_cloud(path)
    points_np = np.asarray(pcd.points, dtype=float)
    return points_np.tolist()


if __name__ == "__main__":
    client = Client(base_url="http://localhost:8000")
    points = load_points_from_pcd("masked.pcd")

    detect_request = DetectRequest(points=points)

    with client as client:
        result: DetectResponse = detect_grasps.sync(
            client=client,
            body=detect_request,
        )
        print("Parsed result:", result)