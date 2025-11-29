import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.orbitgrasp_inference.pipeline import OrbitGraspPipeline
from scripts.orbitgrasp_inference.visualization import save_visualized_grasp_pcd


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
        help="Save a new PCD/PLY with a green sphere at the grasp point.",
    )

    args = parser.parse_args()

    pcd_path = Path(args.pcd_path)
    config_path = Path(args.config)

    pipeline = OrbitGraspPipeline(config_path=config_path)

    best_pose, tip1, tip2 = pipeline.grasp_from_pcd(
        pcd_path=pcd_path,
        voxel_size=0.0055,
        sphere_radius=0.05,
        num_centers=10,
        min_allowed_points=None,
        max_allowed_points=None,
        gripper_width=0.04,
    )

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
