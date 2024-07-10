import pathlib
import subprocess
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tyro


@dataclass
class GeneratePointCloudArgs:
    nerf_is_z_up: bool
    nerf_config: pathlib.Path
    output_dir: pathlib.Path
    num_points: int = 5000
    timeout: float = 60.0

    def __post_init__(self):
        self._bounding_box = PointCloudBoundingBox(nerf_is_z_up=self.nerf_is_z_up)

    @staticmethod
    def tuple_to_str(tup: Tuple[float, ...]) -> str:
        return " ".join([str(x) for x in tup])

    @property
    def obb_scale(self) -> str:
        return self.tuple_to_str(self._bounding_box.obb_scale)

    @property
    def obb_center(self) -> str:
        return self.tuple_to_str(self._bounding_box.obb_center)

    @property
    def obb_rotation(self) -> str:
        return self.tuple_to_str(self._bounding_box.obb_rotation)


@dataclass
class PointCloudBoundingBox:
    nerf_is_z_up: bool

    @property
    def obb_scale(self) -> Tuple[float, float, float]:
        bb_min = nerf_to_point_cloud_bounding_box_min(self.nerf_is_z_up)
        bb_max = nerf_to_point_cloud_bounding_box_max(self.nerf_is_z_up)
        obb_scale = bb_max - bb_min
        return (obb_scale[0], obb_scale[1], obb_scale[2])

    @property
    def obb_center(self) -> Tuple[float, float, float]:
        bb_min = nerf_to_point_cloud_bounding_box_min(self.nerf_is_z_up)
        bb_max = nerf_to_point_cloud_bounding_box_max(self.nerf_is_z_up)
        obb_center = (bb_max + bb_min) / 2
        return (obb_center[0], obb_center[1], obb_center[2])

    @property
    def obb_rotation(self) -> Tuple[float, float, float]:
        obb_rotation = np.array([0.0, 0.0, 0.0])
        return (obb_rotation[0], obb_rotation[1], obb_rotation[2])


# Hardcoded intentionally for consistency
NERF_TO_POINT_CLOUD_MIN_HEIGHT = 0
NERF_TO_POINT_CLOUD_MAX_HEIGHT = 0.3
NERF_TO_POINT_CLOUD_MIN_WIDTH = -0.2
NERF_TO_POINT_CLOUD_MAX_WIDTH = 0.2


def nerf_to_point_cloud_bounding_box_min(nerf_is_z_up: bool) -> np.ndarray:
    min_height = NERF_TO_POINT_CLOUD_MIN_HEIGHT
    min_width = NERF_TO_POINT_CLOUD_MIN_WIDTH

    if nerf_is_z_up:
        min_z = min_height

        min_x = min_width
        min_y = min_width
    else:
        min_y = min_height

        min_x = min_width
        min_z = min_width
    return np.array([min_x, min_y, min_z])


def nerf_to_point_cloud_bounding_box_max(nerf_is_z_up: bool) -> np.ndarray:
    max_height = NERF_TO_POINT_CLOUD_MAX_HEIGHT
    max_width = NERF_TO_POINT_CLOUD_MAX_WIDTH

    if nerf_is_z_up:
        max_z = max_height

        max_x = max_width
        max_y = max_width
    else:
        max_y = max_height

        max_x = max_width
        max_z = max_width
    return np.array([max_x, max_y, max_z])


def generate_point_cloud(
    args: GeneratePointCloudArgs,
) -> None:
    command = " ".join(
        [
            "ns-export",
            "pointcloud",
            f"--load-config {str(args.nerf_config)}",
            f"--output-dir {str(args.output_dir)}",
            "--normal-method open3d",
            f"--obb-scale {args.obb_scale}",
            f"--obb-center {args.obb_center}",
            f"--obb-rotation {args.obb_rotation}",
            f"--num-points {args.num_points}",
        ]
    )

    # NOTE: Some nerfs are terrible by bad luck, so computing point clouds never finishes, so include timeout
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True, timeout=args.timeout)


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GeneratePointCloudArgs])
    generate_point_cloud(args)


if __name__ == "__main__":
    main()
