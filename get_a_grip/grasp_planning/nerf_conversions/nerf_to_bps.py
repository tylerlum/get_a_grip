from __future__ import annotations

from typing import Tuple

import numpy as np
from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.grasp_planning.utils.nerfstudio_point_cloud import ExportPointCloud
from get_a_grip.model_training.scripts.create_bps_grasp_dataset import (
    crop_single_point_cloud,
    get_bps,
    get_fixed_basis_points,
    process_single_point_cloud,
)
from get_a_grip.utils.generate_point_cloud import (
    PointCloudBoundingBox,
)
from get_a_grip.utils.point_utils import (
    transform_points,
)


def nerf_to_bps(
    nerf_pipeline: Pipeline,
    nerf_is_z_up: bool,
    X_N_By: np.ndarray,
    num_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounding_box = PointCloudBoundingBox(nerf_is_z_up=nerf_is_z_up)
    point_cloud_exporter = ExportPointCloud(
        normal_method="open3d",
        obb_center=bounding_box.obb_center,
        obb_rotation=bounding_box.obb_rotation,
        obb_scale=bounding_box.obb_scale,
        num_points=num_points,
    )
    point_cloud = point_cloud_exporter.main(nerf_pipeline)

    #### BELOW IS TIGHTLY CONNECTED TO create_grasp_bps_dataset ####
    # Load point cloud
    points = process_single_point_cloud(point_cloud)
    final_points = crop_single_point_cloud(points)
    assert final_points is not None, f"Cropping failed, points.shape = {points.shape}"

    # Frames
    final_points_N = final_points

    # BPS
    basis_points_By = get_fixed_basis_points()
    assert basis_points_By.shape == (
        4096,
        3,
    ), f"Expected shape (4096, 3), got {basis_points_By.shape}"
    basis_points_N = transform_points(T=X_N_By, points=basis_points_By)
    bps_values = get_bps(
        all_points=final_points_N[None],
        basis_points=basis_points_N,
    ).squeeze(axis=0)
    assert bps_values.shape == (
        4096,
    ), f"Expected shape (4096,), got {bps_values.shape}"

    X_By_N = np.linalg.inv(X_N_By)
    _final_points_By = transform_points(T=X_By_N, points=final_points_N)
    return bps_values, basis_points_N, final_points_N
