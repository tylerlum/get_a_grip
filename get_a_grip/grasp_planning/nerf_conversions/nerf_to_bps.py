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
from get_a_grip.model_training.utils.point_utils import (
    transform_points,
)


def nerf_to_bps(
    nerf_pipeline: Pipeline,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    num_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert lb_N.shape == (3,)
    assert ub_N.shape == (3,)

    # TODO: This is slow because it loads NeRF from file and then outputs point cloud to file
    point_cloud_exporter = ExportPointCloud(
        normal_method="open3d",
        bounding_box_min=(lb_N[0], lb_N[1], lb_N[2]),
        bounding_box_max=(ub_N[0], ub_N[1], ub_N[2]),
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
    )
    assert bps_values.shape == (
        4096,
    ), f"Expected shape (4096,), got {bps_values.shape}"

    X_By_N = np.linalg.inv(X_N_By)
    final_points_By = transform_points(T=X_By_N, points=final_points_N)
    return bps_values, basis_points_By, final_points_By
