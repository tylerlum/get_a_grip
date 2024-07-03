from __future__ import annotations

import math
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import pypose as pp
import torch
import trimesh
import tyro
import wandb
from nerfstudio.pipelines.base_pipeline import Pipeline
from tqdm import tqdm

from get_a_grip import get_package_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_fingertip_dirs,
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from get_a_grip.dataset_generation.utils.pose_conversion import hand_config_to_pose
from get_a_grip.grasp_planning.config.grasp_metric_config import GraspMetricConfig
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    RandomSamplingConfig,
)
from get_a_grip.grasp_planning.scripts.optimizer import (
    sample_random_rotate_transforms_only_around_y,
)
from get_a_grip.grasp_planning.utils import (
    ablation_utils,
    train_nerf_return_trainer,
)
from get_a_grip.grasp_planning.utils.ablation_optimizer import RandomSamplingOptimizer
from get_a_grip.grasp_planning.utils.nerfstudio_point_cloud import ExportPointCloud
from get_a_grip.grasp_planning.utils.optimizer_utils import (
    AllegroGraspConfig,
    AllegroHandConfig,
    hand_config_to_hand_model,
)
from get_a_grip.model_training.models.dex_evaluator import DexEvaluator
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.model_training.utils.nerf_utils import (
    compute_centroid_from_nerf,
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
    pointcloud_exporter = ExportPointCloud(
        normal_method="open3d",
        bounding_box_min=(lb_N[0], lb_N[1], lb_N[2]),
        bounding_box_max=(ub_N[0], ub_N[1], ub_N[2]),
        num_points=num_points,
    )
    point_cloud = pointcloud_exporter.main(nerf_pipeline)

    #### BELOW IS TIGHTLY CONNECTED TO create_grasp_bps_dataset ####
    # Load point cloud
    from get_a_grip.model_training.scripts.create_bps_grasp_dataset import (
        process_point_cloud,
    )

    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    points = np.asarray(point_cloud.points)

    inlier_points = process_point_cloud(points)
    N_PTS = inlier_points.shape[0]
    assert inlier_points.shape == (
        N_PTS,
        3,
    ), f"inlier_points.shape = {inlier_points.shape}"

    MIN_N_POINTS = 3000
    assert (
        N_PTS >= MIN_N_POINTS
    ), f"Expected at least {MIN_N_POINTS} points, but got {N_PTS}"
    final_points = inlier_points[:MIN_N_POINTS]

    # Frames
    final_points_N = final_points

    # BPS
    from bps import bps

    N_BASIS_PTS = 4096
    basis_point_path = get_package_folder() / "model_training" / "basis_points.npy"
    assert basis_point_path.exists(), f"{basis_point_path} does not exist"
    with open(basis_point_path, "rb") as f:
        basis_points_By = np.load(f)
    assert basis_points_By.shape == (
        N_BASIS_PTS,
        3,
    ), f"Expected shape ({N_BASIS_PTS}, 3), got {basis_points_By.shape}"
    basis_points_N = transform_points(T=X_N_By, points=basis_points_By)
    bps_values = bps.encode(
        final_points_N[None],
        bps_arrangement="custom",
        bps_cell_type="dists",
        custom_basis=basis_points_N,
        verbose=0,
    ).squeeze(axis=0)
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"

    X_By_N = np.linalg.inv(X_N_By)
    final_points_By = transform_points(T=X_By_N, points=final_points_N)
    return bps_values, basis_points_By, final_points_By

