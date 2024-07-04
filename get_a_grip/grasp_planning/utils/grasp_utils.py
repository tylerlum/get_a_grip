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
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_bps import nerf_to_bps

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


def normalize_with_warning(v: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    B = v.shape[0]
    assert v.shape == (B, 3), f"Expected shape ({B}, 3), got {v.shape}"
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norm < atol):
        print("^" * 80)
        print(
            f"Warning: Found {np.sum(norm < atol)} vectors with norm less than {atol}"
        )
        print("^" * 80)
    return v / (norm + atol)


def rot6d_to_matrix(rot6d: np.ndarray, check: bool = True) -> np.ndarray:
    B = rot6d.shape[0]
    assert rot6d.shape == (B, 6), f"Expected shape ({B}, 6), got {rot6d.shape}"

    # Step 1: Reshape to (B, 3, 2)
    rot3x2 = rot6d.reshape(B, 3, 2)

    # Step 2: Normalize the first column
    col1 = rot3x2[:, :, 0]
    col1_normalized = normalize_with_warning(col1)

    # Step 3: Orthogonalize the second column with respect to the first column
    col2 = rot3x2[:, :, 1]
    dot_product = np.sum(col1_normalized * col2, axis=1, keepdims=True)
    col2_orthogonal = col2 - dot_product * col1_normalized

    # Step 4: Normalize the second column
    col2_normalized = normalize_with_warning(col2_orthogonal)

    # Step 5: Compute the cross product to obtain the third column
    col3 = np.cross(col1_normalized, col2_normalized)

    # Combine the columns to form the rotation matrix
    rotation_matrices = np.stack((col1_normalized, col2_normalized, col3), axis=-1)

    # Step 6: Check orthogonality and determinant
    if check:
        for i in range(B):
            mat = rotation_matrices[i]
            assert np.allclose(
                np.dot(mat.T, mat), np.eye(3), atol=1e-3
            ), f"Matrix {i} is not orthogonal, got {np.dot(mat.T, mat)}"
            assert np.allclose(
                np.linalg.det(mat), 1.0, atol=1e-3
            ), f"Matrix {i} does not have determinant 1, got {np.linalg.det(mat)}"

    assert rotation_matrices.shape == (
        B,
        3,
        3,
    ), f"Expected shape ({B}, 3, 3), got {rotation_matrices.shape}"
    return rotation_matrices


def compute_grasp_orientations(
    grasp_dirs: torch.Tensor,
    wrist_pose: pp.LieTensor,
    joint_angles: torch.Tensor,
) -> pp.LieTensor:
    B = grasp_dirs.shape[0]
    N_FINGERS = 4
    assert grasp_dirs.shape == (
        B,
        N_FINGERS,
        3,
    ), f"Expected shape ({B}, {N_FINGERS}, 3), got {grasp_dirs.shape}"
    assert wrist_pose.lshape == (B,), f"Expected shape ({B},), got {wrist_pose.lshape}"

    # Normalize
    z_dirs = grasp_dirs
    z_dirs = z_dirs / z_dirs.norm(dim=-1, keepdim=True)

    # Get hand model
    hand_config = AllegroHandConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=joint_angles,
    )
    hand_model = hand_config_to_hand_model(
        hand_config=hand_config,
    )

    # Math to get x_dirs, y_dirs
    (center_to_right_dirs, center_to_tip_dirs) = compute_fingertip_dirs(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )
    option_1_ok = (
        torch.cross(center_to_tip_dirs, z_dirs).norm(dim=-1, keepdim=True) > 1e-4
    )

    y_dirs = torch.where(
        option_1_ok,
        center_to_tip_dirs
        - (center_to_tip_dirs * z_dirs).sum(dim=-1, keepdim=True) * z_dirs,
        center_to_right_dirs
        - (center_to_right_dirs * z_dirs).sum(dim=-1, keepdim=True) * z_dirs,
    )

    assert (y_dirs.norm(dim=-1).min() > 0).all()
    y_dirs = y_dirs / y_dirs.norm(dim=-1, keepdim=True)

    x_dirs = torch.cross(y_dirs, z_dirs)
    assert (x_dirs.norm(dim=-1).min() > 0).all()
    x_dirs = x_dirs / x_dirs.norm(dim=-1, keepdim=True)
    grasp_orientations = torch.stack([x_dirs, y_dirs, z_dirs], dim=-1)
    # Make sure y and z are orthogonal
    assert (torch.einsum("...l,...l->...", y_dirs, z_dirs).abs().max() < 1e-3).all(), (
        f"y_dirs = {y_dirs}",
        f"z_dirs = {z_dirs}",
        f"torch.einsum('...l,...l->...', y_dirs, z_dirs).abs().max() = {torch.einsum('...l,...l->...', y_dirs, z_dirs).abs().max()}",
    )
    assert grasp_orientations.shape == (
        B,
        N_FINGERS,
        3,
        3,
    ), f"Expected shape ({B}, {N_FINGERS}, 3, 3), got {grasp_orientations.shape}"
    grasp_orientations = pp.from_matrix(
        grasp_orientations,
        pp.SO3_type,
    )
    assert grasp_orientations.lshape == (
        B,
        N_FINGERS,
    ), f"Expected shape ({B}, {N_FINGERS}), got {grasp_orientations.lshape}"

    return grasp_orientations


def grasp_to_grasp_config(grasp: torch.Tensor) -> AllegroGraspConfig:
    device = grasp.device

    N_FINGERS = 4
    GRASP_DIM = 3 + 6 + 16 + 4 * 3
    x = grasp
    B = x.shape[0]

    assert x.shape == (
        B,
        GRASP_DIM,
    ), f"Expected shape ({B}, {GRASP_DIM}), got {x.shape}"
    trans = x[:, :3].detach().cpu().numpy()
    rot6d = x[:, 3:9].detach().cpu().numpy()
    joint_angles = x[:, 9:25].detach().cpu().numpy()
    grasp_dirs = x[:, 25:37].reshape(B, N_FINGERS, 3).detach().cpu().numpy()

    rot = rot6d_to_matrix(rot6d)

    wrist_pose_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1).float()
    wrist_pose_matrix[:, :3, :3] = torch.from_numpy(rot).float().to(device)
    wrist_pose_matrix[:, :3, 3] = torch.from_numpy(trans).float().to(device)

    try:
        wrist_pose = pp.from_matrix(
            wrist_pose_matrix,
            pp.SE3_type,
        ).to(device)
    except ValueError as e:
        print("Error in pp.from_matrix")
        print(e)
        print("rot = ", rot)
        print("Orthogonalization did not work, running with looser tolerances")
        wrist_pose = pp.from_matrix(
            wrist_pose_matrix,
            pp.SE3_type,
            atol=1e-3,
            rtol=1e-3,
        ).to(device)

    assert wrist_pose.lshape == (B,)

    grasp_orientations = compute_grasp_orientations(
        grasp_dirs=torch.from_numpy(grasp_dirs).float().to(device),
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
    )

    # Convert to AllegroGraspConfig to dict
    grasp_configs = AllegroGraspConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
        grasp_orientations=grasp_orientations,
    )
    return grasp_configs
