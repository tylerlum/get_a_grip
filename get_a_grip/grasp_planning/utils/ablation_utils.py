from __future__ import annotations
import time
from tqdm import tqdm
import trimesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
)
from dataclasses import dataclass
from nerf_grasping.dexdiffuser.dex_evaluator import DexEvaluator
from nerf_grasping.nerfstudio_train import train_nerfs_return_trainer
from nerf_grasping.grasp_utils import load_nerf_pipeline
import nerf_grasping
import math
import pypose as pp
from collections import defaultdict
from nerf_grasping.optimizer import (
    sample_random_rotate_transforms_only_around_y,
)
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
    GraspMetric,
    DepthImageGraspMetric,
    predict_in_collision_with_object,
    predict_in_collision_with_table,
    get_hand_surface_points_Oy,
    get_joint_limits,
)
from dataclasses import asdict
from nerf_grasping.config.optimization_config import OptimizationConfig
import pathlib
import torch
from nerf_grasping.classifier import Classifier, Simple_CNN_LSTM_Classifier
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimizer_config import (
    SGDOptimizerConfig,
    CEMOptimizerConfig,
    RandomSamplingConfig,
)
from typing import Tuple, Union, Dict, Literal
import nerf_grasping
from functools import partial
import numpy as np
import tyro
import wandb

from rich.console import Console
from rich.table import Table

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import plotly.graph_objects as go
import numpy as np
import pathlib
import pytorch_kinematics as pk
from pytorch_kinematics.chain import Chain
import pypose as pp
import torch

import nerf_grasping
from nerf_grasping import grasp_utils

from typing import List, Tuple, Dict, Any, Iterable, Union, Optional
from nerfstudio.fields.base_field import Field
from nerfstudio.models.base_model import Model
from nerf_grasping.classifier import (
    Classifier,
    DepthImageClassifier,
    Simple_CNN_LSTM_Classifier,
)
from nerf_grasping.learned_metric.DexGraspNet_batch_data import (
    BatchDataInput,
    DepthImageBatchDataInput,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    transform_point,
    transform_points,
)
from nerf_grasping.nerf_utils import (
    get_cameras,
    render,
    get_densities_in_grid,
    get_density,
)
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.config.fingertip_config import UnionFingertipConfig
from nerf_grasping.config.camera_config import CameraConfig
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.dataset.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
from nerfstudio.pipelines.base_pipeline import Pipeline

import open3d as o3d
from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
from nerf_grasping.dexgraspnet_utils.hand_model_type import (
    HandModelType,
)
from nerf_grasping.dexgraspnet_utils.pose_conversion import (
    hand_config_to_pose,
)
from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)

import trimesh
from typing import List, Tuple
from nerf_grasping.dexdiffuser.diffusion import Diffusion
from nerf_grasping.dexdiffuser.diffusion_config import Config, TrainingConfig
from tqdm import tqdm
from nerf_grasping.dexdiffuser.dex_evaluator import DexEvaluator
import nerf_grasping
import math
import pypose as pp
from collections import defaultdict
from nerf_grasping.optimizer import (
    sample_random_rotate_transforms_only_around_y,
)
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
    AllegroHandConfig,
    GraspMetric,
    DepthImageGraspMetric,
    predict_in_collision_with_object,
    predict_in_collision_with_table,
    get_hand_surface_points_Oy,
    get_joint_limits,
    hand_config_to_hand_model,
)
from dataclasses import asdict
from nerf_grasping.config.optimization_config import OptimizationConfig
import pathlib
import torch
from nerf_grasping.classifier import Classifier, Simple_CNN_LSTM_Classifier
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimizer_config import (
    SGDOptimizerConfig,
    CEMOptimizerConfig,
    RandomSamplingConfig,
)
from typing import Tuple, Union, Dict
import nerf_grasping
from functools import partial
import numpy as np
import tyro
import wandb

from rich.console import Console
from rich.table import Table

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import plotly.graph_objects as go
import numpy as np
import pathlib
import pytorch_kinematics as pk
from pytorch_kinematics.chain import Chain
import pypose as pp
import torch

import nerf_grasping
from nerf_grasping import grasp_utils

from typing import List, Tuple, Dict, Any, Iterable, Union, Optional
from nerfstudio.fields.base_field import Field
from nerfstudio.models.base_model import Model
from nerf_grasping.classifier import (
    Classifier,
    DepthImageClassifier,
    Simple_CNN_LSTM_Classifier,
)
from nerf_grasping.learned_metric.DexGraspNet_batch_data import (
    BatchDataInput,
    DepthImageBatchDataInput,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    transform_point,
    transform_points,
)
from nerf_grasping.nerf_utils import (
    get_cameras,
    render,
    get_densities_in_grid,
    get_density,
)
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.config.fingertip_config import UnionFingertipConfig
from nerf_grasping.config.camera_config import CameraConfig
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.dataset.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
    compute_fingertip_dirs,
)

import open3d as o3d


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
    from nerf_grasping.nerfstudio_point_cloud_copy import ExportPointCloud

    cfg = ExportPointCloud(
        normal_method="open3d",
        bounding_box_min=(lb_N[0], lb_N[1], lb_N[2]),
        bounding_box_max=(ub_N[0], ub_N[1], ub_N[2]),
        num_points=num_points,
    )
    point_cloud = cfg.main(nerf_pipeline)

    #### BELOW IS TIGHTLY CONNECTED TO create_grasp_bps_dataset ####
    # Load point cloud
    from nerf_grasping.dexdiffuser.create_grasp_bps_dataset import process_point_cloud

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
    basis_point_path = (
        pathlib.Path(nerf_grasping.get_package_root())
        / "dexdiffuser"
        / "basis_points.npy"
    )
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


def visualize_point_cloud_and_bps_and_grasp(
    grasp: torch.Tensor,
    X_W_Oy: np.array,
    basis_points: np.array,
    bps: np.array,
    mesh: trimesh.Trimesh,
    point_cloud_points: Optional[np.array],
    GRASP_IDX: int,
    object_code: str,
    passed_eval: int,
) -> None:
    # Extract data from grasp
    assert grasp.shape == (
        3 + 6 + 16 + 4 * 3,
    ), f"Expected shape (3 + 6 + 16 + 4 * 3), got {grasp.shape}"
    assert X_W_Oy.shape == (4, 4), f"Expected shape (4, 4), got {X_W_Oy.shape}"
    assert basis_points.shape == (
        4096,
        3,
    ), f"Expected shape (4096, 3), got {basis_points.shape}"
    assert bps.shape == (4096,), f"Expected shape (4096,), got {bps.shape}"
    if point_cloud_points is not None:
        B = point_cloud_points.shape[0]
        assert point_cloud_points.shape == (
            B,
            3,
        ), f"Expected shape ({B}, 3), got {point_cloud_points.shape}"

    grasp = grasp.detach().cpu().numpy()
    grasp_trans, grasp_rot6d, grasp_joints, grasp_dirs = (
        grasp[:3],
        grasp[3:9],
        grasp[9:25],
        grasp[25:].reshape(4, 3),
    )
    grasp_rot = np.zeros((3, 3))
    grasp_rot[:3, :2] = grasp_rot6d.reshape(3, 2)
    grasp_rot[:3, 0] = grasp_rot[:3, 0] / np.linalg.norm(grasp_rot[:3, 0])

    # make grasp_rot[:3, 1] orthogonal to grasp_rot[:3, 0]
    grasp_rot[:3, 1] = (
        grasp_rot[:3, 1] - np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) * grasp_rot[:3, 0]
    )
    grasp_rot[:3, 1] = grasp_rot[:3, 1] / np.linalg.norm(grasp_rot[:3, 1])
    assert (
        np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) < 1e-3
    ), f"Expected dot product < 1e-3, got {np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1])}"
    grasp_rot[:3, 2] = np.cross(grasp_rot[:3, 0], grasp_rot[:3, 1])

    grasp_transform = np.eye(4)  # X_Oy_H
    grasp_transform[:3, :3] = grasp_rot
    grasp_transform[:3, 3] = grasp_trans
    grasp_transform = X_W_Oy @ grasp_transform  # X_W_H = X_W_Oy @ X_Oy_H
    grasp_trans = grasp_transform[:3, 3]
    grasp_rot = grasp_transform[:3, :3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_pose = hand_config_to_pose(
        grasp_trans[None], grasp_rot[None], grasp_joints[None]
    ).to(device)
    hand_model_type = HandModelType.ALLEGRO_HAND
    grasp_orientations = np.zeros(
        (4, 3, 3)
    )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
    grasp_orientations[:, :, 2] = (
        grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    )
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)
    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.8)

    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=torch.from_numpy(grasp_orientations[None]).to(device),
    )
    new_hand_pose = hand_config_to_pose(
        grasp_trans[None],
        grasp_rot[None],
        optimized_joint_angle_targets.detach().cpu().numpy(),
    ).to(device)
    hand_model.set_parameters(new_hand_pose)
    hand_plotly_optimized = hand_model.get_plotly_data(
        i=0, opacity=0.3, color="lightgreen"
    )

    fig = go.Figure()
    breakpoint()
    fig.add_trace(
        go.Scatter3d(
            x=basis_points[:, 0],
            y=basis_points[:, 1],
            z=basis_points[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=bps,
                colorscale="rainbow",
                colorbar=dict(title="Basis points", orientation="h"),
            ),
            name="Basis points",
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name="Object",
            color="white",
            opacity=0.5,
        )
    )
    if point_cloud_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud_points[:, 0],
                y=point_cloud_points[:, 1],
                z=point_cloud_points[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="black"),
                name="Point cloud",
            )
        )
    fig.update_layout(
        title=dict(
            text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, Passed Eval: {passed_eval}"
        ),
    )
    VISUALIZE_HAND = True
    if VISUALIZE_HAND:
        for trace in hand_plotly:
            fig.add_trace(trace)
        for trace in hand_plotly_optimized:
            fig.add_trace(trace)
    # fig.write_html("/home/albert/research/nerf_grasping/dex_diffuser_debug.html")  # if headless
    fig.show()


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


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_pipeline: Pipeline,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    ckpt_path: str,
    optimize: bool,
) -> dict:
    BATCH_SIZE = cfg.eval_batch_size

    N_BASIS_PTS = 4096
    device = torch.device("cuda")
    dex_evaluator = DexEvaluator(in_grasp=3 + 6 + 16 + 12, in_bps=N_BASIS_PTS).to(
        device
    )
    dex_evaluator.eval()

    if pathlib.Path(ckpt_path).exists():
        dex_evaluator.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"{ckpt_path} does not exist")

    # Get BPS
    bps_values, _, _ = nerf_to_bps(
        nerf_pipeline=nerf_pipeline,
        lb_N=lb_N,
        ub_N=ub_N,
        X_N_By=X_N_By,
    )
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"

    bps_values_repeated = torch.from_numpy(bps_values).float().to(device)
    num_repeats = max(BATCH_SIZE, cfg.optimizer.num_grasps)
    bps_values_repeated = bps_values_repeated.unsqueeze(dim=0).repeat(num_repeats, 1)
    assert bps_values_repeated.shape == (
        num_repeats,
        N_BASIS_PTS,
    ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

    # Load grasp configs
    # TODO: Find a way to load a particular split of the grasp_data.
    init_grasp_config_dict = np.load(
        cfg.init_grasp_config_dict_path, allow_pickle=True
    ).item()

    num_grasps_in_dict = init_grasp_config_dict["trans"].shape[0]
    print(f"Found {num_grasps_in_dict} grasps in grasp config dict dataset")

    if (
        cfg.max_num_grasps_to_eval is not None
        and num_grasps_in_dict > cfg.max_num_grasps_to_eval
    ):
        print(f"Limiting to {cfg.max_num_grasps_to_eval} grasps from dataset.")
        # randomize the order, keep at most max_num_grasps_to_eval
        init_grasp_config_dict = {
            k: v[
                np.random.choice(
                    a=v.shape[0], size=cfg.max_num_grasps_to_eval, replace=False
                )
            ]
            for k, v in init_grasp_config_dict.items()
        }

    init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
        init_grasp_config_dict
    )
    print(f"Loaded {init_grasp_configs.batch_size} initial grasp configs.")

    # Put this here to ensure that the random seed is set before sampling random rotations.
    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    all_success_preds = []
    with torch.no_grad():
        # Sample random rotations
        N_SAMPLES = 1 + cfg.n_random_rotations_per_grasp
        new_grasp_configs_list = []
        for i in range(N_SAMPLES):
            new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                init_grasp_config_dict
            )
            if i != 0:
                random_rotate_transforms = (
                    sample_random_rotate_transforms_only_around_y(
                        new_grasp_configs.batch_size
                    )
                )
                new_grasp_configs.hand_config.set_wrist_pose(
                    random_rotate_transforms @ new_grasp_configs.hand_config.wrist_pose
                )
            new_grasp_configs_list.append(new_grasp_configs)

        new_grasp_config_dicts = defaultdict(list)
        for i in range(N_SAMPLES):
            config_dict = new_grasp_configs_list[i].as_dict()
            for k, v in config_dict.items():
                new_grasp_config_dicts[k].append(v)
        for k, v in new_grasp_config_dicts.items():
            new_grasp_config_dicts[k] = np.concatenate(v, axis=0)
        new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            new_grasp_config_dicts
        )
        assert new_grasp_configs.batch_size == init_grasp_configs.batch_size * N_SAMPLES

        # Filter grasps that are less IK feasible
        if cfg.filter_less_feasible_grasps:
            wrist_pose_matrix = new_grasp_configs.wrist_pose.matrix()
            x_dirs = wrist_pose_matrix[:, :, 0]
            z_dirs = wrist_pose_matrix[:, :, 2]

            fingers_forward_cos_theta = math.cos(
                math.radians(cfg.fingers_forward_theta_deg)
            )
            palm_upwards_cos_theta = math.cos(math.radians(cfg.palm_upwards_theta_deg))
            fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
            palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta
            new_grasp_configs = new_grasp_configs[fingers_forward & ~palm_upwards]
            print(
                f"Filtered less feasible grasps. New batch size: {new_grasp_configs.batch_size}"
            )

        # Evaluate grasp metric and collisions
        n_batches = math.ceil(new_grasp_configs.batch_size / BATCH_SIZE)
        for batch_i in tqdm(
            range(n_batches), desc=f"Evaling grasp metric with batch_size={BATCH_SIZE}"
        ):
            start_idx = batch_i * BATCH_SIZE
            end_idx = np.clip(
                (batch_i + 1) * BATCH_SIZE,
                a_min=None,
                a_max=new_grasp_configs.batch_size,
            )
            this_batch_size = end_idx - start_idx

            temp_grasp_configs = new_grasp_configs[start_idx:end_idx].to(device=device)
            wrist_trans_array = temp_grasp_configs.wrist_pose.translation().float()
            wrist_rot_array = temp_grasp_configs.wrist_pose.rotation().matrix().float()
            joint_angles_array = temp_grasp_configs.joint_angles.float()
            grasp_dirs_array = temp_grasp_configs.grasp_dirs.float()
            N_FINGERS = 4
            assert wrist_trans_array.shape == (this_batch_size, 3)
            assert wrist_rot_array.shape == (this_batch_size, 3, 3)
            assert joint_angles_array.shape == (this_batch_size, 16)
            assert grasp_dirs_array.shape == (this_batch_size, N_FINGERS, 3)
            g_O = torch.cat(
                [
                    wrist_trans_array,
                    wrist_rot_array[:, :, :2].reshape(this_batch_size, 6),
                    joint_angles_array,
                    grasp_dirs_array.reshape(this_batch_size, 12),
                ],
                dim=1,
            ).to(device=device)
            assert g_O.shape == (this_batch_size, 3 + 6 + 16 + 12)

            f_O = bps_values_repeated[:this_batch_size]
            assert f_O.shape == (this_batch_size, N_BASIS_PTS)

            success_preds = (
                dex_evaluator(f_O=f_O, g_O=g_O)[:, -1].detach().cpu().numpy()
            )
            assert success_preds.shape == (
                this_batch_size,
            ), f"success_preds.shape = {success_preds.shape}, expected ({this_batch_size},)"
            all_success_preds.append(success_preds)

        # Aggregate
        all_success_preds = np.concatenate(all_success_preds)
        assert all_success_preds.shape == (new_grasp_configs.batch_size,)

        # Sort by success_preds
        new_all_success_preds = all_success_preds
        ordered_idxs_best_first = np.argsort(new_all_success_preds)[::-1].copy()

        new_grasp_configs = new_grasp_configs[ordered_idxs_best_first]
        sorted_success_preds = new_all_success_preds[ordered_idxs_best_first][
            : cfg.optimizer.num_grasps
        ]

    init_grasp_configs = new_grasp_configs[: cfg.optimizer.num_grasps]

    if optimize:
        print(f"Optimizing {cfg.optimizer.num_grasps} grasps with random sampling")
        initial_losses_np = 1 - sorted_success_preds

        from nerf_grasping.ablation_optimizer import RandomSamplingOptimizer

        wrist_trans_array = init_grasp_configs.wrist_pose.translation().float()
        wrist_rot_array = init_grasp_configs.wrist_pose.rotation().matrix().float()
        joint_angles_array = init_grasp_configs.joint_angles.float()
        grasp_dirs_array = init_grasp_configs.grasp_dirs.float()
        N_FINGERS = 4
        assert wrist_trans_array.shape == (cfg.optimizer.num_grasps, 3)
        assert wrist_rot_array.shape == (cfg.optimizer.num_grasps, 3, 3)
        assert joint_angles_array.shape == (cfg.optimizer.num_grasps, 16)
        assert grasp_dirs_array.shape == (cfg.optimizer.num_grasps, N_FINGERS, 3)
        g_O = torch.cat(
            [
                wrist_trans_array,
                wrist_rot_array[:, :, :2].reshape(cfg.optimizer.num_grasps, 6),
                joint_angles_array,
                grasp_dirs_array.reshape(cfg.optimizer.num_grasps, 12),
            ],
            dim=1,
        ).to(device=device)
        assert g_O.shape == (cfg.optimizer.num_grasps, 3 + 6 + 16 + 12)

        random_sampling_optimizer = RandomSamplingOptimizer(
            dex_evaluator=dex_evaluator,
            bps=bps_values_repeated[: cfg.optimizer.num_grasps],
            init_grasps=g_O,
        )
        N_STEPS = cfg.optimizer.num_steps

        if N_STEPS > 0:
            for i in range(N_STEPS):
                losses = random_sampling_optimizer.step()
            losses_np = losses.detach().cpu().numpy()
        else:
            losses_np = initial_losses_np

        diff_losses = losses_np - initial_losses_np

        import sys

        print(
            f"Init Losses:  {[f'{x:.4f}' for x in initial_losses_np.tolist()]}",
            file=sys.stderr,
        )
        print(
            f"Final Losses: {[f'{x:.4f}' for x in losses_np.tolist()]}", file=sys.stderr
        )
        print(
            f"Diff Losses:  {[f'{x:.4f}' for x in diff_losses.tolist()]}",
            file=sys.stderr,
        )
        grasp_config = grasp_to_grasp_config(grasp=random_sampling_optimizer.grasps)
        grasp_config_dict = grasp_config.as_dict()
        grasp_config_dict["loss"] = losses_np
    else:
        print(f"Skipping optimization of grasps")
        grasp_config_dict = init_grasp_configs.as_dict()
        grasp_config_dict["loss"] = 1 - sorted_success_preds

    print(f"Saving final grasp config dict to {cfg.output_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cfg.output_path), grasp_config_dict, allow_pickle=True)

    if wandb.run is not None:
        wandb.finish()
    return grasp_config_dict


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    bps_evaluator_ckpt_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_DexEvaluator_models/ckpt_yp920sn0_final.pth"
    )
    nerfdata_path: Optional[pathlib.Path] = None
    nerfcheckpoint_path: Optional[pathlib.Path] = None
    num_grasps: int = 32
    max_num_iterations: int = 400
    overwrite: bool = False

    optimize: bool = False
    classifier_config_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_FINAL_LABELED_GRASPS_NOISE_AND_NONOISE_cnn-3d-xyz-global-cnn-cropped_CONTINUE/config.yaml"
    )
    init_grasp_config_dict_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-06-03_FINAL_INFERENCE_GRASPS/good_nonoise_one_per_object/grasps.npy"
    )
    optimizer_type: Literal["sgd", "cem", "random-sampling"] = "random-sampling"
    num_steps: int = 50
    n_random_rotations_per_grasp: int = 0
    eval_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.nerfdata_path is not None and self.nerfcheckpoint_path is None:
            assert self.nerfdata_path.exists(), f"{self.nerfdata_path} does not exist"
            assert (
                self.nerfdata_path / "transforms.json"
            ).exists(), f"{self.nerfdata_path / 'transforms.json'} does not exist"
            assert (
                self.nerfdata_path / "images"
            ).exists(), f"{self.nerfdata_path / 'images'} does not exist"
        elif self.nerfdata_path is None and self.nerfcheckpoint_path is not None:
            assert (
                self.nerfcheckpoint_path.exists()
            ), f"{self.nerfcheckpoint_path} does not exist"
            assert (
                self.nerfcheckpoint_path.suffix == ".yml"
            ), f"{self.nerfcheckpoint_path} does not have a .yml suffix"
        else:
            raise ValueError(
                "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
            )


def run_ablation_sim_eval(args: CommandlineArgs) -> None:
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

    # Get object name
    if args.nerfdata_path is not None:
        object_name = args.nerfdata_path.name
    elif args.nerfcheckpoint_path is not None:
        object_name = args.nerfcheckpoint_path.parents[2].name
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
        )
    print(f"object_name = {object_name}")

    # Prepare output folder
    args.output_folder.mkdir(exist_ok=True, parents=True)
    output_file = args.output_folder / f"{object_name}.npy"
    if output_file.exists():
        if not args.overwrite:
            print(f"{output_file} already exists, skipping")
            return

        print(f"{output_file} already exists, overwriting")

    # Prepare nerf model
    if args.nerfdata_path is not None:
        start_time = time.time()
        nerf_checkpoints_folder = args.output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerfs_return_trainer.train_nerf(
            args=train_nerfs_return_trainer.Args(
                nerfdata_folder=args.nerfdata_path,
                nerfcheckpoints_folder=nerf_checkpoints_folder,
                max_num_iterations=args.max_num_iterations,
            )
        )
        nerf_pipeline = nerf_trainer.pipeline
        nerf_model = nerf_trainer.pipeline.model
        nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
        end_time = time.time()
        print("@" * 80)
        print(f"Time to train_nerf: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    elif args.nerfcheckpoint_path is not None:
        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(
            args.nerfcheckpoint_path, test_mode="test"
        )  # Need this for point cloud
        nerf_model = nerf_pipeline.model
        nerf_config = args.nerfcheckpoint_path
        end_time = time.time()
        print("@" * 80)
        print(f"Time to load_nerf_pipeline: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
        )
    args.nerf_config = nerf_config

    # Compute centroid
    nerf_centroid_N = compute_centroid_from_nerf(
        nerf_model.field,
        lb=np.array([-0.2, 0.0, -0.2]),
        ub=np.array([0.2, 0.3, 0.2]),
        level=15,
        num_pts_x=100,
        num_pts_y=100,
        num_pts_z=100,
    )
    assert nerf_centroid_N.shape == (
        3,
    ), f"Expected shape (3,), got {nerf_centroid_N.shape}"
    obj_y = nerf_centroid_N[1]
    X_By_Oy = trimesh.transformations.translation_matrix([0, obj_y, 0])
    X_Oy_By = np.linalg.inv(X_By_Oy)
    X_N_By = np.eye(4)  # Same for sim nerfs
    X_N_Oy = X_N_By @ X_By_Oy

    # Get optimized grasps
    UNUSED_OUTPUT_PATH = pathlib.Path("UNUSED")

    print("\n" + "=" * 80)
    print("Step 5: Run ablation")
    print("=" * 80 + "\n")
    EVALUATOR_CKPT_PATH = args.bps_evaluator_ckpt_path

    # B frame is at base of object z up frame
    # By frame is at base of object y up frame
    from nerf_grasping import ablation_utils

    optimized_grasp_config_dict = ablation_utils.get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=False,  # Not used because causes issues with logging
            init_grasp_config_dict_path=args.init_grasp_config_dict_path,
            grasp_metric=GraspMetricConfig(
                nerf_checkpoint_path=nerf_config,
                classifier_config_path=args.classifier_config_path,
                X_N_Oy=X_N_Oy,
            ),  # This is not used
            optimizer=RandomSamplingConfig(
                num_grasps=args.num_grasps,
                num_steps=args.num_steps,
            ),  # This optimizer is not used, but the num_grasps is used and the num_steps is used
            output_path=UNUSED_OUTPUT_PATH,
            random_seed=0,
            n_random_rotations_per_grasp=0,
            eval_batch_size=args.eval_batch_size,
            wandb=None,
            filter_less_feasible_grasps=False,  # Do not filter for sim
        ),
        nerf_pipeline=nerf_pipeline,
        lb_N=np.array([-0.2, 0.0, -0.2]),
        ub_N=np.array([0.2, 0.3, 0.2]),
        X_N_By=X_N_By,
        ckpt_path=EVALUATOR_CKPT_PATH,
        optimize=args.optimize,
    )

    grasp_config_dict = optimized_grasp_config_dict

    print(f"Saving grasp_config_dict to {output_file}")
    np.save(output_file, grasp_config_dict)


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    run_ablation_sim_eval(args)


if __name__ == "__main__":
    main()
