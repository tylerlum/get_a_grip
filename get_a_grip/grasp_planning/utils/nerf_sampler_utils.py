from __future__ import annotations
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
from nerf_grasping.other_utils import get_points_in_grid
from typing import List, Tuple
from nerf_grasping.ablation_utils import (
    nerf_to_bps,
    visualize_point_cloud_and_bps_and_grasp,
)
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
from nerf_grasping.dexdiffuser_utils import (
    rot6d_to_matrix,
    compute_grasp_orientations,
)

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
    NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
    compute_fingertip_dirs,
)

import open3d as o3d


def DEBUG_plot_grasp(
    fig,
    grasp: torch.Tensor,
    mesh_Oy: trimesh.Trimesh,
):
    assert grasp.shape == (37,), f"grasp.shape: {grasp.shape}"
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

    fig.add_trace(
        go.Mesh3d(
            x=mesh_Oy.vertices[:, 0],
            y=mesh_Oy.vertices[:, 1],
            z=mesh_Oy.vertices[:, 2],
            i=mesh_Oy.faces[:, 0],
            j=mesh_Oy.faces[:, 1],
            k=mesh_Oy.faces[:, 2],
            name="Object",
            color="white",
            opacity=0.5,
        )
    )
    fig.update_layout(
        title=dict(
            text="Grasp Visualization",
        ),
    )
    VISUALIZE_HAND = True
    if VISUALIZE_HAND:
        for trace in hand_plotly:
            fig.add_trace(trace)
        for trace in hand_plotly_optimized:
            fig.add_trace(trace)
    # fig.write_html("/home/albert/research/nerf_grasping/dex_diffuser_debug.html")  # if headless


def DEBUG_plot(
    fig,
    densities: torch.Tensor,
    query_points: torch.Tensor,
    name: str,
    opacity: float = 1.0,
):
    B = densities.shape[0]
    assert densities.shape == (B,), f"densities.shape: {densities.shape}"
    assert query_points.shape == (B, 3), f"query_points.shape: {query_points.shape}"
    fig.add_trace(
        go.Scatter3d(
            x=query_points[:, 0].detach().cpu().numpy(),
            y=query_points[:, 1].detach().cpu().numpy(),
            z=query_points[:, 2].detach().cpu().numpy(),
            mode="markers",
            marker=dict(
                size=3,
                opacity=opacity,
                color=densities.detach().cpu().numpy(),
                colorscale="Viridis",
                colorbar=dict(title="Density"),
            ),
            name=name,
        )
    )


def DEBUG_plot_mesh(fig, mesh):
    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name="object",
            opacity=0.5,
        )
    )


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_model: Model,
    X_N_Oy: np.ndarray,
    ckpt_path: str,
    return_exactly_requested_num_grasps: bool = True,
    sample_grasps_multiplier: int = 10,
) -> dict:
    ckpt_path = pathlib.Path(ckpt_path)

    NUM_GRASPS = cfg.optimizer.num_grasps

    config = Config(
        training=TrainingConfig(
            log_path=ckpt_path.parent,
        ),
        use_nerf_sampler=True,
    )
    runner = Diffusion(config, load_multigpu_ckpt=True)
    runner.load_checkpoint(config, name=ckpt_path.stem)
    runner.model.HACK_MODE_FOR_PERFORMANCE = True  # Big hack to speed up from sampling wasting dumb compute
    device = runner.device

    # Get nerf densities global cropped
    # THIS IS WRONG! IT ONLY WORKS IF X_N_Oy HAS NO ROTATION COMPONENT
    # THIS IS BECAUSE IF WE ROTATE FIRST VS. AFTER SAMPLING THE GRID, IT IS NOT THE SAME
    # lb_N = transform_point(T=X_N_Oy, point=lb_Oy)
    # ub_N = transform_point(T=X_N_Oy, point=ub_Oy)
    # nerf_densities_global, query_points_N = get_densities_in_grid(
    #     field=nerf_model.field,
    #     num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
    #     num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
    #     num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
    #     lb=lb_N,
    #     ub=ub_N,
    # )

    # Must sample points in grid in Oy before transforming to Nz
    query_points_Oy = get_points_in_grid(
        lb=lb_Oy,
        ub=ub_Oy,
        num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
        num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
        num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
    )
    assert query_points_Oy.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    )
    query_points_Nz = transform_points(
        T=X_N_Oy,
        points=query_points_Oy.reshape(-1, 3),
    ).reshape(
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    )
    nerf_densities_global = (
        get_density(
            field=nerf_model.field,
            positions=torch.from_numpy(query_points_Nz).cuda(),
        )[0]
        .squeeze(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .reshape(
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    )

    assert nerf_densities_global.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
    ), f"Expected shape ({NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global.shape}"
    assert query_points_Nz.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    ), f"Expected shape ({NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}, 3), got {query_points_N.shape}"

    # Crop
    start_x = (NERF_DENSITIES_GLOBAL_NUM_X - NERF_DENSITIES_GLOBAL_NUM_X_CROPPED) // 2
    start_y = (NERF_DENSITIES_GLOBAL_NUM_Y - NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED) // 2
    start_z = (NERF_DENSITIES_GLOBAL_NUM_Z - NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED) // 2
    end_x = start_x + NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
    end_y = start_y + NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
    end_z = start_z + NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
    nerf_densities_global_cropped = nerf_densities_global[
        start_x:end_x,
        start_y:end_y,
        start_z:end_z,
    ]
    query_points_Nz_cropped = query_points_Nz[
        start_x:end_x,
        start_y:end_y,
        start_z:end_z,
    ]
    X_Oy_N = np.linalg.inv(X_N_Oy)
    query_points_Oy_cropped = transform_points(
        T=X_Oy_N,
        points=query_points_Nz_cropped.reshape(-1, 3),
    ).reshape(
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        3,
    )

    CHECK = False
    if CHECK:
        from nerf_grasping.dexdiffuser.grasp_nerf_dataset import (
            coords_global_cropped,
        )

        query_points_Oy_cropped_2 = coords_global_cropped(
            device=torch.device("cpu"), dtype=torch.float, batch_size=1
        ).squeeze(0)
        assert query_points_Oy_cropped_2.shape == (
            3,
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        )
        query_points_Oy_cropped_2 = query_points_Oy_cropped_2.permute(
            1, 2, 3, 0
        ).numpy()
        assert query_points_Oy_cropped_2.shape == (
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
            3,
        )
        diff = query_points_Oy_cropped - query_points_Oy_cropped_2
        assert np.max(np.abs(diff)) < 1e-6, f"diff = {np.max(np.abs(diff))}"

    nerf_densities_global_with_coords = np.concatenate(
        [nerf_densities_global_cropped[..., None], query_points_Oy_cropped], axis=-1
    )
    assert nerf_densities_global_with_coords.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        4,
    ), f"Expected shape ({NERF_DENSITIES_GLOBAL_NUM_X_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED}, 4), got {nerf_densities_global_with_coords.shape}"
    nerf_densities_global_with_coords = nerf_densities_global_with_coords.transpose(
        3, 0, 1, 2
    )
    assert nerf_densities_global_with_coords.shape == (
        4,
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
    ), f"Expected shape (4, {NERF_DENSITIES_GLOBAL_NUM_X_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED}), got {nerf_densities_global_with_coords.shape}"

    nerf_densities_global_with_coords = (
        torch.from_numpy(nerf_densities_global_with_coords).float().to(device)
    )

    # We sample more grasps than needed to account for filtering
    NUM_GRASP_SAMPLES = sample_grasps_multiplier * NUM_GRASPS
    MAX_BATCH_SIZE = 400
    if NUM_GRASP_SAMPLES > MAX_BATCH_SIZE:
        nerf_densities_global_with_coords_repeated = (
            nerf_densities_global_with_coords.unsqueeze(0).repeat(
                MAX_BATCH_SIZE, 1, 1, 1, 1
            )
        )

        assert nerf_densities_global_with_coords_repeated.shape == (
            MAX_BATCH_SIZE,
            4,
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        ), f"Expected shape ({MAX_BATCH_SIZE}, 4, {NERF_DENSITIES_GLOBAL_NUM_X_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED}), got {nerf_densities_global_with_coords_repeated.shape}"

        n_batches = int(math.ceil(NUM_GRASP_SAMPLES / MAX_BATCH_SIZE))
        x_list = []
        for i in tqdm(range(n_batches), desc="Sampling grasps in batches"):
            start_idx = i * MAX_BATCH_SIZE
            end_idx = min((i + 1) * MAX_BATCH_SIZE, NUM_GRASP_SAMPLES)
            num_samples = end_idx - start_idx

            xT = torch.randn(num_samples, config.data.grasp_dim, device=runner.device)
            x = runner.sample(
                xT=xT,
                cond=nerf_densities_global_with_coords_repeated[:num_samples],
            ).cpu()
            x_list.append(x)
        x = torch.cat(x_list, dim=0)
    else:
        nerf_densities_global_with_coords_repeated = (
            nerf_densities_global_with_coords.unsqueeze(0).repeat(
                NUM_GRASP_SAMPLES, 1, 1, 1, 1
            )
        )

        assert nerf_densities_global_with_coords_repeated.shape == (
            NUM_GRASP_SAMPLES,
            4,
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        ), f"Expected shape ({NUM_GRASP_SAMPLES}, 4, {NERF_DENSITIES_GLOBAL_NUM_X_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED}, {NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED}), got {nerf_densities_global_with_coords_repeated.shape}"

        # Sample grasps
        xT = torch.randn(NUM_GRASP_SAMPLES, config.data.grasp_dim, device=runner.device)
        x = runner.sample(xT=xT, cond=nerf_densities_global_with_coords_repeated)

    PLOT = False
    if PLOT:
        fig = go.Figure()
        nerf_densities_global_flattened = nerf_densities_global_cropped.reshape(-1)
        query_points_global_N_flattened = query_points_Oy_cropped.reshape(-1, 3)
        THRESHOLD = 15
        DEBUG_plot(
            fig=fig,
            densities=torch.from_numpy(
                nerf_densities_global_flattened[
                    nerf_densities_global_flattened > THRESHOLD
                ]
            ),
            query_points=torch.from_numpy(
                query_points_global_N_flattened[
                    nerf_densities_global_flattened > THRESHOLD
                ]
            ),
            name="global",
        )

        X_Oy_N = np.linalg.inv(X_N_Oy)
        mesh_N = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_Oy = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_Oy.apply_transform(X_Oy_N)

        GRASP_IDX = 0
        breakpoint()
        DEBUG_plot_grasp(
            fig=fig,
            grasp=x[GRASP_IDX],
            mesh_Oy=mesh_Oy,
        )
        fig.show()

    # grasp to AllegroGraspConfig
    # TODO: make the numpy torch conversions less bad
    N_FINGERS = 4
    assert x.shape == (
        NUM_GRASP_SAMPLES,
        config.data.grasp_dim,
    ), f"Expected shape ({NUM_GRASP_SAMPLES}, {config.data.grasp_dim}), got {x.shape}"
    trans = x[:, :3].detach().cpu().numpy()
    rot6d = x[:, 3:9].detach().cpu().numpy()
    joint_angles = x[:, 9:25].detach().cpu().numpy()
    grasp_dirs = (
        x[:, 25:37].reshape(NUM_GRASP_SAMPLES, N_FINGERS, 3).detach().cpu().numpy()
    )

    rot = rot6d_to_matrix(rot6d)

    wrist_pose_matrix = (
        torch.eye(4, device=device).unsqueeze(0).repeat(NUM_GRASP_SAMPLES, 1, 1).float()
    )
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

    assert wrist_pose.lshape == (NUM_GRASP_SAMPLES,)

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

    if cfg.filter_less_feasible_grasps:
        wrist_pose_matrix = grasp_configs.wrist_pose.matrix()
        x_dirs = wrist_pose_matrix[:, :, 0]
        z_dirs = wrist_pose_matrix[:, :, 2]

        fingers_forward_cos_theta = math.cos(math.radians(cfg.fingers_forward_theta_deg))
        palm_upwards_cos_theta = math.cos(math.radians(cfg.palm_upwards_theta_deg))
        fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
        palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta
        grasp_configs = grasp_configs[fingers_forward & ~palm_upwards]
        print(f"Filtered less feasible grasps. New batch size: {grasp_configs.batch_size}")

    if len(grasp_configs) < NUM_GRASPS:
        print(
            f"WARNING: After filtering, only {len(grasp_configs)} grasps remain, less than the requested {NUM_GRASPS} grasps"
        )

    if return_exactly_requested_num_grasps:
        grasp_configs = grasp_configs[:NUM_GRASPS]
    print(f"Returning {grasp_configs.batch_size} grasps")

    grasp_config_dicts = grasp_configs.as_dict()
    grasp_config_dicts["loss"] = np.linspace(
        0, 0.001, len(grasp_configs)
    )  # HACK: Currently don't have a loss, but need something here to sort

    return grasp_config_dicts
