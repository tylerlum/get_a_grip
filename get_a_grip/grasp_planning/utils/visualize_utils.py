from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh

from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from get_a_grip.dataset_generation.utils.pose_conversion import hand_config_to_pose


def plot_point_cloud_and_bps_and_mesh_and_grasp(
    grasp: torch.Tensor,
    X_W_Oy: np.ndarray,
    basis_points: np.ndarray,
    bps: np.ndarray,
    mesh: trimesh.Trimesh,
    point_cloud_points: Optional[np.ndarray],
    GRASP_IDX: int,
    object_code: str,
    y_PGS: int,
) -> None:
    # TODO: Clean up

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
    grasp_orientations = np.zeros(
        (4, 3, 3)
    )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
    grasp_orientations[:, :, 2] = (
        grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    )
    hand_model = HandModel(device=device)
    hand_model.set_parameters(hand_pose)
    assert hand_model.hand_pose is not None
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
            text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, y_PGS: {y_PGS}"
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


def plot_mesh_and_grasp(
    fig: go.Figure,
    mesh_Oy: trimesh.Trimesh,
    grasp: torch.Tensor,
):
    # TODO: clean and move
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
    grasp_orientations = np.zeros(
        (4, 3, 3)
    )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
    grasp_orientations[:, :, 2] = (
        grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    )
    hand_model = HandModel(device=device)
    hand_model.set_parameters(hand_pose)
    assert hand_model.hand_pose is not None
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


def plot_nerf_densities(
    fig: go.Figure,
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


def plot_mesh(fig: go.Figure, mesh: trimesh.Trimesh) -> None:
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
