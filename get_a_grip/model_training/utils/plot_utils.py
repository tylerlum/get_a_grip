from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.utils.point_utils import transform_point, transform_points


def get_scene_dict() -> Dict[str, Any]:
    return dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    )


def get_yup_camera() -> Dict[str, Any]:
    return dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.0, y=1.0, z=0.0),
    )


def get_zup_camera() -> Dict[str, Any]:
    return dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.0, y=0.0, z=1.0),
    )


def plot_mesh(mesh: trimesh.Trimesh, color="lightpink") -> go.Figure:
    vertices = mesh.vertices
    faces = mesh.faces

    # Create the mesh3d trace
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5,
        name="Mesh",
    )

    # Create the layout
    layout = go.Layout(
        scene=get_scene_dict(),
        showlegend=True,
        title="Mesh",
    )

    # Create the figure
    fig = go.Figure(data=[mesh_plot], layout=layout)

    # Return the figure
    return fig


def plot_mesh_and_transforms(
    mesh: trimesh.Trimesh,
    transforms: List[np.ndarray],
    num_fingers: int,
    title: str = "Mesh and Transforms",
    highlight_idx: Optional[int] = None,
) -> go.Figure:
    assert len(transforms) == num_fingers, f"{len(transforms)} != {num_fingers}"

    # Add the scatter plot to a figure and display it
    fig = plot_mesh(mesh)
    for finger_idx in range(num_fingers):
        transform = transforms[finger_idx]
        length = 0.02
        origin = np.array([0, 0, 0])
        x_axis = np.array([length, 0, 0])
        y_axis = np.array([0, length, 0])
        z_axis = np.array([0, 0, length])

        new_origin = transform @ np.concatenate([origin, [1]])
        new_x_axis = transform @ np.concatenate([x_axis, [1]])
        new_y_axis = transform @ np.concatenate([y_axis, [1]])
        new_z_axis = transform @ np.concatenate([z_axis, [1]])
        x_plot = go.Scatter3d(
            x=[new_origin[0], new_x_axis[0]],
            y=[new_origin[1], new_x_axis[1]],
            z=[new_origin[2], new_x_axis[2]],
            mode="lines",
            marker=dict(
                size=8,
                color="orange" if finger_idx is highlight_idx else "red",
                colorscale="viridis",
            ),
            name=f"Finger {finger_idx} X Axis",
        )
        y_plot = go.Scatter3d(
            x=[new_origin[0], new_y_axis[0]],
            y=[new_origin[1], new_y_axis[1]],
            z=[new_origin[2], new_y_axis[2]],
            mode="lines",
            marker=dict(
                size=8,
                color="orange" if finger_idx is highlight_idx else "green",
                colorscale="viridis",
            ),
            name=f"Finger {finger_idx} Y Axis",
        )
        z_plot = go.Scatter3d(
            x=[new_origin[0], new_z_axis[0]],
            y=[new_origin[1], new_z_axis[1]],
            z=[new_origin[2], new_z_axis[2]],
            mode="lines",
            marker=dict(
                size=8,
                color="blue",
                colorscale="viridis",
            ),
            name=f"Finger {finger_idx} Z Axis",
        )

        fig.add_trace(x_plot)
        fig.add_trace(y_plot)
        fig.add_trace(z_plot)

    fig.update_layout(legend_orientation="h", title_text=title, scene=get_scene_dict())
    return fig


def plot_mesh_and_query_points(
    mesh: trimesh.Trimesh,
    all_query_points: np.ndarray,
    all_query_points_colors: np.ndarray,
    num_fingers: int,
    title: str = "Query Points",
) -> go.Figure:
    num_pts = all_query_points.shape[1]
    assert all_query_points.shape == (
        num_fingers,
        num_pts,
        3,
    ), f"{all_query_points.shape}"
    assert all_query_points_colors.shape == (
        num_fingers,
        num_pts,
    ), f"{all_query_points_colors.shape}"

    fig = plot_mesh(mesh)
    for finger_idx in range(num_fingers):
        query_points = all_query_points[finger_idx]
        query_points_colors = all_query_points_colors[finger_idx]
        assert query_points.shape == (num_pts, 3)
        assert query_points_colors.shape == (num_pts,)

        query_point_plot = go.Scatter3d(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=query_points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=query_points_colors,
                colorscale="viridis",
                colorbar=dict(title="Density Scale") if finger_idx == 0 else {},
            ),
            name=f"Query Point Densities Finger {finger_idx}",
        )
        fig.add_trace(query_point_plot)

    fig.update_layout(
        legend_orientation="h",
        scene=dict(
            # xaxis=dict(nticks=4, range=[-0.3, 0.3]),
            # yaxis=dict(nticks=4, range=[-0.3, 0.3]),
            # zaxis=dict(nticks=4, range=[-0.3, 0.3]),
        ),
        title_text=title,
    )  # Avoid overlapping legend
    fig.update_layout(scene_aspectmode="data")
    return fig


def plot_mesh_and_high_density_points(
    mesh: trimesh.Trimesh,
    all_query_points: np.ndarray,
    all_query_points_colors: np.ndarray,
    density_threshold: float,
) -> go.Figure:
    num_pts = all_query_points.shape[0]
    assert all_query_points.shape == (num_pts, 3), f"{all_query_points.shape}"
    assert all_query_points_colors.shape == (
        num_pts,
    ), f"{all_query_points_colors.shape}"

    fig = plot_mesh(mesh)

    # Filter
    query_points = all_query_points[all_query_points_colors > density_threshold]
    query_points_colors = all_query_points_colors[
        all_query_points_colors > density_threshold
    ]

    query_point_plot = go.Scatter3d(
        x=query_points[:, 0],
        y=query_points[:, 1],
        z=query_points[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=query_points_colors,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
        name="Query Point Densities",
    )
    fig.add_trace(query_point_plot)

    fig.update_layout(
        legend_orientation="h",
    )  # Avoid overlapping legend
    fig.update_layout(scene_aspectmode="data")
    return fig


def plot_grasp_and_mesh_and_more(
    fig: Optional[go.Figure] = None,
    grasp: Optional[torch.Tensor] = None,
    X_N_Oy: Optional[np.ndarray] = None,
    visualize_target_hand: bool = False,
    visualize_pre_hand: bool = False,
    visualize_Oy: bool = False,
    mesh: Optional[trimesh.Trimesh] = None,
    basis_points: Optional[np.ndarray] = None,
    bps: Optional[np.ndarray] = None,
    raw_point_cloud_points: Optional[np.ndarray] = None,
    processed_point_cloud_points: Optional[np.ndarray] = None,
    nerf_global_grids_with_coords: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    nerf_is_z_up: bool = False,
) -> go.Figure:
    """Create a plot in NeRF (N) frame"""
    if fig is None:
        fig = go.Figure()

    if grasp is not None:
        # Extract data from grasp
        assert grasp.shape == (
            3 + 6 + 16 + 4 * 3,
        ), f"Expected shape (3 + 6 + 16 + 4 * 3), got {grasp.shape}"
        if X_N_Oy is not None:
            assert X_N_Oy.shape == (4, 4), f"Expected shape (4, 4), got {X_N_Oy.shape}"

        grasp_config = AllegroGraspConfig.from_grasp(grasp[None])
        hand_model = grasp_config.hand_config.as_hand_model()
        hand_plotly = hand_model.get_plotly_data(
            i=0, opacity=0.8, color="lightblue", pose=X_N_Oy
        )
        for trace in hand_plotly:
            fig.add_trace(trace)

        if visualize_target_hand:
            target_hand_model = grasp_config.get_target_hand_config().as_hand_model()
            target_hand_plotly = target_hand_model.get_plotly_data(
                i=0, opacity=0.3, color="lightgreen", pose=X_N_Oy
            )
            for trace in target_hand_plotly:
                fig.add_trace(trace)
        if visualize_pre_hand:
            pre_hand_model = grasp_config.get_pre_hand_config().as_hand_model()
            pre_hand_plotly = pre_hand_model.get_plotly_data(
                i=0, opacity=0.3, color="lightgreen", pose=X_N_Oy
            )
            for trace in pre_hand_plotly:
                fig.add_trace(trace)

    if visualize_Oy:
        assert X_N_Oy is not None, "X_N_Oy must be provided to visualize Oy"
        assert X_N_Oy.shape == (4, 4), f"Expected shape (4, 4), got {X_N_Oy.shape}"

        add_transform_plot(
            fig=fig,
            X_A_B=X_N_Oy,
            A="N",
            B="Oy",
        )

    if mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                name="Object",
                color="lightpink",
                opacity=0.5,
            )
        )

    if basis_points is not None and bps is not None:
        assert basis_points.shape == (
            4096,
            3,
        ), f"Expected shape (4096, 3), got {basis_points.shape}"
        assert bps.shape == (4096,), f"Expected shape (4096,), got {bps.shape}"

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

    if raw_point_cloud_points is not None:
        N_raw_pts = raw_point_cloud_points.shape[0]
        assert raw_point_cloud_points.shape == (
            N_raw_pts,
            3,
        ), f"Expected shape ({N_raw_pts}, 3), got {raw_point_cloud_points.shape}"

        fig.add_trace(
            go.Scatter3d(
                x=raw_point_cloud_points[:, 0],
                y=raw_point_cloud_points[:, 1],
                z=raw_point_cloud_points[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="blue"),
                name="Raw point cloud",
            )
        )

    if processed_point_cloud_points is not None:
        N_processed_pts = processed_point_cloud_points.shape[0]
        assert (
            processed_point_cloud_points.shape
            == (
                N_processed_pts,
                3,
            )
        ), f"Expected shape ({N_processed_pts}, 3), got {processed_point_cloud_points.shape}"

        fig.add_trace(
            go.Scatter3d(
                x=processed_point_cloud_points[:, 0],
                y=processed_point_cloud_points[:, 1],
                z=processed_point_cloud_points[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="black"),
                name="Processed point cloud",
            )
        )

    if nerf_global_grids_with_coords is not None:
        assert (
            nerf_global_grids_with_coords.shape
            == (
                3 + 1,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape (3 + 1, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_global_grids_with_coords.shape}"
        assert X_N_Oy is not None, "X_N_Oy must be provided to plot high density points"

        query_points_colors = nerf_global_grids_with_coords[0].reshape(-1)
        query_points = nerf_global_grids_with_coords[1:].reshape(3, -1).T

        query_points = query_points[query_points_colors > 15]
        query_points_colors = query_points_colors[query_points_colors > 15]

        query_points = transform_points(T=X_N_Oy, points=query_points)

        query_point_plot = go.Scatter3d(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=query_points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=query_points_colors,
                colorscale="viridis",
                colorbar=dict(title="Density Scale"),
            ),
            name="Nerf High Density Points",
        )
        fig.add_trace(query_point_plot)

    if title is not None:
        fig.update_layout(
            title=dict(
                text=title,
            ),
        )
    scene = get_scene_dict()
    scene_camera = get_zup_camera() if nerf_is_z_up else get_yup_camera()
    fig.update_layout(scene=scene, scene_camera=scene_camera)

    return fig


def add_transform_plot(
    fig: go.Figure,
    X_A_B: np.ndarray,
    A: str,
    B: str,
    length: float = 0.1,
) -> None:
    import plotly.graph_objects as go

    assert X_A_B.shape == (4, 4), f"{X_A_B.shape}"

    origin = np.array([0, 0, 0])
    x_end = np.array([length, 0, 0])
    y_end = np.array([0, length, 0])
    z_end = np.array([0, 0, length])

    transformed_origin = transform_point(T=X_A_B, point=origin)
    transformed_x_end = transform_point(T=X_A_B, point=x_end)
    transformed_y_end = transform_point(T=X_A_B, point=y_end)
    transformed_z_end = transform_point(T=X_A_B, point=z_end)

    fig.add_trace(
        go.Scatter3d(
            x=[transformed_origin[0], transformed_x_end[0]],
            y=[transformed_origin[1], transformed_x_end[1]],
            z=[transformed_origin[2], transformed_x_end[2]],
            mode="lines",
            line=dict(color="red"),
            name=f"X_{A}_{B}_x",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[transformed_origin[0], transformed_y_end[0]],
            y=[transformed_origin[1], transformed_y_end[1]],
            z=[transformed_origin[2], transformed_y_end[2]],
            mode="lines",
            line=dict(color="green"),
            name=f"X_{A}_{B}_y",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[transformed_origin[0], transformed_z_end[0]],
            y=[transformed_origin[1], transformed_z_end[1]],
            z=[transformed_origin[2], transformed_z_end[2]],
            mode="lines",
            line=dict(color="blue"),
            name=f"X_{A}_{B}_z",
        )
    )
    return
