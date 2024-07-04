from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import trimesh


def get_scene_dict() -> Dict[str, Any]:
    return dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
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

    fig.update_layout(legend_orientation="h", title_text=title)
    return fig


def plot_mesh_and_query_points(
    mesh: trimesh.Trimesh,
    query_points: np.ndarray,
    query_points_colors: np.ndarray,
    num_fingers: int,
    title: str = "Query Points",
) -> go.Figure:
    num_pts = query_points.shape[1]
    assert query_points.shape == (
        num_fingers,
        num_pts,
        3,
    ), f"{query_points.shape}"
    assert query_points_colors.shape == (
        num_fingers,
        num_pts,
    ), f"{query_points_colors.shape}"

    fig = plot_mesh(mesh)
    for finger_idx in range(num_fingers):
        query_points = query_points[finger_idx]
        query_points_colors = query_points_colors[finger_idx]
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
    query_points: np.ndarray,
    query_points_colors: np.ndarray,
    density_threshold: float,
) -> go.Figure:
    num_pts = query_points.shape[0]
    assert query_points.shape == (num_pts, 3), f"{query_points.shape}"
    assert query_points_colors.shape == (num_pts,), f"{query_points_colors.shape}"

    fig = plot_mesh(mesh)

    # Filter
    query_points = query_points[query_points_colors > density_threshold]
    query_points_colors = query_points_colors[query_points_colors > density_threshold]

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
