from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots

from get_a_grip.model_training.utils.plot_utils import (
    get_scene_dict,
    get_yup_camera,
    get_zup_camera,
)


def plot_mesh(
    mesh: trimesh.Trimesh,
    color: str = "lightpink",
    is_z_up: bool = False,
    fig: Optional[go.Figure] = None,
    name: str = "Mesh",
    title: Optional[str] = None,
    opacity: float = 0.5,
) -> go.Figure:
    vertices = mesh.vertices
    faces = mesh.faces

    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
    )

    layout = go.Layout(
        scene=get_scene_dict(),
        showlegend=True,
    )
    if title is not None:
        layout.title = title

    if fig is None:
        fig = go.Figure(data=[mesh_plot], layout=layout)
    else:
        fig.add_trace(mesh_plot)
        fig.update_layout(layout)

    fig.update_layout(scene_camera=get_zup_camera() if is_z_up else get_yup_camera())
    return fig


def plot_point_cloud(
    points: np.ndarray,
    colors: Optional[str] = None,
    size: int = 4,
    is_z_up: bool = False,
    fig: Optional[go.Figure] = None,
    name: str = "Point Cloud",
    title: Optional[str] = None,
) -> go.Figure:
    if colors is None:
        colors = points[:, 2] if is_z_up else points[:, 1]

    scatter_plot = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=colors,
            colorscale="viridis",
            colorbar=dict(title="Z" if is_z_up else "Y"),
        ),
        name=name,
    )

    layout = go.Layout(
        scene=get_scene_dict(),
        showlegend=True,
    )
    if title is not None:
        layout.title = title

    if fig is None:
        fig = go.Figure(data=[scatter_plot], layout=layout)
    else:
        fig.add_trace(scatter_plot)
        fig.update_layout(layout)

    fig.update_layout(scene_camera=get_zup_camera() if is_z_up else get_yup_camera())
    return fig


def make_fig_grid(
    figs: List[go.Figure],
    fig_titles: List[str],
    title: Optional[str] = None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> go.Figure:
    n_plots = len(figs)
    assert n_plots == len(fig_titles), f"{n_plots} != {len(fig_titles)}!"

    # Handle n_rows n_cols
    if n_rows is None and n_cols is None:
        n_rows = int(np.sqrt(n_plots))
        n_cols = int(np.ceil(n_plots / n_rows))
    elif n_rows is None:
        n_rows = int(np.ceil(n_plots / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(n_plots / n_rows))
    else:
        assert n_rows * n_cols >= n_plots, f"{n_rows} * {n_cols} < {n_plots}!"
    assert n_rows is not None and n_cols is not None, f"{n_rows} {n_cols}!"

    fig_grid = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=fig_titles + [""] * (n_rows * n_cols - n_plots),
        specs=[[{"type": "scene"}] * n_cols for _ in range(n_rows)],
        # specs=[[{"type": "mesh3d"}] * n_cols for _ in range(n_rows)],
    )

    for i, fig in enumerate(figs):
        row, col = divmod(i, n_cols)
        for data in fig.data:
            fig_grid.add_trace(
                data,
                row=row + 1,
                col=col + 1,
            )

    # plotly uses scene, scene2, scene3, ...
    new_scene_dict = {}
    for i, fig in enumerate(figs):
        scene_name = f"scene{i + 1}" if i != 0 else "scene"
        print(f"fig.layout.scene = {fig.layout.scene.to_plotly_json()}")
        new_scene_dict[scene_name] = {
            **fig.layout.scene.to_plotly_json(),
            "camera": fig.layout.scene.camera,
        }

    # Update layout
    fig_grid.update_layout(
        title=title if title is not None else "",
        showlegend=False,
        **new_scene_dict,
    )
    return fig_grid


def get_table(
    bounds: np.ndarray, is_z_up: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert bounds.shape == (2, 3), bounds.shape

    if is_z_up:
        min_object_z = bounds[0, 2]
        table_z = min_object_z

        table_pos = np.array([0, 0, table_z])
        table_normal = np.array([0, 0, 1])
        table_parallel = np.array([1, 0, 0])
    else:
        min_object_y = bounds[0, 1]
        table_y = min_object_y

        table_pos = np.array([0, table_y, 0])
        table_normal = np.array([0, 1, 0])
        table_parallel = np.array([1, 0, 0])

    assert table_pos.shape == table_normal.shape == table_parallel.shape == (3,)
    return table_pos, table_normal, table_parallel


def get_table_mesh(
    bounds: np.ndarray,
    width: float = 0.4,
    height: float = 0.4,
    is_z_up: bool = False,
) -> trimesh.Trimesh:
    W, H = width, height

    table_pos, table_normal, table_parallel = get_table(bounds=bounds, is_z_up=is_z_up)
    assert table_pos.shape == table_normal.shape == (3,)

    table_parallel_2 = np.cross(table_normal, table_parallel)
    corner1 = table_pos + W / 2 * table_parallel + H / 2 * table_parallel_2
    corner2 = table_pos + W / 2 * table_parallel - H / 2 * table_parallel_2
    corner3 = table_pos - W / 2 * table_parallel + H / 2 * table_parallel_2
    corner4 = table_pos - W / 2 * table_parallel - H / 2 * table_parallel_2

    x = np.array([corner1[0], corner2[0], corner3[0], corner4[0]])
    y = np.array([corner1[1], corner2[1], corner3[1], corner4[1]])
    z = np.array([corner1[2], corner2[2], corner3[2], corner4[2]])

    i = [0, 0, 1]
    j = [1, 2, 2]
    k = [2, 3, 3]

    table_mesh = trimesh.Trimesh(
        vertices=np.stack([x, y, z], axis=1), faces=np.stack([i, j, k], axis=1)
    )
    return table_mesh


def plot_table(
    bounds: np.ndarray,
    width: float = 0.4,
    height: float = 0.4,
    is_z_up: bool = False,
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    table_mesh = get_table_mesh(
        bounds=bounds, width=width, height=height, is_z_up=is_z_up
    )
    return plot_mesh(
        mesh=table_mesh,
        color="beige",
        is_z_up=is_z_up,
        fig=fig,
        name="Table",
        opacity=1.0,
    )
