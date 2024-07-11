from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import trimesh
import tyro

from get_a_grip.grasp_planning.utils.frames import (
    GraspMotionPlanningFrames,
    GraspPlanningFrames,
)
from get_a_grip.model_training.utils.plot_utils import add_transform_plot


@dataclass
class VisualizeFramesArgs:
    nerf_is_z_up: bool = True
    object_centroid_N_x: float = 0.1
    object_centroid_N_y: float = 0.2
    object_centroid_N_z: float = 0.3
    nerf_frame_offset_W_x: Optional[float] = None
    nerf_frame_offset_W_y: Optional[float] = None
    nerf_frame_offset_W_z: Optional[float] = None


def get_table_mesh(is_z_up: bool, W: float = 0.4, H: float = 0.4) -> trimesh.Trimesh:
    if is_z_up:
        table_pos, table_normal, table_parallel = (
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
        )
    else:
        table_pos, table_normal, table_parallel = (
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
        )

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


def add_table(
    fig: go.Figure, is_z_up: bool, color: str = "lightgreen", opacity: float = 0.5
) -> None:
    table_mesh = get_table_mesh(is_z_up=is_z_up)
    table_vertices = table_mesh.vertices
    fig.add_trace(
        go.Mesh3d(
            x=table_vertices[:, 0],
            y=table_vertices[:, 1],
            z=table_vertices[:, 2],
            i=table_mesh.faces[:, 0],
            j=table_mesh.faces[:, 1],
            k=table_mesh.faces[:, 2],
            color=color,
            opacity=opacity,
            name="table",
        )
    )


def plot_frames(
    frames: GraspPlanningFrames,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()

    # Plot frames
    A = "N"  # Base
    for B in frames.available_frames:
        X_A_B = frames.X_(A=A, B=B)
        add_transform_plot(fig=fig, X_A_B=X_A_B, A=A, B=B)

    # Plot centroid
    fig.add_trace(
        go.Scatter3d(
            x=[frames.object_centroid_N[0]],
            y=[frames.object_centroid_N[1]],
            z=[frames.object_centroid_N[2]],
            mode="markers",
            marker=dict(size=5, color="black"),
            name="object_centroid_N",
        )
    )

    add_table(fig=fig, is_z_up=frames.nerf_is_z_up)

    fig.update_layout(
        go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title="Frames",
        )
    )
    fig.show()


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeFramesArgs])
    if (
        args.nerf_is_z_up
        and args.nerf_frame_offset_W_x is not None
        and args.nerf_frame_offset_W_y is not None
        and args.nerf_frame_offset_W_z is not None
    ):
        frames = GraspMotionPlanningFrames(
            nerf_is_z_up=args.nerf_is_z_up,
            object_centroid_N_x=args.object_centroid_N_x,
            object_centroid_N_y=args.object_centroid_N_y,
            object_centroid_N_z=args.object_centroid_N_z,
            nerf_frame_offset_W_x=args.nerf_frame_offset_W_x,
            nerf_frame_offset_W_y=args.nerf_frame_offset_W_y,
            nerf_frame_offset_W_z=args.nerf_frame_offset_W_z,
        )
    else:
        frames = GraspPlanningFrames(
            nerf_is_z_up=args.nerf_is_z_up,
            object_centroid_N_x=args.object_centroid_N_x,
            object_centroid_N_y=args.object_centroid_N_y,
            object_centroid_N_z=args.object_centroid_N_z,
        )

    plot_frames(frames=frames)


if __name__ == "__main__":
    main()
