from __future__ import annotations

import plotly.graph_objects as go
import torch
import trimesh


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
