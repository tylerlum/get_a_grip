"""
Utils for working with NeRFs, including:
* getting densities in a grid
* computing the centroid of the field
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.base_field import Field

from get_a_grip.utils.point_utils import get_points_in_grid


def get_densities_in_grid(
    field: Field,
    lb: np.ndarray,
    ub: np.ndarray,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_points_in_region_np = get_points_in_grid(
        lb=lb,
        ub=ub,
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        num_pts_z=num_pts_z,
    )
    assert query_points_in_region_np.shape == (num_pts_x, num_pts_y, num_pts_z, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_points_in_region = (
        torch.from_numpy(query_points_in_region_np).float().to(device)
    )

    nerf_densities_in_region = (
        get_density(
            field=field,
            positions=query_points_in_region,
        )[0]
        .squeeze(dim=-1)
        .reshape(
            num_pts_x,
            num_pts_y,
            num_pts_z,
        )
    )
    return nerf_densities_in_region, query_points_in_region


def get_densities_in_grid_np(
    field: Field,
    lb: np.ndarray,
    ub: np.ndarray,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nerf_densities_in_region, query_points_in_region = get_densities_in_grid(
        field=field,
        lb=lb,
        ub=ub,
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        num_pts_z=num_pts_z,
    )
    return (
        nerf_densities_in_region.detach().cpu().numpy(),
        query_points_in_region.detach().cpu().numpy(),
    )


def compute_centroid_from_nerf(
    field: Field,
    lb: np.ndarray,
    ub: np.ndarray,
    level: float,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> np.ndarray:
    """
    Compute the centroid of the field.
    """
    nerf_densities_in_region, query_points_in_region = get_densities_in_grid_np(
        field=field,
        lb=lb,
        ub=ub,
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        num_pts_z=num_pts_z,
    )

    points_to_keep_with_nans = np.where(
        (nerf_densities_in_region > level)[..., None].repeat(3, axis=-1),
        query_points_in_region,
        np.nan,
    )
    # Check there are some non-nan
    assert not np.all(np.isnan(points_to_keep_with_nans))

    avg_point_no_nans = np.nanmean(points_to_keep_with_nans.reshape(-1, 3), axis=0)
    assert avg_point_no_nans.shape == (3,)
    return avg_point_no_nans


def get_density(
    field: Field,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes and returns the densities.
    Should be same as nerfacto.get_density, but takes in a positions tensor instead of a ray_samples
    """
    assert positions.shape[-1] == 3

    if field.spatial_distortion is not None:
        positions = field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
    else:
        positions = SceneBox.get_normalized_positions(positions, field.aabb)
    # Make sure the tcnn gets inputs between 0 and 1.
    selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
    positions = positions * selector[..., None]
    field._sample_locations = positions
    if not field._sample_locations.requires_grad:
        field._sample_locations.requires_grad = True
    positions_flat = positions.view(-1, 3)
    h = field.mlp_base(positions_flat).view(*positions.shape[:-1], -1)
    assert h.shape[-1] == 1 + field.geo_feat_dim
    density_before_activation, base_mlp_out = torch.split(
        h, [1, field.geo_feat_dim], dim=-1
    )
    field._density_before_activation = density_before_activation

    # Rectifying the density with an exponential is much more stable than a ReLU or
    # softplus, because it enables high post-activation (float32) density outputs
    # from smaller internal (float16) parameters.
    density = field.average_init_density * trunc_exp(
        density_before_activation.to(positions)
    )
    density = density * selector[..., None]
    return density, base_mlp_out
