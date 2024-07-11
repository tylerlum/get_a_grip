import numpy as np
import torch

from get_a_grip.utils.point_utils import (
    get_points_in_grid,
)

# Hardcoded values
(
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
) = (30, 30, 30)

NERF_DENSITIES_GLOBAL_lb_Oy = np.array([-0.15, -0.15, -0.15])
NERF_DENSITIES_GLOBAL_ub_Oy = np.array([0.15, 0.15, 0.15])

# Ensure deltas are the same
NERF_DENSITIES_GLOBAL_DELTA_X = (
    NERF_DENSITIES_GLOBAL_ub_Oy[0] - NERF_DENSITIES_GLOBAL_lb_Oy[0]
) / NERF_DENSITIES_GLOBAL_NUM_X
NERF_DENSITIES_GLOBAL_DELTA_Y = (
    NERF_DENSITIES_GLOBAL_ub_Oy[1] - NERF_DENSITIES_GLOBAL_lb_Oy[1]
) / NERF_DENSITIES_GLOBAL_NUM_Y
NERF_DENSITIES_GLOBAL_DELTA_Z = (
    NERF_DENSITIES_GLOBAL_ub_Oy[2] - NERF_DENSITIES_GLOBAL_lb_Oy[2]
) / NERF_DENSITIES_GLOBAL_NUM_Z

assert np.isclose(NERF_DENSITIES_GLOBAL_DELTA_X, NERF_DENSITIES_GLOBAL_DELTA_Y)
assert np.isclose(NERF_DENSITIES_GLOBAL_DELTA_X, NERF_DENSITIES_GLOBAL_DELTA_Z)
assert np.isclose(NERF_DENSITIES_GLOBAL_DELTA_Y, NERF_DENSITIES_GLOBAL_DELTA_Z)

NERF_DENSITIES_GLOBAL_DELTA = NERF_DENSITIES_GLOBAL_DELTA_X


def get_coords_global(
    device: torch.device, dtype: torch.dtype, batch_size: int
) -> torch.Tensor:
    points = get_points_in_grid(
        lb=NERF_DENSITIES_GLOBAL_lb_Oy,
        ub=NERF_DENSITIES_GLOBAL_ub_Oy,
        num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
        num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
        num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
    )

    assert points.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    )
    points = torch.from_numpy(points).to(device=device, dtype=dtype)
    points = points.permute(3, 0, 1, 2)
    points = points[None, ...].repeat_interleave(batch_size, dim=0)
    assert points.shape == (
        batch_size,
        3,
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
    )
    return points


def add_coords_to_global_grids(
    global_grids: torch.Tensor, coords_global: torch.Tensor
) -> torch.Tensor:
    B = global_grids.shape[0]
    assert (
        global_grids.shape
        == (
            B,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape ({B}, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {global_grids.shape}"
    assert (
        coords_global.shape
        == (
            B,
            3,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape (B, 3, NERF_DENSITIES_GLOBAL_NUM_X, NERF_DENSITIES_GLOBAL_NUM_Y, NERF_DENSITIES_GLOBAL_NUM_Z), got {coords_global.shape}"

    global_grids_with_coords = torch.cat(
        (
            global_grids.unsqueeze(dim=1),
            coords_global,
        ),
        dim=1,
    )
    assert (
        global_grids_with_coords.shape
        == (
            B,
            3 + 1,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape (B, 3 + 1, NERF_DENSITIES_GLOBAL_NUM_X, NERF_DENSITIES_GLOBAL_NUM_Y, NERF_DENSITIES_GLOBAL_NUM_Z), got {global_grids_with_coords.shape}"
    return global_grids_with_coords
