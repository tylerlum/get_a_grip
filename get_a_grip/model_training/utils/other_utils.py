# Made this because we need these functions without nerfstudio imports
import numpy as np


def get_points_in_grid(
    lb: np.ndarray,
    ub: np.ndarray,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> np.ndarray:
    x_min, y_min, z_min = lb
    x_max, y_max, z_max = ub
    x_coords = np.linspace(x_min, x_max, num_pts_x)
    y_coords = np.linspace(y_min, y_max, num_pts_y)
    z_coords = np.linspace(z_min, z_max, num_pts_z)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    query_points_in_region = np.stack([xx, yy, zz], axis=-1)
    assert query_points_in_region.shape == (num_pts_x, num_pts_y, num_pts_z, 3)
    return query_points_in_region
