"""
Utils for working with NeRF rays, including:
* computing ray origins for fingertip trajectories
* computing ray samples along fingertip trajectories
"""

from __future__ import annotations

import pypose as pp
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples

from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)


def get_ray_origins_finger_frame_helper(
    num_pts_x: int,
    num_pts_y: int,
    finger_width_mm: float,
    finger_height_mm: float,
    z_offset_mm: float = 0.0,
) -> torch.Tensor:
    """
    Create grid of grasp origins in finger frame with shape (num_pts_x, num_pts_y, 3)
    so that grid_of_points[2, 3] = [x, y, z], where x, y, z are the coordinates of the
    ray origin for the [2, 3] "pixel" in the finger frame.
    Origin of transform is at center of xy at z=z_offset_mm
    x is width, y is height, z is depth
    """
    gripper_finger_width_m = finger_width_mm / 1000.0
    gripper_finger_height_m = finger_height_mm / 1000.0
    z_offset_m = z_offset_mm / 1000.0

    x_coords = torch.linspace(
        -gripper_finger_width_m / 2, gripper_finger_width_m / 2, num_pts_x
    )
    y_coords = torch.linspace(
        -gripper_finger_height_m / 2, gripper_finger_height_m / 2, num_pts_y
    )

    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
    zz = torch.zeros_like(xx) + z_offset_m

    assert xx.shape == yy.shape == zz.shape == (num_pts_x, num_pts_y)
    ray_origins = torch.stack([xx, yy, zz], axis=-1)

    assert ray_origins.shape == (num_pts_x, num_pts_y, 3)

    return ray_origins


def get_ray_origins_finger_frame(
    fingertip_config: EvenlySpacedFingertipConfig,
) -> torch.Tensor:
    ray_origins_finger_frame = get_ray_origins_finger_frame_helper(
        num_pts_x=fingertip_config.num_pts_x,
        num_pts_y=fingertip_config.num_pts_y,
        finger_width_mm=fingertip_config.finger_width_mm,
        finger_height_mm=fingertip_config.finger_height_mm,
    )
    return ray_origins_finger_frame


def get_ray_bundles(
    ray_origins_finger_frame: torch.Tensor,
    transform: pp.LieTensor,
) -> RayBundle:
    num_pts_x, num_pts_y = ray_origins_finger_frame.shape[:2]
    assert ray_origins_finger_frame.shape == (num_pts_x, num_pts_y, 3)
    batch_dims = transform.lshape

    # Device / dtype cast for the transform.
    ray_origins_finger_frame = ray_origins_finger_frame.to(
        device=transform.device, dtype=transform.dtype
    )

    # Add batch dims for transform.
    for _ in range(len(batch_dims)):
        ray_origins_finger_frame = ray_origins_finger_frame.unsqueeze(0)

    # Add batch dims for finger-frame points.
    transform = transform.unsqueeze(-2).unsqueeze(-2)
    assert transform.lshape == (*batch_dims, 1, 1)

    # Apply transform.
    ray_origins_world_frame = (
        transform @ ray_origins_finger_frame
    )  # shape [*batch_dims, num_pts_x, num_pts_y, 3]
    assert ray_origins_world_frame.shape == (*batch_dims, num_pts_x, num_pts_y, 3)

    ray_dirs_finger_frame = torch.tensor(
        [0.0, 0.0, 1.0], device=transform.device, dtype=transform.dtype
    )

    # Expand ray_dirs to add the batch dims.
    for _ in range(len(batch_dims)):
        ray_dirs_finger_frame = ray_dirs_finger_frame.unsqueeze(0)

    # Rotate ray directions (hence SO3 cast).
    ray_dirs_world_frame = (
        transform.rotation() @ ray_dirs_finger_frame
    )  # [*batch_dims, num_pts_x,  num_pts_y, 3]

    assert ray_dirs_world_frame.shape == (*batch_dims, 1, 1, 3)

    # Create dummy pixel areas object.
    pixel_area = (
        torch.ones_like(ray_dirs_world_frame[..., 0]).unsqueeze(-1).float().contiguous()
    )  # [*batch_dims, num_pts_x, num_pts_y, 1]

    ray_bundle = RayBundle(ray_origins_world_frame, ray_dirs_world_frame, pixel_area)

    return ray_bundle


def get_ray_samples(
    ray_origins_finger_frame: torch.Tensor,
    transform: pp.LieTensor,
    fingertip_config: EvenlySpacedFingertipConfig,
) -> RaySamples:
    return get_ray_samples_helper(
        ray_origins_finger_frame=ray_origins_finger_frame,
        transform=transform,
        num_pts_z=fingertip_config.num_pts_z,
        grasp_depth_mm=fingertip_config.grasp_depth_mm,
    )


def get_ray_samples_helper(
    ray_origins_finger_frame: torch.Tensor,
    transform: pp.LieTensor,
    num_pts_z: int,
    grasp_depth_mm: float,
) -> RaySamples:
    num_pts_x, num_pts_y = ray_origins_finger_frame.shape[:2]
    assert ray_origins_finger_frame.shape == (num_pts_x, num_pts_y, 3)

    ray_bundles = get_ray_bundles(ray_origins_finger_frame, transform)
    grasp_depth_m = grasp_depth_mm / 1000.0

    # Work out sample lengths.
    sample_dists = torch.linspace(
        0.0,
        grasp_depth_m,
        steps=num_pts_z,
        dtype=transform.dtype,
        device=transform.device,
    )
    assert sample_dists.shape == (num_pts_z,)

    for _ in range(len(transform.lshape)):
        sample_dists = sample_dists.unsqueeze(0)

    sample_dists = sample_dists.expand(
        *ray_origins_finger_frame.shape[:-1], num_pts_z
    ).unsqueeze(-1)
    assert sample_dists.shape == (num_pts_x, num_pts_y, num_pts_z, 1)

    # Pull ray samples -- note these are degenerate, i.e., the deltas field is meaningless.
    return ray_bundles.get_ray_samples(sample_dists, sample_dists)
