"""
Utils for working with NeRFs, including:
* rendering depth / uncertainty images from the NeRF
* getting densities in a grid
* computing the centroid of the field
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Tuple

import numpy as np
import pypose as pp
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.base_field import Field
from nerfstudio.models.base_model import Model

from get_a_grip.model_training.config.camera_config import CameraConfig
from get_a_grip.model_training.utils.point_utils import get_points_in_grid

GRASP_TO_OPENCV = pp.euler2SO3(torch.tensor([np.pi, 0, 0])).unsqueeze(0)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def get_cameras(
    grasp_transforms: pp.LieTensor,
    camera_config: CameraConfig,
) -> Cameras:
    """
    Get cameras from the grasp transforms and camera config.
    NOTE: Cameras requires the inputs to be on cpu not gpu
          Issue in nerfstudio/cameras/cameras.py:166 torch.Tensor instead of torch.tensor
    """

    # Flip rotations so -z points along the grasp dir.
    c2w_rotations = grasp_transforms.rotation().cpu() @ GRASP_TO_OPENCV
    # c2w_rotations = grasp_transforms.rotation()
    c2w_translations = grasp_transforms.translation().cpu()

    cameras = Cameras(
        camera_to_worlds=torch.cat(
            (c2w_rotations.matrix(), c2w_translations.unsqueeze(-1)), dim=-1
        ),
        camera_type=CameraType.PERSPECTIVE,
        fx=camera_config.fx,
        fy=camera_config.fy,
        cx=camera_config.cx,
        cy=camera_config.cy,
        width=camera_config.W,
        height=camera_config.H,
    )

    return cameras


def render(
    cameras: Cameras,
    nerf_model: Model,
    depth_mode: Literal["median", "expected"] = "expected",
    near_plane: float = 1e-3,
    far_plane: float = 1e-1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_shape = cameras.shape
    cameras = cameras.reshape(-1)
    assert len(cameras.shape) == 1
    num_imgs = cameras.shape[0]

    # TODO: make sure we have enough VRAM to render all the cameras at once.
    ray_bundle = cameras.generate_rays(
        torch.arange(
            cameras.camera_to_worlds.shape[0], device=cameras.camera_to_worlds.device
        ).unsqueeze(-1),
    )
    ray_bundle.nears = torch.ones_like(ray_bundle.pixel_area) * near_plane
    ray_bundle.fars = torch.ones_like(ray_bundle.pixel_area) * far_plane

    depth, uncertainty = _render_depth_and_uncertainty_for_camera_ray_bundle(
        nerf_model, ray_bundle, depth_mode
    )

    height, width, _ = ray_bundle.shape
    assert_equals(ray_bundle.shape, (height, width, num_imgs))
    assert_equals(depth.shape, ray_bundle.shape)
    assert_equals(uncertainty.shape, ray_bundle.shape)

    depth = depth.reshape(height, width, *original_shape)
    uncertainty = uncertainty.reshape(height, width, *original_shape)
    return depth, uncertainty


def get_ray_samples_from_cameras(
    cameras: Cameras,
    nerf_model: Model,
    near_plane: float = 1e-3,
    far_plane: float = 1e-1,
) -> RaySamples:
    assert len(cameras.shape) == 1, f"Expected cameras to be 1D, got {cameras.shape}"

    ray_bundle = cameras.generate_rays(
        torch.arange(
            cameras.camera_to_worlds.shape[0], device=cameras.camera_to_worlds.device
        ).unsqueeze(-1),
    )

    print(cameras.shape)
    print(ray_bundle.shape)
    ray_bundle = ray_bundle.flatten()
    print(ray_bundle.shape)

    ray_bundle.nears = torch.ones_like(ray_bundle.pixel_area) * near_plane
    ray_bundle.fars = torch.ones_like(ray_bundle.pixel_area) * far_plane

    # Query proposal sampler.
    ray_samples, _, _ = nerf_model.proposal_sampler(
        ray_bundle, density_fns=nerf_model.density_fns
    )

    return ray_samples


def _render_depth_and_uncertainty_for_camera_ray_bundle(
    nerf_model: Model,
    camera_ray_bundle: RayBundle,
    depth_mode: Literal["median", "expected"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Takes in camera parameters and computes the output of the model.

    Args:
        camera_ray_bundle: ray bundle to calculate outputs over
    """
    num_rays_per_chunk = nerf_model.config.eval_num_rays_per_chunk
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    num_rays = len(camera_ray_bundle)
    outputs_lists = defaultdict(list)
    for i in range(0, num_rays, num_rays_per_chunk):
        start_idx = i
        end_idx = i + num_rays_per_chunk
        ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
            start_idx, end_idx
        )
        depth, uncertainty = _render_depth_and_uncertainty_for_single_ray_bundle(
            nerf_model, ray_bundle, depth_mode
        )
        outputs_lists["depth"].append(depth)
        outputs_lists["uncertainty"].append(uncertainty)

    return torch.cat(outputs_lists["depth"]).view(
        image_height, image_width, -1
    ), torch.cat(outputs_lists["uncertainty"]).view(image_height, image_width, -1)


def _render_depth_and_uncertainty_for_single_ray_bundle(
    nerf_model: Model, ray_bundle: RayBundle, depth_mode: Literal["median", "expected"]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slight variation on typical nerf rendering that doesn't render RGB but *does*
    compute the uncertainty along the ray.
    """
    EPS = 1e-7

    # Query proposal sampler.
    ray_samples, _, _ = nerf_model.proposal_sampler(
        ray_bundle, density_fns=nerf_model.density_fns
    )

    # Query field for density.
    density, _ = nerf_model.field.get_density(ray_samples)

    weights = ray_samples.get_weights(density)

    # Compute the depth image.

    # Idea: compute IQR of depth values, and use that as the uncertainty.

    # Compute the uncertainty variance.
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    # Render expected depth map (need regardless to compute variance).
    # If all weights are 0, then normalize_weights will be 0, so expected_depth will be the min
    normalized_weights = weights / (weights.sum(dim=-2, keepdim=True) + EPS)
    expected_depth = (normalized_weights * steps).sum(dim=-2)
    expected_depth = torch.clip(expected_depth, steps.min(), steps.max())

    # expected_depth = nerf_model.renderer_expected_depth(
    #     weights=weights, ray_samples=ray_samples
    # )

    depth_variance = (
        normalized_weights * torch.square(steps - expected_depth.unsqueeze(-2))
    ).sum(-2)

    if depth_mode == "median":
        median_depth = nerf_model.renderer_depth(
            weights=weights, ray_samples=ray_samples
        )
        depth = median_depth
    elif depth_mode == "expected":
        depth = expected_depth
    else:
        raise ValueError(f"Invalid depth mode {depth_mode}")

    return depth, depth_variance


def get_densities_in_grid(
    field: Field,
    lb: np.ndarray,
    ub: np.ndarray,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> Tuple[np.ndarray, np.ndarray]:
    query_points_in_region = get_points_in_grid(
        lb=lb,
        ub=ub,
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        num_pts_z=num_pts_z,
    )
    assert query_points_in_region.shape == (num_pts_x, num_pts_y, num_pts_z, 3)

    nerf_densities_in_region = (
        get_density(
            field=field,
            positions=torch.from_numpy(query_points_in_region).cuda(),
        )[0]
        .squeeze(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .reshape(
            num_pts_x,
            num_pts_y,
            num_pts_z,
        )
    )
    return nerf_densities_in_region, query_points_in_region


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
    nerf_densities_in_region, query_points_in_region = get_densities_in_grid(
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
