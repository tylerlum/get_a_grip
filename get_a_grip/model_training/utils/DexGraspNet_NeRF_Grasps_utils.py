import os
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import trimesh
import pathlib
import pypose as pp
import plotly.graph_objects as go
import nerf_grasping
import nerfstudio
from matplotlib import pyplot as plt
from nerf_grasping.grasp_utils import (
    get_ray_samples_helper,
    get_ray_origins_finger_frame_helper,
)
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils_v2 import (
    get_object_code,
    get_object_scale,
    get_object_string,
    parse_object_code_and_scale,
    transform_point,
    transform_points,
    get_scene_dict,
    plot_mesh,
    plot_mesh_and_transforms,
    plot_mesh_and_query_points,
    plot_mesh_and_high_density_points,
)



def get_ray_samples_in_mesh_region(
    mesh: trimesh.Trimesh,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
    beyond_mesh_region_scale: float = 1.5,
) -> RaySamples:
    # Get bounds from mesh
    min_bounds, max_bounds = mesh.bounds * beyond_mesh_region_scale
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    return get_ray_samples_in_region(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        num_pts_z=num_pts_z,
    )


def get_ray_samples_in_region(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> RaySamples:
    # Get bounds from mesh
    finger_width_mm = (x_max - x_min) * 1000
    finger_height_mm = (y_max - y_min) * 1000
    grasp_depth_mm = (z_max - z_min) * 1000

    # Prepare ray samples
    ray_origins_object_frame = get_ray_origins_finger_frame_helper(
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        finger_width_mm=finger_width_mm,
        finger_height_mm=finger_height_mm,
        z_offset_mm=z_min * 1000,
    )
    assert ray_origins_object_frame.shape == (num_pts_x, num_pts_y, 3)
    identity_transform = pp.identity_SE3(1).to(ray_origins_object_frame.device)
    ray_samples = get_ray_samples_helper(
        ray_origins_object_frame,
        identity_transform,
        num_pts_z=num_pts_z,
        grasp_depth_mm=grasp_depth_mm,
    )
    return ray_samples
