"""
Copied mostly from: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/exporter.py
Modified for point cloud export:
  * Pass in pipeline so don't need to waste time loading from file again if already loaded
  * Return point cloud without saving to file
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import (
    RandomCamerasDataManager,
)
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter.exporter_utils import (
    generate_point_cloud,
)
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.pipelines.base_pipeline import Pipeline
from typing_extensions import Literal


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    pass


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self, pipeline: Pipeline) -> o3d.geometry.PointCloud:
        """Export point cloud."""
        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (
                VanillaDataManager,
                ParallelDataManager,
                FullImageDatamanager,
                RandomCamerasDataManager,
            ),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = (
            self.num_rays_per_batch
        )

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if (
            self.obb_center is not None
            and self.obb_rotation is not None
            and self.obb_scale is not None
        ):
            crop_obb = OrientedBox.from_params(
                self.obb_center, self.obb_rotation, self.obb_scale
            )
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=(
                self.normal_output_name
                if self.normal_method == "model_output"
                else None
            ),
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(
                points.shape[0], axis=0
            )[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()
        return pcd
