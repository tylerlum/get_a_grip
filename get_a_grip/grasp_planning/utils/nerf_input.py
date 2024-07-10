import pathlib
from dataclasses import dataclass

import numpy as np
import pypose as pp
import torch
import trimesh
from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.grasp_planning.nerf_conversions.nerf_to_bps import nerf_to_bps
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_mesh import nerf_to_mesh
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_urdf import (
    nerf_to_mesh_bounding_box_max,
    nerf_to_mesh_bounding_box_min,
)
from get_a_grip.grasp_planning.utils.frames import GraspPlanningFrames
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.scripts.create_nerf_grasp_dataset import (
    get_nerf_densities_at_fingertips,
    get_nerf_densities_global,
)
from get_a_grip.model_training.utils.nerf_evaluator_model_batch_data import (
    BatchDataInput,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
)
from get_a_grip.model_training.utils.nerf_utils import (
    compute_centroid_from_nerf,
)


@dataclass
class NerfInput:
    nerf_pipeline: Pipeline
    nerf_is_z_up: bool

    def __post_init__(self):
        centroid_N = self.centroid_N
        assert centroid_N.shape == (3,), f"Expected shape (3,), got {centroid_N.shape}"

        self.frames = GraspPlanningFrames(
            nerf_is_z_up=self.nerf_is_z_up,
            object_centroid_N_x=centroid_N[0],
            object_centroid_N_y=centroid_N[1],
            object_centroid_N_z=centroid_N[2],
        )

    @property
    def bps_values(self) -> torch.Tensor:
        if not hasattr(self, "_bps_values"):
            self._bps_values = self._compute_bps_values()
        return self._bps_values

    @property
    def nerf_densities_global(self) -> torch.Tensor:
        if not hasattr(self, "_nerf_global_coords"):
            self._nerf_global_coords = self._compute_nerf_densities_global()
        return self._nerf_global_coords

    @property
    def mesh_N(self) -> trimesh.Trimesh:
        if not hasattr(self, "_mesh_N"):
            self._mesh_N = self._compute_mesh_N()
        return self._mesh_N

    @property
    def mesh_Oy(self) -> trimesh.Trimesh:
        X_Oy_N = self.frames.X_("Oy", "N")
        return self.mesh_N.apply_transform(X_Oy_N)

    @property
    def centroid_N(self) -> np.ndarray:
        if not hasattr(self, "_centroid_N"):
            self._centroid_N = self._compute_centroid_N()
        return self._centroid_N

    def compute_nerf_densities_at_fingertips(
        self,
        grasp_frame_transforms: pp.LieTensor,
        fingertip_config: EvenlySpacedFingertipConfig,
    ) -> torch.Tensor:
        nerf_densities, _ = get_nerf_densities_at_fingertips(
            nerf_pipeline=self.nerf_pipeline,
            grasp_frame_transforms=grasp_frame_transforms,
            fingertip_config=fingertip_config,
            ray_samples_chunk_size=100,  # Compute batch size: choose as large as can fit in memory
            ray_origins_finger_frame=get_ray_origins_finger_frame(fingertip_config),
            X_N_Oy=self.frames.X_("N", "Oy"),
        )
        return nerf_densities

    def compute_batch_data_input(
        self,
        grasp_frame_transforms: pp.LieTensor,
        grasp_config_tensor: torch.Tensor,
        fingertip_config: EvenlySpacedFingertipConfig,
    ) -> BatchDataInput:
        batch_data_input = BatchDataInput(
            nerf_densities=self.compute_nerf_densities_at_fingertips(
                grasp_frame_transforms=grasp_frame_transforms,
                fingertip_config=fingertip_config,
            ),
            grasp_transforms=grasp_frame_transforms,
            fingertip_config=fingertip_config,
            grasp_configs=grasp_config_tensor,
            nerf_densities_global=self.nerf_densities_global,
        ).to(grasp_frame_transforms.device)
        return batch_data_input

    def clear_cache(self) -> None:
        if hasattr(self, "_bps_values"):
            del self._bps_values
        if hasattr(self, "_nerf_global_coords"):
            del self._nerf_global_coords
        if hasattr(self, "_mesh_N"):
            del self._mesh_N
        if hasattr(self, "_centroid_N"):
            del self._centroid_N

    def _compute_bps_values(self) -> torch.Tensor:
        bps_values, _, _ = nerf_to_bps(
            nerf_pipeline=self.nerf_pipeline,
            nerf_is_z_up=self.nerf_is_z_up,
            X_N_By=self.frames.X_("N", "By"),
        )
        return torch.from_numpy(bps_values).float()

    def _compute_nerf_densities_global(self) -> torch.Tensor:
        nerf_densities_global, _ = get_nerf_densities_global(
            nerf_pipeline=self.nerf_pipeline,
            X_N_Oy=self.frames.X_("N", "Oy"),
        )
        return nerf_densities_global

    def _compute_mesh_N(self) -> trimesh.Trimesh:
        mesh_N = nerf_to_mesh(
            field=self.nerf_pipeline.model.field,
            level=15,
            lb=nerf_to_mesh_bounding_box_min(self.nerf_is_z_up),
            ub=nerf_to_mesh_bounding_box_max(self.nerf_is_z_up),
            save_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
        )
        return mesh_N

    def _compute_centroid_N(self) -> np.ndarray:
        centroid_N = compute_centroid_from_nerf(
            self.nerf_pipeline.model.field,
            lb=nerf_to_mesh_bounding_box_min(self.nerf_is_z_up),
            ub=nerf_to_mesh_bounding_box_max(self.nerf_is_z_up),
            level=15,
            num_pts_x=100,
            num_pts_y=100,
            num_pts_z=100,
        )
        assert centroid_N.shape == (3,), f"Expected shape (3,), got {centroid_N.shape}"
        return centroid_N
