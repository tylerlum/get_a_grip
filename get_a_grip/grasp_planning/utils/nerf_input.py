import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

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
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.frames import GraspPlanningFrames
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    add_coords_to_global_grids,
    get_coords_global,
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
from get_a_grip.model_training.utils.plot_utils import (
    plot_grasp_and_mesh_and_more,
    plot_mesh_and_query_points,
)
from get_a_grip.utils.point_utils import transform_points


@dataclass
class NerfInput:
    # Nerf inputs used for grasp planning
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
            self._bps_values, self._basis_pts_N, self._point_cloud_pts_N = (
                self._compute_bps_values_and_more()
            )
        return self._bps_values

    @property
    def basis_pts_N(self) -> torch.Tensor:
        if not hasattr(self, "_basis_pts_N"):
            self._bps_values, self._basis_pts_N, self._point_cloud_pts_N = (
                self._compute_bps_values_and_more()
            )
        return self._basis_pts_N

    @property
    def point_cloud_pts_N(self) -> torch.Tensor:
        if not hasattr(self, "_point_cloud_pts_N"):
            self._bps_values, self._basis_pts_N, self._point_cloud_pts_N = (
                self._compute_bps_values_and_more()
            )
        return self._point_cloud_pts_N

    @property
    def nerf_densities_global_with_coords(self) -> torch.Tensor:
        if not hasattr(self, "_nerf_densities_global_with_coords"):
            self._nerf_densities_global_with_coords = (
                self._compute_nerf_densities_global_with_coords()
            )
        return self._nerf_densities_global_with_coords

    @property
    def nerf_densities_global(self) -> torch.Tensor:
        nerf_densities_global_with_coords = self.nerf_densities_global_with_coords
        assert (
            nerf_densities_global_with_coords.shape
            == (
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape (4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords.shape}"
        return nerf_densities_global_with_coords[0]

    def nerf_densities_global_repeated(self, N: int) -> torch.Tensor:
        if (
            not hasattr(self, "_nerf_densities_global_repeated")
            or self._nerf_densities_global_repeated.shape[0] != N
        ):
            self._nerf_densities_global_repeated = self.nerf_densities_global.unsqueeze(
                dim=0
            ).repeat_interleave(N, dim=0)
        return self._nerf_densities_global_repeated

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
        grasp_config: AllegroGraspConfig,
        fingertip_config: EvenlySpacedFingertipConfig,
    ) -> BatchDataInput:
        B = len(grasp_config)

        nerf_densities = self.compute_nerf_densities_at_fingertips(
            grasp_frame_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=fingertip_config,
        )
        nerf_densities_global_repeated = self.nerf_densities_global_repeated(N=B)

        assert (
            nerf_densities.shape[0] == B
        ), f"Expected batch size {B}, got {nerf_densities.shape[0]}, shape: {nerf_densities.shape}"
        assert (
            nerf_densities_global_repeated.shape[0] == B
        ), f"Expected batch size {B}, got {nerf_densities_global_repeated.shape[0]}, shape: {nerf_densities_global_repeated.shape}"

        assert (
            nerf_densities_global_repeated.shape
            == (
                B,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape ({B}, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_repeated.shape}"

        batch_data_input = BatchDataInput(
            nerf_densities=nerf_densities,
            grasp_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=fingertip_config,
            grasp_configs=grasp_config.as_tensor(),
            nerf_densities_global=nerf_densities_global_repeated,
        ).to(grasp_config.device)
        return batch_data_input

    def clear_cache(self) -> None:
        if hasattr(self, "_bps_values"):
            del self._bps_values
        if hasattr(self, "_basis_pts_N"):
            del self._basis_pts_N
        if hasattr(self, "_point_cloud_pts_N"):
            del self._point_cloud_pts_N
        if hasattr(self, "_nerf_densities_global_with_coords"):
            del self._nerf_densities_global_with_coords
        if hasattr(self, "_nerf_densities_global_repeated"):
            del self._nerf_densities_global_repeated
        if hasattr(self, "_mesh_N"):
            del self._mesh_N
        if hasattr(self, "_centroid_N"):
            del self._centroid_N

    def plot_all_figs(
        self,
        grasp_config: AllegroGraspConfig,
        i: int = 0,
        # HACK: Use default fingertip_config if not given, should still be a reasonable visualization
        # But if the user's model uses a different fingertip_config, they should pass it in
        fingertip_config: EvenlySpacedFingertipConfig = EvenlySpacedFingertipConfig(),
    ) -> None:
        self.plot_mesh_and_point_cloud_and_bps(grasp_config=grasp_config, i=i)
        self.plot_mesh_and_nerf_densities_global(grasp_config=grasp_config, i=i)
        self.plot_mesh_and_nerf_densities_at_fingertips(
            grasp_config=grasp_config, i=i, fingertip_config=fingertip_config
        )

    def plot_mesh_and_point_cloud_and_bps(
        self,
        grasp_config: Optional[AllegroGraspConfig] = None,
        i: int = 0,
    ) -> None:
        X_N_Oy = self.frames.X_("N", "Oy")

        title = "Mesh, Point Cloud, Basis Points, BPS"
        fig = plot_grasp_and_mesh_and_more(
            X_N_Oy=X_N_Oy,
            visualize_Oy=True,
            mesh=self.mesh_N,
            basis_points=self.basis_pts_N.detach().cpu().numpy(),
            bps=self.bps_values.detach().cpu().numpy(),
            processed_point_cloud_points=self.point_cloud_pts_N.detach().cpu().numpy(),
            title=title,
            nerf_is_z_up=self.nerf_is_z_up,
        )

        if grasp_config is not None:
            plot_grasp_and_mesh_and_more(
                fig=fig,
                grasp=grasp_config.as_grasp()[i],
                X_N_Oy=X_N_Oy,
                visualize_target_hand=True,
                title=f"Grasp {i} | {title}",
                nerf_is_z_up=self.nerf_is_z_up,
            )
        fig.show()

    def plot_mesh_and_nerf_densities_global(
        self,
        grasp_config: Optional[AllegroGraspConfig] = None,
        i: int = 0,
    ) -> None:
        X_N_Oy = self.frames.X_("N", "Oy")

        title = "Mesh and Nerf Densities Global"
        fig = plot_grasp_and_mesh_and_more(
            X_N_Oy=X_N_Oy,
            visualize_Oy=True,
            mesh=self.mesh_N,
            nerf_global_grids_with_coords=self.nerf_densities_global_with_coords.detach()
            .cpu()
            .numpy(),
            title=title,
            nerf_is_z_up=self.nerf_is_z_up,
        )

        if grasp_config is not None:
            plot_grasp_and_mesh_and_more(
                fig=fig,
                grasp=grasp_config.as_grasp()[i],
                X_N_Oy=X_N_Oy,
                visualize_target_hand=True,
                title=f"Grasp {i} | {title}",
                nerf_is_z_up=self.nerf_is_z_up,
            )
        fig.show()

    def plot_mesh_and_nerf_densities_at_fingertips(
        self,
        grasp_config: AllegroGraspConfig,
        fingertip_config: EvenlySpacedFingertipConfig,
        i: int = 0,
    ) -> None:
        # Compute nerf densities at fingertips
        batch_data_input = self.compute_batch_data_input(
            grasp_config=grasp_config,
            fingertip_config=fingertip_config,
        )

        N_FINGERS = fingertip_config.n_fingers
        N_X = fingertip_config.num_pts_x
        N_Y = fingertip_config.num_pts_y
        N_Z = fingertip_config.num_pts_z

        nerf_alphas_with_coords = batch_data_input.nerf_alphas_with_coords[i]
        assert (
            nerf_alphas_with_coords.shape == (N_FINGERS, 4, N_X, N_Y, N_Z)
        ), f"Expected shape ({N_FINGERS}, 4, {N_X}, {N_Y}, {N_Z}), got {nerf_alphas_with_coords.shape}"

        # Permute and reshape to correct shape
        nerf_alphas_with_coords = nerf_alphas_with_coords.permute(0, 2, 3, 4, 1)
        assert (
            nerf_alphas_with_coords.shape == (N_FINGERS, N_X, N_Y, N_Z, 4)
        ), f"Expected shape ({N_FINGERS}, {N_X}, {N_Y}, {N_Z}, 4), got {nerf_alphas_with_coords.shape}"

        nerf_alphas = (
            nerf_alphas_with_coords[..., 0]
            .reshape(N_FINGERS, N_X * N_Y * N_Z)
            .detach()
            .cpu()
            .numpy()
        )
        coords_Oy = (
            nerf_alphas_with_coords[..., 1:]
            .reshape(N_FINGERS, N_X * N_Y * N_Z, 3)
            .detach()
            .cpu()
            .numpy()
        )

        # Transform
        X_N_Oy = self.frames.X_("N", "Oy")
        coords_N = transform_points(
            T=X_N_Oy,
            points=coords_Oy.reshape(N_FINGERS * N_X * N_Y * N_Z, 3),
        ).reshape(N_FINGERS, N_X * N_Y * N_Z, 3)

        fig = plot_mesh_and_query_points(
            mesh=self.mesh_N,
            all_query_points=coords_N,
            all_query_points_colors=nerf_alphas,
            num_fingers=N_FINGERS,
        )

        plot_grasp_and_mesh_and_more(
            fig=fig,
            grasp=grasp_config.as_grasp()[i],
            X_N_Oy=X_N_Oy,
            visualize_target_hand=True,
            visualize_Oy=True,
            title=f"Grasp {i} | Mesh and Nerf Densities at Fingertips",
            nerf_is_z_up=self.nerf_is_z_up,
        )

        fig.show()

    def _compute_bps_values_and_more(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bps_values, basis_pts_N, point_cloud_pts_N = nerf_to_bps(
            nerf_pipeline=self.nerf_pipeline,
            nerf_is_z_up=self.nerf_is_z_up,
            X_N_By=self.frames.X_("N", "By"),
        )
        return (
            torch.from_numpy(bps_values).float(),
            torch.from_numpy(basis_pts_N).float(),
            torch.from_numpy(point_cloud_pts_N).float(),
        )

    def _compute_nerf_densities_global_with_coords(self) -> torch.Tensor:
        nerf_densities_global, _ = get_nerf_densities_global(
            nerf_pipeline=self.nerf_pipeline,
            X_N_Oy=self.frames.X_("N", "Oy"),
        )

        nerf_densities_global = nerf_densities_global.unsqueeze(dim=0)
        assert nerf_densities_global.shape == (
            1,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        coords_global = get_coords_global(
            device=nerf_densities_global.device,
            dtype=nerf_densities_global.dtype,
            batch_size=nerf_densities_global.shape[0],
        )
        nerf_global_grids_with_coords = add_coords_to_global_grids(
            global_grids=nerf_densities_global, coords_global=coords_global
        ).squeeze(dim=0)

        assert nerf_global_grids_with_coords.shape == (
            4,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        return nerf_global_grids_with_coords

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
