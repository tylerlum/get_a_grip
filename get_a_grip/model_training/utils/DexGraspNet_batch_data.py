from __future__ import annotations
from nerf_grasping.other_utils import (
    get_points_in_grid,
)
from nerf_grasping.dataset.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
    NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
)
from nerf_grasping.other_utils import (
    get_points_in_grid,
)
import functools
from dataclasses import dataclass
import pypose as pp
import torch
from typing import List, Optional, Union
import numpy as np
from nerf_grasping.config.fingertip_config import (
    BaseFingertipConfig,
    EvenlySpacedFingertipConfig,
)
from enum import Enum, auto

NUM_XYZ = 3


@functools.lru_cache()
def get_ray_origins_finger_frame_cached(cfg: BaseFingertipConfig) -> torch.Tensor:
    from nerf_grasping.grasp_utils import (
        get_ray_origins_finger_frame,
    )

    ray_origins_finger_frame = get_ray_origins_finger_frame(cfg)
    return ray_origins_finger_frame


class ConditioningType(Enum):
    """Enum for conditioning type."""

    NONE = auto()
    GRASP_TRANSFORM = auto()
    GRASP_CONFIG = auto()

    @property
    def dim(self) -> int:
        if self == ConditioningType.NONE:
            return 0
        elif self == ConditioningType.GRASP_TRANSFORM:
            return 7
        elif self == ConditioningType.GRASP_CONFIG:
            return 7 + 16 + 4
        else:
            raise NotImplementedError()


@dataclass
class BatchDataInput:
    nerf_densities: torch.Tensor
    grasp_transforms: pp.LieTensor
    fingertip_config: BaseFingertipConfig  # have to take this because all these shape checks used to use hardcoded constants.
    grasp_configs: torch.Tensor
    nerf_densities_global: Optional[torch.Tensor]
    object_y_wrt_table: Optional[torch.Tensor]
    random_rotate_transform: Optional[pp.LieTensor] = None
    nerf_density_threshold_value: Optional[float] = None

    def to(self, device) -> BatchDataInput:
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.grasp_configs = self.grasp_configs.to(device)
        if self.nerf_densities_global is not None:
            self.nerf_densities_global = self.nerf_densities_global.to(device)
        if self.object_y_wrt_table is not None:
            self.object_y_wrt_table = self.object_y_wrt_table.to(device)
        self.random_rotate_transform = (
            self.random_rotate_transform.to(device=device)
            if self.random_rotate_transform is not None
            else None
        )
        return self

    @property
    def nerf_alphas(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        delta = (
            self.fingertip_config.grasp_depth_mm
            / (self.fingertip_config.num_pts_z - 1)
            / 1000
        )
        if isinstance(self.fingertip_config, EvenlySpacedFingertipConfig):
            assert delta == self.fingertip_config.distance_between_pts_mm / 1000
        alphas = 1.0 - torch.exp(-delta * self.nerf_densities)

        if self.nerf_density_threshold_value is not None:
            alphas = torch.where(
                self.nerf_densities > self.nerf_density_threshold_value,
                torch.ones_like(alphas),
                torch.zeros_like(alphas),
            )

        return alphas

    @property
    def nerf_alphas_global(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        delta = (
            self.fingertip_config.grasp_depth_mm
            / (self.fingertip_config.num_pts_z - 1)
            / 1000
        )
        if isinstance(self.fingertip_config, EvenlySpacedFingertipConfig):
            assert delta == self.fingertip_config.distance_between_pts_mm / 1000
        alphas = 1.0 - torch.exp(-delta * self.nerf_densities_global)

        if self.nerf_density_threshold_value is not None:
            alphas = torch.where(
                self.nerf_densities_global > self.nerf_density_threshold_value,
                torch.ones_like(alphas),
                torch.zeros_like(alphas),
            )

        return alphas

    @property
    def nerf_densities_global_cropped(self) -> torch.Tensor:
        assert self.nerf_densities_global.shape == (
            self.batch_size,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        start_x = (
            NERF_DENSITIES_GLOBAL_NUM_X - NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
        ) // 2
        start_y = (
            NERF_DENSITIES_GLOBAL_NUM_Y - NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
        ) // 2
        start_z = (
            NERF_DENSITIES_GLOBAL_NUM_Z - NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
        ) // 2
        end_x = start_x + NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
        end_y = start_y + NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
        end_z = start_z + NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
        nerf_densities_global_cropped = self.nerf_densities_global[
            :, start_x:end_x, start_y:end_y, start_z:end_z
        ]
        assert nerf_densities_global_cropped.shape == (
            self.batch_size,
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        )
        return nerf_densities_global_cropped

    @property
    def nerf_alphas_global_cropped(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        delta = (
            self.fingertip_config.grasp_depth_mm
            / (self.fingertip_config.num_pts_z - 1)
            / 1000
        )
        if isinstance(self.fingertip_config, EvenlySpacedFingertipConfig):
            assert delta == self.fingertip_config.distance_between_pts_mm / 1000
        alphas = 1.0 - torch.exp(-delta * self.nerf_densities_global_cropped)

        if self.nerf_density_threshold_value is not None:
            alphas = torch.where(
                self.nerf_densities_global_cropped > self.nerf_density_threshold_value,
                torch.ones_like(alphas),
                torch.zeros_like(alphas),
            )

        return alphas

    @property
    def coords(self) -> torch.Tensor:
        return self._coords_helper(self.grasp_transforms)

    @property
    def augmented_coords(self) -> torch.Tensor:
        return self._coords_helper(self.augmented_grasp_transforms)

    @property
    def coords_global(self) -> torch.Tensor:
        points = get_points_in_grid(
            lb=lb_Oy,
            ub=ub_Oy,
            num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
            num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
            num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
        )

        assert points.shape == (
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
            NUM_XYZ,
        )
        points = torch.from_numpy(points).to(
            device=self.device, dtype=self.nerf_densities.dtype
        )
        points = points.permute(3, 0, 1, 2)
        points = points[None, ...].repeat_interleave(self.batch_size, dim=0)
        assert points.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        return points

    @property
    def coords_global_cropped(self) -> torch.Tensor:
        coords_global = self.coords_global
        assert coords_global.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )

        start_x = (
            NERF_DENSITIES_GLOBAL_NUM_X - NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
        ) // 2
        start_y = (
            NERF_DENSITIES_GLOBAL_NUM_Y - NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
        ) // 2
        start_z = (
            NERF_DENSITIES_GLOBAL_NUM_Z - NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
        ) // 2
        end_x = start_x + NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
        end_y = start_y + NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
        end_z = start_z + NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
        coords_global_cropped = coords_global[
            :, :, start_x:end_x, start_y:end_y, start_z:end_z
        ]
        assert coords_global_cropped.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        )
        return coords_global_cropped

    @property
    def augmented_coords_global(self) -> torch.Tensor:
        coords_global = self.coords_global
        if self.random_rotate_transform is None:
            return coords_global

        assert coords_global.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        return_value = (
            (
                self.random_rotate_transform.unsqueeze(dim=1)
                @ coords_global.permute(0, 2, 3, 4, 1).reshape(
                    self.batch_size, -1, NUM_XYZ
                )  # Must put the NUM_XYZ dimension last for matrix multiplication.
            )
            .permute(0, 2, 1)
            .reshape(
                self.batch_size,
                NUM_XYZ,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        )
        assert return_value.shape == coords_global.shape
        return return_value

    @property
    def augmented_coords_global_cropped(self) -> torch.Tensor:
        augmented_coords_global = self.augmented_coords_global
        assert augmented_coords_global.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )

        start_x = (
            NERF_DENSITIES_GLOBAL_NUM_X - NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
        ) // 2
        start_y = (
            NERF_DENSITIES_GLOBAL_NUM_Y - NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
        ) // 2
        start_z = (
            NERF_DENSITIES_GLOBAL_NUM_Z - NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
        ) // 2
        end_x = start_x + NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
        end_y = start_y + NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
        end_z = start_z + NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED

        augmented_coords_global_cropped = augmented_coords_global[
            :, :, start_x:end_x, start_y:end_y, start_z:end_z
        ]
        assert augmented_coords_global_cropped.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        )
        return augmented_coords_global_cropped

    @property
    def coords_wrt_wrist(self) -> torch.Tensor:
        return self._coords_wrt_wrist_helper(self.coords, self.grasp_configs)

    @property
    def augmented_coords_wrt_wrist(self) -> torch.Tensor:
        return self._coords_wrt_wrist_helper(
            self.augmented_coords, self.augmented_grasp_configs
        )

    @property
    def y_coords_wrt_table(self) -> torch.Tensor:
        return self._y_coords_wrt_table_helper(self.coords)

    @property
    def augmented_y_coords_wrt_table(self) -> torch.Tensor:
        return self._y_coords_wrt_table_helper(self.augmented_coords)

    @property
    def nerf_alphas_with_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.coords)

    @property
    def nerf_alphas_with_coords_v2(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_v2_helper(
            self.coords,
            self.coords_wrt_wrist,
            self.y_coords_wrt_table,
        )

    @property
    def nerf_alphas_with_coords_v3(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_v3_helper(
            self.coords,
            self.y_coords_wrt_table,
        )

    @property
    def nerf_alphas_with_coords_v4(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_v4_helper(
            self.coords,
            self.coords_wrt_wrist,
        )

    @property
    def nerf_alphas_with_augmented_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.augmented_coords)

    @property
    def nerf_alphas_with_augmented_coords_v2(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_v2_helper(
            self.augmented_coords,
            self.augmented_coords_wrt_wrist,
            self.augmented_y_coords_wrt_table,
        )

    @property
    def nerf_alphas_with_augmented_coords_v3(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_v3_helper(
            self.augmented_coords,
            self.augmented_y_coords_wrt_table,
        )

    @property
    def nerf_alphas_with_augmented_coords_v4(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_v4_helper(
            self.augmented_coords,
            self.augmented_coords_wrt_wrist,
        )

    @property
    def nerf_alphas_global_with_coords(self) -> torch.Tensor:
        return self._nerf_alphas_global_with_coords_helper(
            coords_global=self.coords_global,
            nerf_alphas_global=self.nerf_alphas_global,
            x_dim=NERF_DENSITIES_GLOBAL_NUM_X,
            y_dim=NERF_DENSITIES_GLOBAL_NUM_Y,
            z_dim=NERF_DENSITIES_GLOBAL_NUM_Z,
        )

    @property
    def nerf_alphas_global_with_augmented_coords(self) -> torch.Tensor:
        return self._nerf_alphas_global_with_coords_helper(
            coords_global=self.augmented_coords_global,
            nerf_alphas_global=self.nerf_alphas_global,
            x_dim=NERF_DENSITIES_GLOBAL_NUM_X,
            y_dim=NERF_DENSITIES_GLOBAL_NUM_Y,
            z_dim=NERF_DENSITIES_GLOBAL_NUM_Z,
        )

    @property
    def nerf_alphas_global_cropped_with_coords(self) -> torch.Tensor:
        return self._nerf_alphas_global_with_coords_helper(
            coords_global=self.coords_global_cropped,
            nerf_alphas_global=self.nerf_alphas_global_cropped,
            x_dim=NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            y_dim=NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            z_dim=NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        )

    @property
    def nerf_alphas_global_cropped_with_augmented_coords(self) -> torch.Tensor:
        return self._nerf_alphas_global_with_coords_helper(
            coords_global=self.augmented_coords_global_cropped,
            nerf_alphas_global=self.nerf_alphas_global_cropped,
            x_dim=NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
            y_dim=NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
            z_dim=NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
        )

    @property
    def augmented_grasp_transforms(self) -> pp.LieTensor:
        if self.random_rotate_transform is None:
            return self.grasp_transforms

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        return_value = (
            self.random_rotate_transform.unsqueeze(dim=1) @ self.grasp_transforms
        )
        assert (
            return_value.lshape
            == self.grasp_transforms.lshape
            == (self.batch_size, self.fingertip_config.n_fingers)
        )
        return return_value

    @property
    def augmented_grasp_configs(self) -> torch.Tensor:
        if self.random_rotate_transform is None:
            return self.grasp_configs

        # Apply random rotation to grasp config.
        # NOTE: hardcodes grasp_configs ordering
        wrist_pose = pp.SE3(self.grasp_configs[..., :7])
        joint_angles = self.grasp_configs[..., 7:23]
        grasp_orientations = pp.SO3(self.grasp_configs[..., 23:])

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        wrist_pose = self.random_rotate_transform.unsqueeze(1) @ wrist_pose
        grasp_orientations = (
            self.random_rotate_transform.rotation().unsqueeze(1) @ grasp_orientations
        )

        return_value = torch.cat(
            (wrist_pose.data, joint_angles, grasp_orientations.data), axis=-1
        )
        assert (
            return_value.shape
            == self.grasp_configs.shape
            == (
                self.batch_size,
                self.fingertip_config.n_fingers,
                7 + 16 + 4,
            )
        )

        return return_value

    @property
    def batch_size(self) -> int:
        return self.nerf_densities.shape[0]

    @property
    def device(self) -> torch.device:
        return self.nerf_densities.device

    def _coords_helper(self, grasp_transforms: pp.LieTensor) -> torch.Tensor:
        assert grasp_transforms.lshape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
        )
        USE_NERFSTUDIO = False
        # These two should be the same, but avoid use of nerfstudio
        if USE_NERFSTUDIO:
            from nerf_grasping.grasp_utils import (
                get_ray_samples,
            )

            ray_origins_finger_frame = get_ray_origins_finger_frame_cached(
                self.fingertip_config
            )

            all_query_points = get_ray_samples(
                ray_origins_finger_frame.to(
                    device=grasp_transforms.device, dtype=grasp_transforms.dtype
                ),
                grasp_transforms,
                self.fingertip_config,
            ).frustums.get_positions()
        else:
            finger_width_m = self.fingertip_config.finger_width_mm / 1000
            finger_height_m = self.fingertip_config.finger_height_mm / 1000
            grasp_depth_m = self.fingertip_config.grasp_depth_mm / 1000
            query_point_origins = get_points_in_grid(
                lb=np.array([-finger_width_m / 2, -finger_height_m / 2, 0]),
                ub=np.array([finger_width_m / 2, finger_height_m / 2, grasp_depth_m]),
                num_pts_x=self.fingertip_config.num_pts_x,
                num_pts_y=self.fingertip_config.num_pts_y,
                num_pts_z=self.fingertip_config.num_pts_z,
            )
            assert query_point_origins.shape == (
                self.fingertip_config.num_pts_x,
                self.fingertip_config.num_pts_y,
                self.fingertip_config.num_pts_z,
                NUM_XYZ,
            )
            query_point_origins = torch.from_numpy(query_point_origins).to(
                device=grasp_transforms.device, dtype=grasp_transforms.dtype
            )
            all_query_points = grasp_transforms.unsqueeze(
                dim=2
            ) @ query_point_origins.reshape(1, 1, -1, NUM_XYZ)
            assert all_query_points.shape == (
                self.batch_size,
                self.fingertip_config.n_fingers,
                self.fingertip_config.num_pts_x
                * self.fingertip_config.num_pts_y
                * self.fingertip_config.num_pts_z,
                NUM_XYZ,
            )
            all_query_points = all_query_points.reshape(
                self.batch_size,
                self.fingertip_config.n_fingers,
                self.fingertip_config.num_pts_x,
                self.fingertip_config.num_pts_y,
                self.fingertip_config.num_pts_z,
                NUM_XYZ,
            )

        assert all_query_points.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
            NUM_XYZ,
        )
        all_query_points = all_query_points.permute(0, 1, 5, 2, 3, 4)
        assert all_query_points.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return all_query_points

    def _coords_wrt_wrist_helper(
        self, coords_wrt_object: torch.Tensor, grasp_configs: torch.Tensor
    ) -> torch.Tensor:
        assert coords_wrt_object.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        coords_wrt_object = coords_wrt_object.permute(0, 1, 3, 4, 5, 2)
        assert coords_wrt_object.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
            NUM_XYZ,
        )

        assert grasp_configs[..., :7].shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            7,
        )
        wrist_poses_wrt_object = pp.SE3(grasp_configs[:, :, None, None, None, :7])

        assert wrist_poses_wrt_object.lshape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            1,
            1,
        )

        T_wrist_object = wrist_poses_wrt_object.Inv()

        coords_wrt_wrist = T_wrist_object @ coords_wrt_object
        assert coords_wrt_wrist.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
            NUM_XYZ,
        )
        coords_wrt_wrist = coords_wrt_wrist.permute(0, 1, 5, 2, 3, 4)
        assert coords_wrt_wrist.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        return coords_wrt_wrist

    def _y_coords_wrt_table_helper(
        self, coords_wrt_object: torch.Tensor
    ) -> torch.Tensor:
        assert coords_wrt_object.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        y_coords_wrt_object = coords_wrt_object[:, :, 1:2]
        assert y_coords_wrt_object.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        assert self.object_y_wrt_table is not None, "object_y_wrt_table is None"
        assert self.object_y_wrt_table.shape == (
            self.batch_size,
        ), self.object_y_wrt_table.shape
        assert torch.all(self.object_y_wrt_table >= 0), self.object_y_wrt_table
        y_coords_wrt_table = (
            y_coords_wrt_object
            + self.object_y_wrt_table[..., None, None, None, None, None]
        )
        return y_coords_wrt_table

    def _nerf_alphas_with_coords_helper(self, coords: torch.Tensor) -> torch.Tensor:
        assert coords.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        reshaped_nerf_alphas = self.nerf_alphas.reshape(
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return_value = torch.cat(
            [
                reshaped_nerf_alphas,
                coords,
            ],
            dim=2,
        )
        assert return_value.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ + 1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return return_value

    def _nerf_alphas_with_coords_v2_helper(
        self,
        coords: torch.Tensor,
        coords_wrt_wrist: torch.Tensor,
        y_coords_wrt_table: torch.Tensor,
    ) -> torch.Tensor:
        NUM_Y = 1
        assert coords.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        assert coords_wrt_wrist.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        assert y_coords_wrt_table.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_Y,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        reshaped_nerf_alphas = self.nerf_alphas.reshape(
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return_value = torch.cat(
            [
                reshaped_nerf_alphas,
                coords,
                coords_wrt_wrist,
                y_coords_wrt_table,
            ],
            dim=2,
        )
        assert return_value.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ + 1 + NUM_XYZ + NUM_Y,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return return_value

    def _nerf_alphas_with_coords_v3_helper(
        self,
        coords: torch.Tensor,
        y_coords_wrt_table: torch.Tensor,
    ) -> torch.Tensor:
        NUM_Y = 1
        assert coords.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        assert y_coords_wrt_table.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_Y,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        reshaped_nerf_alphas = self.nerf_alphas.reshape(
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return_value = torch.cat(
            [
                reshaped_nerf_alphas,
                coords,
                y_coords_wrt_table,
            ],
            dim=2,
        )
        assert return_value.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ + 1 + NUM_Y,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return return_value

    def _nerf_alphas_with_coords_v4_helper(
        self,
        coords: torch.Tensor,
        coords_wrt_wrist: torch.Tensor,
    ) -> torch.Tensor:
        assert coords.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        assert coords_wrt_wrist.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        reshaped_nerf_alphas = self.nerf_alphas.reshape(
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return_value = torch.cat(
            [
                reshaped_nerf_alphas,
                coords,
                coords_wrt_wrist,
            ],
            dim=2,
        )
        assert return_value.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ + 1 + NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return return_value

    def _nerf_alphas_global_with_coords_helper(
        self,
        coords_global: torch.Tensor,
        nerf_alphas_global: torch.Tensor,
        x_dim: int,
        y_dim: int,
        z_dim: int,
    ) -> torch.Tensor:
        assert coords_global.shape == (
            self.batch_size,
            NUM_XYZ,
            x_dim,
            y_dim,
            z_dim,
        )
        reshaped_nerf_alphas_global = nerf_alphas_global.reshape(
            self.batch_size,
            1,
            x_dim,
            y_dim,
            z_dim,
        )
        return_value = torch.cat(
            [
                reshaped_nerf_alphas_global,
                coords_global,
            ],
            dim=1,
        )
        assert return_value.shape == (
            self.batch_size,
            NUM_XYZ + 1,
            x_dim,
            y_dim,
            z_dim,
        )
        return return_value

    def get_conditioning(self, conditioning_type: ConditioningType) -> torch.Tensor:
        if conditioning_type == ConditioningType.GRASP_TRANSFORM:
            return self.augmented_grasp_transforms.tensor()
        elif conditioning_type == ConditioningType.GRASP_CONFIG:
            return self.augmented_grasp_configs
        else:
            raise NotImplementedError()


@dataclass
class DepthImageBatchDataInput:
    depth_uncertainty_images: torch.Tensor
    grasp_transforms: pp.LieTensor
    fingertip_config: BaseFingertipConfig  # have to take this because all these shape checks used to use hardcoded constants.
    grasp_configs: torch.Tensor
    random_rotate_transform: Optional[pp.LieTensor] = None
    nerf_density_threshold_value: Optional[float] = None

    def to(self, device) -> DepthImageBatchDataInput:
        self.depth_uncertainty_images = self.depth_uncertainty_images.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.random_rotate_transform = (
            self.random_rotate_transform.to(device=device)
            if self.random_rotate_transform is not None
            else None
        )
        self.grasp_configs = self.grasp_configs.to(device)
        return self

    @property
    def augmented_grasp_transforms(self) -> pp.LieTensor:
        if self.random_rotate_transform is None:
            return self.grasp_transforms

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        return_value = (
            self.random_rotate_transform.unsqueeze(dim=1) @ self.grasp_transforms
        )
        assert (
            return_value.lshape
            == self.grasp_transforms.lshape
            == (self.batch_size, self.fingertip_config.n_fingers)
        )
        return return_value

    @property
    def augmented_grasp_configs(self) -> torch.Tensor:
        if self.random_rotate_transform is None:
            return self.grasp_configs

        # Apply random rotation to grasp config.
        # NOTE: hardcodes grasp_configs ordering
        wrist_pose = pp.SE3(self.grasp_configs[..., :7])
        joint_angles = self.grasp_configs[..., 7:23]
        grasp_orientations = pp.SO3(self.grasp_configs[..., 23:])

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        wrist_pose = self.random_rotate_transform.unsqueeze(1) @ wrist_pose
        grasp_orientations = (
            self.random_rotate_transform.rotation().unsqueeze(1) @ grasp_orientations
        )

        return_value = torch.cat(
            (wrist_pose.data, joint_angles, grasp_orientations.data), axis=-1
        )
        assert (
            return_value.shape
            == self.grasp_configs.shape
            == (
                self.batch_size,
                self.fingertip_config.n_fingers,
                7 + 16 + 4,
            )
        )

        return return_value

    @property
    def batch_size(self) -> int:
        return self.depth_uncertainty_images.shape[0]

    @property
    def device(self) -> torch.device:
        return self.depth_uncertainty_images.device

    def get_conditioning(self, conditioning_type: ConditioningType) -> torch.Tensor:
        if conditioning_type == ConditioningType.GRASP_TRANSFORM:
            return self.augmented_grasp_transforms.tensor()
        elif conditioning_type == ConditioningType.GRASP_CONFIG:
            return self.augmented_grasp_configs
        else:
            raise NotImplementedError()


@dataclass
class BatchDataOutput:
    passed_simulation: torch.Tensor
    passed_penetration_threshold: torch.Tensor
    passed_eval: torch.Tensor

    def to(self, device) -> BatchDataOutput:
        self.passed_simulation = self.passed_simulation.to(device)
        self.passed_penetration_threshold = self.passed_penetration_threshold.to(device)
        self.passed_eval = self.passed_eval.to(device)
        return self

    @property
    def batch_size(self) -> int:
        return self.passed_eval.shape[0]

    @property
    def device(self) -> torch.device:
        return self.passed_eval.device


@dataclass
class BatchData:
    input: Union[BatchDataInput, DepthImageBatchDataInput]
    output: BatchDataOutput
    nerf_config: List[str]

    def to(self, device) -> BatchData:
        self.input = self.input.to(device)
        self.output = self.output.to(device)
        return self

    @property
    def batch_size(self) -> int:
        return self.output.batch_size

    @property
    def device(self) -> torch.device:
        return self.output.device
