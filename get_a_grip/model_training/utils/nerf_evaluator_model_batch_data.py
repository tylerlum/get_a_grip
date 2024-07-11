from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np
import pypose as pp
import torch

from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_DELTA,
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    add_coords_to_global_grids,
    get_coords_global,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
    get_ray_samples,
)
from get_a_grip.utils.point_utils import (
    get_points_in_grid,
)

NUM_XYZ = 3


@functools.lru_cache()
def get_ray_origins_finger_frame_cached(
    cfg: EvenlySpacedFingertipConfig,
) -> torch.Tensor:
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
    fingertip_config: EvenlySpacedFingertipConfig  # have to take this because all these shape checks used to use hardcoded constants.
    grasp_configs: torch.Tensor
    nerf_densities_global: Optional[torch.Tensor]
    random_rotate_transform: Optional[pp.LieTensor] = None

    def print_shapes(self) -> None:
        print(f"nerf_densities.shape: {self.nerf_densities.shape}")
        print(f"grasp_transforms.lshape: {self.grasp_transforms.lshape}")
        print(f"grasp_configs.shape: {self.grasp_configs.shape}")
        if self.nerf_densities_global is not None:
            print(f"nerf_densities_global.shape: {self.nerf_densities_global.shape}")
        if self.random_rotate_transform is not None:
            print(
                f"random_rotate_transform.lshape: {self.random_rotate_transform.lshape}"
            )

    def to(self, device) -> BatchDataInput:
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.grasp_configs = self.grasp_configs.to(device)
        if self.nerf_densities_global is not None:
            self.nerf_densities_global = self.nerf_densities_global.to(device)
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
        delta = self.fingertip_config.distance_between_pts_mm / 1000
        alphas = 1.0 - torch.exp(-delta * self.nerf_densities)

        return alphas

    @property
    def nerf_alphas_global(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        delta = NERF_DENSITIES_GLOBAL_DELTA
        alphas = 1.0 - torch.exp(-delta * self.nerf_densities_global)

        return alphas

    @property
    def coords(self) -> torch.Tensor:
        return self._coords_helper(self.grasp_transforms)

    @property
    def augmented_coords(self) -> torch.Tensor:
        return self._coords_helper(self.augmented_grasp_transforms)

    @property
    def coords_global(self) -> torch.Tensor:
        points = get_coords_global(
            device=self.device,
            dtype=self.nerf_densities.dtype,
            batch_size=self.batch_size,
        )

        assert points.shape == (
            self.batch_size,
            NUM_XYZ,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        return points

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
    def nerf_alphas_with_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.coords)

    @property
    def nerf_alphas_with_augmented_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.augmented_coords)

    @property
    def nerf_alphas_global_with_coords(self) -> torch.Tensor:
        return add_coords_to_global_grids(
            global_grids=self.nerf_alphas_global,
            coords_global=self.coords_global,
        )

    @property
    def nerf_alphas_global_with_augmented_coords(self) -> torch.Tensor:
        return add_coords_to_global_grids(
            global_grids=self.nerf_alphas_global,
            coords_global=self.augmented_coords_global,
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
        # These two should be the same
        USE_NERFSTUDIO = False
        if USE_NERFSTUDIO:
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
class BatchDataOutput:
    y_pick: torch.Tensor
    y_coll: torch.Tensor
    y_PGS: torch.Tensor

    def print_shapes(self) -> None:
        print(f"y_pick: {self.y_pick.shape}")
        print(f"y_coll: {self.y_coll.shape}")
        print(f"y_PGS: {self.y_PGS.shape}")

    def to(self, device) -> BatchDataOutput:
        self.y_pick = self.y_pick.to(device)
        self.y_coll = self.y_coll.to(device)
        self.y_PGS = self.y_PGS.to(device)
        return self

    @property
    def batch_size(self) -> int:
        return self.y_PGS.shape[0]

    @property
    def device(self) -> torch.device:
        return self.y_PGS.device


@dataclass
class BatchData:
    input: BatchDataInput
    output: BatchDataOutput
    nerf_config: List[str]

    def print_shapes(self) -> None:
        self.input.print_shapes()
        self.output.print_shapes()
        print(f"nerf_config: {len(self.nerf_config)}")

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
