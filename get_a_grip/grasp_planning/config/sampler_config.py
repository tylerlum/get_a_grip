from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.grasp_planning.utils.sampler import (
    BpsSampler,
    FixedSampler,
    NerfSampler,
    Sampler,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    add_coords_to_global_grids,
    get_coords_global,
)


@dataclass
class SamplerConfig(ABC):
    @abstractmethod
    def create(self, nerf_input: NerfInput) -> Sampler:
        raise NotImplementedError


@dataclass
class BpsSamplerConfig(SamplerConfig):
    ckpt_path: pathlib.Path
    num_grasps: int = 3200

    def __post_init__(self) -> None:
        assert (
            self.num_grasps > 0
        ), f"num_grasps must be positive, got {self.num_grasps}"
        assert self.ckpt_path.exists(), f"{self.ckpt_path} does not exist"
        assert self.ckpt_path.suffix in [
            ".pt",
            ".pth",
        ], f"{self.ckpt_path} does not have a .pt or .pth suffix"

    def create(self, nerf_input: NerfInput) -> BpsSampler:
        bps_values = nerf_input.bps_values
        return BpsSampler(
            bps_values=bps_values, ckpt_path=self.ckpt_path, num_grasps=self.num_grasps
        )


@dataclass
class NerfSamplerConfig(SamplerConfig):
    ckpt_path: pathlib.Path
    num_grasps: int = 3200

    def __post_init__(self) -> None:
        assert (
            self.num_grasps > 0
        ), f"num_grasps must be positive, got {self.num_grasps}"
        assert self.ckpt_path.exists(), f"{self.ckpt_path} does not exist"
        assert self.ckpt_path.suffix in [
            ".pt",
            ".pth",
        ], f"{self.ckpt_path} does not have a .pt or .pth suffix"

    def create(self, nerf_input: NerfInput) -> NerfSampler:
        nerf_densities_global = nerf_input.nerf_densities_global_with_coords.unsqueeze(
            dim=0
        )
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

        return NerfSampler(
            nerf_densities_global_with_coords=nerf_global_grids_with_coords,
            ckpt_path=self.ckpt_path,
            num_grasps=self.num_grasps,
        )


@dataclass
class FixedSamplerConfig(SamplerConfig):
    fixed_grasp_config_dict_path: pathlib.Path
    max_num_grasps: Optional[int] = 3200

    def __post_init__(self) -> None:
        assert (
            self.fixed_grasp_config_dict_path.exists()
        ), f"{self.fixed_grasp_config_dict_path} does not exist"
        assert self.fixed_grasp_config_dict_path.suffix in [
            ".npy",
        ], f"{self.fixed_grasp_config_dict_path} does not have a .npy suffix"

    def create(self, nerf_input: NerfInput) -> FixedSampler:
        return FixedSampler(
            fixed_grasp_config_dict_path=self.fixed_grasp_config_dict_path,
            max_num_grasps=self.max_num_grasps,
        )


SamplerConfigUnion = Union[BpsSamplerConfig, NerfSamplerConfig, FixedSamplerConfig]
