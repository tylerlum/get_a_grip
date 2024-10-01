from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.models.bps_sampler_model import BpsSamplerModel
from get_a_grip.model_training.models.nerf_sampler_model import NerfSamplerModel
from get_a_grip.model_training.utils.diffusion import Diffusion


class Sampler(ABC):
    @abstractmethod
    def sample(self) -> AllegroGraspConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_sample(self) -> int:
        raise NotImplementedError


class BpsSampler(Sampler):
    def __init__(
        self, bps_values: torch.Tensor, ckpt_path: Path, num_grasps: int
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save BPS values
        assert bps_values.shape == (
            4096,
        ), f"Expected shape (4096,), got {bps_values.shape}"
        self.bps_values = bps_values
        self.num_grasps = num_grasps

        # Load model
        config = DiffusionConfig(
            training=TrainingConfig(
                output_dir=ckpt_path.parent,
            )
        )
        self.model = BpsSamplerModel(
            n_pts=config.data.n_pts,
            grasp_dim=config.data.grasp_dim,
        )
        self.runner = Diffusion(
            config=config, model=self.model, load_multigpu_ckpt=True, device=self.device
        )
        self.runner.load_checkpoint(config, filename=ckpt_path.name)

    def sample(self) -> AllegroGraspConfig:
        # Repeat input B times
        # Use cache if appropriate
        if (
            not hasattr(self, "_bps_values_repeated_cached")
            or self._bps_values_repeated_cached.shape[0] != self.n_sample
        ):
            # Repeat BPS values
            self._bps_values_repeated_cached = self.bps_values.unsqueeze(
                dim=0
            ).repeat_interleave(self.n_sample, dim=0)
        bps_values_repeated = self._bps_values_repeated_cached

        assert bps_values_repeated.shape == (
            self.n_sample,
            4096,
        ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

        # Sample
        xT = torch.randn(self.n_sample, self.model.grasp_dim, device=self.runner.device)
        x = self.runner.sample(xT=xT, cond=bps_values_repeated.to(self.runner.device))
        grasp_configs = AllegroGraspConfig.from_grasp(grasp=x)
        return grasp_configs

    @property
    def n_sample(self) -> int:
        return self.num_grasps


class NerfSampler(Sampler):
    def __init__(
        self,
        nerf_densities_global_with_coords: torch.Tensor,
        ckpt_path: Path,
        num_grasps: int,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert (
            nerf_densities_global_with_coords.shape
            == (
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape (4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords.shape}"
        self.nerf_densities_global_with_coords = nerf_densities_global_with_coords
        self.num_grasps = num_grasps

        # Load model
        config = DiffusionConfig(
            training=TrainingConfig(
                output_dir=ckpt_path.parent,
            ),
        )
        self.model = NerfSamplerModel(
            global_grid_shape=(
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            ),
            grasp_dim=config.data.grasp_dim,
        )
        self.model._HACK_MODE_FOR_PERFORMANCE = True  # Big hack to speed up from sampling wasting dumb compute since it has the same cnn each time

        self.runner = Diffusion(
            config=config, model=self.model, load_multigpu_ckpt=True, device=self.device
        )
        self.runner.load_checkpoint(config, filename=ckpt_path.name)

    def sample(self) -> AllegroGraspConfig:
        # Repeat NeRF values
        # Use cache if appropriate
        if not hasattr(self, "_nerf_densities_global_with_coords_repeated_cached") or (
            self._nerf_densities_global_with_coords_repeated_cached.shape[0]
            != self.n_sample
        ):
            self._nerf_densities_global_with_coords_repeated_cached = (
                self.nerf_densities_global_with_coords.unsqueeze(0).repeat(
                    self.n_sample, 1, 1, 1, 1
                )
            )
        nerf_densities_global_with_coords_repeated = (
            self._nerf_densities_global_with_coords_repeated_cached
        )

        assert (
            nerf_densities_global_with_coords_repeated.shape
            == (
                self.n_sample,
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape ({self.n_sample}, 4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords_repeated.shape}"

        # Sample
        xT = torch.randn(self.n_sample, self.model.grasp_dim, device=self.runner.device)
        x = self.runner.sample(
            xT=xT,
            cond=nerf_densities_global_with_coords_repeated.to(self.runner.device),
        )
        grasp_configs = AllegroGraspConfig.from_grasp(grasp=x)
        return grasp_configs

    @property
    def n_sample(self) -> int:
        return self.num_grasps


class FixedSampler(Sampler):
    def __init__(
        self,
        fixed_grasp_config_dict_path: Path,
        max_num_grasps: Optional[int] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fixed_grasp_config_dict_path = fixed_grasp_config_dict_path
        self.max_num_grasps = max_num_grasps

        self.fixed_grasp_config_dict = np.load(
            fixed_grasp_config_dict_path, allow_pickle=True
        ).item()

        self.num_grasps = self.fixed_grasp_config_dict["trans"].shape[0]
        if self.max_num_grasps is not None and self.num_grasps > self.max_num_grasps:
            self.num_grasps = self.max_num_grasps

    def sample(self) -> AllegroGraspConfig:
        grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            self.fixed_grasp_config_dict
        )

        if len(grasp_configs) > self.n_sample:
            random_idxs = np.random.choice(
                len(grasp_configs), size=self.n_sample, replace=False
            )
            grasp_configs = grasp_configs[random_idxs]

        return grasp_configs.to(self.device)

    @property
    def n_sample(self) -> int:
        return self.num_grasps
