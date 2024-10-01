from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.evaluator import (
    Evaluator,
)
from get_a_grip.grasp_planning.utils.optimizer import (
    Optimizer,
)
from get_a_grip.grasp_planning.utils.sampler import (
    Sampler,
)


@dataclass
class FilterConfig:
    enable: bool = True
    fingers_forward_theta_deg: float = 60.0
    palm_upwards_theta_deg: float = 60.0


class Planner:
    def __init__(
        self,
        sampler: Sampler,
        evaluator: Evaluator,
        optimizer: Optimizer,
        filter_cfg: FilterConfig,
        n_random_rotations_per_grasp: int,
    ) -> None:
        self.sampler = sampler
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.filter_cfg = filter_cfg
        self.n_random_rotations_per_grasp = n_random_rotations_per_grasp

    def plan(self) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        # Sample
        sampled_grasp_configs = self.sampler.sample()

        # Sample random rotations around y
        new_grasp_configs = AllegroGraspConfig.from_multiple_grasp_configs(
            [sampled_grasp_configs]
            + [
                sampled_grasp_configs.sample_random_rotations_only_around_y()
                for _ in range(self.n_random_rotations_per_grasp)
            ]
        )
        assert len(new_grasp_configs) == len(sampled_grasp_configs) * (
            1 + self.n_random_rotations_per_grasp
        )

        # Filter
        if self.filter_cfg.enable:
            filtered_grasp_configs = sampled_grasp_configs.filter_less_feasible(
                fingers_forward_theta_deg=self.filter_cfg.fingers_forward_theta_deg,
                palm_upwards_theta_deg=self.filter_cfg.palm_upwards_theta_deg,
            )
        else:
            filtered_grasp_configs = sampled_grasp_configs

        # Evaluate and rank
        sorted_grasp_configs = self.evaluator.sort(filtered_grasp_configs)

        # Optimize
        optimized_grasp_configs, losses = self.optimizer.optimize(sorted_grasp_configs)

        return optimized_grasp_configs, losses
