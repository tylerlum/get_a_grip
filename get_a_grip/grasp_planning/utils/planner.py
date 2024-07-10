from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.evaluator import (
    Evaluator,
    EvaluatorConfigUnion,
)
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.grasp_planning.utils.optimizer import (
    Optimizer,
    OptimizerConfigUnion,
)
from get_a_grip.grasp_planning.utils.sampler import (
    Sampler,
    SamplerConfigUnion,
)


@dataclass
class FilterConfig:
    enable: bool = True
    fingers_forward_theta_deg: float = 60.0
    palm_upwards_theta_deg: float = 60.0


@dataclass
class PlannerConfig:
    sampler: SamplerConfigUnion
    evaluator: EvaluatorConfigUnion
    optimizer: OptimizerConfigUnion
    filter: FilterConfig = field(default_factory=FilterConfig)
    n_random_rotations_per_grasp: int = 0

    def create(self, nerf_input: NerfInput) -> Planner:
        # Prepare sampler
        sampler = self.sampler.create(nerf_input=nerf_input)

        # Prepare evaluator
        evaluator = self.evaluator.create(nerf_input=nerf_input)

        # Prepare optimizer
        optimizer = self.optimizer.create(evaluator=evaluator)

        # Prepare grasp planner
        grasp_planner = Planner(
            sampler=sampler,
            evaluator=evaluator,
            optimizer=optimizer,
            cfg=self,
        )

        return grasp_planner


class Planner:
    def __init__(
        self,
        sampler: Sampler,
        evaluator: Evaluator,
        optimizer: Optimizer,
        cfg: PlannerConfig,
    ) -> None:
        self.sampler = sampler
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.cfg = cfg

    def plan(self) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        # Sample
        sampled_grasp_configs = self.sampler.sample()

        # Sample random rotations around y
        new_grasp_configs = AllegroGraspConfig.from_multiple_grasp_configs(
            [sampled_grasp_configs]
            + [
                sampled_grasp_configs.sample_random_rotations_only_around_y()
                for _ in range(self.cfg.n_random_rotations_per_grasp)
            ]
        )
        assert len(new_grasp_configs) == len(sampled_grasp_configs) * (
            1 + self.cfg.n_random_rotations_per_grasp
        )

        # Filter
        if self.cfg.filter.enable:
            filtered_grasp_configs = sampled_grasp_configs.filter_less_feasible(
                fingers_forward_theta_deg=self.cfg.filter.fingers_forward_theta_deg,
                palm_upwards_theta_deg=self.cfg.filter.palm_upwards_theta_deg,
            )
        else:
            filtered_grasp_configs = sampled_grasp_configs

        # Evaluate and rank
        sorted_grasp_configs = self.evaluator.sort(filtered_grasp_configs)

        # Optimize
        optimized_grasp_configs, losses = self.optimizer.optimize(sorted_grasp_configs)

        return optimized_grasp_configs, losses
