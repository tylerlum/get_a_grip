from __future__ import annotations

from dataclasses import dataclass, field

from get_a_grip.grasp_planning.config.evaluator_config import EvaluatorConfigUnion
from get_a_grip.grasp_planning.config.optimizer_config import OptimizerConfigUnion
from get_a_grip.grasp_planning.config.sampler_config import SamplerConfigUnion
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.grasp_planning.utils.planner import (
    Planner,
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
