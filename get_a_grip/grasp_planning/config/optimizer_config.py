from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from get_a_grip.grasp_planning.utils.evaluator import (
    BpsEvaluator,
    Evaluator,
    NerfEvaluator,
)
from get_a_grip.grasp_planning.utils.optimizer import (
    BpsRandomSamplingOptimizer,
    NerfGradientOptimizer,
    NerfRandomSamplingOptimizer,
    NoOptimizer,
    Optimizer,
)


@dataclass
class OptimizerConfig(ABC):
    num_grasps: int = 32

    @abstractmethod
    def create(self, evaluator: Evaluator) -> Optimizer:
        raise NotImplementedError


@dataclass
class BpsRandomSamplingOptimizerConfig(OptimizerConfig):
    num_steps: int = 100
    trans_noise: float = 0.005
    rot_noise: float = 0.05
    joint_angle_noise: float = 0.01
    grasp_orientation_noise: float = 0.05

    def create(self, evaluator: Evaluator) -> Optimizer:
        assert isinstance(evaluator, BpsEvaluator), f"{evaluator} is not a BpsEvaluator"
        return BpsRandomSamplingOptimizer(
            bps_evaluator=evaluator,
            num_grasps=self.num_grasps,
            num_steps=self.num_steps,
            trans_noise=self.trans_noise,
            rot_noise=self.rot_noise,
            joint_angle_noise=self.joint_angle_noise,
            grasp_orientation_noise=self.grasp_orientation_noise,
        )


@dataclass
class NerfRandomSamplingOptimizerConfig(OptimizerConfig):
    num_steps: int = 100
    trans_noise: float = 0.005
    rot_noise: float = 0.05
    joint_angle_noise: float = 0.1
    grasp_orientation_noise: float = 0.05

    def create(self, evaluator: Evaluator) -> Optimizer:
        assert isinstance(
            evaluator, NerfEvaluator
        ), f"{evaluator} is not a NerfEvaluator"
        return NerfRandomSamplingOptimizer(
            nerf_evaluator=evaluator,
            num_grasps=self.num_grasps,
            num_steps=self.num_steps,
            trans_noise=self.trans_noise,
            rot_noise=self.rot_noise,
            joint_angle_noise=self.joint_angle_noise,
            grasp_orientation_noise=self.grasp_orientation_noise,
        )


@dataclass
class NerfGradientOptimizerConfig(OptimizerConfig):
    num_steps: int = 100
    finger_lr: float = 1e-4
    grasp_dir_lr: float = 1e-4
    wrist_lr: float = 1e-4
    momentum: float = 0.9
    opt_fingers: bool = True
    opt_grasp_dirs: bool = True
    opt_wrist_pose: bool = True
    use_adamw: bool = True

    def create(self, evaluator: Evaluator) -> Optimizer:
        assert isinstance(
            evaluator, NerfEvaluator
        ), f"{evaluator} is not a NerfEvaluator"
        return NerfGradientOptimizer(
            nerf_evaluator=evaluator,
            num_grasps=self.num_grasps,
            num_steps=self.num_steps,
            finger_lr=self.finger_lr,
            grasp_dir_lr=self.grasp_dir_lr,
            wrist_lr=self.wrist_lr,
            momentum=self.momentum,
            opt_fingers=self.opt_fingers,
            opt_grasp_dirs=self.opt_grasp_dirs,
            opt_wrist_pose=self.opt_wrist_pose,
            use_adamw=self.use_adamw,
        )


@dataclass
class NoOptimizerConfig(OptimizerConfig):
    def create(self, evaluator: Evaluator) -> Optimizer:
        return NoOptimizer(num_grasps=self.num_grasps)


OptimizerConfigUnion = Union[
    BpsRandomSamplingOptimizerConfig,
    NerfRandomSamplingOptimizerConfig,
    NerfGradientOptimizerConfig,
    NoOptimizerConfig,
]
