from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import tyro

from get_a_grip.grasp_planning.utils.evaluator import (
    BpsEvaluator,
    Evaluator,
    NerfEvaluator,
    NoEvaluator,
)
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)


@dataclass
class EvaluatorConfig(ABC):
    @abstractmethod
    def create(self, nerf_input: NerfInput) -> Evaluator:
        raise NotImplementedError


@dataclass
class BpsEvaluatorConfig(EvaluatorConfig):
    ckpt_path: pathlib.Path

    def __post_init__(self) -> None:
        assert self.ckpt_path.exists(), f"{self.ckpt_path} does not exist"
        assert self.ckpt_path.suffix in [
            ".pt",
            ".pth",
        ], f"{self.ckpt_path} does not have a .pt or .pth suffix"

    def create(self, nerf_input: NerfInput) -> BpsEvaluator:
        bps_values = nerf_input.bps_values
        return BpsEvaluator(
            bps_values=bps_values,
            ckpt_path=self.ckpt_path,
        )


@dataclass
class NerfEvaluatorConfig(EvaluatorConfig):
    nerf_evaluator_model_config_path: Optional[pathlib.Path] = None
    nerf_evaluator_model_checkpoint: int = -1  # Load latest checkpoint if -1.

    def __post_init__(self) -> None:
        assert (
            self.nerf_evaluator_model_config_path is not None
        ), "Both nerf_evaluator_model_config and nerf_evaluator_model_config_path are None."
        assert (
            self.nerf_evaluator_model_config_path.exists()
        ), f"{self.nerf_evaluator_model_config_path} does not exist"
        assert self.nerf_evaluator_model_config_path.suffix in [
            ".yaml",
            ".yml",
        ], f"Expected .yaml or .yml, got {self.nerf_evaluator_model_config_path.suffix}"
        print(
            f"Loading nerf_evaluator config from {self.nerf_evaluator_model_config_path}"
        )
        self.nerf_evaluator_model_config = tyro.extras.from_yaml(
            NerfEvaluatorModelConfig,
            self.nerf_evaluator_model_config_path.open(),
        )

    def create(self, nerf_input: NerfInput) -> Evaluator:
        return NerfEvaluator(
            nerf_input=nerf_input,
            nerf_evaluator_model_config=self.nerf_evaluator_model_config,
            nerf_evaluator_model_checkpoint=self.nerf_evaluator_model_checkpoint,
            fingertip_config=self.nerf_evaluator_model_config.nerf_grasp_dataset_config.fingertip_config,
        )


@dataclass
class NoEvaluatorConfig(EvaluatorConfig):
    def create(self, nerf_input: NerfInput) -> Evaluator:
        return NoEvaluator()


EvaluatorConfigUnion = Union[BpsEvaluatorConfig, NerfEvaluatorConfig, NoEvaluatorConfig]
