from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import tyro

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)
from get_a_grip.model_training.models.bps_evaluator_model import BpsEvaluatorModel
from get_a_grip.model_training.models.nerf_evaluator_model import (
    NerfEvaluatorModel,
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
    nerf_evaluator_model_config: Optional[NerfEvaluatorModelConfig] = None
    nerf_evaluator_model_config_path: Optional[pathlib.Path] = None
    nerf_evaluator_model_checkpoint: int = -1  # Load latest checkpoint if -1.
    eval_batch_size: int = 32

    def __post_init__(self) -> None:
        # If nerf_evaluator config is None, load nerf_evaluator config from file
        if self.nerf_evaluator_model_config is None:
            assert (
                self.nerf_evaluator_model_config_path is not None
            ), "Both nerf_evaluator_model_config and nerf_evaluator_model_config_path are None."
            assert (
                self.nerf_evaluator_model_config_path.exists()
            ), f"{self.nerf_evaluator_model_config_path} does not exist"
            assert (
                self.nerf_evaluator_model_config_path.suffix in [".yaml", ".yml"]
            ), f"Expected .yaml or .yml, got {self.nerf_evaluator_model_config_path.suffix}"
            print(
                f"Loading nerf_evaluator config from {self.nerf_evaluator_model_config_path}"
            )
            self.nerf_evaluator_model_config = tyro.extras.from_yaml(
                type(self.nerf_evaluator_model_config),
                self.nerf_evaluator_model_config_path.open(),
            )
        elif self.nerf_evaluator_model_config_path is not None:
            print(
                "WARNING: Both nerf_evaluator_model_config and nerf_evaluator_model_config_path are provided, using nerf_evaluator_model_config."
            )

    def create(self, nerf_input: NerfInput) -> Evaluator:
        assert (
            self.nerf_evaluator_model_config is not None
        ), "nerf_evaluator_model_config is None"
        nerf_evaluator_model = load_nerf_evaluator_model(
            nerf_evaluator_model_config=self.nerf_evaluator_model_config,
            nerf_evaluator_model_checkpoint=self.nerf_evaluator_model_checkpoint,
        )
        return NerfEvaluator(
            nerf_input=nerf_input,
            nerf_evaluator_model=nerf_evaluator_model,
            fingertip_config=self.nerf_evaluator_model_config.nerf_grasp_dataset_config.fingertip_config,
        )


@dataclass
class NoEvaluatorConfig(EvaluatorConfig):
    def create(self, nerf_input: NerfInput) -> Evaluator:
        return NoEvaluator()


EvaluatorConfigUnion = Union[BpsEvaluatorConfig, NerfEvaluatorConfig, NoEvaluatorConfig]


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        """Returns losses where lower is better"""
        raise NotImplementedError

    def sort(self, grasp_config: AllegroGraspConfig) -> AllegroGraspConfig:
        B = len(grasp_config)
        losses = self.evaluate(grasp_config)

        assert losses.shape == (
            B,
        ), f"Expected losses.shape == ({B},), got {losses.shape}"

        sorted_idxs = torch.argsort(losses)
        _sorted_losses = losses[sorted_idxs]
        sorted_grasp_config = grasp_config[sorted_idxs]
        assert (
            len(sorted_grasp_config) == B
        ), f"Expected {B}, got {len(sorted_grasp_config)}"
        return sorted_grasp_config


class NerfEvaluator(nn.Module, Evaluator):
    def __init__(
        self,
        nerf_input: NerfInput,
        nerf_evaluator_model: NerfEvaluatorModel,
        fingertip_config: EvenlySpacedFingertipConfig,
    ) -> None:
        super().__init__()
        self.nerf_input = nerf_input
        self.nerf_evaluator_model = nerf_evaluator_model
        self.fingertip_config = fingertip_config

    def forward(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        batch_data_input = self.nerf_input.compute_batch_data_input(
            grasp_frame_transforms=grasp_config.grasp_frame_transforms,
            grasp_config_tensor=grasp_config.as_tensor(),
            fingertip_config=self.fingertip_config,
        )

        return self.nerf_evaluator_model.get_failure_probability(batch_data_input)

    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        return self(grasp_config)


def load_nerf_evaluator_model(
    nerf_evaluator_model_config: NerfEvaluatorModelConfig,
    nerf_evaluator_model_checkpoint: int = -1,
) -> NerfEvaluatorModel:
    # Load nerf_evaluator_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf_evaluator_model = (
        nerf_evaluator_model_config.model_config.create_model(
            fingertip_config=nerf_evaluator_model_config.nerf_grasp_dataset_config.fingertip_config,
            n_tasks=nerf_evaluator_model_config.task_type.n_tasks,
        )
    ).to(device)

    # Load nerf_evaluator_model weights
    assert nerf_evaluator_model_config.checkpoint_workspace.output_dir.exists(), f"checkpoint_workspace.output_dir does not exist at {nerf_evaluator_model_config.checkpoint_workspace.output_dir}"
    print(
        f"Loading checkpoint ({nerf_evaluator_model_config.checkpoint_workspace.output_dir})..."
    )

    output_checkpoint_paths = (
        nerf_evaluator_model_config.checkpoint_workspace.output_checkpoint_paths
    )
    assert (
        len(output_checkpoint_paths) > 0
    ), f"No checkpoints found in {nerf_evaluator_model_config.checkpoint_workspace.output_checkpoint_paths}"
    assert (
        nerf_evaluator_model_checkpoint < len(output_checkpoint_paths)
    ), f"Requested checkpoint {nerf_evaluator_model_checkpoint} does not exist in {nerf_evaluator_model_config.checkpoint_workspace.output_checkpoint_paths}"
    checkpoint_path = output_checkpoint_paths[nerf_evaluator_model_checkpoint]

    checkpoint = torch.load(checkpoint_path)
    nerf_evaluator_model.load_state_dict(checkpoint["nerf_evaluator_model"])

    return nerf_evaluator_model


class BpsEvaluator(Evaluator):
    def __init__(self, bps_values: torch.Tensor, ckpt_path: Path) -> None:
        # Save BPS values
        assert bps_values.shape == (
            4096,
        ), f"Expected shape (4096,), got {bps_values.shape}"
        self.bps_values = bps_values

        self.bps_evaluator = BpsEvaluatorModel(
            in_grasp=3 + 6 + 16 + 4 * 3, in_bps=4096
        ).to(bps_values.device)
        self.bps_evaluator.eval()
        self.bps_evaluator.load_state_dict(
            torch.load(ckpt_path, map_location=bps_values.device)
        )

    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        return self.evaluate_grasps(grasp_config.as_grasp())

    def evaluate_grasps(self, grasps: torch.Tensor) -> torch.Tensor:
        B = grasps.shape[0]

        # Repeat input B times
        # Use cache if appropriate
        if (
            not hasattr(self, "_bps_values_repeated_cached")
            or self._bps_values_repeated_cached.shape[0] != B
        ):
            # Repeat BPS values
            self._bps_values_repeated_cached = self.bps_values.unsqueeze(
                dim=0
            ).repeat_interleave(B, dim=0)
        bps_values_repeated = self._bps_values_repeated_cached

        assert bps_values_repeated.shape == (
            B,
            4096,
        ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

        # Evaluate
        g_O = grasps
        assert g_O.shape == (B, 3 + 6 + 16 + 4 * 3)

        f_O = bps_values_repeated
        assert f_O.shape == (B, 4096)

        success_preds = self.bps_evaluator(f_O=f_O, g_O=g_O)[:, -1]
        assert success_preds.shape == (
            B,
        ), f"success_preds.shape = {success_preds.shape}, expected ({B},)"

        losses = 1 - success_preds
        return losses


class NoEvaluator(Evaluator):
    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        return torch.linspace(0, 0.001, len(grasp_config))
