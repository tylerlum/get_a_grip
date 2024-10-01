from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

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


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        """Returns losses where lower is better"""
        raise NotImplementedError

    @torch.no_grad()
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
        nerf_evaluator_model_config: NerfEvaluatorModelConfig,
        nerf_evaluator_model_checkpoint: int,
        fingertip_config: EvenlySpacedFingertipConfig,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nerf_input = nerf_input
        self.nerf_evaluator_model = load_nerf_evaluator_model(
            nerf_evaluator_model_config=nerf_evaluator_model_config,
            nerf_evaluator_model_checkpoint=nerf_evaluator_model_checkpoint,
        ).to(self.device)
        self.fingertip_config = fingertip_config

    def forward(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        batch_data_input = self.nerf_input.compute_batch_data_input(
            grasp_config=grasp_config,
            fingertip_config=self.fingertip_config,
        ).to(self.device)

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
    nerf_evaluator_model.eval()
    nerf_evaluator_model.load_state_dict(checkpoint["nerf_evaluator_model"])

    return nerf_evaluator_model


class BpsEvaluator(Evaluator):
    def __init__(self, bps_values: torch.Tensor, ckpt_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save BPS values
        assert bps_values.shape == (
            4096,
        ), f"Expected shape (4096,), got {bps_values.shape}"
        self.bps_values = bps_values

        self.bps_evaluator = BpsEvaluatorModel(
            in_grasp=3 + 6 + 16 + 4 * 3, in_bps=4096
        ).to(self.device)
        self.bps_evaluator.eval()
        self.bps_evaluator.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
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
        g_O = grasps.to(self.device)
        assert g_O.shape == (B, 3 + 6 + 16 + 4 * 3)

        f_O = bps_values_repeated.to(self.device)
        assert f_O.shape == (B, 4096)

        success_preds = self.bps_evaluator(f_O=f_O, g_O=g_O)[:, -1]
        assert success_preds.shape == (
            B,
        ), f"success_preds.shape = {success_preds.shape}, expected ({B},)"

        losses = 1 - success_preds
        return losses


class NoEvaluator(Evaluator):
    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        return torch.linspace(0, 0.001, len(grasp_config), device=grasp_config.device)
