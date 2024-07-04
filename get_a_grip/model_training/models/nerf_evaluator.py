from typing import Iterable

import pypose as pp
import torch
import torch.nn as nn

from get_a_grip.model_training.models.components.models import (
    CNN_3D_CNN_3D_Model,
    CNN_3D_Model,
)
from get_a_grip.model_training.utils.nerf_grasp_evaluator_batch_data import (
    BatchDataInput,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class NerfEvaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        raise NotImplementedError

    def get_failure_probability(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        all_logits = self.get_all_logits(batch_data_input)
        assert_equals(len(all_logits.shape), 3)

        n_tasks = all_logits.shape[1]
        assert_equals(
            all_logits.shape, (batch_data_input.batch_size, n_tasks, self.n_classes)
        )

        # REMOVE, using to ensure gradients are non-zero
        # for overfit nerf_evaluator.
        PROB_SCALING = 1e0

        # TODO: Consider scaling differently for each task
        task_probs = nn.functional.softmax(PROB_SCALING * all_logits, dim=-1)
        passed_task_probs = task_probs[..., -1]
        assert_equals(passed_task_probs.shape, (batch_data_input.batch_size, n_tasks))

        # HACK: Modify to either be product or not
        # BRITTLE IDXING
        MODE = "Y_PGS"
        if MODE == "PRODUCT":
            passed_all_probs = torch.prod(passed_task_probs, dim=-1)
        elif MODE == "Y_PICK":
            Y_PICK_IDX = 0
            passed_all_probs = passed_task_probs[:, Y_PICK_IDX]
        elif MODE == "Y_PGS":
            Y_PGS_IDX = -1
            passed_all_probs = passed_task_probs[:, Y_PGS_IDX]
        else:
            raise ValueError(f"Invalid MODE: {MODE}")
        assert_equals(passed_all_probs.shape, (batch_data_input.batch_size,))

        # Return failure probabilities (as loss).
        return 1.0 - passed_all_probs

    def get_all_logits(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        all_logits = self(batch_data_input)
        assert_equals(len(all_logits.shape), 3)

        n_tasks = all_logits.shape[1]
        assert_equals(
            all_logits.shape, (batch_data_input.batch_size, n_tasks, self.n_classes)
        )
        return all_logits

    @property
    def n_tasks(self) -> int:
        if not hasattr(self, "model"):
            raise NotImplementedError
        if not hasattr(self.model, "n_tasks"):
            raise NotImplementedError
        return self.model.n_tasks

    @property
    def n_classes(self) -> int:
        return 2


class CNN_3D_XYZ_NerfEvaluator(NerfEvaluator):
    def __init__(
        self,
        input_shape: Iterable[int],
        conv_channels: Iterable[int],
        mlp_hidden_layers: Iterable[int],
        n_fingers: int,
        n_tasks: int,
    ) -> None:
        super().__init__()
        self.model = CNN_3D_Model(
            input_shape=input_shape,
            conv_channels=conv_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas_with_augmented_coords
        )
        return all_logits


class CNN_3D_XYZ_Global_CNN_NerfEvaluator(NerfEvaluator):
    def __init__(
        self,
        input_shape: Iterable[int],
        conv_channels: Iterable[int],
        mlp_hidden_layers: Iterable[int],
        global_input_shape: Iterable[int],
        global_conv_channels: Iterable[int],
        n_fingers: int,
        n_tasks: int,
    ) -> None:
        super().__init__()
        self.model = CNN_3D_CNN_3D_Model(
            input_shape=input_shape,
            conv_channels=conv_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            global_input_shape=global_input_shape,
            global_conv_channels=global_conv_channels,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas_with_augmented_coords,
            batch_data_input.nerf_alphas_global_with_augmented_coords,
        )
        return all_logits


def main() -> None:
    from get_a_grip.model_training.config.fingertip_config import (
        EvenlySpacedFingertipConfig,
    )

    # Prepare inputs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FINGERTIP_CONFIG = EvenlySpacedFingertipConfig()
    BATCH_SIZE = 4
    NUM_FINGERS = FINGERTIP_CONFIG.n_fingers
    NUM_PTS_X = FINGERTIP_CONFIG.num_pts_x
    NUM_PTS_Y = FINGERTIP_CONFIG.num_pts_y
    NUM_PTS_Z = FINGERTIP_CONFIG.num_pts_z
    N_TASKS = 2
    assert NUM_PTS_X is not None
    assert NUM_PTS_Y is not None
    assert NUM_PTS_Z is not None
    print("=" * 80)
    print(f"DEVICE: {DEVICE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_FINGERS: {NUM_FINGERS}")
    print(f"NUM_PTS_X: {NUM_PTS_X}")
    print(f"NUM_PTS_Y: {NUM_PTS_Y}")
    print(f"NUM_PTS_Z: {NUM_PTS_Z}")
    print(f"N_TASKS: {N_TASKS}")
    print("=" * 80 + "\n")

    # Example input
    nerf_densities = torch.rand(
        BATCH_SIZE, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, device=DEVICE
    )
    grasp_transforms = pp.identity_SE3(BATCH_SIZE, NUM_FINGERS, device=DEVICE)
    grasp_configs = torch.rand(BATCH_SIZE, NUM_FINGERS, 7 + 16 + 4, device=DEVICE)

    batch_data_input = BatchDataInput(
        nerf_densities=nerf_densities,
        grasp_transforms=grasp_transforms,
        fingertip_config=EvenlySpacedFingertipConfig(),
        grasp_configs=grasp_configs,
        nerf_densities_global=None,
    ).to(DEVICE)

    # Create model
    cnn_3d_nerf_evaluator = CNN_3D_XYZ_NerfEvaluator(
        input_shape=batch_data_input.nerf_alphas_with_augmented_coords.shape[-4:],
        conv_channels=[32, 64, 128],
        mlp_hidden_layers=[256, 256],
        n_fingers=NUM_FINGERS,
        n_tasks=N_TASKS,
    ).to(DEVICE)

    # Run model
    cnn_3d_all_logits = cnn_3d_nerf_evaluator.get_all_logits(batch_data_input)
    cnn_3d_scores = cnn_3d_nerf_evaluator.get_failure_probability(batch_data_input)
    print(f"cnn_3d_all_logits.shape = {cnn_3d_all_logits.shape}")
    print(f"cnn_3d_scores: {cnn_3d_scores}")
    print(f"cnn_3d_scores.shape: {cnn_3d_scores.shape}" + "\n")


if __name__ == "__main__":
    main()
