import pypose as pp
import torch
import torch.nn as nn
import nerf_grasping
import pathlib
from nerf_grasping.models.dexgraspnet_models import (
    CNN_3D_Model,
    CNN_3D_CNN_3D_Model,
    CNN_3D_MLP_Model,
    CNN_2D_1D_Model,
    Simple_CNN_2D_1D_Model,
    Simple_CNN_1D_2D_Model,
    Simple_CNN_LSTM_Model,
    DepthImage_CNN_2D_Model,
    ResnetType2d,
    ConvOutputTo1D,
)
from nerf_grasping.learned_metric.DexGraspNet_batch_data import (
    BatchDataInput,
    DepthImageBatchDataInput,
    ConditioningType,
)
from typing import Iterable, Tuple, List
from enum import Enum, auto


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class Classifier(nn.Module):
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
        # for overfit classifier.
        PROB_SCALING = 1e0

        # TODO: Consider scaling differently for each task
        task_probs = nn.functional.softmax(PROB_SCALING * all_logits, dim=-1)
        passed_task_probs = task_probs[..., -1]
        assert_equals(passed_task_probs.shape, (batch_data_input.batch_size, n_tasks))

        # HACK: Modify to either be product or not
        # BRITTLE IDXING
        MODE = "PASSED_EVAL"
        if MODE == "PRODUCT":
            passed_all_probs = torch.prod(passed_task_probs, dim=-1)
        elif MODE == "PASSED_SIM":
            PASSED_SIM_IDX = 0
            passed_all_probs = passed_task_probs[:, PASSED_SIM_IDX]
        elif MODE == "PASSED_EVAL":
            PASSED_EVAL_IDX = -1
            passed_all_probs = passed_task_probs[:, PASSED_EVAL_IDX]
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


class CNN_3D_XYZY_Classifier(Classifier):
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
            batch_data_input.nerf_alphas_with_augmented_coords_v3
        )
        return all_logits


class CNN_3D_XYZXYZY_Classifier(Classifier):
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
            batch_data_input.nerf_alphas_with_augmented_coords_v2
        )
        return all_logits


class CNN_3D_XYZXYZ_Classifier(Classifier):
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
            batch_data_input.nerf_alphas_with_augmented_coords_v4
        )
        return all_logits


class CNN_3D_XYZ_Classifier(Classifier):
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


class CNN_3D_XYZ_Global_CNN_Classifier(Classifier):
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

class CNN_3D_XYZ_Global_CNN_Cropped_Classifier(Classifier):
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
            batch_data_input.nerf_alphas_global_cropped_with_augmented_coords,
        )
        return all_logits


class CNN_3D_XYZ_Global_MLP_Classifier(Classifier):
    def __init__(
        self,
        input_shape: Iterable[int],
        conv_channels: Iterable[int],
        mlp_hidden_layers: Iterable[int],
        global_input_shape: Iterable[int],
        global_mlp_hidden_layers: Iterable[int],
        n_fingers: int,
        n_tasks: int,
    ) -> None:
        super().__init__()
        self.model = CNN_3D_MLP_Model(
            input_shape=input_shape,
            conv_channels=conv_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            global_input_shape=global_input_shape,
            global_mlp_hidden_layers=global_mlp_hidden_layers,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas_with_augmented_coords,
            batch_data_input.nerf_alphas_global.reshape(
                batch_data_input.batch_size, -1
            ),
        )
        return all_logits


class CNN_2D_1D_Classifier(Classifier):
    def __init__(
        self,
        conditioning_type: ConditioningType,
        grid_shape: Tuple[int, int, int],
        n_fingers: int,
        n_tasks: int,
        conv_2d_film_hidden_layers: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        use_pretrained_2d: bool,
        resnet_type_2d: ResnetType2d,
        pooling_method_2d: ConvOutputTo1D,
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning_type
        self.model = CNN_2D_1D_Model(
            grid_shape=grid_shape,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
            conditioning_dim=conditioning_type.dim,
            conv_2d_film_hidden_layers=conv_2d_film_hidden_layers,
            mlp_hidden_layers=mlp_hidden_layers,
            use_pretrained_2d=use_pretrained_2d,
            resnet_type_2d=resnet_type_2d,
            pooling_method_2d=pooling_method_2d,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas,
            batch_data_input.get_conditioning(conditioning_type=self.conditioning_type),
        )

        return all_logits


class Simple_CNN_2D_1D_Classifier(Classifier):
    def __init__(
        self,
        conditioning_type: ConditioningType,
        grid_shape: Tuple[int, int, int],
        n_fingers: int,
        n_tasks: int,
        mlp_hidden_layers: List[int] = [32, 32],
        conv_2d_channels: List[int] = [32, 16, 8, 4],
        conv_1d_channels: List[int] = [4, 8],
        film_2d_hidden_layers: List[int] = [8, 8],
        film_1d_hidden_layers: List[int] = [8, 8],
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning_type
        self.model = Simple_CNN_2D_1D_Model(
            grid_shape=grid_shape,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
            conditioning_dim=conditioning_type.dim,
            mlp_hidden_layers=mlp_hidden_layers,
            conv_2d_channels=conv_2d_channels,
            conv_1d_channels=conv_1d_channels,
            film_2d_hidden_layers=film_2d_hidden_layers,
            film_1d_hidden_layers=film_1d_hidden_layers,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas,
            batch_data_input.get_conditioning(conditioning_type=self.conditioning_type),
        )
        return all_logits


class Simple_CNN_1D_2D_Classifier(Classifier):
    def __init__(
        self,
        conditioning_type: ConditioningType,
        grid_shape: Tuple[int, int, int],
        n_fingers: int,
        n_tasks: int,
        mlp_hidden_layers: List[int] = [32, 32],
        conv_2d_channels: List[int] = [32, 16, 8, 4],
        conv_1d_channels: List[int] = [4, 8],
        film_2d_hidden_layers: List[int] = [8, 8],
        film_1d_hidden_layers: List[int] = [8, 8],
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning_type
        self.model = Simple_CNN_1D_2D_Model(
            grid_shape=grid_shape,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
            conditioning_dim=conditioning_type.dim,
            mlp_hidden_layers=mlp_hidden_layers,
            conv_2d_channels=conv_2d_channels,
            conv_1d_channels=conv_1d_channels,
            film_2d_hidden_layers=film_2d_hidden_layers,
            film_1d_hidden_layers=film_1d_hidden_layers,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas,
            batch_data_input.get_conditioning(conditioning_type=self.conditioning_type),
        )
        return all_logits


class Simple_CNN_LSTM_Classifier(Classifier):
    def __init__(
        self,
        conditioning_type: ConditioningType,
        grid_shape: Tuple[int, int, int],
        n_fingers: int,
        n_tasks: int,
        mlp_hidden_layers: List[int] = [32, 32],
        conv_2d_channels: List[int] = [32, 16, 8, 4],
        film_2d_hidden_layers: List[int] = [8, 8],
        lstm_hidden_size: int = 32,
        num_lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning_type
        self.model = Simple_CNN_LSTM_Model(
            grid_shape=grid_shape,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
            conditioning_dim=conditioning_type.dim,
            mlp_hidden_layers=mlp_hidden_layers,
            conv_2d_channels=conv_2d_channels,
            film_2d_hidden_layers=film_2d_hidden_layers,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.nerf_alphas,
            batch_data_input.get_conditioning(conditioning_type=self.conditioning_type),
        )
        return all_logits


class DepthImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch_data_input: DepthImageBatchDataInput) -> torch.Tensor:
        raise NotImplementedError

    def get_failure_probability(
        self, batch_data_input: DepthImageBatchDataInput
    ) -> torch.Tensor:
        all_logits = self.get_all_logits(batch_data_input)
        assert_equals(len(all_logits.shape), 3)

        n_tasks = all_logits.shape[1]
        assert_equals(
            all_logits.shape, (batch_data_input.batch_size, n_tasks, self.n_classes)
        )

        # REMOVE, using to ensure gradients are non-zero
        # for overfit classifier.
        PROB_SCALING = 1e0

        # TODO: Consider scaling differently for each task
        task_probs = nn.functional.softmax(PROB_SCALING * all_logits, dim=-1)
        passed_task_probs = task_probs[..., -1]
        assert_equals(passed_task_probs.shape, (batch_data_input.batch_size, n_tasks))
        passed_all_probs = torch.prod(passed_task_probs, dim=-1)
        assert_equals(passed_all_probs.shape, (batch_data_input.batch_size,))

        # Return failure probabilities (as loss).
        return 1.0 - passed_all_probs

    def get_all_logits(
        self, batch_data_input: DepthImageBatchDataInput
    ) -> torch.Tensor:
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


class DepthImage_CNN_2D_Classifier(DepthImageClassifier):
    def __init__(
        self,
        conditioning_type: ConditioningType,
        img_shape: Tuple[int, int, int],
        n_fingers: int,
        n_tasks: int,
        conv_2d_film_hidden_layers: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        use_pretrained_2d: bool,
        resnet_type_2d: ResnetType2d,
        pooling_method_2d: ConvOutputTo1D,
    ) -> None:
        super().__init__()
        self.conditioning_type = conditioning_type

        self.model = DepthImage_CNN_2D_Model(
            img_shape=img_shape,
            n_fingers=n_fingers,
            n_tasks=n_tasks,
            conditioning_dim=conditioning_type.dim,
            conv_2d_film_hidden_layers=conv_2d_film_hidden_layers,
            mlp_hidden_layers=mlp_hidden_layers,
            use_pretrained_2d=use_pretrained_2d,
            resnet_type_2d=resnet_type_2d,
            pooling_method_2d=pooling_method_2d,
        )

    def forward(self, batch_data_input: DepthImageBatchDataInput) -> torch.Tensor:
        # Run model
        all_logits = self.model.get_all_logits(
            batch_data_input.depth_uncertainty_images,
            batch_data_input.get_conditioning(conditioning_type=self.conditioning_type),
        )

        return all_logits


def main() -> None:
    from nerf_grasping.config.fingertip_config import (
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
        BATCH_SIZE, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
    ).to(DEVICE)
    grasp_transforms = pp.identity_SE3(BATCH_SIZE, NUM_FINGERS).to(DEVICE)

    batch_data_input = BatchDataInput(
        nerf_densities=nerf_densities,
        grasp_transforms=grasp_transforms,
        fingertip_config=EvenlySpacedFingertipConfig(),
    ).to(DEVICE)

    # Create model
    cnn_3d_classifier = CNN_3D_XYZ_Classifier(
        input_shape=batch_data_input.nerf_alphas_with_augmented_coords.shape[-4:],
        conv_channels=[32, 64, 128],
        mlp_hidden_layers=[256, 256],
        n_fingers=NUM_FINGERS,
        n_tasks=N_TASKS,
    ).to(DEVICE)
    cnn_2d_1d_classifier = CNN_2D_1D_Classifier(
        conditioning_type=ConditioningType.GRASP_TRANSFORM,
        grid_shape=(NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
        n_fingers=NUM_FINGERS,
        n_tasks=N_TASKS,
        conv_2d_film_hidden_layers=(256, 256),
        mlp_hidden_layers=(256, 256),
        use_pretrained_2d=True,
        resnet_type_2d=ResnetType2d.RESNET18,
        pooling_method_2d=ConvOutputTo1D.AVG_POOL_SPATIAL,
    ).to(DEVICE)

    simple_cnn_2d_1d_classifier = Simple_CNN_2D_1D_Classifier(
        conditioning_type=ConditioningType.GRASP_TRANSFORM,
        grid_shape=(NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
        n_fingers=NUM_FINGERS,
        n_tasks=N_TASKS,
        # conv_2d_film_hidden_layers=(32, 32),
        mlp_hidden_layers=(64, 64),
    ).to(DEVICE)

    # Run model
    cnn_3d_all_logits = cnn_3d_classifier.get_all_logits(batch_data_input)
    cnn_3d_scores = cnn_3d_classifier.get_failure_probability(batch_data_input)
    print(f"cnn_3d_all_logits.shape = {cnn_3d_all_logits.shape}")
    print(f"cnn_3d_scores: {cnn_3d_scores}")
    print(f"cnn_3d_scores.shape: {cnn_3d_scores.shape}" + "\n")

    cnn_2d_1d_all_logits = cnn_2d_1d_classifier.get_all_logits(batch_data_input)
    cnn_2d_1d_scores = cnn_2d_1d_classifier.get_failure_probability(batch_data_input)
    print(f"cnn_2d_1d_all_logits.shape = {cnn_2d_1d_all_logits.shape}")
    print(f"cnn_2d_1d_scores: {cnn_2d_1d_scores}")
    print(f"cnn_2d_1d_scores.shape: {cnn_2d_1d_scores.shape}" + "\n")

    simple_cnn_2d_1d_all_logits = simple_cnn_2d_1d_classifier.get_all_logits(
        batch_data_input
    )
    simple_cnn_2d_1d_scores = simple_cnn_2d_1d_classifier.get_failure_probability(
        batch_data_input
    )
    print(f"simple_cnn_2d_1d_all_logits.shape = {simple_cnn_2d_1d_all_logits.shape}")
    print(f"simple_cnn_2d_1d_scores: {simple_cnn_2d_1d_scores}")
    print(f"simple_cnn_2d_1d_scores.shape: {simple_cnn_2d_1d_scores.shape}" + "\n")


if __name__ == "__main__":
    main()
