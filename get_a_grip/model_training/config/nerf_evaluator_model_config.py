from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Literal, Optional, Tuple, Union

import tyro

from get_a_grip import get_data_folder
from get_a_grip.model_training.config.datetime_str import get_datetime_str
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.config.nerf_grasp_dataset_config import (
    NerfGraspDatasetConfig,
)
from get_a_grip.model_training.config.wandb_config import WandbConfig
from get_a_grip.model_training.models.nerf_evaluator_model import (
    CnnXyzGlobalCnnNerfEvaluatorModel,
    CnnXyzNerfEvaluatorModel,
    NerfEvaluatorModel,
)

DEFAULT_WANDB_PROJECT = "nerf_evaluator_model"


class TaskType(Enum):
    """Enum for task type."""

    Y_PICK = auto()
    Y_COLL = auto()
    Y_PGS = auto()
    Y_PICK_AND_Y_COLL = auto()
    Y_PICK_AND_Y_COLL_AND_Y_PGS = auto()

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    @property
    def task_names(self) -> List[str]:
        if self == TaskType.Y_PICK:
            return ["y_pick"]
        elif self == TaskType.Y_COLL:
            return ["y_coll"]
        elif self == TaskType.Y_PGS:
            return ["y_PGS"]
        elif self == TaskType.Y_PICK_AND_Y_COLL:
            return [
                "y_pick",
                "y_coll",
            ]
        elif self == TaskType.Y_PICK_AND_Y_COLL_AND_Y_PGS:
            return [
                "y_pick",
                "y_coll",
                "y_PGS",
            ]
        else:
            raise ValueError(f"Unknown task_type: {self}")


@dataclass(frozen=True)
class DataConfig:
    """Parameters for dataset loading."""

    frac_val: float = 0.1
    frac_test: float = 0.1
    frac_train: float = 1 - frac_val - frac_test

    max_num_data_points: Optional[int] = None
    """Maximum number of data points to use from the dataset. If None, use all."""

    use_random_rotations: bool = True
    """Flag to add random rotations to augment the dataset."""

    debug_shuffle_labels: bool = False
    """Flag to randomize all the labels to see what memorization looks like."""


@dataclass(frozen=True)
class DataLoaderConfig:
    """Parameters for dataloader."""

    batch_size: int = 128

    num_workers: int = 8
    """Number of workers for the dataloader."""

    pin_memory: bool = True
    """Flag to pin memory for the dataloader."""


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for training."""

    grad_clip_val: float = 1.0
    """Maximimum value of the gradient norm."""

    lr: float = 1e-4
    """Learning rate."""

    weight_decay: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    """Adam optimizer parameters."""

    label_smoothing: float = 0.0
    """Cross entropy loss label smoothing"""

    lr_scheduler_name: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant_with_warmup",
        "piecewise_constant",
    ] = "constant"
    """Strategy for learning rate scheduling."""

    lr_scheduler_num_warmup_steps: int = 0
    """(if applicable) number of warmup steps for learning rate scheduling."""

    n_epochs: int = 1000
    """Number of epochs to train for."""

    val_freq: int = 5
    """Number of iterations between validation steps."""

    val_on_epoch_0: bool = False
    """Flag to run validation on epoch 0."""

    save_checkpoint_freq: int = 5
    """Number of iterations between saving checkpoints."""

    save_checkpoint_on_epoch_0: bool = False
    """Flag to save checkpoint on epoch 0."""

    loss_fn: Literal[
        "cross_entropy",
        "l1",
        "l2",
    ] = "l1"


@dataclass(frozen=True)
class CheckpointWorkspaceConfig:
    """Parameters for paths to checkpoints."""

    input_dir: Optional[pathlib.Path] = None
    """Input directory to LOAD a checkpoint and potentially resume a run."""

    output_dir: pathlib.Path = (
        get_data_folder() / "models/NEW/nerf_evaluator_model" / get_datetime_str()
    )
    """Output directory to SAVE checkpoints and run information."""

    @property
    def input_checkpoint_paths(self) -> List[pathlib.Path]:
        return self.checkpoint_paths(self.input_dir)

    @property
    def latest_input_checkpoint_path(self) -> Optional[pathlib.Path]:
        """Path to the latest checkpoint in the input directory."""
        return self.latest_checkpoint_path(self.input_dir)

    @property
    def output_checkpoint_paths(self) -> List[pathlib.Path]:
        return self.checkpoint_paths(self.output_dir)

    @property
    def latest_output_checkpoint_path(self) -> Optional[pathlib.Path]:
        """Path to the latest checkpoint in the output directory."""
        return self.latest_checkpoint_path(self.output_dir)

    @staticmethod
    def checkpoint_paths(
        checkpoint_dir: Optional[pathlib.Path],
    ) -> List[pathlib.Path]:
        if checkpoint_dir is None:
            return []

        """Get all the checkpoint paths in a directory."""
        checkpoint_filepaths = sorted(
            [x for x in checkpoint_dir.glob("*.pt")]
            + [x for x in checkpoint_dir.glob("*.pth")],
            key=lambda x: x.stat().st_mtime,
        )
        return checkpoint_filepaths

    @staticmethod
    def latest_checkpoint_path(
        checkpoint_dir: Optional[pathlib.Path],
    ) -> Optional[pathlib.Path]:
        """Path to the latest checkpoint in a directory."""
        checkpoint_filepaths = CheckpointWorkspaceConfig.checkpoint_paths(
            checkpoint_dir
        )
        if len(checkpoint_filepaths) == 0:
            print("No checkpoint found")
            return None

        if len(checkpoint_filepaths) > 1:
            print(
                f"Found multiple checkpoints: {checkpoint_filepaths}. Returning most recent one."
            )
        return checkpoint_filepaths[-1]


@dataclass(frozen=True)
class ModelConfig(ABC):
    """Default (abstract) parameters for the nerf_evaluator."""

    @abstractmethod
    def create_model(
        self,
        fingertip_config: EvenlySpacedFingertipConfig,
        n_tasks: int,
    ) -> NerfEvaluatorModel:
        """Helper method to return the correct nerf_evaluator_model from config."""
        raise NotImplementedError


@dataclass(frozen=True)
class CnnXyzModelConfig(ModelConfig):
    """Parameters for the CnnXyzNerfEvaluatorModel."""

    conv_channels: Tuple[int, ...] = (32, 64, 128)
    """List of channels for each convolutional layer. Length specifies number of layers."""

    mlp_hidden_layers: Tuple[int, ...] = (256, 256)
    """List of hidden layer sizes for the MLP. Length specifies number of layers."""

    n_fingers: int = 4
    """Number of fingers."""

    def input_shape_from_fingertip_config(
        self, fingertip_config: EvenlySpacedFingertipConfig
    ) -> Tuple[int, int, int, int]:
        n_density_channels = 1
        n_xyz_channels = 3
        return (
            n_density_channels + n_xyz_channels,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        )

    def create_model(
        self,
        fingertip_config: EvenlySpacedFingertipConfig,
        n_tasks: int,
    ) -> NerfEvaluatorModel:
        """Helper method to return the correct nerf_evaluator from config."""

        input_shape = self.input_shape_from_fingertip_config(fingertip_config)
        return CnnXyzNerfEvaluatorModel(
            input_shape=input_shape,
            n_fingers=fingertip_config.n_fingers,
            n_tasks=n_tasks,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


@dataclass(frozen=True)
class CnnXyzGlobalCnnModelConfig(ModelConfig):
    """Parameters for the CnnXyzGlobalCnnNerfEvaluatorModel."""

    conv_channels: Tuple[int, ...] = (32, 64, 128)
    """List of channels for each convolutional layer. Length specifies number of layers."""

    mlp_hidden_layers: Tuple[int, ...] = (256, 256)
    """List of hidden layer sizes for the MLP. Length specifies number of layers."""

    global_conv_channels: Tuple[int, ...] = (32, 64, 128)
    """List of channels for each convolutional layer for the global CNN. Length specifies number of layers."""

    n_fingers: int = 4
    """Number of fingers."""

    def input_shape_from_fingertip_config(
        self, fingertip_config: EvenlySpacedFingertipConfig
    ) -> Tuple[int, int, int, int]:
        n_density_channels = 1
        n_xyz_channels = 3
        return (
            n_density_channels + n_xyz_channels,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        )

    def global_input_shape(self) -> Tuple[int, int, int, int]:
        n_density_channels = 1
        n_xyz_channels = 3
        return (
            n_density_channels + n_xyz_channels,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )

    def create_model(
        self,
        fingertip_config: EvenlySpacedFingertipConfig,
        n_tasks: int,
    ) -> NerfEvaluatorModel:
        """Helper method to return the correct nerf_evaluator from config."""

        input_shape = self.input_shape_from_fingertip_config(fingertip_config)
        global_input_shape = self.global_input_shape()
        return CnnXyzGlobalCnnNerfEvaluatorModel(
            input_shape=input_shape,
            n_fingers=fingertip_config.n_fingers,
            n_tasks=n_tasks,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
            global_input_shape=global_input_shape,
            global_conv_channels=self.global_conv_channels,
        )


@dataclass
class PlotConfig:
    """Parameters for plotting."""

    scatter_predicted_vs_actual: bool = False
    """Flag to plot predicted vs actual scatter plot."""

    histogram_predictions: bool = False
    """Flag to plot histogram of predictions."""

    confusion_matrices: bool = False
    """Flag to plot confusion matrices."""

    batch_data: bool = False
    """Flag to plot batch data."""


@dataclass
class NerfEvaluatorModelConfig:
    model_config: Union[CnnXyzModelConfig, CnnXyzGlobalCnnModelConfig] = field(
        default_factory=CnnXyzGlobalCnnModelConfig
    )
    nerf_grasp_dataset_config: NerfGraspDatasetConfig = field(
        default_factory=NerfGraspDatasetConfig
    )
    nerf_grasp_dataset_config_path: Optional[pathlib.Path] = None
    train_dataset_path: Optional[pathlib.Path] = None
    val_dataset_path: Optional[pathlib.Path] = None
    test_dataset_path: Optional[pathlib.Path] = None
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint_workspace: CheckpointWorkspaceConfig = field(
        default_factory=CheckpointWorkspaceConfig
    )
    task_type: TaskType = TaskType.Y_PICK_AND_Y_COLL_AND_Y_PGS
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(project=DEFAULT_WANDB_PROJECT)
    )
    plot: PlotConfig = field(default_factory=PlotConfig)

    random_seed: int = 42

    def __post_init__(self):
        """
        If a nerf grasp dataset config path was passed, load that config object.
        Otherwise use defaults.
        """
        if self.nerf_grasp_dataset_config_path is not None:
            print(f"Loading nerfdata config from {self.nerf_grasp_dataset_config_path}")
            self.nerf_grasp_dataset_config = tyro.extras.from_yaml(
                type(self.nerf_grasp_dataset_config),
                self.nerf_grasp_dataset_config_path.open(),
            )

        assert (
            (self.val_dataset_path is None and self.test_dataset_path is None)
            or (
                self.val_dataset_path is not None and self.test_dataset_path is not None
            )
        ), f"Must specify both val and test dataset paths, or neither. Got val: {self.val_dataset_path}, test: {self.test_dataset_path}"

    @property
    def actual_train_dataset_path(self) -> pathlib.Path:
        if self.train_dataset_path is None:
            assert self.nerf_grasp_dataset_config.output_filepath is not None
            return self.nerf_grasp_dataset_config.output_filepath
        return self.train_dataset_path

    @property
    def create_val_test_from_train(self) -> bool:
        return self.val_dataset_path is None and self.test_dataset_path is None

    @property
    def actual_val_dataset_path(self) -> pathlib.Path:
        if self.val_dataset_path is None:
            raise ValueError("Must specify val dataset filepath")
        return self.val_dataset_path

    @property
    def actual_test_dataset_path(self) -> pathlib.Path:
        if self.test_dataset_path is None:
            raise ValueError("Must specify test dataset filepath")
        return self.test_dataset_path


if __name__ == "__main__":
    cfg = tyro.cli(tyro.conf.FlagConversionOff[NerfEvaluatorModelConfig])
    print(cfg)
