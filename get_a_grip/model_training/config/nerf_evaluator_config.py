from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Literal, Optional, Tuple

import tyro

from get_a_grip import get_data_folder
from get_a_grip.model_training.config.base import CONFIG_DATETIME_STR, WandbConfig
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.config.nerfdata_config import (
    GridNerfDataConfig,
)
from get_a_grip.model_training.models.nerf_evaluator import (
    CNN_3D_XYZ_Global_CNN_NerfEvaluator,
    CNN_3D_XYZ_NerfEvaluator,
    NerfEvaluator,
)

DEFAULT_WANDB_PROJECT = "learned_metric"


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

    nerf_density_threshold_value: Optional[float] = None
    """Threshold used to convert nerf density values to binary 0/1 occupancy values, None for no thresholding."""


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

    root_dir: pathlib.Path = get_data_folder() / "logs/nerf_grasp_evaluator"
    """Root directory for checkpoints."""

    input_leaf_dir_name: Optional[str] = None
    """Leaf directory name to LOAD a checkpoint and potentially resume a run."""

    output_leaf_dir_name: str = CONFIG_DATETIME_STR
    """Leaf directory name to SAVE checkpoints and run information."""

    @property
    def input_dir(self) -> Optional[pathlib.Path]:
        """Input directory for checkpoints."""
        return (
            self.root_dir / self.input_leaf_dir_name
            if self.input_leaf_dir_name is not None
            else None
        )

    @property
    def output_dir(self) -> pathlib.Path:
        """Output directory for checkpoints."""
        return self.root_dir / self.output_leaf_dir_name

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
class ModelConfig:
    """Default (abstract) parameters for the nerf_evaluator."""

    def get_nerf_evaluator_from_fingertip_config(
        self,
        fingertip_config: EvenlySpacedFingertipConfig,
        n_tasks: int,
    ) -> NerfEvaluator:
        """Helper method to return the correct nerf_evaluator from config."""
        raise NotImplementedError("Implement in subclass.")


@dataclass(frozen=True)
class CNN_3D_XYZ_ModelConfig(ModelConfig):
    """Parameters for the CNN_3D_XYZ_NerfEvaluator."""

    conv_channels: List[int]
    """List of channels for each convolutional layer. Length specifies number of layers."""

    mlp_hidden_layers: List[int]
    """List of hidden layer sizes for the MLP. Length specifies number of layers."""

    n_fingers: int = 4
    """Number of fingers."""

    def input_shape_from_fingertip_config(
        self, fingertip_config: EvenlySpacedFingertipConfig
    ):
        n_density_channels = 1
        n_xyz_channels = 3
        return [
            n_density_channels + n_xyz_channels,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        ]

    def get_nerf_evaluator_from_fingertip_config(
        self,
        fingertip_config: EvenlySpacedFingertipConfig,
        n_tasks: int,
    ) -> NerfEvaluator:
        """Helper method to return the correct nerf_evaluator from config."""

        input_shape = self.input_shape_from_fingertip_config(fingertip_config)
        return CNN_3D_XYZ_NerfEvaluator(
            input_shape=input_shape,
            n_fingers=fingertip_config.n_fingers,
            n_tasks=n_tasks,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


@dataclass(frozen=True)
class CNN_3D_XYZ_Global_CNN_ModelConfig(ModelConfig):
    """Parameters for the CNN_3D_XYZ_Global_CNN_NerfEvaluator."""

    conv_channels: List[int]
    """List of channels for each convolutional layer. Length specifies number of layers."""

    mlp_hidden_layers: List[int]
    """List of hidden layer sizes for the MLP. Length specifies number of layers."""

    global_conv_channels: List[int]
    """List of channels for each convolutional layer for the global CNN. Length specifies number of layers."""

    n_fingers: int = 4
    """Number of fingers."""

    def input_shape_from_fingertip_config(
        self, fingertip_config: EvenlySpacedFingertipConfig
    ):
        n_density_channels = 1
        n_xyz_channels = 3
        return [
            n_density_channels + n_xyz_channels,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        ]

    def global_input_shape(self):
        n_density_channels = 1
        n_xyz_channels = 3
        return (
            n_density_channels + n_xyz_channels,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )

    def get_nerf_evaluator_from_fingertip_config(
        self,
        fingertip_config: EvenlySpacedFingertipConfig,
        n_tasks: int,
    ) -> NerfEvaluator:
        """Helper method to return the correct nerf_evaluator from config."""

        input_shape = self.input_shape_from_fingertip_config(fingertip_config)
        global_input_shape = self.global_input_shape()
        return CNN_3D_XYZ_Global_CNN_NerfEvaluator(
            input_shape=input_shape,
            n_fingers=fingertip_config.n_fingers,
            n_tasks=n_tasks,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
            global_input_shape=global_input_shape,
            global_conv_channels=self.global_conv_channels,
        )


DEFAULT_CNN_3D_XYZ_ModelConfig = CNN_3D_XYZ_ModelConfig(
    conv_channels=[32, 64, 128], mlp_hidden_layers=[256, 256]
)
DEFAULT_CNN_3D_XYZ_Global_CNN_ModelConfig = CNN_3D_XYZ_Global_CNN_ModelConfig(
    conv_channels=[32, 64, 128],
    mlp_hidden_layers=[256, 256],
    global_conv_channels=[32, 64, 128],
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
class NerfEvaluatorConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: DEFAULT_CNN_3D_XYZ_Global_CNN_ModelConfig
    )
    nerfdata_config: GridNerfDataConfig = field(default_factory=GridNerfDataConfig)
    nerfdata_config_path: Optional[pathlib.Path] = None
    train_dataset_path: Optional[pathlib.Path] = None
    val_dataset_path: Optional[pathlib.Path] = None
    test_dataset_path: Optional[pathlib.Path] = None
    data: DataConfig = DataConfig()
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint_workspace: CheckpointWorkspaceConfig = field(
        default_factory=CheckpointWorkspaceConfig
    )
    task_type: TaskType = TaskType.Y_PICK_AND_Y_COLL_AND_Y_PGS
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(project=DEFAULT_WANDB_PROJECT)
    )
    name: Optional[str] = None
    plot: PlotConfig = PlotConfig()

    random_seed: int = 42

    def __post_init__(self):
        """
        If a nerfdata config path was passed, load that config object.
        Otherwise use defaults.

        Then load the correct model config based on the nerfdata config.
        """
        if self.nerfdata_config_path is not None:
            print(f"Loading nerfdata config from {self.nerfdata_config_path}")
            self.nerfdata_config = tyro.extras.from_yaml(
                type(self.nerfdata_config), self.nerfdata_config_path.open()
            )

        assert (
            (self.val_dataset_path is None and self.test_dataset_path is None)
            or (
                self.val_dataset_path is not None and self.test_dataset_path is not None
            )
        ), f"Must specify both val and test dataset paths, or neither. Got val: {self.val_dataset_path}, test: {self.test_dataset_path}"

        # Set the name of the run if given
        # HACK: don't want to overwrite these if we're loading this config from a file
        #       can tell if loading by file if self.checkpoint_workspace.output_dir exists
        if self.name is not None and not self.checkpoint_workspace.output_dir.exists():
            name_with_date = f"{self.name}_{CONFIG_DATETIME_STR}"
            self.checkpoint_workspace = CheckpointWorkspaceConfig(
                output_leaf_dir_name=name_with_date
            )
            self.wandb = WandbConfig(project=DEFAULT_WANDB_PROJECT, name=name_with_date)

    @property
    def actual_train_dataset_path(self) -> pathlib.Path:
        if self.train_dataset_path is None:
            assert self.nerfdata_config.output_filepath is not None
            return self.nerfdata_config.output_filepath
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


DEFAULTS_DICT = {
    "cnn-3d-xyz": NerfEvaluatorConfig(
        model_config=DEFAULT_CNN_3D_XYZ_ModelConfig,
        nerfdata_config=GridNerfDataConfig(),
    ),
    "cnn-3d-xyz-global-cnn": NerfEvaluatorConfig(
        model_config=DEFAULT_CNN_3D_XYZ_Global_CNN_ModelConfig,
        nerfdata_config=GridNerfDataConfig(),
    ),
}

UnionNerfEvaluatorConfig = tyro.extras.subcommand_type_from_defaults(DEFAULTS_DICT)

if __name__ == "__main__":
    cfg = tyro.cli(UnionNerfEvaluatorConfig)
    print(cfg)
