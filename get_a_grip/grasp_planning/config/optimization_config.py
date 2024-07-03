import pathlib
from dataclasses import dataclass, field
from typing import Optional, Union

import tyro

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.config.grasp_metric_config import (
    GraspMetricConfig,
)
from get_a_grip.grasp_planning.config.optimizer_config import (
    CEMOptimizerConfig,
    SGDOptimizerConfig,
)
from get_a_grip.model_training.config.base import WandbConfig

DEFAULT_WANDB_PROJECT = "optimize_metric"


@dataclass
class OptimizationConfig:
    """Top-level config for optimizing grasp metric."""

    optimizer: Union[SGDOptimizerConfig, CEMOptimizerConfig]
    grasp_metric: GraspMetricConfig = field(default_factory=GraspMetricConfig)
    init_grasp_config_dict_path: pathlib.Path = (
        get_data_folder()
        / "2023-01-03_mugs_smaller0-075_noise_lightshake_mid_opt"
        / "evaled_grasp_config_dicts"
        / "core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy"
    )
    output_path: Optional[pathlib.Path] = None
    wandb: Optional[WandbConfig] = field(
        default_factory=lambda: WandbConfig(
            project=DEFAULT_WANDB_PROJECT,
        )
    )
    use_rich: bool = False
    """Whether to use rich for logging (rich is nice but makes breakpoint() not work)."""
    eval_batch_size: int = 32
    max_num_grasps_to_eval: Optional[int] = 10000
    """Batch size for evaluating grasp metric on dataset of grasps before optimization."""
    print_freq: int = 5
    save_grasps_freq: int = 5
    random_seed: Optional[int] = None
    n_random_rotations_per_grasp: int = 5
    filter_less_feasible_grasps: bool = True
    fingers_forward_theta_deg: float = 60.0
    palm_upwards_theta_deg: float = 60.0

    def __post_init__(self):
        """
        Set default output path if not specified.
        """
        if self.output_path is None:
            print("Using default output path.")
            filename = self.grasp_metric.object_name
            input_folder_path = self.init_grasp_config_dict_path.parent
            output_folder_path = (
                input_folder_path.parent / f"{input_folder_path.name}_optimized"
            )
            self.output_path = output_folder_path / f"{filename}.npy"


if __name__ == "__main__":
    cfg = tyro.cli(OptimizationConfig)
    print(cfg)
