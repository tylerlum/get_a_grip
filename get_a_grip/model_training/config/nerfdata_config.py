import pathlib
from dataclasses import dataclass
from typing import Optional

import tyro

from get_a_grip import get_data_folder
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
    UnionFingertipConfig,
)

EXPERIMENT_NAME = "DEFAULT_EXPERIMENT_NAME"


@dataclass
class BaseNerfDataConfig:
    """Top-level config for NeRF data generation."""

    meshdata_root: pathlib.Path = get_data_folder() / "large/meshes"
    evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / EXPERIMENT_NAME / "evaled_grasp_config_dicts"
    )
    nerf_checkpoints_path: pathlib.Path = (
        get_data_folder() / EXPERIMENT_NAME / "nerfcheckpoints"
    )
    output_filepath: Optional[pathlib.Path] = None
    plot_only_one: bool = False
    config_dict_visualize_index: Optional[int] = 0
    grasp_visualize_index: Optional[int] = 0
    save_dataset: bool = True
    print_timing: bool = True
    limit_num_configs: Optional[int] = None  # None for no limit
    max_num_data_points_per_file: Optional[int] = (
        None  # None for count actual num data points, set to avoid OOM
    )
    ray_samples_chunk_size: int = 50  # ~8GB on GPU
    cameras_samples_chunk_size: int = 2000  # ~14GB on GPU
    plot_all_high_density_points: bool = True
    plot_alphas_each_finger_1D: bool = True
    plot_alpha_images_each_finger: bool = True

    fingertip_config: Optional[UnionFingertipConfig] = EvenlySpacedFingertipConfig()

    @property
    def config_filepath(self) -> pathlib.Path:
        return self.output_filepath.parent / "config.yml"


@dataclass
class GridNerfDataConfig(BaseNerfDataConfig):
    plot_alphas_each_finger_1D: bool = True
    plot_alpha_images_each_finger: bool = True


UnionNerfDataConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "grid": GridNerfDataConfig(),
    }
)

if __name__ == "__main__":
    cfg = tyro.cli(UnionNerfDataConfig)
    print(cfg)
