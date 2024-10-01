import pathlib
from dataclasses import dataclass, field
from typing import Optional

import tyro

from get_a_grip import get_data_folder
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)

EXPERIMENT_NAME = "DEFAULT_EXPERIMENT_NAME"


@dataclass
class NerfGraspDatasetConfig:
    """Top-level config for NeRF data generation."""

    input_nerfcheckpoints_path: pathlib.Path = (
        get_data_folder() / EXPERIMENT_NAME / "nerfcheckpoints"
    )
    input_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / EXPERIMENT_NAME / "evaled_grasp_config_dicts"
    )
    output_filepath: Optional[pathlib.Path] = None

    fingertip_config: EvenlySpacedFingertipConfig = field(
        default_factory=EvenlySpacedFingertipConfig
    )
    print_timing: bool = True

    limit_num_objects: Optional[int] = None  # None for no limit
    max_num_data_points_per_file: Optional[int] = (
        None  # None for count actual num data points, set to avoid OOM
    )

    ray_samples_chunk_size: int = 50  # ~8GB on GPU

    @property
    def config_filepath(self) -> pathlib.Path:
        assert self.output_filepath is not None
        return self.output_filepath.parent / "config.yml"


if __name__ == "__main__":
    cfg = tyro.cli(tyro.conf.FlagConversionOff[NerfGraspDatasetConfig])
    print(cfg)
