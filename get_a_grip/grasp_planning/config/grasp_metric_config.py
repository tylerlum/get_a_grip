import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from get_a_grip import get_data_folder, get_repo_folder
from get_a_grip.model_training.config.classifier_config import (
    DEFAULTS_DICT as CLASSIFIER_DEFAULTS_DICT,
)
from get_a_grip.model_training.config.classifier_config import (
    ClassifierConfig,
)


@dataclass
class GraspMetricConfig:
    """Top-level config for creating a grasp metric."""

    classifier_config: ClassifierConfig = CLASSIFIER_DEFAULTS_DICT[
        "grasp-cond-simple-cnn-2d-1d"
    ]
    classifier_config_path: Optional[pathlib.Path] = (
        get_repo_folder()
        / "nerf_grasp_evaluator_workspaces"
        / "mugs_grid_grasp-cond-simple-cnn-2d-1d"
        / "config.yaml"
    )
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    nerf_config: pathlib.Path = (
        get_data_folder()
        / "2023-01-03_mugs_smaller0-075_noise_lightshake_mid_opt"
        / "nerfcheckpoints"
        / "core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750"
        / "nerfacto"
        / "2024-01-03_235839"
        / "config.yml"
    )
    X_N_Oy: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Load classifier config from file if classifier config is not None.
        """
        if self.classifier_config_path is not None:
            print(f"Loading classifier config from {self.classifier_config_path}")
            self.classifier_config = tyro.extras.from_yaml(
                type(self.classifier_config), self.classifier_config_path.open()
            )
        else:
            print("Loading default classifier config.")

        if self.X_N_Oy is None:
            # HACK: Should try to compute this or use cfg to be told what to do
            import trimesh

            X_N_O = np.eye(4)

            # Z-up
            IS_Z_UP = False
            if IS_Z_UP:
                X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            else:
                X_O_Oy = np.eye(4)

            self.X_N_Oy = X_N_O @ X_O_Oy

    @property
    def object_name(self) -> str:
        return self.nerf_config.parents[2].stem


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
