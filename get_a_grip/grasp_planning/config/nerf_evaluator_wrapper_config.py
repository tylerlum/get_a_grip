import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tyro

from get_a_grip import get_data_folder
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    DEFAULTS_DICT as NERF_EVALUATOR_DEFAULTS_DICT,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)


@dataclass
class NerfEvaluatorWrapperConfig:
    """Top-level config for creating a grasp metric."""

    nerf_evaluator_config: NerfEvaluatorModelConfig = field(
        default_factory=lambda: NERF_EVALUATOR_DEFAULTS_DICT["cnn-3d-xyz-global-cnn"]
    )
    nerf_evaluator_config_path: Optional[pathlib.Path] = (
        get_data_folder()
        / "logs/nerf_grasp_evaluator/MY_NERF_EXPERIMENT_NAME_2024-07-05_17-47-28-051229/config.yaml"
    )

    nerf_evaluator_checkpoint: int = -1  # Load latest checkpoint if -1.
    nerf_config: pathlib.Path = (
        get_data_folder()
        / "NEW_DATASET/nerfcheckpoints/core-bottle-11fc9827d6b467467d3aa3bae1f7b494_0_0726/nerfacto/2024-07-04_210957/config.yml"
    )
    X_N_Oy: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Load nerf_evaluator config from file if nerf_evaluator config is not None.
        """
        if self.nerf_evaluator_config_path is not None:
            print(
                f"Loading nerf_evaluator config from {self.nerf_evaluator_config_path}"
            )
            self.nerf_evaluator_config = tyro.extras.from_yaml(
                type(self.nerf_evaluator_config), self.nerf_evaluator_config_path.open()
            )
        else:
            print("Loading default nerf_evaluator config.")

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
    cfg = tyro.cli(NerfEvaluatorWrapperConfig)
    print(cfg)
