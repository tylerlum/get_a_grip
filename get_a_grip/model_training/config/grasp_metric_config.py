from dataclasses import dataclass
import tyro
import pathlib
import nerf_grasping
from nerf_grasping.config.classifier_config import (
    DEFAULTS_DICT as CLASSIFIER_DEFAULTS_DICT,
    ClassifierConfig,
)
from typing import Optional
import numpy as np


@dataclass
class GraspMetricConfig:
    """Top-level config for creating a grasp metric."""

    classifier_config: ClassifierConfig = CLASSIFIER_DEFAULTS_DICT["grasp-cond-simple-cnn-2d-1d"]
    classifier_config_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "mugs_grid_grasp-cond-simple-cnn-2d-1d"
        / "config.yaml"
    )
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    nerf_checkpoint_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
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

            APPLY_TRANSLATION = False
            if APPLY_TRANSLATION:
                # X_N_O = trimesh.transformations.translation_matrix([-0.04092566, -0.05782086,  0.04981683])
                # X_N_O = trimesh.transformations.translation_matrix([-0.02423195, -0.00194203,  0.13271753])
                # X_N_O = trimesh.transformations.translation_matrix([0, 0, 0.1])
                X_N_O = trimesh.transformations.translation_matrix([0.01965157, -0.00010462, 0.05522743])
            else:
                X_N_O = np.eye(4)


            # Z-up
            IS_Z_UP = False
            if IS_Z_UP:
                X_O_Oy = trimesh.transformations.rotation_matrix(
                    np.pi / 2, [1, 0, 0]
                )
            else:
                X_O_Oy = np.eye(4)

            self.X_N_Oy = X_N_O @ X_O_Oy


    @property
    def object_name(self) -> str:
        return self.nerf_checkpoint_path.parents[2].stem


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
