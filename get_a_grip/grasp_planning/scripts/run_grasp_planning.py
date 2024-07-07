import pathlib
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.grasp_planning.utils.planner import PlannerConfig
from get_a_grip.grasp_planning.utils.train_nerf import (
    TrainNerfArgs,
    train_nerf_return_trainer,
)
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline


@dataclass
class NerfArgs:
    nerf_is_z_up: bool
    nerf_config: Optional[pathlib.Path] = None
    nerfdata_path: Optional[pathlib.Path] = None
    max_num_iterations: int = 400

    def __post_init__(self) -> None:
        # Only
        if self.nerf_config is None:
            assert self.nerfdata_path is not None, "nerfdata_path must be specified"
            self.verify_nerfdata_path(self.nerfdata_path)
        else:
            self.verify_nerf_config(self.nerf_config)

            if self.nerfdata_path is not None:
                print(
                    f"WARNING: Ignoring nerfdata_path {self.nerfdata_path} because given nerf_config {self.nerf_config}"
                )

    def verify_nerfdata_path(self, nerfdata_path: pathlib.Path) -> None:
        assert nerfdata_path.exists(), f"{nerfdata_path} does not exist"
        assert (
            nerfdata_path / "transforms.json"
        ).exists(), f"{nerfdata_path / 'transforms.json'} does not exist"
        assert (
            nerfdata_path / "images"
        ).exists(), f"{nerfdata_path / 'images'} does not exist"

    def verify_nerf_config(self, nerf_config: pathlib.Path) -> None:
        assert nerf_config.exists(), f"{nerf_config} does not exist"
        assert (
            nerf_config.suffix == ".yml"
        ), f"{nerf_config} does not have a .yml suffix"

    def load_nerf_pipeline(
        self,
        test_mode: Literal[
            "test", "inference"
        ] = "test",  # Must be test for point cloud
        print_timing: bool = True,
    ) -> Pipeline:
        assert self.nerf_config is not None, "nerf_config must be specified"

        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(self.nerf_config, test_mode=test_mode)
        end_time = time.time()
        if print_timing:
            print("@" * 80)
            print(f"Time to load_nerf_pipeline: {end_time - start_time:.2f}s")
            print("@" * 80 + "\n")
        return nerf_pipeline

    def train_nerf_pipeline(
        self, output_folder: pathlib.Path, print_timing: bool = True
    ) -> Pipeline:
        assert self.nerfdata_path is not None, "nerfdata_path must be specified"
        self.verify_nerfdata_path(self.nerfdata_path)

        start_time = time.time()
        nerfcheckpoints_folder = output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerf_return_trainer(
            args=TrainNerfArgs(
                nerfdata_folder=self.nerfdata_path,
                nerfcheckpoints_folder=nerfcheckpoints_folder,
                max_num_iterations=self.max_num_iterations,
            )
        )
        nerf_pipeline = nerf_trainer.pipeline
        self.nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"

        end_time = time.time()
        if print_timing:
            print("@" * 80)
            print(f"Time to train_nerf: {end_time - start_time:.2f}s")
            print("@" * 80 + "\n")
        return nerf_pipeline

    @property
    def object_name(self) -> str:
        if self.nerfdata_path is not None and self.nerf_config is None:
            object_name = self.nerfdata_path.name
        elif self.nerfdata_path is None and self.nerf_config is not None:
            object_name = self.nerf_config.parents[2].name
        else:
            raise ValueError(
                "Exactly one of nerfdata_path or nerf_config must be specified"
            )
        return object_name


@dataclass
class GraspPlanningArgs:
    nerf: NerfArgs
    planner: PlannerConfig
    output_folder: pathlib.Path
    overwrite: bool = False


def run_grasp_planning(
    args: GraspPlanningArgs,
) -> Tuple[AllegroGraspConfig, NerfInput, torch.Tensor]:
    # Get object name
    object_name = args.nerf.object_name
    print(f"object_name = {object_name}")

    # Prepare output folder
    args.output_folder.mkdir(exist_ok=True, parents=True)
    output_file = args.output_folder / f"{object_name}.npy"
    if output_file.exists():
        if not args.overwrite:
            raise ValueError(f"{output_file} already exists, skipping")

        print(f"{output_file} already exists, overwriting")

    # Prepare nerf pipeline
    if args.nerf.nerf_config is not None:
        nerf_pipeline = args.nerf.load_nerf_pipeline()
    else:
        nerf_pipeline = args.nerf.train_nerf_pipeline(output_folder=args.output_folder)

    # Prepare nerf input
    nerf_input = NerfInput(
        nerf_pipeline=nerf_pipeline, nerf_is_z_up=args.nerf.nerf_is_z_up
    )

    # Prepare grasp planner
    grasp_planner = args.planner.create(nerf_input=nerf_input)

    # Plan grasps
    planned_grasp_config, losses = grasp_planner.plan()

    # Save planned grasp config
    planned_grasp_config_dict = planned_grasp_config.as_dict()
    planned_grasp_config_dict["loss"] = losses.detach().cpu().numpy()
    print(f"Saving grasp_config_dict to {output_file}")
    np.save(file=output_file, arr=planned_grasp_config_dict)

    return planned_grasp_config, nerf_input, losses


def main() -> None:
    args = tyro.cli(GraspPlanningArgs)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")
    run_grasp_planning(args)


if __name__ == "__main__":
    main()
