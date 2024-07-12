import pathlib
import time
from dataclasses import dataclass
from typing import Literal, Optional

from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.utils.train_nerf import (
    TrainNerfArgs,
    train_nerf_return_trainer,
)


@dataclass
class NerfArgs:
    nerf_is_z_up: bool
    nerf_config: Optional[pathlib.Path] = None
    nerfdata_path: Optional[pathlib.Path] = None
    max_num_iterations: int = 400

    def __post_init__(self) -> None:
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
        ] = "test",  # Must be test for point cloud and BPS, can be inference for sampling densities
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
