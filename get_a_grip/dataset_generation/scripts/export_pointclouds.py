import pathlib
import random
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    is_object_code_and_scale_str,
)


@dataclass
class ExportPointcloudsArgs:
    experiment_name: str
    nerf_is_z_up: bool
    input_nerfcheckpoints_name: str = "nerfcheckpoints"
    output_pointclouds_name: str = "pointclouds"
    nerf_grasping_data_path: pathlib.Path = get_data_folder()
    randomize_order_seed: Optional[int] = None
    num_points: int = 5000
    timeout: float = 60.0

    @property
    def bounding_box_min(self) -> str:
        if self.nerf_is_z_up:
            return "-0.2 -0.2 0.0"
        else:
            return "-0.2 0.0 -0.2"

    @property
    def bounding_box_max(self) -> str:
        if self.nerf_is_z_up:
            return "0.2 0.2 0.3"
        else:
            return "0.2 0.3 0.2"

    @property
    def obb_scale(self) -> str:
        bb_min = np.array([float(x) for x in self.bounding_box_min.split()])
        bb_max = np.array([float(x) for x in self.bounding_box_max.split()])
        obb_scale = bb_max - bb_min
        return " ".join([str(x) for x in obb_scale])

    @property
    def obb_center(self) -> str:
        bb_min = np.array([float(x) for x in self.bounding_box_min.split()])
        bb_max = np.array([float(x) for x in self.bounding_box_max.split()])
        obb_center = (bb_max + bb_min) / 2
        return " ".join([str(x) for x in obb_center])

    @property
    def obb_rotation(self) -> str:
        obb_rotation = np.array([0.0, 0.0, 0.0])
        return " ".join([str(x) for x in obb_rotation])


def get_latest_nerf_config(nerfcheckpoint_path: pathlib.Path) -> pathlib.Path:
    nerf_configs = list(nerfcheckpoint_path.glob("nerfacto/*/config.yml"))
    assert len(nerf_configs) > 0, f"No NERF configs found in {nerfcheckpoint_path}"
    latest_nerf_config = max(nerf_configs, key=lambda p: p.stat().st_ctime)
    return latest_nerf_config


def export_pointclouds(args: ExportPointcloudsArgs) -> pathlib.Path:
    assert (
        args.nerf_grasping_data_path.exists()
    ), f"{args.nerf_grasping_data_path} does not exist"
    experiment_path = args.nerf_grasping_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"

    nerfcheckpoints_path = experiment_path / args.input_nerfcheckpoints_name
    assert nerfcheckpoints_path.exists(), f"{nerfcheckpoints_path} does not exist"

    output_pointclouds_path = experiment_path / args.output_pointclouds_name
    output_pointclouds_path.mkdir(exist_ok=True)

    nerfcheckpoint_paths = sorted(
        [
            nerfcheckpoint_path
            for nerfcheckpoint_path in nerfcheckpoints_path.iterdir()
            if nerfcheckpoint_path.is_dir()
        ]
    )
    print(f"Found {len(nerfcheckpoint_paths)} NERF checkpoints")

    if args.randomize_order_seed is not None:
        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(nerfcheckpoint_paths)

    for nerfcheckpoint_path in tqdm(
        nerfcheckpoint_paths, dynamic_ncols=True, desc="Exporting Pointclouds"
    ):
        nerf_config = get_latest_nerf_config(nerfcheckpoint_path)

        object_code_and_scale_str = nerfcheckpoint_path.stem
        assert is_object_code_and_scale_str(
            object_code_and_scale_str
        ), f"object_code_and_scale_str: {object_code_and_scale_str} is not valid."

        output_path_to_be_created = output_pointclouds_path / object_code_and_scale_str
        if output_path_to_be_created.exists():
            print(f"Skipping {output_path_to_be_created} because it already exists")
            continue

        # Need to do this here to avoid error when saving the pointclouds
        print(f"Creating {output_path_to_be_created}")
        output_path_to_be_created.mkdir(exist_ok=False)

        command = " ".join(
            [
                "ns-export",
                "pointcloud",
                f"--load-config {str(nerf_config)}",
                f"--output-dir {str(output_path_to_be_created)}",
                "--normal-method open3d",
                f"--obb-scale {args.obb_scale}",
                f"--obb-center {args.obb_center}",
                f"--obb-rotation {args.obb_rotation}",
                f"--num-points {args.num_points}",
            ]
        )

        # NOTE: Some nerfs are terrible, so computing point clouds never finishes.
        # In this case, we should know about this and move on.
        print(f"Running: {command}")
        try:
            subprocess.run(command, shell=True, check=True, timeout=args.timeout)
            print(f"Finished generating {output_path_to_be_created}")
            print("=" * 80)
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {args.timeout} seconds: {command}")
            print("~" * 80)

            timeout_path = (
                experiment_path
                / f"{args.input_nerfcheckpoints_name}_{args.output_pointclouds_name}_timeout.txt"
            )
            print(f"Writing to {timeout_path}")
            with open(timeout_path, "a") as f:
                f.write(f"{object_code_and_scale_str}\n")

    return output_pointclouds_path


def main() -> None:
    args = tyro.cli(ExportPointcloudsArgs)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    export_pointclouds(args)


if __name__ == "__main__":
    main()
