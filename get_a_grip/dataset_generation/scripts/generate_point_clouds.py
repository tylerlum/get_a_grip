import pathlib
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.utils.generate_point_cloud import (
    GeneratePointCloudArgs,
    generate_point_cloud,
)
from get_a_grip.utils.nerf_load_utils import get_latest_nerf_config
from get_a_grip.utils.parse_object_code_and_scale import (
    is_object_code_and_scale_str,
)


@dataclass
class GeneratePointCloudsArgs:
    nerf_is_z_up: bool
    input_nerfcheckpoints_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/nerfcheckpoints"
    )
    output_point_clouds_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/point_clouds"
    )
    randomize_order_seed: Optional[int] = datetime.now().microsecond
    num_points: int = 5000
    timeout: float = 60.0


def generate_point_clouds(args: GeneratePointCloudsArgs) -> pathlib.Path:
    assert (
        args.input_nerfcheckpoints_path.exists()
    ), f"{args.input_nerfcheckpoints_path} does not exist"

    args.output_point_clouds_path.mkdir(exist_ok=True)

    nerfcheckpoint_paths = sorted(
        [
            nerfcheckpoint_path
            for nerfcheckpoint_path in args.input_nerfcheckpoints_path.iterdir()
            if nerfcheckpoint_path.is_dir()
        ]
    )
    print(f"Found {len(nerfcheckpoint_paths)} NERF checkpoints")

    if args.randomize_order_seed is not None:
        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(nerfcheckpoint_paths)

    for nerfcheckpoint_path in tqdm(
        nerfcheckpoint_paths, dynamic_ncols=True, desc="Generating PointClouds"
    ):
        nerf_config = get_latest_nerf_config(nerfcheckpoint_path)

        object_code_and_scale_str = nerfcheckpoint_path.stem
        assert is_object_code_and_scale_str(
            object_code_and_scale_str
        ), f"object_code_and_scale_str: {object_code_and_scale_str} is not valid."

        output_path_to_be_created = (
            args.output_point_clouds_path / object_code_and_scale_str
        )
        if output_path_to_be_created.exists():
            print(f"Skipping {output_path_to_be_created} because it already exists")
            continue

        # Need to do this here to avoid error when saving the point_clouds
        print(f"Creating {output_path_to_be_created}")
        output_path_to_be_created.mkdir(exist_ok=False)

        # NOTE: Very rarely nerfs are terrible by bad luck, so computing point clouds never finishes.
        # In this case, we should know about this and move on.
        try:
            generate_point_cloud(
                GeneratePointCloudArgs(
                    nerf_is_z_up=args.nerf_is_z_up,
                    nerf_config=nerf_config,
                    output_dir=output_path_to_be_created,
                    num_points=args.num_points,
                    timeout=args.timeout,
                )
            )
            print(f"Finished generating {output_path_to_be_created}")
            print("=" * 80)
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {args.timeout} seconds")
            print("~" * 80)

            timeout_path = (
                args.output_point_clouds_path.parent
                / f"{args.input_nerfcheckpoints_path.stem}_{args.output_point_clouds_path.stem}_timeout.txt"
            )
            print(f"Writing to {timeout_path}")
            with open(timeout_path, "a") as f:
                f.write(f"{object_code_and_scale_str}\n")

    return args.output_point_clouds_path


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GeneratePointCloudsArgs])
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    generate_point_clouds(args)


if __name__ == "__main__":
    main()
