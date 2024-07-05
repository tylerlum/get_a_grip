import pathlib
import subprocess
from dataclasses import dataclass
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    is_object_code_and_scale_str,
)


@dataclass
class TrainNerfsArgs:
    experiment_name: str
    max_num_iterations: int = 400
    input_nerfdata_name: str = "nerfdata"
    output_nerfcheckpoints_name: str = "nerfcheckpoints"
    root_data_path: pathlib.Path = get_data_folder()
    randomize_order_seed: Optional[int] = None


def print_and_run(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def train_nerfs(args: TrainNerfsArgs) -> pathlib.Path:
    assert args.root_data_path.exists(), f"{args.root_data_path} does not exist"
    experiment_path = args.root_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"

    nerfdata_path = experiment_path / args.input_nerfdata_name
    assert nerfdata_path.exists(), f"{nerfdata_path} does not exist"

    output_nerfcheckpoints_path = experiment_path / args.output_nerfcheckpoints_name
    output_nerfcheckpoints_path.mkdir(exist_ok=True)

    object_nerfdata_paths = sorted(list(nerfdata_path.iterdir()))

    if args.randomize_order_seed is not None:
        import random

        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(object_nerfdata_paths)

    for object_nerfdata_path in tqdm(
        object_nerfdata_paths, dynamic_ncols=True, desc="Training NERF"
    ):
        if not object_nerfdata_path.is_dir():
            print(f"Skipping {object_nerfdata_path} because it is not a directory")
            continue
        assert is_object_code_and_scale_str(
            object_nerfdata_path.name
        ), f"{object_nerfdata_path.name} is not an object code and scale"

        output_path_to_be_created = (
            output_nerfcheckpoints_path / object_nerfdata_path.name
        )
        if output_path_to_be_created.exists():
            print(f"Skipping {output_path_to_be_created} because it already exists")
            continue

        command = " ".join(
            [
                "ns-train nerfacto",
                f"--data {str(object_nerfdata_path)}",
                f"--max-num-iterations {args.max_num_iterations}",
                f"--output-dir {str(output_nerfcheckpoints_path)}",
                "--vis wandb",
                "--pipeline.model.disable-scene-contraction True",
                "nerfstudio-data",
                "--auto-scale-poses False",
                "--scale-factor 1.",
                "--center-method none",
                "--orientation-method none",
            ]
        )
        print_and_run(command)
    return output_nerfcheckpoints_path


def main() -> None:
    args = tyro.cli(TrainNerfsArgs)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    train_nerfs(args)


if __name__ == "__main__":
    main()
