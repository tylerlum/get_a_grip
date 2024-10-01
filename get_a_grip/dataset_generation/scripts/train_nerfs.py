import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.utils.parse_object_code_and_scale import (
    is_object_code_and_scale_str,
)
from get_a_grip.utils.train_nerf import (
    TrainNerfArgs,
    train_nerf,
    train_nerf_return_trainer,
)


@dataclass
class TrainNerfsArgs:
    input_nerfdata_path: pathlib.Path = get_data_folder() / "dataset/NEW/nerfdata"
    output_nerfcheckpoints_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/nerfcheckpoints"
    )
    max_num_iterations: int = 400
    randomize_order_seed: Optional[int] = datetime.now().microsecond


def train_nerfs(args: TrainNerfsArgs) -> pathlib.Path:
    assert (
        args.input_nerfdata_path.exists()
    ), f"{args.input_nerfdata_path} does not exist"

    args.output_nerfcheckpoints_path.mkdir(exist_ok=True, parents=True)

    object_nerfdata_paths = sorted(list(args.input_nerfdata_path.iterdir()))

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
            args.output_nerfcheckpoints_path / object_nerfdata_path.name
        )
        if output_path_to_be_created.exists():
            print(f"Skipping {output_path_to_be_created} because it already exists")
            continue

        # Both options are equivalent
        OPTION = "train_nerf"
        if OPTION == "train_nerf":
            train_nerf(
                TrainNerfArgs(
                    nerfdata_folder=object_nerfdata_path,
                    nerfcheckpoints_folder=args.output_nerfcheckpoints_path,
                    max_num_iterations=args.max_num_iterations,
                )
            )
        elif OPTION == "train_nerf_return_trainer":
            raise ValueError(
                "This option is not supported. Because of nerfstudio's implementation, its experiment config does not update on each loop, so it keeps saving all nerfs to the same directory, overwriting itself"
            )
            train_nerf_return_trainer(
                TrainNerfArgs(
                    nerfdata_folder=object_nerfdata_path,
                    nerfcheckpoints_folder=args.output_nerfcheckpoints_path,
                    max_num_iterations=args.max_num_iterations,
                )
            )
        else:
            raise ValueError(f"Invalid option: {OPTION}")

    return args.output_nerfcheckpoints_path


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[TrainNerfsArgs])
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    train_nerfs(args)


if __name__ == "__main__":
    main()
