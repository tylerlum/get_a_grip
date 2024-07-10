import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    object_code_and_scale_to_str,
)


@dataclass
class GenerateObjectCodeAndScalesTxtArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "large/meshes"
    output_object_code_and_scales_txt_path: pathlib.Path = (
        get_data_folder() / "NEW_DATASET/object_code_and_scales.txt"
    )
    min_object_scale: float = 0.05
    max_object_scale: float = 0.10
    max_num_objects: Optional[int] = None
    overwrite: bool = False


def main() -> None:
    args = tyro.cli(GenerateObjectCodeAndScalesTxtArgs)

    object_codes = sorted([path.name for path in args.meshdata_root_path.iterdir()])
    print(f"First 10 in object_codes: {object_codes[:10]}")
    print(f"len(object_codes): {len(object_codes)}")

    object_scales = np.random.uniform(
        low=args.min_object_scale,
        high=args.max_object_scale,
        size=(len(object_codes),),
    )

    object_code_and_scale_strs = [
        object_code_and_scale_to_str(object_code, object_scale)
        for object_code, object_scale in zip(object_codes, object_scales)
    ]

    if args.output_object_code_and_scales_txt_path.exists():
        print(f"{args.output_object_code_and_scales_txt_path} already exists.")
        if args.overwrite:
            print(
                f"Overwriting {args.output_object_code_and_scales_txt_path} with {len(object_code_and_scale_strs)} object code and scales."
            )
        else:
            print(
                f"Exiting since {args.output_object_code_and_scales_txt_path} already exists, rerun with --overwrite to overwrite."
            )
            return

    args.output_object_code_and_scales_txt_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    with open(args.output_object_code_and_scales_txt_path, "w") as f:
        for object_code_and_scale_str in tqdm(
            object_code_and_scale_strs, desc="Writing object code and scales"
        ):
            f.write(f"{object_code_and_scale_str}\n")


if __name__ == "__main__":
    main()