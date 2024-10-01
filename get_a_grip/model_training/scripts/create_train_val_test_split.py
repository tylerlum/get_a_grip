import math
import pathlib
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.utils.parse_object_code_and_scale import (
    object_code_and_scale_to_str,
    parse_object_code_and_scale,
)


@dataclass
class CreateTrainValTestSplitArgs:
    input_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/final_evaled_grasp_config_dicts"
    )
    frac_train: float = 0.9
    frac_val: float = 0.025
    random_seed: int = 42

    @property
    def frac_test(self) -> float:
        assert self.frac_train + self.frac_val <= 1
        return 1 - self.frac_train - self.frac_val


def get_object_code_and_scale_strs(
    object_codes: List[str], object_code_to_scales: Dict[str, List[float]]
) -> List[str]:
    return [
        object_code_and_scale_to_str(object_code, object_scale)
        for object_code in object_codes
        for object_scale in object_code_to_scales[object_code]
    ]


def print_and_run(cmd: str) -> None:
    print("=" * 80)
    print(f"Running: {cmd}")
    print("=" * 80 + "\n")
    subprocess.run(cmd, shell=True, check=True)


def create_symlinks(
    src_folderpath: pathlib.Path, dest_folderpath: pathlib.Path, filenames: List[str]
) -> None:
    dest_folderpath.mkdir(exist_ok=True)
    for filename in filenames:
        src_filename = src_folderpath / filename
        dest_filename = dest_folderpath / filename
        assert src_filename.exists(), f"{src_filename} does not exist"
        if not dest_filename.exists():
            print_and_run(f"ln -sr {str(src_filename)} {str(dest_filename)}")


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[CreateTrainValTestSplitArgs])
    INPUT_PATH = args.input_evaled_grasp_config_dicts_path
    assert INPUT_PATH.exists(), f"{INPUT_PATH} does not exist"

    # Get all object codes and scales
    object_code_and_scale_strs = sorted([f.stem for f in INPUT_PATH.iterdir()])
    print(object_code_and_scale_strs)
    object_code_to_scales = defaultdict(list)
    for object_code_and_scale_str in object_code_and_scale_strs:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        object_code_to_scales[object_code].append(object_scale)

    # Randomize the order of object codes
    object_codes = sorted(list(object_code_to_scales.keys()))
    random.Random(args.random_seed).shuffle(object_codes)

    # Split into train, val, and test
    n_train, n_val = (
        math.ceil(args.frac_train * len(object_codes)),
        math.floor(args.frac_val * len(object_codes)),
    )
    n_test = len(object_codes) - n_train - n_val
    print(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
    print()

    train_object_codes = object_codes[:n_train]
    val_object_codes = object_codes[n_train : n_train + n_val]
    test_object_codes = object_codes[n_train + n_val :]

    train_object_codes_and_scale_strs = get_object_code_and_scale_strs(
        object_codes=train_object_codes, object_code_to_scales=object_code_to_scales
    )
    val_object_codes_and_scale_strs = get_object_code_and_scale_strs(
        object_codes=val_object_codes, object_code_to_scales=object_code_to_scales
    )
    test_object_codes_and_scale_strs = get_object_code_and_scale_strs(
        object_codes=test_object_codes, object_code_to_scales=object_code_to_scales
    )
    assert set(train_object_codes_and_scale_strs).isdisjoint(
        val_object_codes_and_scale_strs
    )
    assert set(train_object_codes_and_scale_strs).isdisjoint(
        test_object_codes_and_scale_strs
    )
    assert set(val_object_codes_and_scale_strs).isdisjoint(
        test_object_codes_and_scale_strs
    )

    # Create symlinks
    create_symlinks(
        src_folderpath=INPUT_PATH,
        dest_folderpath=(INPUT_PATH.parent / f"{INPUT_PATH.stem}_train"),
        filenames=[f"{x}.npy" for x in train_object_codes_and_scale_strs],
    )
    create_symlinks(
        src_folderpath=INPUT_PATH,
        dest_folderpath=(INPUT_PATH.parent / f"{INPUT_PATH.stem}_val"),
        filenames=[f"{x}.npy" for x in val_object_codes_and_scale_strs],
    )
    create_symlinks(
        src_folderpath=INPUT_PATH,
        dest_folderpath=(INPUT_PATH.parent / f"{INPUT_PATH.stem}_test"),
        filenames=[f"{x}.npy" for x in test_object_codes_and_scale_strs],
    )

    # Save to file
    with open((INPUT_PATH.parent / f"{INPUT_PATH.stem}_train.txt"), "w") as f:
        for object_code_and_scale_str in tqdm(
            train_object_codes_and_scale_strs, desc="Writing object code and scales"
        ):
            f.write(f"{object_code_and_scale_str}\n")
    with open((INPUT_PATH.parent / f"{INPUT_PATH.stem}_val.txt"), "w") as f:
        for object_code_and_scale_str in tqdm(
            val_object_codes_and_scale_strs, desc="Writing object code and scales"
        ):
            f.write(f"{object_code_and_scale_str}\n")
    with open((INPUT_PATH.parent / f"{INPUT_PATH.stem}_test.txt"), "w") as f:
        for object_code_and_scale_str in tqdm(
            test_object_codes_and_scale_strs, desc="Writing object code and scales"
        ):
            f.write(f"{object_code_and_scale_str}\n")


if __name__ == "__main__":
    main()
