import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder


@dataclass
class CreateFixedSamplerGraspConfigDict:
    input_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/final_evaled_grasp_config_dicts_train"
    )
    output_grasp_config_dict_path: pathlib.Path = (
        get_data_folder() / "fixed_sampler_grasp_config_dicts/all_good_grasps.npy"
    )
    y_PGS_threshold: float = 0.9
    max_grasps_per_object: Optional[int] = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        assert (
            self.output_grasp_config_dict_path.suffix == ".npy"
        ), f"output_grasp_config_dict_path {self.output_grasp_config_dict_path} must have a .npy suffix"


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[CreateFixedSamplerGraspConfigDict])
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

    if not args.input_evaled_grasp_config_dicts_path.exists():
        raise ValueError(
            f"input_evaled_grasp_config_dicts_path {args.input_evaled_grasp_config_dicts_path} doesn't exist"
        )

    if args.output_grasp_config_dict_path.exists() and not args.overwrite:
        raise ValueError(
            f"output_grasp_config_dict_path {args.output_grasp_config_dict_path} already exists, use --overwrite True to overwrite"
        )

    # Read in all npy files
    input_evaled_grasp_config_dict_filepaths = sorted(
        list(args.input_evaled_grasp_config_dicts_path.rglob("*.npy"))
    )
    assert (
        len(input_evaled_grasp_config_dict_filepaths) > 0
    ), f"No npy files found in {args.input_evaled_grasp_config_dicts_path}"
    print(
        f"Found {len(input_evaled_grasp_config_dict_filepaths)} input_config_dict_filepaths"
    )
    filepath_to_evaled_grasp_config_dict = {
        filepath: np.load(filepath, allow_pickle=True).item()
        for filepath in tqdm(
            input_evaled_grasp_config_dict_filepaths, desc="Loading npy files"
        )
    }

    total_num_grasps = sum(
        [
            dict["y_PGS"].shape[0]
            for dict in filepath_to_evaled_grasp_config_dict.values()
        ]
    )
    print(f"total_num_grasps: {total_num_grasps}")

    # Get all good grasps
    good_grasp_config_dicts = []
    for dict in tqdm(
        filepath_to_evaled_grasp_config_dict.values(),
        total=len(filepath_to_evaled_grasp_config_dict),
        desc="Getting good grasps",
    ):
        good_idxs = np.where(dict["y_PGS"] > args.y_PGS_threshold)[0]
        if good_idxs.shape[0] == 0:
            continue
        good_grasp_config_dict = {k: v[good_idxs] for k, v in dict.items()}
        good_grasp_config_dicts.append(good_grasp_config_dict)
    for dict in good_grasp_config_dicts:
        assert dict["y_PGS"].shape[0] > 0
    print(
        f"Found {len(good_grasp_config_dicts)} good_grasp_config_dicts that pass the threshold of {args.y_PGS_threshold}"
    )
    total_num_good_grasps = sum(
        [dict["y_PGS"].shape[0] for dict in good_grasp_config_dicts]
    )
    print(f"Total number of good grasps: {total_num_good_grasps}")

    # Merge all good grasps
    key_in_all = set.intersection(
        *[set(dict.keys()) for dict in good_grasp_config_dicts]
    )
    merged_good_grasp_config_dict = {
        k: np.concatenate(
            [dict[k][: args.max_grasps_per_object] for dict in good_grasp_config_dicts],
            axis=0,
        )
        for k in tqdm(key_in_all, desc="Merging good grasps")
    }

    # Save
    args.output_grasp_config_dict_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_grasp_config_dict_path, merged_good_grasp_config_dict)
    print(f"Saved to {args.output_grasp_config_dict_path}")


if __name__ == "__main__":
    main()
