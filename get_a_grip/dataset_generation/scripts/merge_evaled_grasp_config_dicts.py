import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder


@dataclass
class MergeEvaledGraspConfigDictsArgs:
    input_evaled_grasp_config_dicts_paths: List[pathlib.Path] = field(
        default_factory=lambda: [
            get_data_folder() / "dataset/NEW/evaled_grasp_config_dicts",
            get_data_folder() / "dataset/NEW/augmented_evaled_grasp_config_dicts",
        ]
    )
    output_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/final_evaled_grasp_config_dicts"
    )


# These are the only keys that matter for merging evaled grasp config dicts.
EVALED_GRASP_CONFIG_DICT_KEYS = [
    "trans",
    "rot",
    "joint_angles",
    "grasp_orientations",
    "y_coll",
    "y_pick",
    "y_PGS",
    "object_states_before_grasp",
]


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[MergeEvaledGraspConfigDictsArgs])

    # Create dir
    args.output_evaled_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)

    # Get list of all filenames
    input_config_dict_filepaths = [
        filepath
        for input_config_dicts_path in args.input_evaled_grasp_config_dicts_paths
        for filepath in input_config_dicts_path.rglob("*.npy")
    ]
    unique_filenames = list(set([path.name for path in input_config_dict_filepaths]))

    for filename in tqdm(unique_filenames, desc="Merging evaled grasp config dicts"):
        filepaths_with_filename = [
            path for path in input_config_dict_filepaths if path.name == filename
        ]
        print(f"Found {len(filepaths_with_filename)} files with filename {filename}")

        # Append all
        num_grasps = 0
        grasp_config_dict_list = defaultdict(list)
        for filepath in filepaths_with_filename:
            grasp_config_dict = np.load(filepath, allow_pickle=True).item()

            num_grasps += grasp_config_dict["trans"].shape[0]
            for key in EVALED_GRASP_CONFIG_DICT_KEYS:
                grasp_config_dict_list[key].append(grasp_config_dict[key])

        # Concatenate all
        combined_grasp_config_dict = {
            key: np.concatenate(grasp_config_dict_list[key], axis=0)
            for key in EVALED_GRASP_CONFIG_DICT_KEYS
        }

        # Shape checks.
        for key in EVALED_GRASP_CONFIG_DICT_KEYS:
            assert (
                combined_grasp_config_dict[key].shape[0] == num_grasps
            ), f"key: {key}, shape: {combined_grasp_config_dict[key].shape}, num_grasps: {num_grasps}"

        new_filepath = args.output_evaled_grasp_config_dicts_path / filename
        print(f"Saving {num_grasps} grasps to {new_filepath}")
        np.save(file=new_filepath, arr=combined_grasp_config_dict, allow_pickle=True)


if __name__ == "__main__":
    main()
