from __future__ import annotations

import multiprocessing
import pathlib
import random
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.scripts.eval_grasp_config_dict import (
    EvalGraspConfigDictArgs,
    eval_grasp_config_dict,
)
from get_a_grip.dataset_generation.utils.isaac_validator import ValidationType
from get_a_grip.dataset_generation.utils.process_utils import (
    get_object_codes_and_scales_to_process,
)


@dataclass
class EvalAllGraspConfigDictsArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "large/meshes"
    input_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "NEW_DATASET/grasp_config_dicts"
    )
    output_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "NEW_DATASET/evaled_grasp_config_dicts"
    )
    validation_type: ValidationType = ValidationType.GRAVITY_AND_TABLE
    num_random_pose_noise_samples_per_grasp: Optional[int] = None
    gpu: int = 0
    max_grasps_per_batch: int = 5000
    move_fingers_back_at_init: bool = False

    use_cpu: bool = False
    randomize_order_seed: Optional[int] = None
    mid_optimization_steps: List[int] = field(default_factory=list)
    use_multiprocess: bool = True
    num_workers: int = 3


def get_object_code_and_scale_strs_to_process(
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
) -> List[str]:
    input_object_code_and_scale_strs = [
        path.stem for path in list(input_grasp_config_dicts_path.glob("*.npy"))
    ]
    print(
        f"Found {len(input_object_code_and_scale_strs)} object codes in input_grasp_config_dicts_path ({input_grasp_config_dicts_path})"
    )

    # Compare input and output directories
    existing_object_code_and_scale_strs = (
        [
            path.stem
            for path in list(output_evaled_grasp_config_dicts_path.glob("*.npy"))
        ]
        if output_evaled_grasp_config_dicts_path.exists()
        else []
    )
    print(
        f"Found {len(existing_object_code_and_scale_strs)} object codes in {output_evaled_grasp_config_dicts_path}"
    )

    # Sanity check that we are going into the right folder
    only_in_output = set(existing_object_code_and_scale_strs) - set(
        input_object_code_and_scale_strs
    )
    print(f"Num only in output: {len(only_in_output)}")
    assert len(only_in_output) == 0, f"Object codes only in output: {only_in_output}"

    # Don't redo old work
    only_in_input = set(input_object_code_and_scale_strs) - set(
        existing_object_code_and_scale_strs
    )
    print(f"Num codes only in input: {len(only_in_input)}")

    return list(only_in_input)


def run_eval_grasp_config_dict(
    object_code_and_scale_str: str,
    args: EvalAllGraspConfigDictsArgs,
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
):
    try:
        eval_grasp_config_dict(
            EvalGraspConfigDictArgs(
                meshdata_root_path=args.meshdata_root_path,
                input_grasp_config_dicts_path=input_grasp_config_dicts_path,
                output_evaled_grasp_config_dicts_path=output_evaled_grasp_config_dicts_path,
                object_code_and_scale_str=object_code_and_scale_str,
                validation_type=args.validation_type,
                num_random_pose_noise_samples_per_grasp=args.num_random_pose_noise_samples_per_grasp,
                gpu=args.gpu,
                max_grasps_per_batch=args.max_grasps_per_batch,
                move_fingers_back_at_init=args.move_fingers_back_at_init,
                use_cpu=args.use_cpu,
            )
        )
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_code_and_scale_str} and continuing")


def eval_all_grasp_config_dicts(
    args: EvalAllGraspConfigDictsArgs,
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
) -> None:
    # Read in object codes and scales
    input_object_code_and_scale_strs_from_folder = [
        path.stem for path in list(input_grasp_config_dicts_path.glob("*.npy"))
    ]
    input_object_code_and_scale_strs = get_object_codes_and_scales_to_process(
        input_object_code_and_scale_strs=input_object_code_and_scale_strs_from_folder,
        meshdata_root_path=args.meshdata_root_path,
        output_folder_path=output_evaled_grasp_config_dicts_path,
        no_continue=False,
    )

    if args.randomize_order_seed is not None:
        random.Random(args.randomize_order_seed).shuffle(
            input_object_code_and_scale_strs
        )

    print(f"Processing {len(input_object_code_and_scale_strs)} object codes")
    print(f"First 10 object codes: {input_object_code_and_scale_strs[:10]}")

    map_fn = partial(
        run_eval_grasp_config_dict,
        args=args,
        input_grasp_config_dicts_path=input_grasp_config_dicts_path,
        output_evaled_grasp_config_dicts_path=output_evaled_grasp_config_dicts_path,
    )

    if args.use_multiprocess:
        with multiprocessing.Pool(args.num_workers) as p:
            p.map(
                map_fn,
                input_object_code_and_scale_strs,
            )
    else:
        pbar = tqdm(input_object_code_and_scale_strs, dynamic_ncols=True)
        for object_code_and_scale_str in pbar:
            pbar.set_description(f"Processing {object_code_and_scale_str}")

            run_eval_grasp_config_dict(
                object_code_and_scale_str=object_code_and_scale_str,
                args=args,
                input_grasp_config_dicts_path=input_grasp_config_dicts_path,
                output_evaled_grasp_config_dicts_path=output_evaled_grasp_config_dicts_path,
            )


def main() -> None:
    args = tyro.cli(EvalAllGraspConfigDictsArgs)
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    eval_all_grasp_config_dicts(
        args=args,
        input_grasp_config_dicts_path=args.input_grasp_config_dicts_path,
        output_evaled_grasp_config_dicts_path=args.output_evaled_grasp_config_dicts_path,
    )

    for mid_optimization_step in args.mid_optimization_steps:
        print("!" * 80)
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_grasp_config_dicts_path = (
            args.input_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_grasp_config_dicts_path = (
            args.output_evaled_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        eval_all_grasp_config_dicts(
            args=args,
            input_grasp_config_dicts_path=mid_optimization_input_grasp_config_dicts_path,
            output_evaled_grasp_config_dicts_path=mid_optimization_output_grasp_config_dicts_path,
        )


if __name__ == "__main__":
    main()
