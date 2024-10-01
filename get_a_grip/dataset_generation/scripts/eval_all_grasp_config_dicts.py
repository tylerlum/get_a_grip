from __future__ import annotations

import multiprocessing
import pathlib
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder, get_package_folder
from get_a_grip.dataset_generation.scripts.eval_grasp_config_dict import (
    EvalGraspConfigDictArgs,
    eval_grasp_config_dict,
)
from get_a_grip.dataset_generation.utils.hand_model import HandModelType
from get_a_grip.dataset_generation.utils.isaac_validator import ValidationType
from get_a_grip.dataset_generation.utils.process_utils import (
    get_object_codes_and_scales_to_process,
)


@dataclass
class EvalAllGraspConfigDictsArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    input_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/grasp_config_dicts"
    )
    output_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/evaled_grasp_config_dicts"
    )
    validation_type: ValidationType = ValidationType.GRAVITY_AND_TABLE
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    num_random_pose_noise_samples_per_grasp: Optional[int] = None
    gpu: int = 0
    max_grasps_per_batch: int = 5000  # Reasonable default, but some extreme objects require more GPU memory, so this can be lowered for those
    move_fingers_back_at_init: bool = False
    use_cpu: bool = False

    randomize_order_seed: Optional[int] = datetime.now().microsecond
    all_mid_optimization_steps: bool = False
    use_multiprocess: bool = True
    num_workers: int = 3


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
                hand_model_type=args.hand_model_type,
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


def run_eval_grasp_config_dict_multiprocess(
    object_code_and_scale_str: str,
    args: EvalAllGraspConfigDictsArgs,
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
):
    # HACK: Should do the same thing as run_eval_grasp_config_dict
    # However, cannot start multiple isaacgym instances unless use subprocess
    # This code is more brittle because the strings cannot be auto-updated if we rename
    script_path = (
        get_package_folder() / "dataset_generation/scripts/eval_grasp_config_dict.py"
    )
    assert script_path.exists(), f"Script path {script_path} doesn't exist"
    command = (
        f"python {script_path} "
        + f"--meshdata_root_path {args.meshdata_root_path} "
        + f"--input_grasp_config_dicts_path {input_grasp_config_dicts_path} "
        + f"--output_evaled_grasp_config_dicts_path {output_evaled_grasp_config_dicts_path} "
        + f"--object_code_and_scale_str {object_code_and_scale_str} "
        + f"--validation_type {args.validation_type} "
        + f"--hand_model_type {args.hand_model_type} "
        + f"--num_random_pose_noise_samples_per_grasp {args.num_random_pose_noise_samples_per_grasp} "
        + f"--gpu {args.gpu} "
        + f"--max_grasps_per_batch {args.max_grasps_per_batch} "
        + f"--move_fingers_back_at_init {args.move_fingers_back_at_init} "
        + f"--use_cpu {args.use_cpu}"
    )
    try:
        subprocess.run(command, shell=True, check=True)
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
        continue_ok=True,
    )

    if args.randomize_order_seed is not None:
        random.Random(args.randomize_order_seed).shuffle(
            input_object_code_and_scale_strs
        )

    map_fn = partial(
        run_eval_grasp_config_dict_multiprocess,
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
    args = tyro.cli(tyro.conf.FlagConversionOff[EvalAllGraspConfigDictsArgs])
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    eval_all_grasp_config_dicts(
        args=args,
        input_grasp_config_dicts_path=args.input_grasp_config_dicts_path,
        output_evaled_grasp_config_dicts_path=args.output_evaled_grasp_config_dicts_path,
    )

    if not args.all_mid_optimization_steps:
        return

    mid_optimization_steps = (
        sorted(
            [
                int(pp.name)
                for pp in args.input_grasp_config_dicts_path.glob("mid_optimization/*")
            ]
        )
        if (args.input_grasp_config_dicts_path / "mid_optimization").exists()
        else []
    )
    print(f"mid_optimization_steps: {mid_optimization_steps}")

    for mid_optimization_step in mid_optimization_steps:
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_path = (
            args.input_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_path = (
            args.output_evaled_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        eval_all_grasp_config_dicts(
            args=args,
            input_grasp_config_dicts_path=mid_optimization_input_path,
            output_evaled_grasp_config_dicts_path=mid_optimization_output_path,
        )


if __name__ == "__main__":
    main()
