import multiprocessing
import pathlib
import random
import subprocess
from dataclasses import dataclass
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder, get_package_folder
from get_a_grip.dataset_generation.scripts.generate_nerfdata_one_object_one_scale import (
    GenerateNerfDataOneObjectOneScaleArgs,
    generate_nerfdata_one_object_one_scale,
)
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.dataset_generation.utils.process_utils import (
    get_object_codes_and_scales_to_process,
)


@dataclass
class GenerateNerfDataArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "large/meshes"
    input_object_code_and_scales_txt_path: pathlib.Path = (
        get_data_folder() / "NEW_DATASET/object_code_and_scales.txt"
    )
    output_nerfdata_path: pathlib.Path = get_data_folder() / "NEW_DATASET/nerfdata"
    num_cameras: int = 250
    randomize_order_seed: Optional[int] = None
    gpu: int = 0
    use_multiprocess: bool = True
    num_workers: int = 4
    continue_ok: bool = True


def run_command(
    object_code: str,
    object_scale: float,
    args: GenerateNerfDataArgs,
):
    try:
        generate_nerfdata_one_object_one_scale(
            GenerateNerfDataOneObjectOneScaleArgs(
                meshdata_root_path=args.meshdata_root_path,
                output_nerfdata_path=args.output_nerfdata_path,
                object_code=object_code,
                object_scale=object_scale,
                num_cameras=args.num_cameras,
                gpu=args.gpu,
            )
        )
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_code} and continuing")
    print(f"Finished object {object_code}.")


def run_command_multiprocess(
    object_code: str,
    object_scale: float,
    args: GenerateNerfDataArgs,
):
    # HACK: Should do the same thing as run_command
    # However, cannot start multiple isaacgym instances unless use subprocess
    # This code is more brittle because the strings cannot be auto-updated if we rename
    script_path = (
        get_package_folder()
        / "dataset_generation/scripts/generate_nerfdata_one_object_one_scale.py"
    )
    assert script_path.exists(), f"Script path {script_path} doesn't exist"

    command = (
        f"python {script_path} "
        + f"--meshdata_root_path {args.meshdata_root_path} "
        + f"--output_nerfdata_path {args.output_nerfdata_path} "
        + f"--object_code {object_code} "
        + f"--object_scale {object_scale} "
        + f"--num_cameras {args.num_cameras} "
        + f"--gpu {args.gpu}"
    )
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_code} and continuing")
    print(f"Finished object {object_code}.")


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GenerateNerfDataArgs])

    # Read in object codes and scales
    if not args.input_object_code_and_scales_txt_path.exists():
        raise ValueError(
            f"input_object_code_and_scales_txt_path {args.input_object_code_and_scales_txt_path} doesn't exist"
        )
    with open(args.input_object_code_and_scales_txt_path, "r") as f:
        input_object_code_and_scale_strs_from_file = f.read().splitlines()
    input_object_code_and_scale_strs = get_object_codes_and_scales_to_process(
        input_object_code_and_scale_strs=input_object_code_and_scale_strs_from_file,
        meshdata_root_path=args.meshdata_root_path,
        output_folder_path=args.output_nerfdata_path,
        continue_ok=args.continue_ok,
    )
    assert len(input_object_code_and_scale_strs) > 0

    # Randomize order
    if args.randomize_order_seed is not None:
        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(
            input_object_code_and_scale_strs
        )

    input_object_codes, input_object_scales = [], []
    for object_code_and_scale_str in input_object_code_and_scale_strs:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        input_object_codes.append(object_code)
        input_object_scales.append(object_scale)

    if args.use_multiprocess:
        print(f"Using multiprocessing with {args.num_workers} workers.")
        with multiprocessing.Pool(args.num_workers) as p:
            p.starmap(
                run_command_multiprocess,
                zip(
                    input_object_codes,
                    input_object_scales,
                    [args] * len(input_object_codes),
                ),
            )
    else:
        for i, (object_code, object_scale) in tqdm(
            enumerate(zip(input_object_codes, input_object_scales)),
            desc="Generating NeRF data for all objects",
            dynamic_ncols=True,
            total=len(input_object_codes),
        ):
            run_command(
                object_code=object_code,
                object_scale=object_scale,
                args=args,
            )


if __name__ == "__main__":
    main()
