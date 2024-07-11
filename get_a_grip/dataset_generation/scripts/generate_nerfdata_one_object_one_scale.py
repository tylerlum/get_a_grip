import pathlib
from dataclasses import dataclass

import tyro
from clean_loop_timer import LoopTimer

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.isaac_validator import (
    IsaacValidator,
    ValidationType,
)
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    object_code_and_scale_to_str,
)
from get_a_grip.dataset_generation.utils.seed import set_seed


@dataclass
class GenerateNerfDataOneObjectOneScaleArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "large/meshes"
    output_nerfdata_path: pathlib.Path = get_data_folder() / "NEW_DATASET/nerfdata"
    object_code: str = "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5"
    object_scale: float = 0.1
    num_cameras: int = 100
    gpu: int = 0
    generate_seg: bool = False
    generate_depth: bool = False
    debug_with_gui: bool = False


def generate_nerfdata_one_object_one_scale(
    args: GenerateNerfDataOneObjectOneScaleArgs,
) -> None:
    set_seed(42)

    object_code_and_scale_str = object_code_and_scale_to_str(
        args.object_code, args.object_scale
    )
    output_nerf_object_path = args.output_nerfdata_path / object_code_and_scale_str
    if output_nerf_object_path.exists():
        print(f"{output_nerf_object_path} exists, skipping {object_code_and_scale_str}")
        return

    loop_timer = LoopTimer()

    with loop_timer.add_section_timer("create sim"):
        sim = IsaacValidator(
            gpu=args.gpu,
            validation_type=ValidationType.GRAVITY_AND_TABLE,  # Object on table
            mode="gui" if args.debug_with_gui else "headless",
        )

    with loop_timer.add_section_timer("set obj asset"):
        args.output_nerfdata_path.mkdir(parents=True, exist_ok=True)
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / args.object_code / "coacd"),
            obj_file="coacd.urdf",
            vhacd_enabled=False,  # Disable vhacd because it should be faster and not needed for nerfdata generation
        )

    with loop_timer.add_section_timer("add env"):
        sim.add_env_nerfdata_collection(
            obj_scale=args.object_scale,
        )

    with loop_timer.add_section_timer("run_sim_till_object_settles_upright"):
        is_valid, log_text = sim.run_sim_till_object_settles_upright()

    with loop_timer.add_section_timer("log success or failure to txt"):
        if is_valid:
            log_successes_path = pathlib.Path(
                str(args.output_nerfdata_path) + "_settled_successes.txt"
            )
            with open(log_successes_path, "a") as f:
                f.write(f"{object_code_and_scale_str}\n")
        else:
            log_failures_path = pathlib.Path(
                str(args.output_nerfdata_path) + "_settled_failures.txt"
            )
            print(
                f"Skipping {object_code_and_scale_str} because {log_text}, writing to {log_failures_path}"
            )
            with open(log_failures_path, "a") as f:
                f.write(f"{object_code_and_scale_str}: {log_text}\n")

    with loop_timer.add_section_timer("save images light"):
        sim.save_images_lightweight(
            folder=str(output_nerf_object_path),
            generate_seg=args.generate_seg,
            generate_depth=args.generate_depth,
            num_cameras=args.num_cameras,
        )

    with loop_timer.add_section_timer("create no split data"):
        sim.create_no_split_data(folder=str(output_nerf_object_path))

    with loop_timer.add_section_timer("destroy"):
        sim.reset_simulator()
        sim.destroy()

    loop_timer.pretty_print_section_times()
    return


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GenerateNerfDataOneObjectOneScaleArgs])
    generate_nerfdata_one_object_one_scale(args)


if __name__ == "__main__":
    main()
