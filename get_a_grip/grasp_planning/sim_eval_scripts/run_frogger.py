import pathlib
from dataclasses import dataclass

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.scripts.run_frogger_grasp_planning import (
    FroggerGraspPlanningArgs,
    run_frogger_grasp_planning,
)
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs
from get_a_grip.utils.nerf_load_utils import (
    get_latest_nerf_config,
)


@dataclass
class Args:
    output_folder: pathlib.Path = get_data_folder() / "sim_eval_script_outputs/frogger"
    nerfcheckpoints_path: pathlib.Path = (
        get_data_folder() / "dataset" / "large" / "nerfcheckpoints"
    )
    object_code_and_scales_txt_path: pathlib.Path = (
        pathlib.Path(__file__).parent / "test_object_code_and_scales.txt"
    )

    def __post_init__(self) -> None:
        assert (
            self.nerfcheckpoints_path.exists()
        ), f"{self.nerfcheckpoints_path} does not exist"
        assert (
            self.object_code_and_scales_txt_path.exists()
            and self.object_code_and_scales_txt_path.suffix == ".txt"
        ), f"{self.object_code_and_scales_txt_path} does not exist or does not have a .txt suffix"


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[Args])

    with open(args.object_code_and_scales_txt_path, "r") as f:
        input_object_code_and_scale_strs_from_file = f.read().splitlines()
    nerf_configs = [
        get_latest_nerf_config(args.nerfcheckpoints_path / object_code_and_scale_str)
        for object_code_and_scale_str in input_object_code_and_scale_strs_from_file
    ]

    assert (
        len(nerf_configs) > 0
    ), f"No NERF configs found in {args.nerfcheckpoints_path}"
    print(f"Found {len(nerf_configs)} NERF configs")

    for nerf_config in tqdm(nerf_configs, desc="nerf_configs"):
        run_frogger_grasp_planning(
            FroggerGraspPlanningArgs(
                nerf=NerfArgs(
                    nerf_is_z_up=False,
                    nerf_config=nerf_config,
                ),
                num_grasps=5,
                output_folder=args.output_folder,
            )
        )


if __name__ == "__main__":
    main()
