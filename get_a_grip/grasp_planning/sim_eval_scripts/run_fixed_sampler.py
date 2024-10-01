import pathlib
from dataclasses import dataclass
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.config.evaluator_config import NoEvaluatorConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    NoOptimizerConfig,
)
from get_a_grip.grasp_planning.config.planner_config import PlannerConfig
from get_a_grip.grasp_planning.config.sampler_config import FixedSamplerConfig
from get_a_grip.grasp_planning.scripts.run_grasp_planning import (
    GraspPlanningArgs,
    run_grasp_planning,
)
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs
from get_a_grip.utils.nerf_load_utils import (
    get_latest_nerf_config,
)


@dataclass
class Args:
    fixed_sampler_grasp_config_dict_path: pathlib.Path = (
        get_data_folder()
        / "fixed_sampler_grasp_config_dicts"
        / "given"
        / "one_good_grasp_per_object.npy"
    )
    output_folder: pathlib.Path = (
        get_data_folder() / "sim_eval_script_outputs/fixed_sampler"
    )
    nerfcheckpoints_path: pathlib.Path = (
        get_data_folder() / "dataset" / "large" / "nerfcheckpoints"
    )
    object_code_and_scales_txt_path: pathlib.Path = (
        pathlib.Path(__file__).parent / "test_object_code_and_scales.txt"
    )
    visualize_idx: Optional[int] = None

    def __post_init__(self) -> None:
        assert (
            self.fixed_sampler_grasp_config_dict_path.exists()
        ), f"{self.fixed_sampler_grasp_config_dict_path} does not exist"
        assert (
            self.fixed_sampler_grasp_config_dict_path.suffix == ".npy"
        ), f"{self.fixed_sampler_grasp_config_dict_path} does not have a .pt or .pth suffix"
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
    ), f"No NERF configs found in {args.nerfcheckpoints_path} for the object codes and scales in {args.object_code_and_scales_txt_path}"

    print(f"Found {len(nerf_configs)} NERF configs")

    # Set up the config
    planner_cfg = PlannerConfig(
        sampler=FixedSamplerConfig(
            fixed_grasp_config_dict_path=args.fixed_sampler_grasp_config_dict_path,
            max_num_grasps=500,
        ),
        evaluator=NoEvaluatorConfig(),
        optimizer=NoOptimizerConfig(num_grasps=5),
    )

    for nerf_config in tqdm(nerf_configs, desc="nerf_configs"):
        run_grasp_planning(
            GraspPlanningArgs(
                nerf=NerfArgs(
                    nerf_is_z_up=False,
                    nerf_config=nerf_config,
                ),
                planner=planner_cfg,
                output_folder=args.output_folder,
                overwrite=True,
                visualize_idx=args.visualize_idx,
            )
        )


if __name__ == "__main__":
    main()
