import pathlib
from dataclasses import dataclass

import tyro

from get_a_grip import get_data_folder
from get_a_grip.grasp_motion_planning.scripts.run_grasp_motion_planning import (
    GraspMotionPlanningArgs,
    run_grasp_motion_planning,
)
from get_a_grip.grasp_motion_planning.utils.trajopt import (
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
)
from get_a_grip.grasp_planning.config.evaluator_config import NerfEvaluatorConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    NerfRandomSamplingOptimizerConfig,
)
from get_a_grip.grasp_planning.config.planner_config import PlannerConfig
from get_a_grip.grasp_planning.config.sampler_config import FixedSamplerConfig
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs


@dataclass
class Args:
    fixed_sampler_grasp_config_dict_path: pathlib.Path = (
        get_data_folder()
        / "fixed_sampler_grasp_config_dicts"
        / "given"
        / "one_good_grasp_per_object.npy"
    )
    nerf_evaluator_model_config_path: pathlib.Path = (
        get_data_folder() / "models" / "pretrained" / "nerf_evaluator_model" / "TODO"
    )
    nerf_config: pathlib.Path = (
        get_data_folder()
        / "dataset"
        / "large"
        / "nerfcheckpoints"
        / "core-bottle-1071fa4cddb2da2fc8724d5673a063a6_0_0709"
        / "nerfacto"
        / "2024-07-12_192529"
        / "config.yml"
    )
    output_folder: pathlib.Path = (
        get_data_folder() / "sim_eval_script_outputs/fixed_sampler_nerf_evaluator"
    )

    def __post_init__(self) -> None:
        assert (
            self.fixed_sampler_grasp_config_dict_path.exists()
        ), f"{self.fixed_sampler_grasp_config_dict_path} does not exist"
        assert (
            self.fixed_sampler_grasp_config_dict_path.suffix == ".npy"
        ), f"{self.fixed_sampler_grasp_config_dict_path} does not have a .npy suffix"
        assert (
            self.nerf_evaluator_model_config_path.exists()
        ), f"{self.nerf_evaluator_model_config_path} does not exist"
        assert (
            self.nerf_evaluator_model_config_path.suffix
            in [
                ".yml",
                ".yaml",
            ]
        ), f"{self.nerf_evaluator_model_config_path} does not have a .yml or .yaml suffix"
        assert self.nerf_config.exists(), f"{self.nerf_config} does not exist"
        assert self.nerf_config.suffix in [
            ".yml",
            ".yaml",
        ], f"{self.nerf_config} does not have a .yml or .yaml suffix"


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[Args])

    # Set up the config
    planner_cfg = PlannerConfig(
        sampler=FixedSamplerConfig(
            fixed_grasp_config_dict_path=args.fixed_sampler_grasp_config_dict_path,
        ),
        evaluator=NerfEvaluatorConfig(
            nerf_evaluator_model_config_path=args.nerf_evaluator_model_config_path,
        ),
        optimizer=NerfRandomSamplingOptimizerConfig(),
    )

    run_grasp_motion_planning(
        GraspMotionPlanningArgs(
            nerf=NerfArgs(
                nerf_is_z_up=False,
                nerf_config=args.nerf_config,
            ),
            planner=planner_cfg,
            output_folder=args.output_folder,
            overwrite=True,
            visualize_loop=True,
        ),
        q_fr3_start=DEFAULT_Q_FR3,
        q_algr_start=DEFAULT_Q_ALGR,
    )


if __name__ == "__main__":
    main()
