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
from get_a_grip.grasp_planning.config.evaluator_config import BpsEvaluatorConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    BpsRandomSamplingOptimizerConfig,
)
from get_a_grip.grasp_planning.config.planner_config import PlannerConfig
from get_a_grip.grasp_planning.config.sampler_config import BpsSamplerConfig
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs


@dataclass
class Args:
    bps_sampler_ckpt_path: pathlib.Path = (
        get_data_folder()
        / "models"
        / "pretrained"
        / "bps_sampler_model"
        / "ckpt_162600.pth"
    )
    bps_evaluator_ckpt_path: pathlib.Path = (
        get_data_folder()
        / "models"
        / "pretrained"
        / "bps_evaluator_model"
        / "ckpt-3ev4ifrt-step-810.pth"
    )
    nerf_config: pathlib.Path = get_data_folder() / "TODO" / "config.yml"
    output_folder: pathlib.Path = (
        get_data_folder() / "real_eval_script_outputs/bps_sampler_bps_evaluator"
    )

    def __post_init__(self) -> None:
        assert (
            self.bps_sampler_ckpt_path.exists()
        ), f"{self.bps_sampler_ckpt_path} does not exist"
        assert self.bps_sampler_ckpt_path.suffix in [
            ".pt",
            ".pth",
        ], f"{self.bps_sampler_ckpt_path} does not have a .pt or .pth suffix"
        assert (
            self.bps_evaluator_ckpt_path.exists()
        ), f"{self.bps_evaluator_ckpt_path} does not exist"
        assert self.bps_evaluator_ckpt_path.suffix in [
            ".pt",
            ".pth",
        ], f"{self.bps_evaluator_ckpt_path} does not have a .pt or .pth suffix"
        assert self.nerf_config.exists(), f"{self.nerf_config} does not exist"
        assert self.nerf_config.suffix in [
            ".yml",
            ".yaml",
        ], f"{self.nerf_config} does not have a .yml or .yaml suffix"


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[Args])

    # Set up the config
    planner_cfg = PlannerConfig(
        sampler=BpsSamplerConfig(
            ckpt_path=args.bps_sampler_ckpt_path,
        ),
        evaluator=BpsEvaluatorConfig(
            ckpt_path=args.bps_evaluator_ckpt_path,
        ),
        optimizer=BpsRandomSamplingOptimizerConfig(),
    )

    run_grasp_motion_planning(
        GraspMotionPlanningArgs(
            nerf=NerfArgs(
                nerf_is_z_up=True,
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
