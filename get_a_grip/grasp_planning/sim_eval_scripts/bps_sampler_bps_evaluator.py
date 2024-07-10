import pathlib
from dataclasses import dataclass

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.config.evaluator_config import BpsEvaluatorConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    BpsRandomSamplingOptimizerConfig,
)
from get_a_grip.grasp_planning.config.planner_config import PlannerConfig
from get_a_grip.grasp_planning.config.sampler_config import BpsSamplerConfig
from get_a_grip.grasp_planning.scripts.run_grasp_planning import (
    GraspPlanningArgs,
    run_grasp_planning,
)
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs
from get_a_grip.model_training.utils.nerf_load_utils import (
    get_nerf_configs,
)


@dataclass
class Args:
    bps_sampler_ckpt_path: pathlib.Path
    bps_evaluator_ckpt_path: pathlib.Path
    nerfcheckpoints_path: pathlib.Path
    output_folder: pathlib.Path = get_data_folder() / "sim_eval_script_outputs"

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
        assert (
            self.nerfcheckpoints_path.exists()
        ), f"{self.nerfcheckpoints_path} does not exist"


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[Args])

    nerf_configs = get_nerf_configs(args.nerfcheckpoints_path)
    assert (
        len(nerf_configs) > 0
    ), f"No NERF configs found in {args.nerfcheckpoints_path}"
    print(f"Found {len(nerf_configs)} NERF configs")

    # Set up the config
    planner_cfg = PlannerConfig(
        sampler=BpsSamplerConfig(
            ckpt_path=args.bps_sampler_ckpt_path,
            num_grasps=500,
        ),
        evaluator=BpsEvaluatorConfig(
            ckpt_path=args.bps_evaluator_ckpt_path,
        ),
        optimizer=BpsRandomSamplingOptimizerConfig(
            num_grasps=5,
        ),
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
            )
        )


if __name__ == "__main__":
    main()
