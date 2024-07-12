import pathlib
from dataclasses import dataclass

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
    get_nerf_configs,
)


@dataclass
class Args:
    fixed_sampler_grasp_config_dict_path: pathlib.Path
    nerf_evaluator_model_config_path: pathlib.Path
    nerfcheckpoints_path: pathlib.Path
    output_folder: pathlib.Path = (
        get_data_folder() / "sim_eval_script_outputs/fixed_sampler"
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
        sampler=FixedSamplerConfig(
            fixed_grasp_config_dict_path=args.fixed_sampler_grasp_config_dict_path,
        ),
        evaluator=NoEvaluatorConfig(),
        optimizer=NoOptimizerConfig(
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
