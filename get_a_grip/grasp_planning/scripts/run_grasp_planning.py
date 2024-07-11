import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import tyro

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.grasp_planning.utils.planner import PlannerConfig


@dataclass
class GraspPlanningArgs:
    nerf: NerfArgs
    planner: PlannerConfig
    output_folder: pathlib.Path = get_data_folder() / "grasp_planning_outputs"
    overwrite: bool = False


def run_grasp_planning(
    args: GraspPlanningArgs,
) -> Tuple[AllegroGraspConfig, NerfInput, torch.Tensor, Dict[str, np.ndarray]]:
    # Get object name
    object_name = args.nerf.object_name
    print(f"object_name = {object_name}")

    # Prepare output folder
    args.output_folder.mkdir(exist_ok=True, parents=True)
    output_file = args.output_folder / f"{object_name}.npy"
    if output_file.exists():
        if not args.overwrite:
            raise ValueError(
                f"{output_file} already exists, use --overwrite True to overwrite"
            )

        print(f"{output_file} already exists, overwriting")

    # Prepare nerf pipeline
    if args.nerf.nerf_config is not None:
        nerf_pipeline = args.nerf.load_nerf_pipeline()
    else:
        nerf_pipeline = args.nerf.train_nerf_pipeline(output_folder=args.output_folder)

    # Prepare nerf input
    nerf_input = NerfInput(
        nerf_pipeline=nerf_pipeline, nerf_is_z_up=args.nerf.nerf_is_z_up
    )

    # Prepare grasp planner
    grasp_planner = args.planner.create(nerf_input=nerf_input)

    # Plan grasps
    planned_grasp_config, losses = grasp_planner.plan()

    # Save planned grasp config
    planned_grasp_config_dict = planned_grasp_config.as_dict()
    planned_grasp_config_dict["loss"] = losses.detach().cpu().numpy()
    print(f"Saving grasp_config_dict to {output_file}")
    np.save(file=output_file, arr=planned_grasp_config_dict)

    return planned_grasp_config, nerf_input, losses, planned_grasp_config_dict


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GraspPlanningArgs])
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")
    run_grasp_planning(args)


if __name__ == "__main__":
    main()
