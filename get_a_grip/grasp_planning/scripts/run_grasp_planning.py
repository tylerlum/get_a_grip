import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tyro

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.config.planner_config import PlannerConfig
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput


@dataclass
class GraspPlanningArgs:
    nerf: NerfArgs
    planner: PlannerConfig
    output_folder: pathlib.Path = get_data_folder() / "grasp_planning_outputs"
    overwrite: bool = False
    visualize_idx: Optional[int] = None
    visualize_loop: bool = False


def run_visualize_loop(
    nerf_input: NerfInput,
    planned_grasp_config: AllegroGraspConfig,
    losses: np.ndarray,
) -> None:
    B = len(planned_grasp_config)
    assert losses.shape == (B,), f"Expected losses.shape = {B}, got {losses.shape}"

    IDX = 0
    nerf_input.plot_all_figs(
        grasp_config=planned_grasp_config,
        i=IDX,
    )
    while True:
        input_options = "\n".join(
            [
                "=====================",
                f"Currently on grasp {IDX}",
                "-------------------------",
                "OPTIONS",
                "b for breakpoint",
                "f to show figure of grasp and nerf inputs",
                "n to go to next grasp",
                "p to go to prev grasp",
                "0 to go to grasp idx 0 (or any other int to go to that idx)",
                "q to quit",
                f"idxs = {np.arange(len(losses))}",
                f"losses = {np.round(losses, 2)}",
                "=====================",
            ]
        )
        x = input("\n" + input_options + "\n\n")
        if x == "b":
            print("Breakpoint")
            breakpoint()
        elif x == "f":
            nerf_input.plot_all_figs(
                grasp_config=planned_grasp_config,
                i=IDX,
            )
        elif x == "n":
            IDX += 1
            if IDX >= len(planned_grasp_config):
                IDX = 0
            print(f"Updated to grasp {IDX}")
        elif x == "p":
            IDX -= 1
            if IDX < 0:
                IDX = len(planned_grasp_config) - 1
            print(f"Updated to grasp {IDX}")
        elif x.isdigit():
            new_idx = int(x)
            if new_idx >= len(planned_grasp_config) or new_idx < 0:
                print(
                    f"Invalid index {new_idx}, must be between 0 and {len(planned_grasp_config) - 1} inclusive"
                )
            else:
                IDX = new_idx
                print(f"Updated to grasp {IDX}")
        elif x == "q":
            print("Quitting")
            break
        else:
            print(f"Invalid input: {x}")


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

    if args.visualize_loop:
        run_visualize_loop(
            nerf_input=nerf_input,
            planned_grasp_config=planned_grasp_config,
            losses=losses.detach().cpu().numpy(),
        )
    elif args.visualize_idx is not None:
        nerf_input.plot_all_figs(
            grasp_config=planned_grasp_config,
            i=args.visualize_idx,
        )

    return planned_grasp_config, nerf_input, losses, planned_grasp_config_dict


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GraspPlanningArgs])
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")
    run_grasp_planning(args)


if __name__ == "__main__":
    main()
