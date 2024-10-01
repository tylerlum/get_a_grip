import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tyro

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.scripts.run_grasp_planning import run_visualize_loop
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.frames import GraspMotionPlanningFrames
from get_a_grip.grasp_planning.utils.frogger_utils import (
    FroggerArgs,
    custom_coll_callback,
    frogger_to_grasp_config_dict,
)
from get_a_grip.grasp_planning.utils.nerf_args import NerfArgs
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput


@dataclass
class FroggerGraspPlanningArgs:
    nerf: NerfArgs
    num_grasps: int = 32
    output_folder: pathlib.Path = get_data_folder() / "frogger_outputs"

    visualize_idx: Optional[int] = 0
    visualize_loop: bool = False


def run_frogger_grasp_planning(
    args: FroggerGraspPlanningArgs,
) -> Tuple[AllegroGraspConfig, NerfInput, torch.Tensor, Dict[str, np.ndarray]]:
    # Prepare nerf pipeline
    if args.nerf.nerf_config is not None:
        nerf_pipeline = args.nerf.load_nerf_pipeline()
    else:
        nerf_pipeline = args.nerf.train_nerf_pipeline(output_folder=args.output_folder)

    # Prepare nerf input
    nerf_input = NerfInput(
        nerf_pipeline=nerf_pipeline, nerf_is_z_up=args.nerf.nerf_is_z_up
    )

    # Export mesh_N
    # Must be absolute path so that curobo can find it
    mesh_N_path = (args.output_folder / f"{args.nerf.object_name}.obj").absolute()
    nerf_input.mesh_N.export(mesh_N_path)

    # Handle frames
    frames = GraspMotionPlanningFrames(
        nerf_is_z_up=nerf_input.nerf_is_z_up,
        object_centroid_N_x=nerf_input.frames.object_centroid_N_x,
        object_centroid_N_y=nerf_input.frames.object_centroid_N_y,
        object_centroid_N_z=nerf_input.frames.object_centroid_N_z,
        nerf_frame_offset_W_x=0.65,
        nerf_frame_offset_W_y=0.0,
        nerf_frame_offset_W_z=0.0,
    )

    # Run frogger
    # Frogger requires X_W_O, which is the O frame wrt W frame
    # with O frame at the horizontal position of the object, but vertical position of the table surface, with z-up
    # This is the same as Bz frame in our system
    X_W_Bz = frames.X_("W", "Bz")
    grasp_config_dict = frogger_to_grasp_config_dict(
        args=FroggerArgs(
            obj_filepath=mesh_N_path,
            obj_scale=0.9999,
            obj_code=args.nerf.object_name,
            obj_is_yup=nerf_input.nerf_is_z_up,
            num_grasps=args.num_grasps,
            output_grasp_config_dicts_folder=args.output_folder,
            visualize=False,
            max_time=60,
        ),
        X_W_O=X_W_Bz,
        mesh=None,
        custom_coll_callback=custom_coll_callback,
    )
    losses = torch.from_numpy(grasp_config_dict["loss"]).float()
    grasp_config = AllegroGraspConfig.from_grasp_config_dict(grasp_config_dict)

    if args.visualize_loop:
        run_visualize_loop(
            nerf_input=nerf_input,
            planned_grasp_config=grasp_config,
            losses=losses.detach().cpu().numpy(),
        )
    elif args.visualize_idx is not None:
        nerf_input.plot_all_figs(
            grasp_config=grasp_config,
            i=args.visualize_idx,
        )

    return grasp_config, nerf_input, losses, grasp_config_dict


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[FroggerGraspPlanningArgs])
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")
    run_frogger_grasp_planning(args)


if __name__ == "__main__":
    main()
