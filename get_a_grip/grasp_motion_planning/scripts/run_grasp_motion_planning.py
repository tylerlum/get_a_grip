import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tyro

from get_a_grip.grasp_motion_planning.utils.grasp_motion_planner import (
    GraspMotionPlanner,
    GraspMotionPlannerConfig,
)
from get_a_grip.grasp_motion_planning.utils.trajopt import (
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
)
from get_a_grip.grasp_planning.scripts.run_grasp_planning import (
    GraspPlanningArgs,
    run_grasp_planning,
)
from get_a_grip.grasp_planning.utils.frames import GraspMotionPlanningFrames
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput
from get_a_grip.grasp_planning.utils.sort_grasps import get_sorted_grasps_from_dict


@dataclass
class GraspMotionPlanningArgs(GraspPlanningArgs):
    # This technically should not be a derived class in the semantic senes
    # But this is very convenient because we can use the exact same arguments for
    # grasp_planning and grasp_motion_planning
    grasp_motion_planner: GraspMotionPlannerConfig = field(
        default_factory=GraspMotionPlannerConfig
    )
    skip_grasp_planning_config_dict: Optional[pathlib.Path] = (
        None  # If not None, skip grasp planning and use these grasps instead
    )
    visualize: bool = True


def run_grasp_motion_planning(
    args: GraspMotionPlanningArgs, q_fr3_start: np.ndarray, q_algr_start: np.ndarray
) -> None:
    assert q_fr3_start.shape == (7,)
    assert q_algr_start.shape == (16,)

    # Run grasp planning
    if args.skip_grasp_planning_config_dict is None:
        start_grasp_planning = time.time()
        _, nerf_input, _, planned_grasp_config_dict = run_grasp_planning(args)
        end_grasp_planning = time.time()
        print("@" * 80)
        print(
            f"Time to run_grasp_planning: {end_grasp_planning - start_grasp_planning:.2f}s"
        )
        print("@" * 80 + "\n")
    else:
        print("@" * 80)
        print(
            f"Skipping grasp planning, reading from {args.skip_grasp_planning_config_dict} instead"
        )
        print("@" * 80 + "\n")

        assert (
            args.skip_grasp_planning_config_dict.exists()
        ), f"{args.skip_grasp_planning_config_dict} does not exist"

        # Load planned_grasp_config_dict
        planned_grasp_config_dict = np.load(
            args.skip_grasp_planning_config_dict, allow_pickle=True
        ).item()

        # Prepare nerf pipeline
        if args.nerf.nerf_config is not None:
            nerf_pipeline = args.nerf.load_nerf_pipeline()
        else:
            nerf_pipeline = args.nerf.train_nerf_pipeline(
                output_folder=args.output_folder
            )

        # Prepare nerf input
        nerf_input = NerfInput(
            nerf_pipeline=nerf_pipeline, nerf_is_z_up=args.nerf.nerf_is_z_up
        )

    # Export mesh_N
    # Must be absolute path so that curobo can find it
    mesh_N_path = (args.output_folder / f"{args.nerf.object_name}.obj").absolute()
    nerf_input.mesh_N.export(mesh_N_path)

    # Sort grasps
    X_Oy_Hs, _, q_algrs_postgrasp, q_algrs_pregrasp, sorted_losses = (
        get_sorted_grasps_from_dict(
            planned_grasp_config_dict,
            print_best=False,
            check=False,
        )
    )

    n_grasps = X_Oy_Hs.shape[0]
    assert X_Oy_Hs.shape == (n_grasps, 4, 4)
    assert q_algrs_postgrasp.shape == (n_grasps, 16)
    assert q_algrs_pregrasp.shape == (n_grasps, 16)
    assert sorted_losses.shape == (n_grasps,)

    frames = GraspMotionPlanningFrames(
        nerf_is_z_up=nerf_input.nerf_is_z_up,
        object_centroid_N_x=nerf_input.frames.object_centroid_N_x,
        object_centroid_N_y=nerf_input.frames.object_centroid_N_y,
        object_centroid_N_z=nerf_input.frames.object_centroid_N_z,
        nerf_frame_offset_W_x=0.65,
        nerf_frame_offset_W_y=0.0,
        nerf_frame_offset_W_z=0.0,
    )

    X_W_Oy = frames.X_("W", "Oy")
    X_W_Hs = np.stack([X_W_Oy @ X_Oy_Hs[i] for i in range(n_grasps)], axis=0)
    assert X_W_Hs.shape == (n_grasps, 4, 4)

    # Initialize grasp motion planner
    grasp_motion_planner = GraspMotionPlanner(
        cfg=args.grasp_motion_planner,
        frames=frames,
    )

    # Prepare curobo
    start_prepare_trajopt_batch = time.time()
    grasp_motion_planner.prepare(num_grasps=n_grasps, warmup=False)
    end_prepare_trajopt_batch = time.time()
    print("@" * 80)
    print(
        f"Time to prepare_trajopt_batch: {end_prepare_trajopt_batch - start_prepare_trajopt_batch:.2f}s"
    )
    print("@" * 80 + "\n")

    # Run curobo
    start_run_curobo = time.time()
    q_trajs, qd_trajs, T_trajs, final_success_idxs, DEBUG_TUPLE, log_dict = (
        grasp_motion_planner.plan(
            X_W_Hs=X_W_Hs,
            q_algrs_pregrasp=q_algrs_pregrasp,
            q_algrs_postgrasp=q_algrs_postgrasp,
            q_fr3_start=q_fr3_start,
            q_algr_start=q_algr_start,
            object_mesh_N_path=mesh_N_path,
        )
    )
    curobo_time = time.time()
    print("@" * 80)
    print(f"Time to run_curobo: {curobo_time - start_run_curobo:.2f}s")
    print("@" * 80 + "\n")

    print("@" * 80)
    print(f"sorted_losses = {sorted_losses}")
    print(
        f"sorted_losses of successful grasps: {[sorted_losses[i] for i in final_success_idxs]}"
    )
    print("@" * 80 + "\n")

    # Visualize
    if args.visualize:
        grasp_motion_planner.visualize(
            qs=q_trajs,
            T_trajs=T_trajs,
            success_idxs=final_success_idxs,
            sorted_losses=sorted_losses,
            DEBUG_TUPLE=DEBUG_TUPLE,
            object_mesh_N_path=mesh_N_path,
        )


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GraspMotionPlanningArgs])
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")
    run_grasp_motion_planning(
        args=args, q_fr3_start=DEFAULT_Q_FR3, q_algr_start=DEFAULT_Q_ALGR
    )


if __name__ == "__main__":
    main()
