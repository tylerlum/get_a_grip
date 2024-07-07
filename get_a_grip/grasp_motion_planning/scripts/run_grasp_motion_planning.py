import time
from dataclasses import dataclass

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
from get_a_grip.grasp_planning.utils.sort_grasps import get_sorted_grasps_from_file


@dataclass
class GraspMotionPlanningArgs:
    grasp_planning: GraspPlanningArgs
    grasp_motion_planner: GraspMotionPlannerConfig
    visualize: bool = True

    def __post_init__(self) -> None:
        assert (
            self.grasp_planning.planner.optimizer.num_grasps
            == self.grasp_motion_planner.num_grasps
        ), f"num_grasps must be the same in both configs, got {self.grasp_planning.planner.optimizer.num_grasps} and {self.grasp_motion_planner.num_grasps}"


def run_grasp_motion_planning(
    args: GraspMotionPlanningArgs, q_fr3_start: np.ndarray, q_algr_start: np.ndarray
) -> None:
    assert q_fr3_start.shape == (7,)
    assert q_algr_start.shape == (16,)

    # Run grasp planning
    start_grasp_planning = time.time()
    planned_grasp_config, nerf_input, losses = run_grasp_planning(args.grasp_planning)
    end_grasp_planning = time.time()
    print("@" * 80)
    print(
        f"Time to run_grasp_planning: {end_grasp_planning - start_grasp_planning:.2f}s"
    )
    print("@" * 80 + "\n")

    # TODO: Save mesh to file so that it can be used for motion planning

    # Sort grasps
    X_Oy_Hs, _, q_algrs_postgrasp, q_algrs_pregrasp, sorted_losses = (
        get_sorted_grasps_from_file(
            optimized_grasp_config_dict_filepath=args.grasp_planning.output_folder
            / f"{args.grasp_planning.nerf.object_name}.npy",
        )
    )

    n_grasps = len(planned_grasp_config)
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
    grasp_motion_planner.prepare(warmup=False)
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
        )
    )
    # TODO: Print sorted losses that pass
    curobo_time = time.time()
    print("@" * 80)
    print(f"Time to run_curobo: {curobo_time - start_run_curobo:.2f}s")
    print("@" * 80 + "\n")

    # Visualize
    if args.visualize:
        grasp_motion_planner.visualize(
            qs=q_trajs,
            T_trajs=T_trajs,
            success_idxs=final_success_idxs,
            sorted_losses=planned_grasp_config.loss,
            DEBUG_TUPLE=DEBUG_TUPLE,
        )


def main() -> None:
    args = tyro.cli(GraspMotionPlanningArgs)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")
    run_grasp_motion_planning(
        args=args, q_fr3_start=DEFAULT_Q_FR3, q_algr_start=DEFAULT_Q_ALGR
    )


if __name__ == "__main__":
    main()
