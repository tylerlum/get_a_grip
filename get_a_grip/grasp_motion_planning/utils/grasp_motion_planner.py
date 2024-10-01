from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import transforms3d
import trimesh
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKResult, IKSolver
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenResult,
)

from get_a_grip.grasp_motion_planning.utils.ik import (
    max_penetration_from_q,
)
from get_a_grip.grasp_motion_planning.utils.trajopt_batch import (
    compute_over_limit_factors,
    get_trajectories_from_result,
    prepare_trajopt_batch,
    solve_prepared_trajopt_batch,
)
from get_a_grip.grasp_motion_planning.utils.world import (
    get_world_cfg,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.frames import GraspMotionPlanningFrames
from get_a_grip.grasp_planning.utils.nerf_input import NerfInput


@dataclass
class GraspMotionTimingConfig:
    approach_time: float = 3.0
    stay_open_time: float = 0.2
    close_time: float = 0.5
    stay_closed_time: float = 0.2
    lift_time: float = 1.0


@dataclass
class GraspMotionPlannerConfig:
    timing: GraspMotionTimingConfig = field(default_factory=GraspMotionTimingConfig)
    DEBUG_turn_off_object_collision: bool = False  # Turn off object collision for debugging why motion planning may be failing


@dataclass
class MotionPlanningComponents:
    robot_cfg: RobotConfig
    ik_solver: IKSolver
    ik_solver2: IKSolver
    motion_gen: MotionGen
    motion_gen_config: MotionGenConfig

    @classmethod
    def from_tuple(
        cls,
        components: Tuple[RobotConfig, IKSolver, IKSolver, MotionGen, MotionGenConfig],
    ) -> MotionPlanningComponents:
        robot_cfg, ik_solver, ik_solver2, motion_gen, motion_gen_config = components
        return MotionPlanningComponents(
            robot_cfg=robot_cfg,
            ik_solver=ik_solver,
            ik_solver2=ik_solver2,
            motion_gen=motion_gen,
            motion_gen_config=motion_gen_config,
        )

    def update_world(self, world_cfg: WorldConfig, visualize: bool = False) -> None:
        self.ik_solver.update_world(world_cfg)
        self.ik_solver2.update_world(world_cfg)
        self.motion_gen.update_world(world_cfg)

        if visualize:
            self.visualize_world()

    def get_world_mesh(
        self,
        bounding_box_min: Optional[np.ndarray] = None,
        bounding_box_max: Optional[np.ndarray] = None,
    ) -> trimesh.Trimesh:
        if bounding_box_min is None:
            bounding_box_min = np.array([-1, -1, -1])
        if bounding_box_max is None:
            bounding_box_max = np.array([1, 1, 1])
        assert (
            bounding_box_min.shape == (3,) and bounding_box_max.shape == (3,)
        ), f"Expected shape (3,), got {bounding_box_min.shape} and {bounding_box_max.shape}"

        pos = (bounding_box_min + bounding_box_max) / 2
        dims = bounding_box_max - bounding_box_min

        mesh = self.motion_gen.world_coll_checker.get_mesh_in_bounding_box(
            cuboid=Cuboid(
                name="DEBUG",
                pose=[*pos, 1, 0, 0, 0],
                dims=[*dims],
            )
        )
        trimesh_mesh = mesh.get_trimesh_mesh()
        return trimesh_mesh

    def visualize_world(
        self,
        bounding_box_min: Optional[np.ndarray] = None,
        bounding_box_max: Optional[np.ndarray] = None,
    ) -> None:
        # Debugging visualization to make sure world updates are working properly
        # Since curobo can sometimes silently fail if the world is not updated in the exact way it expects
        trimesh_mesh = self.get_world_mesh(
            bounding_box_min=bounding_box_min, bounding_box_max=bounding_box_max
        )
        trimesh.Scene([trimesh_mesh]).show()

    def plan(
        self,
        X_W_Hs: np.ndarray,
        q_algrs_pregrasp: np.ndarray,
        q_fr3s_start: np.ndarray,
        q_algrs_start: np.ndarray,
        timeout: float,
    ) -> Tuple[MotionGenResult, IKResult, IKResult]:
        n_grasps = X_W_Hs.shape[0]
        assert X_W_Hs.shape == (n_grasps, 4, 4)
        assert q_algrs_pregrasp.shape == (n_grasps, 16)
        assert q_fr3s_start.shape == (n_grasps, 7)
        assert q_algrs_start.shape == (n_grasps, 16)

        motion_gen_result, ik_result, ik_result2 = solve_prepared_trajopt_batch(
            X_W_Hs=X_W_Hs,
            q_algrs=q_algrs_pregrasp,
            robot_cfg=self.robot_cfg,
            ik_solver=self.ik_solver,
            ik_solver2=self.ik_solver2,
            motion_gen=self.motion_gen,
            motion_gen_config=self.motion_gen_config,
            q_fr3s_start=q_fr3s_start,
            q_algrs_start=q_algrs_start,
            enable_graph=True,
            enable_opt=False,
            timeout=timeout,
        )
        return motion_gen_result, ik_result, ik_result2


class GraspMotionPlanner:
    """Motion planner for grasping using Curobo's parallelized graph search

    The motion consists of:
    1. Approach: Move from start configuration to pre-grasp pose [requires collision-free motion planning]
    2. Stay open: Stay at pre-grasp pose [no motion planning required]
    3. Close: Close the hand from pre-grasp pose to post-grasp pose [no motion planning required]
    4. Stay closed: Stay at post-grasp pose [no motion planning required]
    5. Lift: Lift the object from grasp pose to lift pose [requires collision-free motion planning]
    """

    def __init__(
        self, cfg: GraspMotionPlannerConfig, frames: GraspMotionPlanningFrames
    ) -> None:
        self.cfg = cfg
        self.frames = frames
        self._prepared = False

    def prepare(self, num_grasps: int, warmup: bool = False) -> None:
        """Prepares Curobo's motion planning components

        Curobo's CUDAGraphs require a predefined number of grasps to be planned in parallel.

        If warmup=True, the CUDA kernels are warmed up, so that subsequent motion planning calls are faster.
        This can be done before grasp planning to amortize the cost of motion planning (if done concurrently).
        However, this will take longer overall, so it is not recommended for serial programs.
        """
        # HACK: Need to include a mesh into the world for the motion_gen warmup or else it will not prepare mesh buffers
        # TODO: There should be another way to do this to tell it there will be a mesh without actually having to load it
        #       Probably with collision_cache={"obb": 10, "mesh": 10} or similar
        dummy_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        dummy_mesh.export("/tmp/dummy.obj")
        FAR_AWAY_OBJ_XYZ = (10.0, 0.0, 0.0)
        self.approach_motion_planning = MotionPlanningComponents.from_tuple(
            prepare_trajopt_batch(
                n_grasps=num_grasps,
                collision_check_object=(
                    True if not self.cfg.DEBUG_turn_off_object_collision else False
                ),
                obj_filepath=pathlib.Path("/tmp/dummy.obj"),
                obj_xyz=FAR_AWAY_OBJ_XYZ,
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                collision_check_table=True,
                use_cuda_graph=True,
                collision_sphere_buffer=0.001,
                warmup=warmup,
            )
        )
        self.lift_motion_planning = MotionPlanningComponents.from_tuple(
            prepare_trajopt_batch(
                n_grasps=num_grasps,
                collision_check_object=(
                    True if not self.cfg.DEBUG_turn_off_object_collision else False
                ),
                obj_filepath=pathlib.Path("/tmp/dummy.obj"),
                obj_xyz=FAR_AWAY_OBJ_XYZ,
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                collision_check_table=True,
                use_cuda_graph=True,
                collision_sphere_buffer=0.001,
                warmup=warmup,
            )
        )

        self._prepared = True

    def reset(self) -> None:
        del self.approach_motion_planning
        del self.lift_motion_planning
        self._prepared = False

    def plan(
        self,
        X_W_Hs: np.ndarray,
        q_algrs_pregrasp: np.ndarray,
        q_algrs_postgrasp: np.ndarray,
        q_fr3_start: np.ndarray,
        q_algr_start: np.ndarray,
        object_mesh_N_path: Optional[pathlib.Path] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[int], tuple, dict]:
        n_grasps = X_W_Hs.shape[0]
        assert X_W_Hs.shape == (n_grasps, 4, 4)
        assert q_algrs_pregrasp.shape == (n_grasps, 16)
        assert q_fr3_start.shape == (7,)
        assert q_algr_start.shape == (16,)

        if not self._prepared:
            self.prepare(num_grasps=n_grasps, warmup=False)

        # Timing
        APPROACH_TIME = self.cfg.timing.approach_time
        STAY_OPEN_TIME = self.cfg.timing.stay_open_time
        CLOSE_TIME = self.cfg.timing.close_time
        STAY_CLOSED_TIME = self.cfg.timing.stay_closed_time
        LIFT_TIME = self.cfg.timing.lift_time

        print("\n" + "=" * 80)
        print("Step 1: Solve approach motion gen")
        print("=" * 80 + "\n")
        object_world_cfg = get_world_cfg(
            collision_check_object=(
                True if not self.cfg.DEBUG_turn_off_object_collision else False
            ),
            obj_filepath=object_mesh_N_path,
            obj_xyz=self.obj_xyz,
            obj_quat_wxyz=self.obj_quat_wxyz,
            collision_check_table=True,
            obj_name="NERF_OBJECT",  # HACK: MUST BE DIFFERENT FROM EXISTING OBJECT NAME "object" OR ELSE COLLISION DETECTION WILL FAIL
        )
        self.approach_motion_planning.update_world(object_world_cfg)
        approach_motion_gen_result, approach_ik_result, approach_ik_result2 = (
            self.approach_motion_planning.plan(
                X_W_Hs=X_W_Hs,
                q_algrs_pregrasp=q_algrs_pregrasp,
                q_fr3s_start=q_fr3_start[None, ...].repeat(n_grasps, axis=0),
                q_algrs_start=q_algr_start[None, ...].repeat(n_grasps, axis=0),
                timeout=2.0,
            )
        )
        approach_qs, approach_qds, approach_dts = get_trajectories_from_result(
            result=approach_motion_gen_result,
            desired_trajectory_time=APPROACH_TIME,
        )

        assert approach_motion_gen_result.success is not None
        approach_motion_gen_success_idxs = (
            approach_motion_gen_result.success.flatten().nonzero().flatten().tolist()
        )
        approach_ik_success_idxs = (
            approach_ik_result.success.flatten().nonzero().flatten().tolist()
        )
        approach_ik_success_idxs2 = (
            approach_ik_result2.success.flatten().nonzero().flatten().tolist()
        )
        approach_nonzero_q_idxs = [
            i
            for i, approach_q in enumerate(approach_qs)
            if np.absolute(approach_q).sum() > 1e-2
        ]
        approach_overall_success_idxs = sorted(
            list(
                set(approach_motion_gen_success_idxs)
                .intersection(
                    set(approach_ik_success_idxs).intersection(
                        set(approach_ik_success_idxs2)
                    )
                )
                .intersection(set(approach_nonzero_q_idxs))
            )
        )  # All must be successful or else it may be successful for the wrong trajectory

        print("\n" + "=" * 80)
        print(
            "Motion generation without trajectory optimization complete, printing results"
        )
        print("=" * 80 + "\n")
        print(
            f"approach_motion_gen_success_idxs: {approach_motion_gen_success_idxs} ({len(approach_motion_gen_success_idxs)} / {n_grasps} = {len(approach_motion_gen_success_idxs) / n_grasps * 100:.2f}%)"
        )
        print(
            f"approach_ik_success_idxs: {approach_ik_success_idxs} ({len(approach_ik_success_idxs)} / {n_grasps} = {len(approach_ik_success_idxs) / n_grasps * 100:.2f}%)"
        )
        print(
            f"approach_ik_success_idxs2: {approach_ik_success_idxs2} ({len(approach_ik_success_idxs2)} / {n_grasps} = {len(approach_ik_success_idxs2) / n_grasps * 100:.2f}%)"
        )
        print(
            f"approach_nonzero_q_idxs: {approach_nonzero_q_idxs} ({len(approach_nonzero_q_idxs)} / {n_grasps} = {len(approach_nonzero_q_idxs) / n_grasps * 100:.2f}%)"
        )
        print(
            f"approach_overall_success_idxs: {approach_overall_success_idxs} ({len(approach_overall_success_idxs)} / {n_grasps} = {len(approach_overall_success_idxs) / n_grasps * 100:.2f}%)"
        )

        # Fix issue with going over limit
        over_limit_factors_approach_qds = compute_over_limit_factors(
            qds=approach_qds, dts=approach_dts
        )
        approach_qds = [
            approach_qd / over_limit_factor
            for approach_qd, over_limit_factor in zip(
                approach_qds, over_limit_factors_approach_qds
            )
        ]
        approach_dts = [
            dt * over_limit_factor
            for dt, over_limit_factor in zip(
                approach_dts, over_limit_factors_approach_qds
            )
        ]

        print("\n" + "=" * 80)
        print("Step 2: Add closing motion")
        print("=" * 80 + "\n")
        closing_qs, closing_qds = [], []
        for i, (approach_q, approach_qd, approach_dt) in enumerate(
            zip(approach_qs, approach_qds, approach_dts)
        ):
            # Keep arm joints same, change hand joints
            open_q = approach_q[-1]
            close_q = np.concatenate([open_q[:7], q_algrs_postgrasp[i]])

            # Stay open
            N_STAY_OPEN_STEPS = int(STAY_OPEN_TIME / approach_dt)
            interpolated_qs0 = interpolate(
                start=open_q, end=open_q, N=N_STAY_OPEN_STEPS
            )
            assert interpolated_qs0.shape == (N_STAY_OPEN_STEPS, 23)

            # Close
            N_CLOSE_STEPS = int(CLOSE_TIME / approach_dt)
            interpolated_qs1 = interpolate(start=open_q, end=close_q, N=N_CLOSE_STEPS)
            assert interpolated_qs1.shape == (N_CLOSE_STEPS, 23)

            # Stay closed
            N_STAY_CLOSED_STEPS = int(STAY_CLOSED_TIME / approach_dt)
            interpolated_qs2 = interpolate(
                start=close_q, end=close_q, N=N_STAY_CLOSED_STEPS
            )
            assert interpolated_qs2.shape == (N_STAY_CLOSED_STEPS, 23)

            closing_q = np.concatenate(
                [interpolated_qs0, interpolated_qs1, interpolated_qs2], axis=0
            )
            assert closing_q.shape == (
                N_STAY_OPEN_STEPS + N_CLOSE_STEPS + N_STAY_CLOSED_STEPS,
                23,
            )

            closing_qd = np.diff(closing_q, axis=0) / approach_dt
            closing_qd = np.concatenate([closing_qd, closing_qd[-1:]], axis=0)

            closing_qs.append(closing_q)
            closing_qds.append(closing_qd)

        print("\n" + "=" * 80)
        print("Step 3: Solve lift motion gen")
        print("=" * 80 + "\n")
        # Using same approach_qs found from motion gen to ensure they are not starting in collision
        # Not using closing_qs because they potentially could have issues?
        q_start_lifts = np.array([approach_q[-1] for approach_q in approach_qs])
        assert q_start_lifts.shape == (n_grasps, 23)

        X_W_H_lifts = X_W_Hs.copy()
        X_W_H_lifts[:, :3, 3] = np.array([0.440870285, 0.0, 0.563780367])

        # HACK: If motion_gen above fails, then it leaves q as all 0s, which causes next step to fail
        #       So we populate those with another valid one
        assert (
            len(approach_overall_success_idxs) > 0
        ), "approach_overall_success_idxs is empty"
        valid_idx = approach_overall_success_idxs[0]
        for i in range(n_grasps):
            if i in approach_overall_success_idxs:
                continue
            q_start_lifts[i] = q_start_lifts[valid_idx]
            X_W_H_lifts[i] = X_W_H_lifts[valid_idx]

        # Update world to remove object collision check
        no_object_world_cfg = get_world_cfg(
            collision_check_object=False,
            obj_filepath=object_mesh_N_path,
            obj_xyz=self.obj_xyz,
            obj_quat_wxyz=self.obj_quat_wxyz,
            collision_check_table=True,
            obj_name="NO_OBJECT",  # HACK: MUST BE DIFFERENT FROM EXISTING OBJECT NAME "object" OR ELSE COLLISION DETECTION WILL FAIL
        )
        self.lift_motion_planning.update_world(no_object_world_cfg)
        lift_motion_gen_result, lift_ik_result, lift_ik_result2 = (
            self.lift_motion_planning.plan(
                X_W_Hs=X_W_H_lifts,
                q_algrs_pregrasp=q_algrs_pregrasp,
                q_fr3s_start=q_start_lifts[:, :7],
                q_algrs_start=q_start_lifts[:, 7:],
                timeout=1.0,
            )
        )
        raw_lift_qs, raw_lift_qds, raw_lift_dts = get_trajectories_from_result(
            result=lift_motion_gen_result, desired_trajectory_time=LIFT_TIME
        )

        assert lift_motion_gen_result.success is not None
        lift_motion_gen_success_idxs = (
            lift_motion_gen_result.success.flatten().nonzero().flatten().tolist()
        )
        lift_ik_success_idxs = (
            lift_ik_result.success.flatten().nonzero().flatten().tolist()
        )
        lift_ik_success_idxs2 = (
            lift_ik_result2.success.flatten().nonzero().flatten().tolist()
        )
        lift_nonzero_q_idxs = [
            i
            for i, raw_lift_q in enumerate(raw_lift_qs)
            if np.absolute(raw_lift_q).sum() > 1e-2
        ]
        lift_overall_success_idxs = sorted(
            list(
                set(lift_motion_gen_success_idxs)
                .intersection(
                    set(lift_ik_success_idxs).intersection(set(lift_ik_success_idxs2))
                )
                .intersection(set(lift_nonzero_q_idxs))
            )
        )  # All must be successful or else it may be successful for the wrong trajectory

        print(
            f"lift_motion_gen_success_idxs: {lift_motion_gen_success_idxs} ({len(lift_motion_gen_success_idxs)} / {n_grasps} = {len(lift_motion_gen_success_idxs) / n_grasps * 100:.2f}%)"
        )
        print(
            f"lift_ik_success_idxs: {lift_ik_success_idxs} ({len(lift_ik_success_idxs)} / {n_grasps} = {len(lift_ik_success_idxs) / n_grasps * 100:.2f}%)"
        )
        print(
            f"lift_ik_success_idxs2: {lift_ik_success_idxs2} ({len(lift_ik_success_idxs2)} / {n_grasps} = {len(lift_ik_success_idxs2) / n_grasps * 100:.2f}%)"
        )
        print(
            f"lift_nonzero_q_idxs: {lift_nonzero_q_idxs} ({len(lift_nonzero_q_idxs)} / {n_grasps} = {len(lift_nonzero_q_idxs) / n_grasps * 100:.2f}%"
        )
        print(
            f"lift_overall_success_idxs: {lift_overall_success_idxs} ({len(lift_overall_success_idxs)} / {n_grasps} = {len(lift_overall_success_idxs) / n_grasps * 100:.2f}%)"
        )

        # Need to adjust raw_lift_qs, raw_lift_qds, raw_lift_dts to match dts from trajopt
        new_raw_lift_qs, new_raw_lift_qds, new_raw_lift_dts = [], [], []
        for i, (raw_lift_q, raw_lift_qd, raw_lift_dt, approach_dt) in enumerate(
            zip(raw_lift_qs, raw_lift_qds, raw_lift_dts, approach_dts)
        ):
            n_timepoints = raw_lift_q.shape[0]
            assert raw_lift_q.shape == (n_timepoints, 23)
            total_time = n_timepoints * raw_lift_dt

            # Interpolate with new timepoints
            n_new_timepoints = int(total_time / approach_dt)
            new_raw_lift_q = np.zeros((n_new_timepoints, 23))
            for j in range(23):
                new_raw_lift_q[:, j] = np.interp(
                    np.arange(n_new_timepoints) * approach_dt,
                    np.linspace(0, total_time, n_timepoints),
                    raw_lift_q[:, j],
                )

            new_raw_lift_qd = np.diff(new_raw_lift_q, axis=0) / approach_dt
            new_raw_lift_qd = np.concatenate(
                [new_raw_lift_qd, new_raw_lift_qd[-1:]], axis=0
            )

            new_raw_lift_qs.append(new_raw_lift_q)
            new_raw_lift_qds.append(new_raw_lift_qd)
            new_raw_lift_dts.append(approach_dt)

        raw_lift_qs, raw_lift_qds, raw_lift_dts = (
            new_raw_lift_qs,
            new_raw_lift_qds,
            new_raw_lift_dts,
        )

        # Handle exceeding joint limits
        over_limit_factors_raw_lift_qds = compute_over_limit_factors(
            qds=raw_lift_qds, dts=raw_lift_dts
        )
        new2_raw_lift_qs, new2_raw_lift_qds = [], []
        for i, (raw_lift_q, raw_lift_qd, raw_lift_dt, over_limit_factor) in enumerate(
            zip(
                raw_lift_qs, raw_lift_qds, raw_lift_dts, over_limit_factors_raw_lift_qds
            )
        ):
            assert over_limit_factor >= 1.0
            if over_limit_factor > 1.0:
                print(f"Rescaling raw_lift_qs by {over_limit_factor} for grasp {i}")
                n_timepoints = raw_lift_q.shape[0]
                assert raw_lift_q.shape == (n_timepoints, 23)

                previous_total_time = n_timepoints * raw_lift_dt
                new2_total_time = previous_total_time * over_limit_factor
                new2_n_timepoints = int(new2_total_time / raw_lift_dt)

                # Interpolate with new timepoints
                new2_raw_lift_q = np.zeros((new2_n_timepoints, 23))
                for j in range(23):
                    new2_raw_lift_q[:, j] = np.interp(
                        np.arange(new2_n_timepoints) * raw_lift_dt,
                        np.linspace(0, new2_total_time, n_timepoints),
                        raw_lift_q[:, j],
                    )

                new2_raw_lift_qd = np.diff(new2_raw_lift_q, axis=0) / raw_lift_dt
                new2_raw_lift_qd = np.concatenate(
                    [new2_raw_lift_qd, new2_raw_lift_qd[-1:]], axis=0
                )

                new2_raw_lift_qs.append(new2_raw_lift_q)
                new2_raw_lift_qds.append(new2_raw_lift_qd)
            else:
                new2_raw_lift_qs.append(raw_lift_q)
                new2_raw_lift_qds.append(raw_lift_qd)
        raw_lift_qs, raw_lift_qds = new2_raw_lift_qs, new2_raw_lift_qds

        # Adjust the lift qs to have the same hand position as the closing qs
        # We only want the arm position of the lift qs
        adjusted_lift_qs, adjusted_lift_qds = [], []
        for i, (
            closing_q,
            closing_qd,
            approach_dt,
            raw_lift_q,
            raw_lift_qd,
            raw_lift_dt,
        ) in enumerate(
            zip(
                closing_qs,
                closing_qds,
                approach_dts,
                raw_lift_qs,
                raw_lift_qds,
                raw_lift_dts,
            )
        ):
            # TODO: Figure out how to handle if lift_qs has different dt, only a problem if set enable_opt=True
            assert (
                approach_dt == raw_lift_dt
            ), f"dt: {approach_dt}, lift_dt: {raw_lift_dt}"

            # Only want the arm position of the lift closing_q (keep same hand position as before)
            adjusted_lift_q = raw_lift_q.copy()
            last_closing_q = closing_q[-1]
            adjusted_lift_q[:, 7:] = last_closing_q[None, 7:]

            adjusted_lift_qd = raw_lift_qd.copy()
            adjusted_lift_qd[:, 7:] = 0.0

            adjusted_lift_qs.append(adjusted_lift_q)
            adjusted_lift_qds.append(adjusted_lift_qd)

        print("\n" + "=" * 80)
        print("Step 4: Aggregate qs and qds")
        print("=" * 80 + "\n")
        q_trajs, qd_trajs = [], []
        for (
            approach_q,
            approach_qd,
            closing_q,
            closing_qd,
            lift_q,
            lift_qd,
            approach_dt,
        ) in zip(
            approach_qs,
            approach_qds,
            closing_qs,
            closing_qds,
            adjusted_lift_qs,
            adjusted_lift_qds,
            approach_dts,
        ):
            q_traj = np.concatenate([approach_q, closing_q, lift_q], axis=0)
            qd_traj = np.diff(q_traj, axis=0) / approach_dt
            qd_traj = np.concatenate([qd_traj, qd_traj[-1:]], axis=0)
            q_trajs.append(q_traj)
            qd_trajs.append(qd_traj)

        print("\n" + "=" * 80)
        print("Step 5: Compute T_trajs")
        print("=" * 80 + "\n")
        T_trajs = []
        for q_traj, approach_dt in zip(q_trajs, approach_dts):
            n_timesteps = q_traj.shape[0]
            T_trajs.append(n_timesteps * approach_dt)

        over_limit_factors_qd_trajs = compute_over_limit_factors(
            qds=qd_trajs, dts=raw_lift_dts
        )
        no_crazy_jumps_idxs = [
            i
            for i, over_limit_factor in enumerate(over_limit_factors_qd_trajs)
            if over_limit_factor
            <= 2.0  # Actually just 1.0, but higher to be safe against numerical errors and other weirdness
        ]
        print(
            f"no_crazy_jumps_idxs: {no_crazy_jumps_idxs} ({len(no_crazy_jumps_idxs)} / {n_grasps} = {len(no_crazy_jumps_idxs) / n_grasps * 100:.2f}%)"
        )
        final_success_idxs = sorted(
            list(
                set(approach_overall_success_idxs)
                .intersection(set(lift_overall_success_idxs))
                .intersection(set(no_crazy_jumps_idxs))
            )
        )
        print("\n" + "~" * 80)
        print(
            f"final_success_idxs: {final_success_idxs} ({len(final_success_idxs)} / {n_grasps} = {len(final_success_idxs) / n_grasps * 100:.2f}%)"
        )
        print("~" * 80 + "\n")

        DEBUG_TUPLE = (
            approach_motion_gen_result,
            approach_ik_result,
            approach_ik_result2,
            lift_motion_gen_result,
            lift_ik_result,
            lift_ik_result2,
        )

        log_dict = {
            "approach_qs": approach_qs,
            "approach_qds": approach_qds,
            "dts": approach_dts,
            "closing_qs": closing_qs,
            "closing_qds": closing_qds,
            "raw_lift_qs": raw_lift_qs,
            "raw_lift_qds": raw_lift_qds,
            "adjusted_lift_qs": adjusted_lift_qs,
            "adjusted_lift_qds": adjusted_lift_qds,
            "approach_motion_gen_success_idxs": approach_motion_gen_success_idxs,
            "approach_ik_success_idxs": approach_ik_success_idxs,
            "approach_ik_success_idxs2": approach_ik_success_idxs2,
            "approach_overall_success_idxs": approach_overall_success_idxs,
            "lift_motion_gen_success_idxs": lift_motion_gen_success_idxs,
            "lift_ik_success_idxs": lift_ik_success_idxs,
            "lift_ik_success_idxs2": lift_ik_success_idxs2,
            "lift_overall_success_idxs": lift_overall_success_idxs,
            "final_success_idxs": final_success_idxs,
            "no_crazy_jumps_idxs": no_crazy_jumps_idxs,
            "over_limit_factors_qd_trajs": over_limit_factors_qd_trajs,
            "over_limit_factors_approach_qds": over_limit_factors_approach_qds,
            "over_limit_factors_raw_lift_qds": over_limit_factors_raw_lift_qds,
            "q_start_lifts": q_start_lifts,
        }
        return (
            q_trajs,
            qd_trajs,
            T_trajs,
            final_success_idxs,
            DEBUG_TUPLE,
            log_dict,
        )

    def visualize_loop(
        self,
        qs: List[np.ndarray],
        T_trajs: List[float],
        success_idxs: List[int],
        sorted_losses: np.ndarray,
        DEBUG_TUPLE: tuple,
        object_mesh_N_path: Optional[pathlib.Path] = None,
        nerf_input: Optional[NerfInput] = None,
        sorted_grasp_configs: Optional[AllegroGraspConfig] = None,
    ) -> None:
        # Import here to avoid pybullet build time if visualizer isn't used
        from get_a_grip.grasp_motion_planning.utils.visualizer import (
            animate_robot,
            create_urdf,
            draw_collision_spheres_default_config,
            remove_collision_spheres_default_config,
            set_robot_state,
            start_visualizer,
        )

        # Visualize
        print("\n" + "=" * 80)
        print("Visualizing")
        print("=" * 80 + "\n")

        OBJECT_URDF_PATH = (
            create_urdf(obj_path=object_mesh_N_path)
            if object_mesh_N_path is not None
            else None
        )
        pb_robot = start_visualizer(
            object_urdf_path=OBJECT_URDF_PATH,
            obj_xyz=self.obj_xyz,
            obj_quat_wxyz=self.obj_quat_wxyz,
        )
        draw_collision_spheres_default_config(pb_robot)
        time.sleep(1.0)

        if len(success_idxs) == 0:
            print("WARNING: No successful trajectories")

        TRAJ_IDX = success_idxs[0] if len(success_idxs) > 0 else 0

        dts = []
        for q, T_traj in zip(qs, T_trajs):
            n_timesteps = q.shape[0]
            dt = T_traj / n_timesteps
            dts.append(dt)

        remove_collision_spheres_default_config()
        q, dt = qs[TRAJ_IDX], dts[TRAJ_IDX]
        print(f"Visualizing trajectory {TRAJ_IDX}")
        animate_robot(robot=pb_robot, qs=q, dt=dt)
        if nerf_input is not None and sorted_grasp_configs is not None:
            nerf_input.plot_all_figs(
                grasp_config=sorted_grasp_configs,
                i=TRAJ_IDX,
            )

        while True:
            input_options = "\n".join(
                [
                    "=====================",
                    "OPTIONS",
                    "b for breakpoint",
                    "v to visualize traj",
                    "f to show figure of grasp and nerf inputs",
                    "i to move hand to exact X_W_H and q_algr_pre IK solution",
                    "n to go to next traj",
                    "p to go to prev traj",
                    "0 to go to traj idx 0 (or any other int to go to that idx)",
                    "c to draw collision spheres",
                    "r to remove collision spheres",
                    "q to quit",
                    f"success_idxs = {success_idxs}",
                    f"sorted_losses = {np.round([sorted_losses[i] for i in success_idxs], 2)}",
                    "=====================",
                ]
            )
            x = input("\n" + input_options + "\n\n")
            if x == "b":
                print("Breakpoint")
                breakpoint()
            elif x == "v":
                q, dt = qs[TRAJ_IDX], dts[TRAJ_IDX]
                print(f"Visualizing trajectory {TRAJ_IDX}")
                animate_robot(robot=pb_robot, qs=q, dt=dt)
            elif x == "f":
                if nerf_input is not None and sorted_grasp_configs is not None:
                    nerf_input.plot_all_figs(
                        grasp_config=sorted_grasp_configs,
                        i=TRAJ_IDX,
                    )
                else:
                    print("nerf_input and/or grasp_config is None, cannot plot figures")
            elif x == "i":
                print(
                    f"Moving hand to exact X_W_H and q_algr_pre of trajectory {TRAJ_IDX} with IK collision check"
                )
                ik_result2 = DEBUG_TUPLE[2]  # BRITTLE
                ik_q = ik_result2.solution[TRAJ_IDX].flatten().detach().cpu().numpy()
                assert ik_q.shape == (23,)
                set_robot_state(robot=pb_robot, q=ik_q)
                d_world, d_self = max_penetration_from_q(
                    q=ik_q,
                    include_object=True,
                    obj_filepath=object_mesh_N_path,
                    obj_xyz=self.obj_xyz,
                    obj_quat_wxyz=self.obj_quat_wxyz,
                    include_table=True,
                )
                print(f"np.max(d_world): {np.max(d_world)}")
                print(f"np.max(d_self): {np.max(d_self)}")
            elif x == "n":
                TRAJ_IDX += 1
                if TRAJ_IDX >= len(qs):
                    TRAJ_IDX = 0
                print(f"Updated to trajectory {TRAJ_IDX}")
            elif x == "p":
                TRAJ_IDX -= 1
                if TRAJ_IDX < 0:
                    TRAJ_IDX = len(qs) - 1
                print(f"Updated to trajectory {TRAJ_IDX}")
            elif x.isdigit():
                new_idx = int(x)
                if new_idx >= len(qs) or new_idx < 0:
                    print(
                        f"Invalid index {new_idx}, must be between 0 and {len(qs) - 1} inclusive"
                    )
                else:
                    TRAJ_IDX = new_idx
                    print(f"Updated to trajectory {TRAJ_IDX}")
            elif x == "c":
                print("Drawing collision spheres")
                draw_collision_spheres_default_config(robot=pb_robot)
            elif x == "r":
                print("Removing collision spheres")
                remove_collision_spheres_default_config()
            elif x == "q":
                print("Quitting")
                break
            else:
                print(f"Invalid input: {x}")

    @property
    def obj_xyz(self) -> Tuple[float, float, float]:
        X_W_N = self.frames.X_("W", "N")
        obj_xyz = X_W_N[:3, 3]
        return (obj_xyz[0], obj_xyz[1], obj_xyz[2])

    @property
    def obj_quat_wxyz(self) -> Tuple[float, float, float, float]:
        X_W_N = self.frames.X_("W", "N")
        obj_quat_wxyz = transforms3d.quaternions.mat2quat(X_W_N[:3, :3])
        return (obj_quat_wxyz[0], obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3])


def interpolate(start: np.ndarray, end: np.ndarray, N: int) -> np.ndarray:
    d = start.shape[0]
    assert start.shape == end.shape == (d,)
    interpolated = np.zeros((N, d))
    for i in range(d):
        interpolated[:, i] = np.linspace(start[i], end[i], N)
    return interpolated
