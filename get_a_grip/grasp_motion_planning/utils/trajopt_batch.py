import pathlib
from typing import List, Optional, Tuple

import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
)
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKResult, IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
)

from get_a_grip.dataset_generation.utils.hand_model import (
    HandModelType,
)
from get_a_grip.dataset_generation.utils.torch_quat_utils import (
    matrix_to_quat_wxyz,
)
from get_a_grip.grasp_motion_planning.utils.joint_limit_utils import (
    modify_robot_cfg_to_add_joint_limit_buffer,
)
from get_a_grip.grasp_motion_planning.utils.trajopt import (
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
)
from get_a_grip.grasp_motion_planning.utils.world import get_world_cfg
from get_a_grip.grasp_planning.utils.joint_limit_utils import (
    is_in_limits_np,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def debug_start_state_invalid(
    motion_gen_config: MotionGenConfig,
    start_state: JointState,
) -> None:
    # DEBUG
    graph_planner = motion_gen_config.graph_planner
    x_init_batch = start_state.position
    node_set = x_init_batch
    mask = graph_planner.mask_samples(x_init_batch)
    if not mask.all():
        print("^" * 80)
        print("PROBLEM WITH START_STATE")
        print("^" * 80 + "\n")
        print(f"mask: {mask}")
        print(f"x_init_batch: {x_init_batch}")
        print(f"mask.nonzero(): {mask.nonzero()}")
        print(f"mask.nonzero().shape: {mask.nonzero().shape}")
        print(f"torch.logical_not(mask).nonzero(): {torch.logical_not(mask).nonzero()}")
        print(
            f"torch.logical_not(mask).nonzero().shape: {torch.logical_not(mask).nonzero().shape}"
        )
        act_seq = node_set.unsqueeze(1)
        state = graph_planner.safety_rollout_fn.dynamics_model.forward(
            graph_planner.safety_rollout_fn.start_state, act_seq
        )
        metrics = graph_planner.safety_rollout_fn.constraint_fn(
            state, use_batch_env=False
        )
        bound_constraint = graph_planner.safety_rollout_fn.bound_constraint.forward(
            state.state_seq
        )
        coll_constraint = (
            graph_planner.safety_rollout_fn.primitive_collision_constraint.forward(
                state.robot_spheres, env_query_idx=None
            )
        )
        self_constraint = (
            graph_planner.safety_rollout_fn.robot_self_collision_constraint.forward(
                state.robot_spheres
            )
        )
        print(f"metrics: {metrics}")
        print(f"bound_constraint: {bound_constraint}")
        print(f"coll_constraint: {coll_constraint}")
        print(f"self_constraint: {self_constraint}")
        breakpoint()
    else:
        print("mask.all() == True, so no issues with start_state")


def prepare_trajopt_batch(
    n_grasps: int,
    collision_check_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = None,
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision_check_table: bool = True,
    use_cuda_graph: bool = True,
    collision_sphere_buffer: Optional[float] = None,
    warmup: bool = True,  # warmup=True helps amortize downstream calls, but makes it slower overall
    rotation_threshold: float = 0.01,
    position_threshold: float = 0.001,
    rotation_threshold2: float = 0.05,
    position_threshold2: float = 0.005,
) -> Tuple[RobotConfig, IKSolver, IKSolver, MotionGen, MotionGenConfig]:
    """Sets up the necessary objects for solving trajopt_batch and runs warmup on motion_gen"""
    # robot_cfg
    tensor_args = TensorDeviceType()
    robot_file = "fr3_algr_zed2i_with_fingertips.yml"
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    if collision_sphere_buffer is not None:
        robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_sphere_buffer
    robot_cfg = RobotConfig.from_dict(robot_cfg)
    modify_robot_cfg_to_add_joint_limit_buffer(robot_cfg)

    # world_cfg
    world_cfg = get_world_cfg(
        collision_check_object=collision_check_object,
        obj_filepath=obj_filepath,
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        collision_check_table=collision_check_table,
        obj_name="MY_object",
    )

    # ik_solver
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=rotation_threshold,
        position_threshold=position_threshold,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
    )
    ik_solver = IKSolver(ik_config)

    # ik_solver2
    ik_config2 = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=rotation_threshold2,
        position_threshold=position_threshold2,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
    )
    ik_solver2 = IKSolver(ik_config2)

    # motion_gen
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=use_cuda_graph,
        # collision_cache={"obb": 10, "mesh": 10},  # Possible method to ensure that the cache is large enough so that update_world calls work properly
    )
    motion_gen = MotionGen(motion_gen_config)

    if warmup:
        # Careful: make sure warmup inputs are correct or else causes issues with CUDA graph
        motion_gen.warmup(batch=n_grasps, enable_graph=True, use_link_poses=True)

    return robot_cfg, ik_solver, ik_solver2, motion_gen, motion_gen_config


def solve_prepared_trajopt_batch(
    X_W_Hs: np.ndarray,
    q_algrs: np.ndarray,
    robot_cfg: RobotConfig,
    ik_solver: IKSolver,
    ik_solver2: IKSolver,
    motion_gen: MotionGen,
    motion_gen_config: MotionGenConfig,
    q_fr3s_start: Optional[np.ndarray] = None,
    q_algrs_start: Optional[np.ndarray] = None,
    enable_graph: bool = True,
    enable_opt: bool = False,  # Getting some errors from setting this to True
    timeout: float = 5.0,
) -> Tuple[MotionGenResult, IKResult, IKResult]:
    print("Step 1: Prepare inputs")
    N_GRASPS = X_W_Hs.shape[0]
    assert X_W_Hs.shape == (N_GRASPS, 4, 4), f"X_W_Hs.shape: {X_W_Hs.shape}"
    assert q_algrs.shape == (N_GRASPS, 16), f"q_algrs.shape: {q_algrs.shape}"
    assert is_in_limits_np(
        q_algrs, hand_model_type=HandModelType.ALLEGRO
    ).all(), f"q_algrs: {q_algrs}"

    if q_fr3s_start is None:
        print("Using default q_fr3_starts")
        q_fr3s_start = DEFAULT_Q_FR3[None, ...].repeat(N_GRASPS, axis=0)
    assert q_fr3s_start.shape == (
        N_GRASPS,
        7,
    ), f"q_fr3_starts.shape: {q_fr3s_start.shape}"
    if q_algrs_start is None:
        print("Using default q_algr_starts")
        q_algrs_start = DEFAULT_Q_ALGR[None, ...].repeat(N_GRASPS, axis=0)
    assert q_algrs_start.shape == (
        N_GRASPS,
        16,
    ), f"q_algr_starts.shape: {q_algrs_start.shape}"

    print("Step 2: Prepare target_pose")
    trans = X_W_Hs[:, :3, 3]
    rot_matrix = X_W_Hs[:, :3, :3]
    quat_wxyz = matrix_to_quat_wxyz(torch.from_numpy(rot_matrix).float().to(device))
    target_pose = Pose(
        torch.from_numpy(trans).float().to(device),
        quaternion=quat_wxyz,
    )

    print("Step 3: Solve IK for arm q")
    ik_result = ik_solver.solve_batch(target_pose)

    print("Step 4: Solve FK for fingertip poses")
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    q_fr3s = ik_result.solution[..., :7].squeeze(dim=1).detach().cpu().numpy()
    q = torch.from_numpy(np.concatenate([q_fr3s, q_algrs], axis=1)).float().to(device)
    assert q.shape == (N_GRASPS, 23)
    state = kin_model.get_state(q)

    print("Step 5: Solve IK for arm q and hand q")
    ik_result2 = ik_solver2.solve_batch(
        goal_pose=target_pose, link_poses=state.link_pose
    )

    print("Step 6: Solve FK for new fingertip poses")
    kin_model2 = CudaRobotModel(robot_cfg.kinematics)
    q2 = ik_result2.solution.squeeze(dim=1)
    assert q2.shape == (N_GRASPS, 23)
    state2 = kin_model2.get_state(q2)

    print("Step 7: Set up start_state for motion generation")
    q_starts = (
        torch.from_numpy(
            np.concatenate(
                [
                    q_fr3s_start,
                    q_algrs_start,
                ],
                axis=1,
            )
        )
        .float()
        .to(device)
    )
    # Can't succeed in motion planning if joint limits are violated
    CHECK_JOINT_LIMITS = True
    if CHECK_JOINT_LIMITS:
        joint_limits = robot_cfg.kinematics.kinematics_config.joint_limits.position
        assert joint_limits.shape == (2, 23)
        joint_lower_limits, joint_upper_limits = joint_limits[0], joint_limits[1]
        if (q_starts < joint_lower_limits[None]).any() or (
            q_starts > joint_upper_limits[None]
        ).any():
            print("#" * 80)
            print("q_starts out of joint limits!")
            print(f"joint_lower_limits = {joint_lower_limits}")
            print(f"joint_upper_limits = {joint_upper_limits}")
            print("#" * 80)

            # Check if out of joint limits due to small numerical issues
            eps = 1e-4
            if (q_starts > joint_lower_limits[None] - eps).all() and (
                q_starts < joint_upper_limits[None] + eps
            ).all():
                print(
                    f"q_starts is close to joint limits within {eps}, so clamping to limits with some margin"
                )
                q_starts = torch.clamp(
                    q_starts,
                    joint_lower_limits[None] + eps,
                    joint_upper_limits[None] - eps,
                )
            else:
                FAIL_IF_OUT_OF_JOINT_LIMITS = False
                if FAIL_IF_OUT_OF_JOINT_LIMITS:
                    print("q_starts is far from joint limits, so not clamping")
                    print(
                        f"q_starts = {q_starts}, joint_lower_limits = {joint_lower_limits}, joint_upper_limits = {joint_upper_limits}"
                    )
                    print(
                        f"q_starts < joint_lower_limits = {(q_starts < joint_lower_limits[None]).any()}"
                    )
                    print(
                        f"q_starts > joint_upper_limits = {(q_starts > joint_upper_limits[None]).any()}"
                    )
                    print(
                        f"(q_starts < joint_lower_limits[None]).nonzero() = {(q_starts < joint_lower_limits[None]).nonzero()}"
                    )
                    print(
                        f"(q_starts > joint_upper_limits[None]).nonzero() = {(q_starts > joint_upper_limits[None]).nonzero()}"
                    )
                    raise ValueError("q_starts out of joint limits!")
                else:
                    print(
                        "q_starts is far from joint limits, but still clamping to limits with margin to continue"
                    )
                    q_starts = torch.clamp(
                        q_starts,
                        joint_lower_limits[None] + eps,
                        joint_upper_limits[None] - eps,
                    )
    start_state = JointState.from_position(q_starts)

    DEBUG_START_STATE_INVALID = True
    if DEBUG_START_STATE_INVALID:
        debug_start_state_invalid(
            motion_gen_config=motion_gen_config, start_state=start_state
        )

    print("Step 8: Solve motion generation")
    target_pose2 = Pose(
        state2.ee_position,
        quaternion=state2.ee_quaternion,
    )
    motion_result = motion_gen.plan_batch(
        start_state=start_state,
        goal_pose=target_pose2,
        plan_config=MotionGenPlanConfig(
            enable_graph=enable_graph,
            enable_opt=enable_opt,
            # max_attempts=1,  # Reduced to save time? Actually doesn't make a noticeable difference
            # num_trajopt_seeds=1,  # Reduced to save time? Actually doesn't make a noticeable difference
            num_graph_seeds=1,  # Must be 1 for plan_batch
            timeout=timeout,
        ),
        link_poses=state2.link_pose,
    )
    return motion_result, ik_result, ik_result2


def solve_trajopt_batch(
    X_W_Hs: np.ndarray,
    q_algrs: np.ndarray,
    q_fr3_starts: Optional[np.ndarray] = None,
    q_algr_starts: Optional[np.ndarray] = None,
    collision_check_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = None,
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision_check_table: bool = True,
    use_cuda_graph: bool = True,
    enable_graph: bool = True,
    enable_opt: bool = False,  # Getting some errors from setting this to True
    timeout: float = 5.0,
    collision_sphere_buffer: Optional[float] = None,
) -> Tuple[MotionGenResult, IKResult, IKResult]:
    N_GRASPS = X_W_Hs.shape[0]
    robot_cfg, ik_solver, ik_solver2, motion_gen, motion_gen_config = (
        prepare_trajopt_batch(
            n_grasps=N_GRASPS,
            collision_check_object=collision_check_object,
            obj_filepath=obj_filepath,
            obj_xyz=obj_xyz,
            obj_quat_wxyz=obj_quat_wxyz,
            collision_check_table=collision_check_table,
            use_cuda_graph=use_cuda_graph,
            collision_sphere_buffer=collision_sphere_buffer,
            warmup=False,  # warmup=False because we are not using motion_gen for anything else with this
        )
    )

    return solve_prepared_trajopt_batch(
        X_W_Hs=X_W_Hs,
        q_algrs=q_algrs,
        robot_cfg=robot_cfg,
        ik_solver=ik_solver,
        ik_solver2=ik_solver2,
        motion_gen=motion_gen,
        motion_gen_config=motion_gen_config,
        q_fr3s_start=q_fr3_starts,
        q_algrs_start=q_algr_starts,
        enable_graph=enable_graph,
        enable_opt=enable_opt,
        timeout=timeout,
    )


def get_trajectories_from_result(
    result: MotionGenResult,
    desired_trajectory_time: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    used_trajopt = result.trajopt_time > 0

    paths = result.get_paths()

    qs, qds, dts = [], [], []
    for i, path in enumerate(paths):
        q = path.position.detach().cpu().numpy()
        qd = path.velocity.detach().cpu().numpy()

        n_timesteps = q.shape[0]
        assert q.shape == (n_timesteps, 23)

        if used_trajopt:
            # When using trajopt, interpolation_dt is correct and qd is populated
            assert (np.absolute(qd) > 1e-4).any()  # qd is populated

            if desired_trajectory_time is not None:
                print(
                    "WARNING: desired_trajectory_time is provided, but trajopt is used, so it is ignored"
                )

            dt = result.interpolation_dt
            total_time = n_timesteps * dt
        else:
            # Without trajopt, it is simply a linear interpolation between waypoints
            # with a fixed number of timesteps, so interpolation_dt is way too big
            # and qd is not populated
            assert (np.absolute(qd) < 1e-4).all()  # qd is not populated
            assert (
                n_timesteps * result.interpolation_dt > 60
            )  # interpolation_dt is too big

            assert desired_trajectory_time is not None
            total_time = desired_trajectory_time
            dt = total_time / n_timesteps

            qd = np.diff(q, axis=0) / dt
            qd = np.concatenate([qd, qd[-1:]], axis=0)
            assert qd.shape == q.shape

        qs.append(q)
        qds.append(qd)
        dts.append(dt)

    return (
        qs,
        qds,
        dts,
    )


def compute_over_limit_factors(
    qds: List[np.ndarray], dts: List[float], safety_limit_rescaling: float = 0.25
) -> List[float]:
    """Computes factors to rescale qd to stay within limits for each trajectory
    Lower safety_limit_rescaling is more conservative and further from limits
    """
    n_trajs = len(qds)
    assert len(dts) == n_trajs
    for qd in qds:
        assert qd.shape[1] == 23 and len(qd.shape) == 2
    assert (
        0 < safety_limit_rescaling <= 1
    ), f"safety_limit_rescaling: {safety_limit_rescaling}"

    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    qd_limits = (
        robot_cfg.kinematics.kinematics_config.joint_limits.velocity.detach()
        .cpu()
        .numpy()
    )
    qdd_limits = (
        robot_cfg.kinematics.kinematics_config.joint_limits.acceleration.detach()
        .cpu()
        .numpy()
    )
    qddd_limits = (
        robot_cfg.kinematics.kinematics_config.joint_limits.jerk.detach().cpu().numpy()
    )

    # Rescale limits by safety_limit_rescaling to be conservative
    qd_limits *= safety_limit_rescaling
    qdd_limits *= safety_limit_rescaling
    qddd_limits *= safety_limit_rescaling

    qd_limits_min, qd_limits_max = qd_limits[0], qd_limits[1]
    qdd_limits_min, qdd_limits_max = qdd_limits[0], qdd_limits[1]
    qddd_limits_min, qddd_limits_max = qddd_limits[0], qddd_limits[1]

    assert qd_limits_min.shape == qd_limits_max.shape == (23,)
    assert qdd_limits_min.shape == qdd_limits_max.shape == (23,)
    assert qddd_limits_min.shape == qddd_limits_max.shape == (23,)

    assert (qd_limits_min < 0).all()
    assert (qd_limits_max > 0).all()
    assert (qdd_limits_min < 0).all()
    assert (qdd_limits_max > 0).all()
    assert (qddd_limits_min < 0).all()
    assert (qddd_limits_max > 0).all()

    rescale_factors = []
    for i, (qd, dt) in enumerate(zip(qds, dts)):
        n_timesteps = qd.shape[0]
        assert qd.shape == (n_timesteps, 23)
        qdd = np.diff(qd, axis=0) / dt
        qddd = np.diff(qdd, axis=0) / dt

        qd_min = qd.min(axis=0)
        qd_max = qd.max(axis=0)
        qdd_min = qdd.min(axis=0)
        qdd_max = qdd.max(axis=0)
        qddd_min = qddd.min(axis=0)
        qddd_max = qddd.max(axis=0)

        qd_min_under_scale = np.where(
            qd_min < qd_limits_min,
            np.abs(qd_min / qd_limits_min),
            1.0,
        ).max()
        qd_max_over_scale = np.where(
            qd_max > qd_limits_max,
            np.abs(qd_max / qd_limits_max),
            1.0,
        ).max()
        qdd_min_under_scale = np.where(
            qdd_min < qdd_limits_min,
            np.abs(qdd_min / qdd_limits_min),
            1.0,
        ).max()
        qdd_max_over_scale = np.where(
            qdd_max > qdd_limits_max,
            np.abs(qdd_max / qdd_limits_max),
            1.0,
        ).max()
        qddd_min_under_scale = np.where(
            qddd_min < qddd_limits_min,
            np.abs(qddd_min / qddd_limits_min),
            1.0,
        ).max()
        qddd_max_over_scale = np.where(
            qddd_max > qddd_limits_max,
            np.abs(qddd_max / qddd_limits_max),
            1.0,
        ).max()

        # Rescale qd if needed to stay within limits
        IGNORE_QDD_QDDD = (
            True  # Ignore qdd and qddd for now because it was too conservative
        )
        if IGNORE_QDD_QDDD:
            rescale_factor = max(
                qd_min_under_scale,
                qd_max_over_scale,
            )
        else:
            rescale_factor = max(
                qd_min_under_scale,
                qd_max_over_scale,
                qdd_min_under_scale,
                qdd_max_over_scale,
                qddd_min_under_scale,
                qddd_max_over_scale,
            )
        assert rescale_factor >= 1.0
        rescale_factors.append(rescale_factor)

    return rescale_factors


def main() -> None:
    X_W_H_feasible = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_feasible2 = np.array(
        [
            [0, 0, 1, 0.5],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    X_W_Hs = np.stack([X_W_H_feasible, X_W_H_feasible2], axis=0)
    q_algrs = np.stack([DEFAULT_Q_ALGR, DEFAULT_Q_ALGR], axis=0)

    result, _, _ = solve_trajopt_batch(
        X_W_Hs=X_W_Hs,
        q_algrs=q_algrs,
        collision_check_object=True,
        obj_filepath=None,
        obj_xyz=(0.65, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
        use_cuda_graph=True,
        enable_graph=True,
        enable_opt=True,
        timeout=10.0,
    )
    paths = result.get_paths()
    assert len(paths) == 2

    q = paths[0].position.detach().cpu().numpy()
    q2 = paths[1].position.detach().cpu().numpy()
    N_TIMESTEPS = q.shape[0]
    assert q.shape == (N_TIMESTEPS, 23)
    print(f"Success: {result.success}")
    print(f"q.shape: {q.shape}, q2.shape: {q2.shape}")
    print(f"q[0]: {q[0]}")
    print(f"q2[0]: {q2[0]}")


if __name__ == "__main__":
    main()
