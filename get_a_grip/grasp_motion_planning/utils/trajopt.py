import pathlib
import time
from typing import Optional, Tuple

import numpy as np
import torch
import transforms3d
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
)

from get_a_grip.grasp_motion_planning.utils.joint_limit_utils import (
    modify_robot_cfg_to_add_joint_limit_buffer,
)
from get_a_grip.grasp_motion_planning.utils.world import (
    get_world_cfg,
)

DEFAULT_Q_FR3 = np.array([0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
DEFAULT_Q_ALGR = np.array([0, 1.5, 1, 1, 0, 1.5, 1, 1, 0, 1.5, 1, 1, 0.5, 0.5, 1.5, 1])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def solve_trajopt(
    X_W_H: np.ndarray,
    q_algr_constraint: Optional[np.ndarray] = None,
    q_fr3_start: Optional[np.ndarray] = None,
    collision_check_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = None,
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision_check_table: bool = True,
    use_cuda_graph: bool = True,
    enable_graph: bool = True,
    enable_opt: bool = False,  # Getting some errors from setting this to True
    raise_if_fail: bool = True,
    warn_if_fail: bool = False,
    timeout: float = 5.0,
    collision_sphere_buffer: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, MotionGenResult, MotionGen]:
    assert X_W_H.shape == (4, 4), f"X_W_H.shape: {X_W_H.shape}"
    trans = X_W_H[:3, 3]
    rot_matrix = X_W_H[:3, :3]
    quat_wxyz = transforms3d.quaternions.mat2quat(rot_matrix)

    target_pose = Pose(
        torch.from_numpy(trans).float().to(device),
        quaternion=torch.from_numpy(quat_wxyz).float().to(device),
    )

    tensor_args = TensorDeviceType()
    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    if collision_sphere_buffer is not None:
        robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_sphere_buffer
    robot_cfg = RobotConfig.from_dict(robot_cfg)
    modify_robot_cfg_to_add_joint_limit_buffer(robot_cfg)

    # Apply joint limits
    if q_algr_constraint is not None:
        assert q_algr_constraint.shape == (
            16,
        ), f"q_algr_constraint.shape: {q_algr_constraint.shape}"
        assert robot_cfg.kinematics.kinematics_config.joint_limits.position.shape == (
            2,
            23,
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] = (
            torch.from_numpy(q_algr_constraint).float().to(device) - 0.01
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:] = (
            torch.from_numpy(q_algr_constraint).float().to(device) + 0.01
        )

    world_cfg = get_world_cfg(
        collision_check_object=collision_check_object,
        obj_filepath=obj_filepath,
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        collision_check_table=collision_check_table,
    )

    tensor_args = TensorDeviceType()
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=use_cuda_graph,
        num_ik_seeds=1,  # Reduced to save time?
        num_graph_seeds=1,  # Reduced to save time?
        num_trajopt_seeds=1,  # Reduced to save time?
        num_batch_ik_seeds=1,  # Reduced to save time?
        num_batch_trajopt_seeds=1,  # Reduced to save time?
        num_trajopt_noisy_seeds=1,  # Reduced to save time?
    )
    motion_gen = MotionGen(motion_gen_config)
    # motion_gen.warmup()  # Can cause issues with use_cuda_graph=True

    if q_fr3_start is None:
        q_fr3_start = DEFAULT_Q_FR3
    if q_algr_constraint is None:
        q_algr_constraint = DEFAULT_Q_ALGR
    start_q = np.concatenate([q_fr3_start, q_algr_constraint])
    start_state = JointState.from_position(
        torch.from_numpy(start_q).float().to(device).view(1, -1)
    )

    t_start = time.time()
    result = motion_gen.plan_single(
        start_state=start_state,
        goal_pose=target_pose,
        plan_config=MotionGenPlanConfig(
            enable_graph=enable_graph,
            enable_opt=enable_opt,
            # max_attempts=10,
            max_attempts=1,  # Reduce to save time?
            num_trajopt_seeds=1,  # Reduce to save time?
            num_graph_seeds=1,  # Must be 1 for plan_batch
            timeout=timeout,
        ),
    )
    print("Time taken: ", time.time() - t_start)
    print("Trajectory Generated: ", result.success)
    if result is None:
        raise RuntimeError("IK Failed")

    traj = result.get_interpolated_plan()
    if traj is None:
        raise RuntimeError("Trajectory is None")

    if traj.position is None:
        raise RuntimeError("Trajectory Position is None")

    if not result.success.any():
        if raise_if_fail:
            raise RuntimeError("Trajectory Optimization Failed")
        if warn_if_fail:
            print("WARNING: Trajectory Optimization Failed")

    # If enable_opt=False, then the dt is not set correctly, making the dt way too small
    # For some reason, can sometimes by 2D or 3D
    if len(traj.position.shape) == 2:
        n_timesteps = traj.position.shape[0]
        assert traj.position.shape == (
            n_timesteps,
            23,
        ), f"traj.position.shape: {traj.position.shape}"
        assert traj.velocity.shape == (
            n_timesteps,
            23,
        ), f"traj.velocity.shape: {traj.velocity.shape}"
        assert traj.acceleration.shape == (
            n_timesteps,
            23,
        ), f"traj.acceleration.shape: {traj.acceleration.shape}"

        if enable_opt:
            dt = result.interpolation_dt
        else:
            ASSUMED_TOTAL_TIME = 5.0
            dt = ASSUMED_TOTAL_TIME / n_timesteps

        return (
            traj.position.detach().cpu().numpy(),
            traj.velocity.detach().cpu().numpy(),
            traj.acceleration.detach().cpu().numpy(),
            dt,
            result,
            motion_gen,
        )
    elif len(traj.position.shape) == 3:
        n_trajs = traj.position.shape[0]
        n_timesteps = traj.position.shape[1]
        assert traj.position.shape == (
            n_trajs,
            n_timesteps,
            23,
        ), f"traj.position.shape: {traj.position.shape}"
        assert traj.velocity.shape == (
            n_trajs,
            n_timesteps,
            23,
        ), f"traj.velocity.shape: {traj.velocity.shape}"
        assert traj.acceleration.shape == (
            n_trajs,
            n_timesteps,
            23,
        ), f"traj.acceleration.shape: {traj.acceleration.shape}"

        if enable_opt:
            dt = result.interpolation_dt
        else:
            ASSUMED_TOTAL_TIME = 5.0
            dt = ASSUMED_TOTAL_TIME / n_timesteps

        # HACK: Not sure if there's a smarter way to select
        SELECTED_TRAJ_IDX = 0
        return (
            traj.position[SELECTED_TRAJ_IDX].detach().cpu().numpy(),
            traj.velocity[SELECTED_TRAJ_IDX].detach().cpu().numpy(),
            traj.acceleration[SELECTED_TRAJ_IDX].detach().cpu().numpy(),
            dt,
            result,
            motion_gen,
        )
    else:
        raise ValueError(f"traj.position.shape: {traj.position.shape}")


def main() -> None:
    X_W_H_feasible = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_collide_object = np.array(
        [
            [0, 0, 1, 0.65],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_collide_table = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.10],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q_algr_pre = np.array(
        [
            0.29094562,
            0.7371094,
            0.5108592,
            0.12263706,
            0.12012535,
            0.5845135,
            0.34382993,
            0.605035,
            -0.2684319,
            0.8784579,
            0.8497135,
            0.8972184,
            1.3328283,
            0.34778783,
            0.20921567,
            -0.00650969,
        ]
    )

    q, qd, qdd, dt, result, _ = solve_trajopt(
        X_W_H=X_W_H_feasible,
        q_algr_constraint=q_algr_pre,
    )
    print(
        f"q.shape = {q.shape}, qd.shape = {qd.shape}, qdd.shape = {qdd.shape}, dt = {dt}"
    )

    try:
        q, qd, qdd, dt, _, _ = solve_trajopt(
            X_W_H=X_W_H_collide_object,
            q_algr_constraint=q_algr_pre,
        )
        raise ValueError("Collision with object should have failed")
    except RuntimeError as e:
        print(f"Collision with object failed as expected: {e}")

    try:
        q, qd, qdd, dt, _, _ = solve_trajopt(
            X_W_H=X_W_H_collide_table,
            q_algr_constraint=q_algr_pre,
        )
        raise ValueError("Collision with table should have failed")
    except RuntimeError as e:
        print(f"Collision with table failed as expected: {e}")


if __name__ == "__main__":
    main()
