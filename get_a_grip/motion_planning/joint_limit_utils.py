from curobo.types.robot import RobotConfig


def modify_robot_cfg_to_add_joint_limit_buffer(
    robot_cfg: RobotConfig, buffer_arm: float = 0.0, buffer_hand: float = 0.01
) -> None:
    assert robot_cfg.kinematics.kinematics_config.joint_limits.position.shape == (
        2,
        23,
    )

    # Arm
    low_arm, high_arm = (
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, :7],
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, :7],
    )
    buffer_arm_range = (low_arm + high_arm) / 2 * buffer_arm
    robot_cfg.kinematics.kinematics_config.joint_limits.position[0, :7] += (
        buffer_arm_range
    )
    robot_cfg.kinematics.kinematics_config.joint_limits.position[1, :7] -= (
        buffer_arm_range
    )

    # Hand
    low_hand, high_hand = (
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:],
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:],
    )
    buffer_hand_range = (low_hand + high_hand) / 2 * buffer_hand
    robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] += (
        buffer_hand_range
    )
