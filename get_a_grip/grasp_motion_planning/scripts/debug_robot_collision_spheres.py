import pathlib

import numpy as np
import pybullet as pb
import yaml
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

from get_a_grip.grasp_motion_planning.utils.pybullet_utils import (
    draw_collision_spheres,
)

robot_yml = load_yaml(join_path(get_robot_configs_path(), "fr3_algr_zed2i.yml"))
FR3_ALGR_ZED2I_URDF_PATH = robot_yml["robot_cfg"]["kinematics"]["urdf_path"]
FR3_ALGR_ZED2I_URDF_PATH = pathlib.Path(
    join_path(get_assets_path(), FR3_ALGR_ZED2I_URDF_PATH)
)
assert FR3_ALGR_ZED2I_URDF_PATH.exists()

COLLISION_SPHERES_YAML_PATH = robot_yml["robot_cfg"]["kinematics"]["collision_spheres"]
COLLISION_SPHERES_YAML_PATH = pathlib.Path(
    join_path(get_robot_configs_path(), COLLISION_SPHERES_YAML_PATH)
)
assert COLLISION_SPHERES_YAML_PATH.exists()

DEFAULT_Q_FR3 = np.array(
    [
        1.76261055e-06,
        -1.29018439e00,
        0.00000000e00,
        -2.69272642e00,
        0.00000000e00,
        1.35254201e00,
        7.85400000e-01,
    ]
)
DEFAULT_Q_ALGR = np.array(
    [
        2.90945620e-01,
        7.37109400e-01,
        5.10859200e-01,
        1.22637060e-01,
        1.20125350e-01,
        5.84513500e-01,
        3.43829930e-01,
        6.05035000e-01,
        -2.68431900e-01,
        8.78457900e-01,
        8.49713500e-01,
        8.97218400e-01,
        1.33282830e00,
        3.47787830e-01,
        2.09215670e-01,
        -6.50969000e-03,
    ]
)

c = pb.connect(pb.GUI)
robot = pb.loadURDF(
    str(FR3_ALGR_ZED2I_URDF_PATH),
    useFixedBase=True,
)

num_total_joints = pb.getNumJoints(robot)
assert num_total_joints == 39
actuatable_joint_idxs = [
    i for i in range(num_total_joints) if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
]
num_actuatable_joints = len(actuatable_joint_idxs)
assert num_actuatable_joints == 23
arm_actuatable_joint_idxs = actuatable_joint_idxs[:7]
hand_actuatable_joint_idxs = actuatable_joint_idxs[7:]

for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(robot, joint_idx, DEFAULT_Q_FR3[i])

for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(robot, joint_idx, DEFAULT_Q_ALGR[i])

collision_config = yaml.safe_load(
    open(
        COLLISION_SPHERES_YAML_PATH,
        "r",
    )
)
draw_collision_spheres(robot, collision_config)

print("=" * 80)
print("Setting breakpoint to allow user to inspect collision spheres")
print("=" * 80 + "\n")
breakpoint()
