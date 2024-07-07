import pathlib
import time
from typing import Optional, Tuple

import numpy as np
import pybullet as pb
import yaml
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from tqdm import tqdm

from get_a_grip.grasp_motion_planning.utils.pybullet_utils import (
    draw_collision_spheres,
    remove_collision_spheres,
)
from get_a_grip.grasp_motion_planning.utils.trajopt import (
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
)


def start_visualizer(
    object_urdf_path: Optional[pathlib.Path] = None,
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    FR3_ALGR_ZED2I_URDF_PATH = load_yaml(
        join_path(get_robot_configs_path(), "fr3_algr_zed2i_with_fingertips.yml")
    )["robot_cfg"]["kinematics"]["urdf_path"]
    FR3_ALGR_ZED2I_URDF_PATH = pathlib.Path(
        join_path(get_assets_path(), FR3_ALGR_ZED2I_URDF_PATH)
    )
    assert FR3_ALGR_ZED2I_URDF_PATH.exists()

    pb.connect(pb.GUI)
    robot = pb.loadURDF(
        str(FR3_ALGR_ZED2I_URDF_PATH),
        useFixedBase=True,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )
    num_total_joints = pb.getNumJoints(robot)
    assert num_total_joints == 39

    if object_urdf_path is not None:
        assert object_urdf_path.exists()
        obj_quat_xyzw = obj_quat_wxyz[1:] + obj_quat_wxyz[:1]
        _obj = pb.loadURDF(
            str(object_urdf_path),
            useFixedBase=True,
            basePosition=obj_xyz,
            baseOrientation=obj_quat_xyzw,  # Must be xyzw
        )

    actuatable_joint_idxs = [
        i
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]
    num_actuatable_joints = len(actuatable_joint_idxs)
    assert num_actuatable_joints == 23

    q = np.concatenate([DEFAULT_Q_FR3, DEFAULT_Q_ALGR])
    assert q.shape == (23,)
    set_robot_state(robot, q)

    # For debugging
    _joint_names = [
        pb.getJointInfo(robot, i)[1].decode("utf-8")
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]
    _link_names = [
        pb.getJointInfo(robot, i)[12].decode("utf-8")
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]

    DEBUG = False
    if DEBUG:
        print(f"joint_names: {_joint_names}")
        print(f"link_names: {_link_names}")

    return robot


def draw_collision_spheres_default_config(robot) -> None:
    COLLISION_SPHERES_YAML_PATH = load_yaml(
        join_path(get_robot_configs_path(), "fr3_algr_zed2i_with_fingertips.yml")
    )["robot_cfg"]["kinematics"]["collision_spheres"]
    COLLISION_SPHERES_YAML_PATH = pathlib.Path(
        join_path(get_robot_configs_path(), COLLISION_SPHERES_YAML_PATH)
    )
    assert COLLISION_SPHERES_YAML_PATH.exists()

    collision_config = yaml.safe_load(
        open(
            COLLISION_SPHERES_YAML_PATH,
            "r",
        )
    )
    draw_collision_spheres(
        robot=robot,
        config=collision_config,
    )


def remove_collision_spheres_default_config() -> None:
    remove_collision_spheres()


def set_robot_state(robot, q: np.ndarray) -> None:
    assert q.shape == (23,)

    num_total_joints = pb.getNumJoints(robot)
    assert num_total_joints == 39
    actuatable_joint_idxs = [
        i
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]
    num_actuatable_joints = len(actuatable_joint_idxs)
    assert num_actuatable_joints == 23

    for i, joint_idx in enumerate(actuatable_joint_idxs):
        pb.resetJointState(robot, joint_idx, q[i])


def animate_robot(robot, qs: np.ndarray, dt: float) -> None:
    N_pts = qs.shape[0]
    assert qs.shape == (N_pts, 23)

    last_update_time = time.time()
    for i in tqdm(range(N_pts)):
        q = qs[i]
        assert q.shape == (23,)

        set_robot_state(robot, q)
        time_since_last_update = time.time() - last_update_time
        if time_since_last_update <= dt:
            time.sleep(dt - time_since_last_update)
        else:
            print(f"WARNING: Time since last update {time_since_last_update} > dt {dt}")
        last_update_time = time.time()


def create_urdf(obj_path: pathlib.Path) -> pathlib.Path:
    assert obj_path.suffix == ".obj"
    filename = obj_path.name
    parent_folder = obj_path.parent
    urdf_path = parent_folder / f"{obj_path.stem}.urdf"
    urdf_text = f"""<?xml version="0.0" ?>
<robot name="model.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.8"/>
      <rolling_friction value="0.001"/>g
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
    <origin rpy="0 0 0" xyz="0.01 0.0 0.01"/>
       <mass value=".066"/>
       <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1. 1. 1. 1."/>
      </material>
    </visual>
    <collision>
      <geometry>
    	 	<mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""
    with urdf_path.open("w") as f:
        f.write(urdf_text)
    return urdf_path
