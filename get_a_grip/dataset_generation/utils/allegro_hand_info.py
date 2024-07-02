import torch
import transforms3d
import numpy as np
from get_a_grip import get_assets_folder


ALLEGRO_HAND_JOINT_NAMES = [
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
]

ALLEGRO_HAND_ROTATION = torch.tensor(
    transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes="rzyz"),
    dtype=torch.float,
)

ALLEGRO_HAND_JOINT_ANGLES_MU = torch.tensor(
    [
        0,
        0.5,
        0,
        0,
        0,
        0.5,
        0,
        0,
        0,
        0.5,
        0,
        0,
        1.4,
        0,
        0,
        0,
    ],
    dtype=torch.float,
)

ALLEGRO_HAND_ROOT_HAND_FILE = (
    str(get_assets_folder()),
    "allegro_hand_description/allegro_hand_description_right.urdf",
)

ALLEGRO_HAND_ROOT_HAND_FILE_WITH_VIRTUAL_JOINTS = (
    str(get_assets_folder()),
    "allegro_hand_description/allegro_hand_description_right_with_virtual_joints.urdf",
)

# This is a list of allowed contact link names for precision grasps
ALLEGRO_HAND_ALLOWED_CONTACT_LINK_NAMES = [
    "link_3.0",
    "link_7.0",
    "link_11.0",
    "link_15.0",
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
    # Allow some contact with parts before
    "link_2.0",
    "link_6.0",
    "link_10.0",
    "link_14.0",
    # Allow some contact with parts before
    "link_1.0",
    "link_5.0",
    "link_9.0",
    "link_13.0",
]

ALLEGRO_HAND_FINGER_KEYWORDS = ["3.0", "7.0", "11.0", "15.0"]
