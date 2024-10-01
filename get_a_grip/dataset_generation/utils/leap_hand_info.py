import torch

from get_a_grip import get_assets_folder

# Leap Hand Constants
LEAP_HAND_NUM_FINGERS = 4
LEAP_HAND_NUM_JOINTS = 16

LEAP_HAND_JOINT_NAMES = [
    "1",
    "0",
    "2",
    "3",
    "5",
    "4",
    "6",
    "7",
    "9",
    "8",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
]

# LEAP hand starts with x along finger, y towards thumb-side, z along -ve palm normal
# Want to rotate it so that x towards thumb-side, y along finger, z along palm normal
# So convert x (finger) -> y
#            y (thumb)  -> x
#            z (-palm) -> -z
LEAP_HAND_DEFAULT_ORIENTATION = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=torch.float,
)

LEAP_HAND_DEFAULT_JOINT_ANGLES = torch.tensor(
    [
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
        0,
        1.4,
        0,
        0,
        0,
    ],
    dtype=torch.float,
)

LEAP_HAND_FILE = "leap_hand_simplified/leap_hand_right.urdf"

LEAP_HAND_FILE_WITH_VIRTUAL_JOINTS = (
    "leap_hand_simplified/leap_hand_right_with_virtual_joints.urdf"
)

# This is a list of allowed contact link names for precision grasps
LEAP_HAND_ALLOWED_CONTACT_LINK_NAMES = [
    "fingertip_1",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
    # Allow some contact with parts before
    "dip_1",
    "dip_2",
    "dip_3",
    "thumb_dip",
    # Allow some contact with parts before
    "pip_1",
    "pip_2",
    "pip_3",
    "thumb_pip",
]

LEAP_HAND_FINGERTIP_KEYWORDS = [
    "fingertip_1",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
]
LEAP_HAND_FINGERTIP_NAMES = [
    "fingertip_1",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
]

LEAP_HAND_URDF_PATH = get_assets_folder() / LEAP_HAND_FILE
LEAP_HAND_WITH_VIRTUAL_JOINTS_URDF_PATH = (
    get_assets_folder() / LEAP_HAND_FILE_WITH_VIRTUAL_JOINTS
)

LEAP_HAND_CONTACT_POINTS_PATH = (
    get_assets_folder() / "leap_hand_simplified/contact_points_precision_grasp.json"
)
LEAP_HAND_PENETRATION_POINTS_PATH = (
    get_assets_folder() / "leap_hand_simplified/penetration_points.json"
)

# Sanity checks
assert (
    len(LEAP_HAND_JOINT_NAMES) == LEAP_HAND_NUM_JOINTS
), f"Expected {LEAP_HAND_NUM_JOINTS} joints, got {len(LEAP_HAND_JOINT_NAMES)}"

assert (
    len(LEAP_HAND_DEFAULT_JOINT_ANGLES) == LEAP_HAND_NUM_JOINTS
), f"Expected {LEAP_HAND_NUM_JOINTS} joint angles, got {len(LEAP_HAND_DEFAULT_JOINT_ANGLES)}"

assert LEAP_HAND_URDF_PATH.exists(), f"Could not find {LEAP_HAND_URDF_PATH}"
assert (
    LEAP_HAND_WITH_VIRTUAL_JOINTS_URDF_PATH.exists()
), f"Could not find {LEAP_HAND_WITH_VIRTUAL_JOINTS_URDF_PATH}"
assert (
    LEAP_HAND_CONTACT_POINTS_PATH.exists()
), f"Could not find {LEAP_HAND_CONTACT_POINTS_PATH}"
assert (
    LEAP_HAND_PENETRATION_POINTS_PATH.exists()
), f"Could not find {LEAP_HAND_PENETRATION_POINTS_PATH}"
