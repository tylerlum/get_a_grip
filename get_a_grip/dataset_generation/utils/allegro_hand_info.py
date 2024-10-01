import torch

from get_a_grip import get_assets_folder

# Allegro Hand Constants
ALLEGRO_HAND_NUM_FINGERS = 4
ALLEGRO_HAND_NUM_JOINTS = 16

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

# Allegro hand starts with x along palm normal, y towards thumb-side, z along finger
# Want to rotate it so that x towards thumb-side, y along finger, z along palm normal
# So convert x (palm)   -> z
#            y (thumb)  -> x
#            z (finger) -> y
ALLEGRO_HAND_DEFAULT_ORIENTATION = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float,
)

ALLEGRO_HAND_DEFAULT_JOINT_ANGLES = torch.tensor(
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

ALLEGRO_HAND_FILE = "allegro_hand_description/allegro_hand_description_right.urdf"

ALLEGRO_HAND_FILE_WITH_VIRTUAL_JOINTS = (
    "allegro_hand_description/allegro_hand_description_right_with_virtual_joints.urdf"
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

ALLEGRO_HAND_FINGERTIP_KEYWORDS = ["link_3.0", "link_7.0", "link_11.0", "link_15.0"]
ALLEGRO_HAND_FINGERTIP_NAMES = [
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
]

ALLEGRO_HAND_URDF_PATH = get_assets_folder() / ALLEGRO_HAND_FILE
ALLEGRO_HAND_WITH_VIRTUAL_JOINTS_URDF_PATH = (
    get_assets_folder() / ALLEGRO_HAND_FILE_WITH_VIRTUAL_JOINTS
)

ALLEGRO_HAND_CONTACT_POINTS_PATH = (
    get_assets_folder() / "allegro_hand_description/contact_points_precision_grasp.json"
)
ALLEGRO_HAND_PENETRATION_POINTS_PATH = (
    get_assets_folder() / "allegro_hand_description/penetration_points.json"
)

# Sanity checks
assert (
    len(ALLEGRO_HAND_JOINT_NAMES) == ALLEGRO_HAND_NUM_JOINTS
), f"Expected {ALLEGRO_HAND_NUM_JOINTS} joints, got {len(ALLEGRO_HAND_JOINT_NAMES)}"

assert (
    len(ALLEGRO_HAND_DEFAULT_JOINT_ANGLES) == ALLEGRO_HAND_NUM_JOINTS
), f"Expected {ALLEGRO_HAND_NUM_JOINTS} joint angles, got {len(ALLEGRO_HAND_DEFAULT_JOINT_ANGLES)}"

assert ALLEGRO_HAND_URDF_PATH.exists(), f"Could not find {ALLEGRO_HAND_URDF_PATH}"
assert (
    ALLEGRO_HAND_WITH_VIRTUAL_JOINTS_URDF_PATH.exists()
), f"Could not find {ALLEGRO_HAND_WITH_VIRTUAL_JOINTS_URDF_PATH}"
assert (
    ALLEGRO_HAND_CONTACT_POINTS_PATH.exists()
), f"Could not find {ALLEGRO_HAND_CONTACT_POINTS_PATH}"
assert (
    ALLEGRO_HAND_PENETRATION_POINTS_PATH.exists()
), f"Could not find {ALLEGRO_HAND_PENETRATION_POINTS_PATH}"
