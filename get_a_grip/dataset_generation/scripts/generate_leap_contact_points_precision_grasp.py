import json

import numpy as np
import plotly.graph_objects as go

from get_a_grip import get_assets_folder

# Number of points
N = 6

# Generate evenly spaced points on a unit hemisphere
LEAP_TIP_WIDTH = 0.02
LEAP_TIP_LENGTH = 0.025
LEAP_TIP_THICKNESS = 0.01

# front
z_front, y_front = np.meshgrid(
    np.linspace(-LEAP_TIP_WIDTH / 2, LEAP_TIP_WIDTH / 2, N),
    np.linspace(-LEAP_TIP_LENGTH / 2, LEAP_TIP_LENGTH / 2, N),
)
x_front = np.zeros_like(z_front)
points = np.stack((x_front, y_front, z_front), axis=-1).reshape(-1, 3)

# left
x_left, y_left = np.meshgrid(
    np.linspace(0, LEAP_TIP_THICKNESS, N),
    np.linspace(-LEAP_TIP_LENGTH / 2, LEAP_TIP_LENGTH / 2, N),
)
z_left = np.zeros_like(x_left) + -LEAP_TIP_WIDTH / 2
points_left = np.stack((x_left, y_left, z_left), axis=-1).reshape(-1, 3)

# right
x_right, y_right = np.meshgrid(
    np.linspace(0, LEAP_TIP_THICKNESS, N),
    np.linspace(-LEAP_TIP_LENGTH / 2, LEAP_TIP_LENGTH / 2, N),
)
z_right = np.zeros_like(x_right) + LEAP_TIP_WIDTH / 2
points_right = np.stack((x_right, y_right, z_right), axis=-1).reshape(-1, 3)

# top
x_top, z_top = np.meshgrid(
    np.linspace(0, LEAP_TIP_THICKNESS, N),
    np.linspace(-LEAP_TIP_WIDTH / 2, LEAP_TIP_WIDTH / 2, N),
)
y_top = np.zeros_like(x_top) - LEAP_TIP_LENGTH / 2
points_top = np.stack((x_top, y_top, z_top), axis=-1).reshape(-1, 3)

# All
all_points = np.concatenate((points, points_left, points_right, points_top), axis=0)

# Scatter plot points in 3d using plotly.
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=all_points[:, 0], y=all_points[:, 1], z=all_points[:, 2], mode="markers"
        )
    ]
)
fig.show()

# Store points in dictionary.
FINGERTIP_CENTER = np.array([[-0.01, -0.036, 0.015]])
THUMB_FINGERTIP_CENTER = np.array([-0.01, -0.048, -0.015])

contact_point_dictionary = {
    "fingertip_1": (FINGERTIP_CENTER[None] + all_points).tolist(),
    "fingertip_2": (FINGERTIP_CENTER[None] + all_points).tolist(),
    "fingertip_3": (FINGERTIP_CENTER[None] + all_points).tolist(),
    "thumb_fingertip": (THUMB_FINGERTIP_CENTER[None] + all_points).tolist(),
}

contact_point_dictionary["palm_link"] = []
contact_point_dictionary["mcp_joint_1"] = []
contact_point_dictionary["mcp_joint_2"] = []
contact_point_dictionary["mcp_joint_3"] = []
contact_point_dictionary["pip_1"] = []
contact_point_dictionary["pip_2"] = []
contact_point_dictionary["pip_3"] = []
contact_point_dictionary["pip_4"] = []
contact_point_dictionary["dip_1"] = []
contact_point_dictionary["dip_2"] = []
contact_point_dictionary["dip_3"] = []
contact_point_dictionary["thumb_pip"] = []
contact_point_dictionary["thumb_dip"] = []

for i in range(9):
    contact_point_dictionary[f"palm_link_collision_link_{i}"] = []
for i in range(5):
    contact_point_dictionary[f"mcp_joint_1_collision_link_{i}"] = []
    contact_point_dictionary[f"mcp_joint_2_collision_link_{i}"] = []
    contact_point_dictionary[f"mcp_joint_3_collision_link_{i}"] = []
for i in range(4):
    contact_point_dictionary[f"dip_1_collision_link_{i}"] = []
    contact_point_dictionary[f"dip_2_collision_link_{i}"] = []
    contact_point_dictionary[f"dip_3_collision_link_{i}"] = []
for i in range(3):
    contact_point_dictionary[f"fingertip_1_collision_link_{i}"] = []
    contact_point_dictionary[f"fingertip_2_collision_link_{i}"] = []
    contact_point_dictionary[f"fingertip_3_collision_link_{i}"] = []
for i in range(2):
    contact_point_dictionary[f"thumb_pip_collision_link_{i}"] = []
for i in range(4):
    contact_point_dictionary[f"thumb_dip_collision_link_{i}"] = []
for i in range(3):
    contact_point_dictionary[f"thumb_fingertip_collision_link_{i}"] = []
# Save dictionary to json file.
with open(
    get_assets_folder() / "leap_hand_simplified/contact_points_precision_grasp.json",
    "w",
) as f:
    json.dump(contact_point_dictionary, f)
