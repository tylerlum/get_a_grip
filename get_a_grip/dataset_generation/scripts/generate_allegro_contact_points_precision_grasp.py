import json

import numpy as np
import plotly.graph_objects as go

from get_a_grip import get_assets_folder

# Number of points
num_points = 128

# Generate evenly spaced points on a unit hemisphere
phi = np.random.uniform(-0.6 * np.pi / 2, 0.6 * np.pi / 2, size=num_points)
theta = np.random.uniform(0.3 * np.pi, 0.7 * np.pi, size=num_points)

z = np.cos(theta)
x = np.where(z >= 0, np.sin(theta) * np.cos(phi), np.cos(phi))
y = np.where(z >= 0, np.sin(theta) * np.sin(phi), np.sin(phi))

# Apply the desired radius
radius = 0.012
x *= radius
y *= radius
z *= radius

points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

# Scatter plot points in 3d using plotly.
fig = go.Figure(
    data=[go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers")]
)
fig.show()

# Store points in dictionary.
contact_point_dictionary = {
    f"link_{num:.1f}_tip": points.tolist() for num in [3.0, 7.0, 11.0, 15.0]
}

contact_point_dictionary["palm_link"] = []
for ii in range(16):
    contact_point_dictionary[f"link_{float(ii):.1f}"] = []
# Save dictionary to json file.
with open(
    get_assets_folder()
    / "allegro_hand_description/contact_points_precision_grasp.json",
    "w",
) as f:
    json.dump(contact_point_dictionary, f)
