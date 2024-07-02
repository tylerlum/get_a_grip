import os

import plotly.graph_objects as go
import torch
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_JOINT_ANGLES_MU,
    ALLEGRO_HAND_ROTATION,
)
from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.seed import set_seed

set_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":
    device = torch.device("cpu")
    # hand model
    hand_model = HandModel(n_surface_points=2000, device=device)
    joint_angles = ALLEGRO_HAND_JOINT_ANGLES_MU
    rotation = ALLEGRO_HAND_ROTATION
    hand_pose = torch.cat(
        [
            torch.tensor([0, 0, 0], dtype=torch.float, device=device),
            rotation.T.ravel()[:6],
            joint_angles,
        ]
    )
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # visualize
    hand_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=0.5,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=True,
        with_surface_points=True,
        with_penetration_keypoints=True,
    )
    fig = go.Figure(
        data=(hand_plotly),
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title="Hand Model",
        ),
    )
    fig.update_layout(scene_aspectmode="data")
    fig.show()
