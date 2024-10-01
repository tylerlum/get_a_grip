import os
from dataclasses import dataclass

import plotly.graph_objects as go
import torch
import tyro

from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.utils.seed import set_seed

set_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass
class VisualizeHandModelArgs:
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    n_surface_points: int = 2000


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeHandModelArgs])
    print(args)

    device = torch.device("cpu")

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type,
        n_surface_points=args.n_surface_points,
        device=device,
    )
    joint_angles = hand_model.default_joint_angles
    rotation = hand_model.default_orientation
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
            title=f"Hand Model ({args.hand_model_type})",
        ),
    )
    fig.update_layout(scene_aspectmode="data")
    fig.show()


if __name__ == "__main__":
    main()
