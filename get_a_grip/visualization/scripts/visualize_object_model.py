import os
import pathlib
from dataclasses import dataclass

import plotly.graph_objects as go
import torch
import tyro

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.object_model import ObjectModel
from get_a_grip.utils.seed import set_seed

set_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass
class VisualizeObjectModelArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    object_code: str = "sem-Mug-10f6e09036350e92b3f21f1137c3c347"
    object_scale: float = 0.1


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeObjectModelArgs])

    # object model
    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=1,
        scale=args.object_scale,
        num_samples=2000,
        device=torch.device("cpu"),
    )
    object_model.initialize([args.object_code])

    # visualize
    object_plotly = object_model.get_plotly_data(
        i=0,
        with_surface_points=True,
    )
    fig = go.Figure(
        data=object_plotly,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title=f"Object Model: {args.object_code}",
        ),
    )
    fig.update_layout(scene_aspectmode="data")
    fig.show()


if __name__ == "__main__":
    main()
