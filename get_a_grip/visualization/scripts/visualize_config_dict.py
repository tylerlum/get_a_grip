import math
import os
import pathlib
from dataclasses import dataclass
from typing import Dict

import numpy as np
import tyro
from plotly.subplots import make_subplots

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.object_model import ObjectModel
from get_a_grip.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.visualization.utils.visualize_config_dict_helper import (
    create_config_dict_fig,
    get_scene_dict,
    get_yup_camera,
)


@dataclass
class VisualizeConfigDictArgs:
    input_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/large/evaled_grasp_config_dicts"
    )  # SHOULD be able to hand most types of config dicts
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    object_code_and_scale_str: str = "sem-Ipod-4b6c6248d5c01b3e4eee8d1cb32988b_0_1000"
    idx_to_visualize: int = 0
    visualize_all: bool = False
    save_to_html: bool = False
    device: str = "cpu"

    # Detailed args
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    object_model_num_sampled_pts: int = 2000
    skip_visualize_qpos_start: bool = False
    skip_visualize_grasp_config_dict: bool = False


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeConfigDictArgs])

    # load results
    config_dict: Dict[str, np.ndarray] = np.load(
        args.input_config_dicts_path / f"{args.object_code_and_scale_str}.npy",
        allow_pickle=True,
    ).item()

    # hand model: be careful with this, as it is stateful
    hand_model = HandModel(
        hand_model_type=args.hand_model_type, device=args.device, n_surface_points=1000
    )

    # object model
    try:
        object_code, object_scale = parse_object_code_and_scale(
            args.object_code_and_scale_str
        )
        object_model = ObjectModel(
            meshdata_root_path=str(args.meshdata_root_path),
            batch_size_each=1,
            num_samples=args.object_model_num_sampled_pts,
            device=args.device,
        )
        object_model.initialize(object_code, object_scale)
    except Exception as e:
        print("=" * 80)
        print(f"Exception: {e}")
        print("=" * 80)
        print(f"Skipping {args.object_code_and_scale_str} and continuing")
        object_model = None

    if args.visualize_all:
        MAX_TO_VISUALIZE = 9
        OFFSET = 0
        individual_figs = [
            create_config_dict_fig(
                config_dict=config_dict,
                hand_model=hand_model,
                object_model=object_model,
                skip_visualize_qpos_start=args.skip_visualize_qpos_start,
                skip_visualize_grasp_config_dict=args.skip_visualize_grasp_config_dict,
                idx_to_visualize=i + OFFSET,
                title=f"{args.object_code_and_scale_str} {i + OFFSET}",
            )
            for i in range(MAX_TO_VISUALIZE)
        ]
        # Get titles
        titles = [
            individual_fig.layout.title.text.replace(args.object_code_and_scale_str, "")
            for individual_fig in individual_figs
        ]

        nrows = math.ceil(math.sqrt(MAX_TO_VISUALIZE))
        ncols = math.ceil(MAX_TO_VISUALIZE / nrows)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=titles,
            specs=[[{"type": "mesh3d"} for _ in range(ncols)] for _ in range(nrows)],
        )

        # Adding each element to the main figure
        for i, individual_fig in enumerate(individual_figs):
            for trace in individual_fig.data:
                row = (i // ncols) + 1
                col = (i % ncols) + 1
                fig.add_trace(trace, row=row, col=col)

        # plotly uses scene, scene2, scene3, ...
        new_scene_dict = {}
        for i in range(len(individual_figs)):
            scene_name = f"scene{i+1}" if i > 0 else "scene"
            new_scene_dict[scene_name] = {
                **get_scene_dict(),
                "camera": get_yup_camera(),
            }

        fig.update_layout(
            title=f"{args.object_code_and_scale_str} (all)",
            showlegend=True,
            **new_scene_dict,
        )
    else:
        fig = create_config_dict_fig(
            config_dict=config_dict,
            hand_model=hand_model,
            object_model=object_model,
            skip_visualize_qpos_start=args.skip_visualize_qpos_start,
            skip_visualize_grasp_config_dict=args.skip_visualize_grasp_config_dict,
            idx_to_visualize=args.idx_to_visualize,
            title=f"{args.object_code_and_scale_str} {args.idx_to_visualize}",
        )

    # Output
    if args.save_to_html:
        output_folder = "../html_outputs"
        filename = (
            f"result_{args.object_code_and_scale_str}-{args.idx_to_visualize}.html"
            if not args.visualize_all
            else f"result_{args.object_code_and_scale_str}.html"
        )
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(
            output_folder,
            filename,
        )
        print(f"Saving to {output_filepath}")
        fig.write_html(output_filepath)
    else:
        print("Showing figure...")
        fig.show()


if __name__ == "__main__":
    main()
