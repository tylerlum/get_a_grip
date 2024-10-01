import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.object_model import ObjectModel
from get_a_grip.dataset_generation.utils.pose_conversion import hand_config_np_to_pose
from get_a_grip.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.visualization.utils.visualize_config_dict_helper import (
    create_config_dict_fig,
)
from get_a_grip.visualization.utils.visualize_optimization_helper import (
    create_figure_with_buttons_and_slider,
)


@dataclass
class VisualizeConfigDictOptimizationArgs:
    """Expects a folder with the following structure:
    - <input_config_dicts_mid_optimization_path>
        - 0
            - <object_code_and_scale_str>.npy
        - x
            - <object_code_and_scale_str>.npy
        - 2x
            - <object_code_and_scale_str>.npy
        - 3x
            - <object_code_and_scale_str>.npy
        ...
    """

    input_config_dicts_mid_optimization_path: pathlib.Path = (
        get_data_folder() / "dataset/large/evaled_grasp_config_dicts/mid_optimization"
    )
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    object_code_and_scale_str: str = (
        "mujoco-Olive_Kids_Butterfly_Garden_Pencil_Case_0_1000"
    )
    idx_to_visualize: int = 0
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    frame_duration: int = 200
    transition_duration: int = 100
    device: str = "cpu"
    save_to_html: bool = False
    skip_visualize_grasp_config_dict: bool = False


def get_hand_model_from_config_dicts(
    hand_model_type: HandModelType,
    config_dict: Dict[str, Any],
    device: str,
    idx_to_visualize: int,
) -> HandModel:
    hand_pose = hand_config_np_to_pose(
        trans=config_dict["trans"][idx_to_visualize],
        rot=config_dict["rot"][idx_to_visualize],
        joint_angles=config_dict["joint_angles"][idx_to_visualize],
    ).to(device)
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)
    return hand_model


def get_object_model(
    meshdata_root_path: pathlib.Path,
    object_code_and_scale_str: str,
    device: str,
) -> Optional[ObjectModel]:
    try:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        object_model = ObjectModel(
            meshdata_root_path=str(meshdata_root_path),
            batch_size_each=1,
            device=device,
        )
        object_model.initialize(object_code, object_scale)
    except Exception as e:
        print("=" * 80)
        print(f"Exception: {e}")
        print("=" * 80)
        print(f"Skipping {object_code_and_scale_str} and continuing")
        object_model = None
    return object_model


def create_config_dict_figs_from_folder(
    hand_model_type: HandModelType,
    input_config_dicts_mid_optimization_path: pathlib.Path,
    meshdata_root_path: pathlib.Path,
    object_code_and_scale_str: str,
    idx_to_visualize: int,
    device: str,
    skip_visualize_grasp_config_dict: bool,
) -> Tuple[List[go.Figure], List[int]]:
    filename = f"{object_code_and_scale_str}.npy"

    optimization_steps = sorted(
        [
            int(path.name)
            for path in input_config_dicts_mid_optimization_path.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / filename).exists()
        ]
    )
    assert (
        len(optimization_steps) > 0
    ), f"No folders with {filename} found in {input_config_dicts_mid_optimization_path}"

    figs = []
    for optimization_step in tqdm(optimization_steps, desc="Going through folders..."):
        filepath = (
            input_config_dicts_mid_optimization_path / f"{optimization_step}" / filename
        )
        assert filepath.exists(), f"{filepath} does not exist"

        # Read in data
        config_dict = np.load(filepath, allow_pickle=True).item()
        hand_model = get_hand_model_from_config_dicts(
            hand_model_type=hand_model_type,
            config_dict=config_dict,
            device=device,
            idx_to_visualize=idx_to_visualize,
        )
        object_model = get_object_model(
            meshdata_root_path=meshdata_root_path,
            object_code_and_scale_str=object_code_and_scale_str,
            device=device,
        )

        # Create figure
        fig = create_config_dict_fig(
            config_dict=config_dict,
            hand_model=hand_model,
            object_model=object_model,
            skip_visualize_qpos_start=True,
            skip_visualize_grasp_config_dict=skip_visualize_grasp_config_dict,
            idx_to_visualize=idx_to_visualize,
            title=f"{object_code_and_scale_str} {idx_to_visualize}",
        )
        figs.append(fig)

    return figs, optimization_steps


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeConfigDictOptimizationArgs])

    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    input_figs, optimization_steps = create_config_dict_figs_from_folder(
        hand_model_type=args.hand_model_type,
        input_config_dicts_mid_optimization_path=args.input_config_dicts_mid_optimization_path,
        meshdata_root_path=args.meshdata_root_path,
        object_code_and_scale_str=args.object_code_and_scale_str,
        idx_to_visualize=args.idx_to_visualize,
        device=args.device,
        skip_visualize_grasp_config_dict=args.skip_visualize_grasp_config_dict,
    )

    print("Making figure with buttons and slider...")
    new_fig = create_figure_with_buttons_and_slider(
        input_figs=input_figs,
        optimization_steps=optimization_steps,
        frame_duration=args.frame_duration,
        transition_duration=args.transition_duration,
    )
    print("Done making figure with buttons and slider")

    if args.save_to_html:
        output_folder = "../html_outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(
            output_folder,
            f"optimization_{args.object_code_and_scale_str}_{args.idx_to_visualize}.html",
        )
        print(f"Saving to {output_filepath}")
        new_fig.write_html(output_filepath)
    else:
        print("Showing figure...")
        new_fig.show()


if __name__ == "__main__":
    main()
