import pathlib
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_grasp_orientations as compute_grasp_orientations_external,
)
from get_a_grip.dataset_generation.utils.object_model import ObjectModel
from get_a_grip.dataset_generation.utils.pose_conversion import (
    hand_config_np_to_pose,
)
from get_a_grip.dataset_generation.utils.process_utils import (
    get_object_codes_and_scales_to_process,
)
from get_a_grip.utils.parse_object_code_and_scale import (
    is_object_code_and_scale_str,
    parse_object_code_and_scale,
)
from get_a_grip.utils.seed import set_seed


@dataclass
class GenerateGraspConfigDictsArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    input_hand_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/hand_config_dicts"
    )
    input_object_code_and_scales_txt_path: Optional[pathlib.Path] = None
    output_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/grasp_config_dicts"
    )
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    gpu: int = 0
    all_mid_optimization_steps: bool = False
    seed: int = 42
    continue_ok: bool = True


def compute_grasp_orientations(
    args: GenerateGraspConfigDictsArgs,
    hand_config_dict: Dict[str, np.ndarray],
    object_code: str,
    object_scale: float,
) -> torch.Tensor:
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"

    hand_pose = hand_config_np_to_pose(
        trans=hand_config_dict["trans"],
        rot=hand_config_dict["rot"],
        joint_angles=hand_config_dict["joint_angles"],
    ).to(device)
    batch_size = hand_pose.shape[0]

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)

    # object model
    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=batch_size,
        num_samples=0,
        device=device,
    )
    object_model.initialize(object_code, object_scale)
    grasp_orientations = compute_grasp_orientations_external(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        object_model=object_model,
    )
    assert grasp_orientations.shape == (batch_size, hand_model.num_fingers, 3, 3)
    return grasp_orientations


def generate_grasp_config_dicts(
    args: GenerateGraspConfigDictsArgs,
    input_hand_config_dicts_path: pathlib.Path,
    output_grasp_config_dicts_path: pathlib.Path,
) -> None:
    input_object_code_and_scale_strs_from_folder = [
        path.stem for path in list(input_hand_config_dicts_path.glob("*.npy"))
    ]
    input_object_code_and_scale_strs = get_object_codes_and_scales_to_process(
        input_object_code_and_scale_strs=input_object_code_and_scale_strs_from_folder,
        meshdata_root_path=args.meshdata_root_path,
        output_folder_path=output_grasp_config_dicts_path,
        continue_ok=args.continue_ok,
    )

    # Only do the objects in the text file
    if args.input_object_code_and_scales_txt_path is not None:
        print(
            f"Reading object codes and scales from {args.input_object_code_and_scales_txt_path}"
        )
        with open(args.input_object_code_and_scales_txt_path, "r") as f:
            input_object_code_and_scale_strs_from_file = f.read().splitlines()
        print(f"From folder, there are {len(input_object_code_and_scale_strs)} objects")
        print(
            f"From file, there are {len(input_object_code_and_scale_strs_from_file)} objects"
        )
        input_object_code_and_scale_strs = list(
            set(input_object_code_and_scale_strs).intersection(
                input_object_code_and_scale_strs_from_file
            )
        )
        print(
            f"In intersection of both, there are {len(input_object_code_and_scale_strs)} objects"
        )

    random.Random(args.seed).shuffle(input_object_code_and_scale_strs)

    set_seed(42)  # Want this fixed so deterministic computation
    pbar = tqdm(
        input_object_code_and_scale_strs,
        desc="Generating grasp_config_dicts",
        dynamic_ncols=True,
    )
    for object_code_and_scale_str in pbar:
        hand_config_dict_filepath = (
            input_hand_config_dicts_path / f"{object_code_and_scale_str}.npy"
        )

        assert is_object_code_and_scale_str(
            object_code_and_scale_str
        ), f"object_code_and_scale_str: {object_code_and_scale_str} is not valid."
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        # Read in data
        hand_config_dict: Dict[str, np.ndarray] = np.load(
            hand_config_dict_filepath, allow_pickle=True
        ).item()

        # Compute grasp_orientations
        grasp_orientations = compute_grasp_orientations(
            args=args,
            hand_config_dict=hand_config_dict,
            object_code=object_code,
            object_scale=object_scale,
        )  # shape = (batch_size, num_fingers, 3, 3)

        grasp_config_dict = hand_config_dict.copy()
        grasp_config_dict["grasp_orientations"] = (
            grasp_orientations.detach().cpu().numpy()
        )

        # Save grasp_config_dict
        output_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)
        np.save(
            file=output_grasp_config_dicts_path / f"{object_code_and_scale_str}.npy",
            arr=grasp_config_dict,
            allow_pickle=True,
        )


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[GenerateGraspConfigDictsArgs])

    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    generate_grasp_config_dicts(
        args=args,
        input_hand_config_dicts_path=args.input_hand_config_dicts_path,
        output_grasp_config_dicts_path=args.output_grasp_config_dicts_path,
    )

    if not args.all_mid_optimization_steps:
        return

    mid_optimization_steps = (
        sorted(
            [
                int(pp.name)
                for pp in args.input_hand_config_dicts_path.glob("mid_optimization/*")
            ]
        )
        if (args.input_hand_config_dicts_path / "mid_optimization").exists()
        else []
    )
    print(f"mid_optimization_steps: {mid_optimization_steps}")

    for mid_optimization_step in mid_optimization_steps:
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_path = (
            args.input_hand_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_path = (
            args.output_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        generate_grasp_config_dicts(
            args=args,
            input_hand_config_dicts_path=mid_optimization_input_path,
            output_grasp_config_dicts_path=mid_optimization_output_path,
        )


if __name__ == "__main__":
    main()
