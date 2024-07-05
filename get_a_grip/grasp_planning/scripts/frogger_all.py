import pathlib
from dataclasses import dataclass

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.dataset_generation.utils.process_utils import (
    get_object_codes_and_scales_to_process,
)
from get_a_grip.grasp_planning.utils.frogger_utils import (
    FroggerArgs,
    frogger_to_grasp_config_dict,
)


@dataclass
class Args:
    input_object_code_and_scales_txt_path: pathlib.Path = (
        get_data_folder() / "NEW_DATASET/nerfdata_settled_successes.txt"
    )
    meshdata_root_path: pathlib.Path = pathlib.Path(get_data_folder() / "large/meshes")
    obj_is_yup: bool = True
    num_grasps: int = 3
    output_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "./output_grasp_config_dicts"
    )
    max_time: float = 60.0
    no_continue: bool = False


def main() -> None:
    args = tyro.cli(Args)

    input_object_code_and_scale_strs = get_object_codes_and_scales_to_process(
        input_object_code_and_scales_txt_path=args.input_object_code_and_scales_txt_path,
        meshdata_root_path=args.meshdata_root_path,
        output_folder_path=args.output_grasp_config_dicts_path,
        no_continue=args.no_continue,
    )
    assert len(input_object_code_and_scale_strs) > 0

    for object_code_and_scale_str in tqdm(
        input_object_code_and_scale_strs, desc="Frogger"
    ):
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        obj_filepath = (
            args.meshdata_root_path / object_code / "coacd" / "decomposed.obj"
        )
        assert obj_filepath.exists(), f"{obj_filepath} does not exist"
        frogger_args = FroggerArgs(
            obj_filepath=obj_filepath,
            obj_scale=object_scale,
            obj_is_yup=args.obj_is_yup,
            num_grasps=args.num_grasps,
            output_grasp_config_dicts_folder=args.output_grasp_config_dicts_path,
            visualize=False,
            max_time=args.max_time,
        )
        frogger_to_grasp_config_dict(frogger_args)


if __name__ == "__main__":
    main()
