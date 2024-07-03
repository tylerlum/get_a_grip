from nerf_grasping.frogger_utils import frogger_to_grasp_config_dict, FroggerArgs
from tqdm import tqdm
import tyro
import pathlib
from dataclasses import dataclass

@dataclass
class Args:
    only_objects_in_this_path: pathlib.Path
    meshdata_folder: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata_v2"
    )
    obj_is_yup: bool = True
    num_grasps: int = 3
    output_grasp_config_dicts_folder: pathlib.Path = pathlib.Path(
        "./output_grasp_config_dicts"
    )
    max_time: float = 60.0


def main() -> None:
    args = tyro.cli(Args)
    assert args.only_objects_in_this_path.exists(), f"{args.only_objects_in_this_path} does not exist"

    object_code_and_scale_strs = sorted([p.name for p in args.only_objects_in_this_path.iterdir()])
    assert len(object_code_and_scale_strs) > 0, f"{args.only_objects_in_this_path} is empty"

    for object_code_and_scale_str in tqdm(object_code_and_scale_strs, desc="Frogger"):
        idx = object_code_and_scale_str.index("_0_")
        object_code = object_code_and_scale_str[:idx]
        object_scale = float(object_code_and_scale_str[idx + 1 :].replace("_", "."))

        obj_filepath = args.meshdata_folder / object_code / "coacd" / "decomposed.obj"
        assert obj_filepath.exists(), f"{obj_filepath} does not exist"
        frogger_args = FroggerArgs(
            obj_filepath=obj_filepath,
            obj_scale=object_scale,
            obj_is_yup=args.obj_is_yup,
            num_grasps=args.num_grasps,
            output_grasp_config_dicts_folder=args.output_grasp_config_dicts_folder,
            visualize=False,
            max_time=args.max_time,
        )
        frogger_to_grasp_config_dict(frogger_args)


if __name__ == "__main__":
    main()