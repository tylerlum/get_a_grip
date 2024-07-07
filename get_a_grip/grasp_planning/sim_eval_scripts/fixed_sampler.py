import pathlib
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro
from tqdm import tqdm

from get_a_grip.model_training.utils.nerf_load_utils import (
    get_nerf_configs_through_symlinks,
)


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    nerfdatas_path: Optional[pathlib.Path] = None
    nerfcheckpoints_path: Optional[pathlib.Path] = None
    num_grasps: int = 32
    max_num_iterations: int = 400
    overwrite: bool = False

    nerf_evaluator_config_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_FINAL_LABELED_GRASPS_NOISE_AND_NONOISE_cnn-3d-xyz-global-cnn-cropped_CONTINUE/config.yaml"
    )
    init_grasp_config_dict_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-06-03_FINAL_INFERENCE_GRASPS/good_nonoise_one_per_object/grasps.npy"
    )
    optimizer_type: Literal["sgd", "random-sampling"] = "random-sampling"
    num_steps: int = 50
    n_random_rotations_per_grasp: int = 0
    eval_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.nerfdatas_path is not None and self.nerfcheckpoints_path is None:
            assert self.nerfdatas_path.exists(), f"{self.nerfdatas_path} does not exist"
        elif self.nerfdatas_path is None and self.nerfcheckpoints_path is not None:
            assert (
                self.nerfcheckpoints_path.exists()
            ), f"{self.nerfcheckpoints_path} does not exist"
        else:
            raise ValueError(
                "Exactly one of nerfdatas_path or nerfcheckpoints_path must be specified"
            )


def fixed_sampler_eval(
    init_grasp_config_dict_path: pathlib.Path,
    num_grasps: int,
    object_code_and_scale_str: str,
    output_folder: pathlib.Path,
) -> None:
    init_grasp_config_dict = np.load(
        init_grasp_config_dict_path, allow_pickle=True
    ).item()
    B = init_grasp_config_dict["trans"].shape[0]
    assert (
        B >= num_grasps
    ), f"num_grasps ({num_grasps}) must be less than or equal to B ({B})"

    indices = np.random.choice(B, size=num_grasps, replace=False)
    fixed_sampler_grasp_config_dict = {
        k: v[indices] for k, v in init_grasp_config_dict.items()
    }

    np.save(
        output_folder / f"{object_code_and_scale_str}.npy",
        fixed_sampler_grasp_config_dict,
    )


def main() -> None:
    args = tyro.cli(CommandlineArgs)

    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.nerfdatas_path is not None:
        nerfdata_paths = sorted(list(args.nerfdatas_path.iterdir()))
        print(f"Found {len(nerfdata_paths)} nerfdata paths")
        for nerfdata_path in tqdm(nerfdata_paths, desc="nerfdata_paths"):
            object_name = nerfdata_path.name
            fixed_sampler_eval(
                init_grasp_config_dict_path=args.init_grasp_config_dict_path,
                num_grasps=args.num_grasps,
                object_code_and_scale_str=object_name,
                output_folder=args.output_folder,
            )
    elif args.nerfcheckpoints_path is not None:
        nerf_configs = get_nerf_configs_through_symlinks(args.nerfcheckpoints_path)
        print(f"Found {len(nerf_configs)} NERF configs")
        for nerf_config in tqdm(nerf_configs, desc="nerf_configs"):
            object_name = nerf_config.parents[2].name
            fixed_sampler_eval(
                init_grasp_config_dict_path=args.init_grasp_config_dict_path,
                num_grasps=args.num_grasps,
                object_code_and_scale_str=object_name,
                output_folder=args.output_folder,
            )
    else:
        raise ValueError(
            "Exactly one of nerfdatas_path or nerfcheckpoints_path must be specified"
        )


if __name__ == "__main__":
    main()
