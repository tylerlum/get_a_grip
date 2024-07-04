import pathlib
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from tqdm import tqdm

from get_a_grip.grasp_planning.utils import dexdiffuser_utils
from get_a_grip.model_training.utils.nerf_load_utils import (
    get_nerf_configs_through_symlinks,
)


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    ckpt_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_DexDiffuser_models/ckpt_final.pth"
    )
    nerfdatas_path: Optional[pathlib.Path] = None
    nerfcheckpoints_path: Optional[pathlib.Path] = None
    num_grasps: int = 32
    max_num_iterations: int = 400
    overwrite: bool = False

    classifier_config_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_FINAL_LABELED_GRASPS_NOISE_AND_NONOISE_cnn-3d-xyz-global-cnn-cropped_CONTINUE/config.yaml"
    )
    optimizer_type: Literal["sgd", "cem", "random-sampling"] = "random-sampling"
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


def main() -> None:
    args = tyro.cli(CommandlineArgs)

    if args.nerfdatas_path is not None:
        nerfdata_paths = sorted(list(args.nerfdatas_path.iterdir()))
        print(f"Found {len(nerfdata_paths)} nerfdata paths")
        for nerfdata_path in tqdm(nerfdata_paths, desc="nerfdata_paths"):
            dexdiffuser_utils.run_dexdiffuser_sim_eval(
                args=dexdiffuser_utils.CommandlineArgs(
                    output_folder=args.output_folder,
                    ckpt_path=args.ckpt_path,
                    nerfdata_path=nerfdata_path,
                    nerf_config=None,
                    num_grasps=args.num_grasps,
                    max_num_iterations=args.max_num_iterations,
                    overwrite=args.overwrite,
                    optimize=True,
                    classifier_config_path=args.classifier_config_path,
                    optimizer_type=args.optimizer_type,
                    num_steps=args.num_steps,
                    n_random_rotations_per_grasp=args.n_random_rotations_per_grasp,
                    eval_batch_size=args.eval_batch_size,
                    use_bps_evaluator=True,
                    bps_evaluator_ckpt_path=pathlib.Path(
                        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-06_ALBERT_DexEvaluator_models_5_percent/ckpt-oh6pdm0p-step-final.pth"
                    ),
                )
            )
    elif args.nerfcheckpoints_path is not None:
        nerf_configs = get_nerf_configs_through_symlinks(args.nerfcheckpoints_path)
        print(f"Found {len(nerf_configs)} NERF configs")
        for nerf_config in tqdm(nerf_configs, desc="nerf_configs"):
            dexdiffuser_utils.run_dexdiffuser_sim_eval(
                args=dexdiffuser_utils.CommandlineArgs(
                    output_folder=args.output_folder,
                    ckpt_path=args.ckpt_path,
                    nerfdata_path=None,
                    nerf_config=nerf_config,
                    num_grasps=args.num_grasps,
                    max_num_iterations=args.max_num_iterations,
                    overwrite=args.overwrite,
                    optimize=True,
                    classifier_config_path=args.classifier_config_path,
                    optimizer_type=args.optimizer_type,
                    num_steps=args.num_steps,
                    n_random_rotations_per_grasp=args.n_random_rotations_per_grasp,
                    eval_batch_size=args.eval_batch_size,
                    use_bps_evaluator=True,
                    bps_evaluator_ckpt_path=pathlib.Path(
                        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-06_ALBERT_DexEvaluator_models_5_percent/ckpt-oh6pdm0p-step-final.pth"
                    ),
                )
            )
    else:
        raise ValueError(
            "Exactly one of nerfdatas_path or nerfcheckpoints_path must be specified"
        )


if __name__ == "__main__":
    main()
