import pathlib
from tqdm import tqdm
import tyro
from nerf_grasping import gg_utils
from dataclasses import dataclass
from typing import Optional, Literal
from nerf_grasping.grasp_utils import get_nerf_configs_through_symlinks


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    nerfdatas_path: Optional[pathlib.Path] = None
    nerfcheckpoints_path: Optional[pathlib.Path] = None
    num_grasps: int = 32
    max_num_iterations: int = 400
    overwrite: bool = False

    classifier_config_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_FINAL_LABELED_GRASPS_NOISE_AND_NONOISE_cnn-3d-xyz-global-cnn-cropped_CONTINUE/config.yaml"
    )
    init_grasp_config_dict_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-06-03_FINAL_INFERENCE_GRASPS/good_nonoise_one_per_object/grasps.npy"
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
            gg_utils.run_gg_sim_eval(
                args=gg_utils.CommandlineArgs(
                    output_folder=args.output_folder,
                    nerfdata_path=nerfdata_path,
                    nerfcheckpoint_path=None,
                    num_grasps=args.num_grasps,
                    max_num_iterations=args.max_num_iterations,
                    overwrite=args.overwrite,
                    optimize=True,
                    classifier_config_path=args.classifier_config_path,
                    init_grasp_config_dict_path=args.init_grasp_config_dict_path,
                    optimizer_type=args.optimizer_type,
                    num_steps=args.num_steps,
                    n_random_rotations_per_grasp=args.n_random_rotations_per_grasp,
                    eval_batch_size=args.eval_batch_size,
                )
            )
    elif args.nerfcheckpoints_path is not None:
        nerf_configs = get_nerf_configs_through_symlinks(args.nerfcheckpoints_path)
        print(f"Found {len(nerf_configs)} NERF configs")
        for nerf_config in tqdm(nerf_configs, desc="nerf_configs"):
            gg_utils.run_gg_sim_eval(
                args=gg_utils.CommandlineArgs(
                    output_folder=args.output_folder,
                    nerfdata_path=None,
                    nerfcheckpoint_path=nerf_config,
                    num_grasps=args.num_grasps,
                    max_num_iterations=args.max_num_iterations,
                    overwrite=args.overwrite,
                    optimize=True,
                    classifier_config_path=args.classifier_config_path,
                    init_grasp_config_dict_path=args.init_grasp_config_dict_path,
                    optimizer_type=args.optimizer_type,
                    num_steps=args.num_steps,
                    n_random_rotations_per_grasp=args.n_random_rotations_per_grasp,
                    eval_batch_size=args.eval_batch_size,
                )
            )
    else:
        raise ValueError(
            "Exactly one of nerfdatas_path or nerfcheckpoints_path must be specified"
        )


if __name__ == "__main__":
    main()
