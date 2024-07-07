from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import trimesh
import tyro

from get_a_grip.grasp_planning.config.nerf_evaluator_wrapper_config import (
    NerfEvaluatorWrapperConfig,
)
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    RandomSamplingConfig,
    SGDOptimizerConfig,
)
from get_a_grip.grasp_planning.scripts.optimizer import (
    get_optimized_grasps,
)
from get_a_grip.grasp_planning.utils import (
    train_nerf_return_trainer,
)
from get_a_grip.grasp_planning.utils.nerf_evaluator_wrapper import (
    NerfEvaluatorWrapper,
    load_nerf_evaluator,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.model_training.utils.nerf_utils import (
    compute_centroid_from_nerf,
)


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    bps_evaluator_ckpt_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_DexEvaluator_models/ckpt_yp920sn0_final.pth"
    )
    nerfdata_path: Optional[pathlib.Path] = None
    nerf_config: Optional[pathlib.Path] = None
    num_grasps: int = 32
    max_num_iterations: int = 400
    overwrite: bool = False

    optimize: bool = False
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
        if self.nerfdata_path is not None and self.nerf_config is None:
            assert self.nerfdata_path.exists(), f"{self.nerfdata_path} does not exist"
            assert (
                self.nerfdata_path / "transforms.json"
            ).exists(), f"{self.nerfdata_path / 'transforms.json'} does not exist"
            assert (
                self.nerfdata_path / "images"
            ).exists(), f"{self.nerfdata_path / 'images'} does not exist"
        elif self.nerfdata_path is None and self.nerf_config is not None:
            assert self.nerf_config.exists(), f"{self.nerf_config} does not exist"
            assert (
                self.nerf_config.suffix == ".yml"
            ), f"{self.nerf_config} does not have a .yml suffix"
        else:
            raise ValueError(
                "Exactly one of nerfdata_path or nerf_config must be specified"
            )


def run_nerf_evaluator_sim_eval(args: CommandlineArgs) -> None:
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

    # Get object name
    if args.nerfdata_path is not None:
        object_name = args.nerfdata_path.name
    elif args.nerf_config is not None:
        object_name = args.nerf_config.parents[2].name
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerf_config must be specified"
        )
    print(f"object_name = {object_name}")

    # Prepare output folder
    args.output_folder.mkdir(exist_ok=True, parents=True)
    output_file = args.output_folder / f"{object_name}.npy"
    if output_file.exists():
        if not args.overwrite:
            print(f"{output_file} already exists, skipping")
            return

        print(f"{output_file} already exists, overwriting")

    # Prepare nerf model
    if args.nerfdata_path is not None:
        start_time = time.time()
        nerfcheckpoints_folder = args.output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerf_return_trainer.train_nerf(
            args=train_nerf_return_trainer.TrainNerfReturnTrainerArgs(
                nerfdata_folder=args.nerfdata_path,
                nerfcheckpoints_folder=nerfcheckpoints_folder,
                max_num_iterations=args.max_num_iterations,
            )
        )
        nerf_pipeline = nerf_trainer.pipeline
        nerf_model = nerf_trainer.pipeline.model
        nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
        end_time = time.time()
        print("@" * 80)
        print(f"Time to train_nerf: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    elif args.nerf_config is not None:
        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(
            args.nerf_config, test_mode="test"
        )  # Need this for point cloud
        nerf_model = nerf_pipeline.model
        nerf_config = args.nerf_config
        end_time = time.time()
        print("@" * 80)
        print(f"Time to load_nerf_pipeline: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerf_config must be specified"
        )
    args.nerf_config = nerf_config

    # Compute centroid
    nerf_centroid_N = compute_centroid_from_nerf(
        nerf_model.field,
        lb=np.array([-0.2, 0.0, -0.2]),
        ub=np.array([0.2, 0.3, 0.2]),
        level=15,
        num_pts_x=100,
        num_pts_y=100,
        num_pts_z=100,
    )
    assert nerf_centroid_N.shape == (
        3,
    ), f"Expected shape (3,), got {nerf_centroid_N.shape}"
    obj_y = nerf_centroid_N[1]
    X_By_Oy = trimesh.transformations.translation_matrix([0, obj_y, 0])
    _X_Oy_By = np.linalg.inv(X_By_Oy)
    X_N_By = np.eye(4)  # Same for sim nerfs
    X_N_Oy = X_N_By @ X_By_Oy

    # Get optimized grasps
    UNUSED_OUTPUT_PATH = pathlib.Path("UNUSED")

    print("\n" + "=" * 80)
    print("Step 5: Load grasp metric")
    print("=" * 80 + "\n")
    print(f"Loading nerf_evaluator config from {args.nerf_evaluator_config_path}")
    nerf_evaluator_config = tyro.extras.from_yaml(
        NerfEvaluatorModelConfig, args.nerf_evaluator_config_path.open()
    )

    nerf_evaluator_model = load_nerf_evaluator(
        nerf_evaluator_config=nerf_evaluator_config
    )
    nerf_evaluator_wrapper = NerfEvaluatorWrapper(
        nerf_field=nerf_model.field,
        nerf_evaluator_model=nerf_evaluator_model,
        fingertip_config=nerf_evaluator_config.nerfdata_config.fingertip_config,
        X_N_Oy=X_N_Oy,
    )

    print("\n" + "=" * 80)
    print("Step 6: Optimize grasps")
    print("=" * 80 + "\n")
    if args.optimizer_type == "sgd":
        optimizer = SGDOptimizerConfig(
            num_grasps=args.num_grasps,
            num_steps=args.num_steps,
            # finger_lr=1e-3,
            finger_lr=0,
            # grasp_dir_lr=1e-4,
            grasp_dir_lr=0,
            wrist_lr=1e-3,
        )
    elif args.optimizer_type == "random-sampling":
        optimizer = RandomSamplingConfig(
            num_grasps=args.num_grasps,
            num_steps=args.num_steps,
        )
    else:
        raise ValueError(f"Invalid args.optimizer_type: {args.optimizer_type}")

    optimized_grasp_config_dict = get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=False,  # Not used because causes issues with logging
            init_grasp_config_dict_path=args.init_grasp_config_dict_path,
            nerf_evaluator_wrapper=NerfEvaluatorWrapperConfig(
                nerf_config=nerf_config,
                nerf_evaluator_config_path=args.nerf_evaluator_config_path,
                X_N_Oy=X_N_Oy,
            ),  # This is not used because we are passing in a nerf_evaluator_wrapper
            optimizer=optimizer,
            output_path=UNUSED_OUTPUT_PATH,
            random_seed=0,
            n_random_rotations_per_grasp=0,
            eval_batch_size=args.eval_batch_size,
            wandb=None,
            filter_less_feasible_grasps=False,  # Do not filter for sim
        ),
        nerf_evaluator_wrapper=nerf_evaluator_wrapper,
    )

    grasp_config_dict = optimized_grasp_config_dict

    print(f"Saving grasp_config_dict to {output_file}")
    np.save(output_file, grasp_config_dict)


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    run_nerf_evaluator_sim_eval(args)


if __name__ == "__main__":
    main()
