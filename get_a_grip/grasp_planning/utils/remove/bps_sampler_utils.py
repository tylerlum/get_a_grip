from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import trimesh
import tyro
from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.grasp_planning.config.nerf_evaluator_wrapper_config import (
    NerfEvaluatorWrapperConfig,
)
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    RandomSamplingConfig,
    SGDOptimizerConfig,
)
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_bps import (
    nerf_to_bps,
)
from get_a_grip.grasp_planning.scripts import optimizer as nerf_evaluator_optimizer
from get_a_grip.grasp_planning.utils import (
    train_nerf,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.grasp_planning.utils.nerf_evaluator_wrapper import (
    NerfEvaluatorWrapper,
    load_nerf_evaluator,
)
from get_a_grip.grasp_planning.utils.remove import bps_evaluator_utils
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)
from get_a_grip.model_training.models.bps_sampler_model import BpsSamplerModel
from get_a_grip.model_training.utils.diffusion import Diffusion
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.model_training.utils.nerf_utils import (
    compute_centroid_from_nerf,
)
from get_a_grip.model_training.utils.plot_utils import (
    plot_grasp_and_mesh_and_more,
)


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_pipeline: Pipeline,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    X_Oy_By: np.ndarray,
    ckpt_path: pathlib.Path,
    return_exactly_requested_num_grasps: bool = True,
    sample_grasps_multiplier: int = 10,
    PLOT: bool = False,
) -> dict:
    NUM_GRASPS = cfg.optimizer.num_grasps

    config = DiffusionConfig(
        training=TrainingConfig(
            log_path=ckpt_path.parent,
        )
    )
    model = BpsSamplerModel(
        n_pts=config.data.n_pts,
        grasp_dim=config.data.grasp_dim,
    )
    runner = Diffusion(config=config, model=model, load_multigpu_ckpt=True)
    runner.load_checkpoint(config, filename=ckpt_path.name)
    device = runner.device

    # Get BPS
    N_BASIS_PTS = 4096
    bps_values, basis_points_By, point_cloud_points_By = nerf_to_bps(
        nerf_pipeline=nerf_pipeline,
        lb_N=lb_N,
        ub_N=ub_N,
        X_N_By=X_N_By,
    )
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"

    # We sample more grasps than needed to account for filtering
    NUM_GRASP_SAMPLES = sample_grasps_multiplier * NUM_GRASPS
    bps_values_repeated = torch.from_numpy(bps_values).float().to(device)
    bps_values_repeated = bps_values_repeated.unsqueeze(dim=0).repeat(
        NUM_GRASP_SAMPLES, 1
    )
    assert bps_values_repeated.shape == (
        NUM_GRASP_SAMPLES,
        N_BASIS_PTS,
    ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

    # Sample grasps
    xT = torch.randn(NUM_GRASP_SAMPLES, config.data.grasp_dim, device=runner.device)
    x = runner.sample(xT=xT, cond=bps_values_repeated)

    if PLOT:
        assert X_Oy_By is not None
        X_By_Oy = np.linalg.inv(X_Oy_By)
        X_By_N = np.linalg.inv(X_N_By)

        _mesh_N = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_By = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_By.apply_transform(X_By_N)

        IDX = 0
        while True:
            fig = plot_grasp_and_mesh_and_more(
                grasp=x[IDX],
                X_N_Oy=X_By_Oy,
                basis_points=basis_points_By,
                bps=bps_values_repeated[IDX].detach().cpu().numpy(),
                mesh=mesh_By,
                processed_point_cloud_points=point_cloud_points_By,
            )
            fig.show()
            user_input = input("Next action?")
            if user_input == "q":
                break
            elif user_input == "n":
                IDX += 1
                IDX = IDX % NUM_GRASP_SAMPLES
            elif user_input == "p":
                IDX -= 1
                IDX = IDX % NUM_GRASP_SAMPLES
            else:
                print("Invalid input")
        breakpoint()

    # grasp to AllegroGraspConfig
    assert x.shape == (
        NUM_GRASP_SAMPLES,
        config.data.grasp_dim,
    ), f"Expected shape ({NUM_GRASP_SAMPLES}, {config.data.grasp_dim}), got {x.shape}"

    grasp_configs = AllegroGraspConfig.from_grasp(
        grasp=x,
    )

    if cfg.filter_less_feasible_grasps:
        grasp_configs = grasp_configs.filter_less_feasible(
            fingers_forward_theta_deg=cfg.fingers_forward_theta_deg,
            palm_upwards_theta_deg=cfg.palm_upwards_theta_deg,
        )

    if len(grasp_configs) < NUM_GRASPS:
        print(
            f"WARNING: After filtering, only {len(grasp_configs)} grasps remain, less than the requested {NUM_GRASPS} grasps"
        )

    if return_exactly_requested_num_grasps:
        grasp_configs = grasp_configs[:NUM_GRASPS]
    print(f"Returning {len(grasp_configs)} grasps")

    grasp_config_dicts = grasp_configs.as_dict()
    grasp_config_dicts["loss"] = np.linspace(
        0, 0.001, len(grasp_configs)
    )  # HACK: Currently don't have a loss, but need something here to sort

    return grasp_config_dicts


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    ckpt_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_DexDiffuser_models/ckpt_final.pth"
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
    optimizer_type: Literal["sgd", "random-sampling"] = "random-sampling"
    num_steps: int = 50
    n_random_rotations_per_grasp: int = 0
    eval_batch_size: int = 32

    use_bps_evaluator: bool = False
    bps_evaluator_ckpt_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_DexEvaluator_models/ckpt_yp920sn0_final.pth"
    )

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


def run_bps_sampler_sim_eval(args: CommandlineArgs) -> None:
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
        nerf_trainer = train_nerf.train_nerf_return_trainer(
            args=train_nerf.TrainNerfArgs(
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
    X_Oy_By = np.linalg.inv(X_By_Oy)
    X_N_By = np.eye(4)  # Same for sim nerfs
    X_N_Oy = X_N_By @ X_By_Oy

    # Get optimized grasps
    UNUSED_INIT_GRASP_CONFIG_DICT_PATH = pathlib.Path(
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-06-03_FINAL_INFERENCE_GRASPS/good_nonoise_one_per_object/grasps.npy"
    )
    UNUSED_NERF_EVALUATOR_CONFIG_PATH = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_nonoise_train_val_test_splits_cnn-3d-xyz-global-cnn-cropped_2024-06-02_16-57-36-630877/config.yaml"
    )
    UNUSED_X_N_Oy = np.eye(4)
    UNUSED_OUTPUT_PATH = pathlib.Path("UNUSED")
    grasp_config_dict = get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=False,  # Not used because causes issues with logging
            init_grasp_config_dict_path=UNUSED_INIT_GRASP_CONFIG_DICT_PATH,
            nerf_evaluator_wrapper=NerfEvaluatorWrapperConfig(
                nerf_config=nerf_config,
                nerf_evaluator_config_path=UNUSED_NERF_EVALUATOR_CONFIG_PATH,
                X_N_Oy=UNUSED_X_N_Oy,
            ),  # This is not used
            optimizer=SGDOptimizerConfig(
                num_grasps=args.num_grasps
            ),  # This optimizer is not used, but the num_grasps is used
            output_path=UNUSED_OUTPUT_PATH,
            random_seed=0,
            n_random_rotations_per_grasp=0,
            eval_batch_size=0,
            wandb=None,
            filter_less_feasible_grasps=False,  # Do not filter for sim
        ),
        nerf_pipeline=nerf_pipeline,
        lb_N=np.array([-0.2, 0.0, -0.2]),
        ub_N=np.array([0.2, 0.3, 0.2]),
        X_N_By=X_N_By,
        X_Oy_By=X_Oy_By,
        ckpt_path=args.ckpt_path,
        return_exactly_requested_num_grasps=True if not args.optimize else False,
    )

    if args.optimize:
        # Save this to file for next stage
        given_grasp_config_dict = grasp_config_dict.copy()
        NEW_init_grasp_config_dict_path = pathlib.Path("/tmp/temp.npy")
        np.save(NEW_init_grasp_config_dict_path, given_grasp_config_dict)

        USE_NERF_EVALUATOR = not args.use_bps_evaluator
        if USE_NERF_EVALUATOR:
            print("\n" + "=" * 80)
            print("Step 5: Load grasp metric")
            print("=" * 80 + "\n")
            print(
                f"Loading nerf_evaluator config from {args.nerf_evaluator_config_path}"
            )
            nerf_evaluator_config = tyro.extras.from_yaml(
                NerfEvaluatorModelConfig, args.nerf_evaluator_config_path.open()
            )

            nerf_evaluator_model = load_nerf_evaluator(
                nerf_evaluator_config=nerf_evaluator_config
            )
            nerf_evaluator_wrapper = NerfEvaluatorWrapper(
                nerf_field=nerf_pipeline.model.field,
                nerf_evaluator_model=nerf_evaluator_model,
                fingertip_config=nerf_evaluator_config.nerf_grasp_dataset_config.fingertip_config,
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

            optimized_grasp_config_dict = nerf_evaluator_optimizer.get_optimized_grasps(
                cfg=OptimizationConfig(
                    use_rich=False,  # Not used because causes issues with logging
                    init_grasp_config_dict_path=NEW_init_grasp_config_dict_path,
                    nerf_evaluator_wrapper=NerfEvaluatorWrapperConfig(
                        nerf_config=nerf_config,
                        nerf_evaluator_config_path=args.nerf_evaluator_config_path,
                        X_N_Oy=X_N_Oy,
                    ),  # This is not used because we are passing in a nerf_evaluator_wrapper
                    optimizer=optimizer,
                    output_path=pathlib.Path(
                        args.output_folder
                        / "optimized_grasp_config_dicts"
                        / f"{object_name}.npy"
                    ),
                    random_seed=0,
                    n_random_rotations_per_grasp=args.n_random_rotations_per_grasp,
                    eval_batch_size=args.eval_batch_size,
                    wandb=None,
                    filter_less_feasible_grasps=False,  # Do not filter for sim
                ),
                nerf_evaluator_wrapper=nerf_evaluator_wrapper,
            )

            grasp_config_dict = optimized_grasp_config_dict
        else:
            print("\n" + "=" * 80)
            print("Step 5: Run bps_evaluator")
            print("=" * 80 + "\n")
            EVALUATOR_CKPT_PATH = args.bps_evaluator_ckpt_path

            # B frame is at base of object z up frame
            # By frame is at base of object y up frame

            optimized_grasp_config_dict = bps_evaluator_utils.get_optimized_grasps(
                cfg=OptimizationConfig(
                    use_rich=False,  # Not used because causes issues with logging
                    init_grasp_config_dict_path=NEW_init_grasp_config_dict_path,
                    nerf_evaluator_wrapper=NerfEvaluatorWrapperConfig(
                        nerf_config=nerf_config,
                        nerf_evaluator_config_path=args.nerf_evaluator_config_path,
                        X_N_Oy=X_N_Oy,
                    ),  # This is not used
                    optimizer=RandomSamplingConfig(
                        num_grasps=args.num_grasps,
                        num_steps=args.num_steps,
                    ),  # This optimizer is not used, but the num_grasps is used and the num_steps is used
                    output_path=UNUSED_OUTPUT_PATH,
                    random_seed=0,
                    n_random_rotations_per_grasp=0,
                    eval_batch_size=args.eval_batch_size,
                    wandb=None,
                    filter_less_feasible_grasps=False,  # Do not filter for sim
                ),
                nerf_pipeline=nerf_pipeline,
                lb_N=np.array([-0.2, 0.0, -0.2]),
                ub_N=np.array([0.2, 0.3, 0.2]),
                X_N_By=X_N_By,
                ckpt_path=EVALUATOR_CKPT_PATH,
                optimize=True,  # Run refinement
            )

            grasp_config_dict = optimized_grasp_config_dict

    print(f"Saving grasp_config_dict to {output_file}")
    np.save(output_file, grasp_config_dict)


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    run_bps_sampler_sim_eval(args)


if __name__ == "__main__":
    main()
