from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pypose as pp
import torch
import trimesh
import tyro
from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_fingertip_dirs,
)
from get_a_grip.grasp_planning.config.grasp_metric_config import GraspMetricConfig
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    CEMOptimizerConfig,
    RandomSamplingConfig,
    SGDOptimizerConfig,
)
from get_a_grip.grasp_planning.scripts import optimizer as gg_optimizer
from get_a_grip.grasp_planning.utils import (
    ablation_utils,
    train_nerf_return_trainer,
)
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_bps import (
    nerf_to_bps,
)
from get_a_grip.grasp_planning.utils.visualize_utils import (
    plot_point_cloud_and_bps_and_mesh_and_grasp,
)
from get_a_grip.grasp_planning.utils.optimizer_utils import (
    AllegroGraspConfig,
    AllegroHandConfig,
    GraspMetric,
    hand_config_to_hand_model,
    load_classifier,
)
from get_a_grip.model_training.config.classifier_config import ClassifierConfig
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.utils.diffusion import Diffusion
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.model_training.utils.nerf_utils import (
    compute_centroid_from_nerf,
)

from get_a_grip.grasp_planning.utils.grasp_utils import (
    rot6d_to_matrix,
    compute_grasp_orientations,
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
    runner = Diffusion(config, load_multigpu_ckpt=True)
    runner.load_checkpoint(config, name=ckpt_path.stem)
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
            plot_point_cloud_and_bps_and_mesh_and_grasp(
                grasp=x[IDX],
                X_W_Oy=X_By_Oy,
                basis_points=basis_points_By,
                bps=bps_values_repeated[IDX].detach().cpu().numpy(),
                mesh=mesh_By,
                point_cloud_points=point_cloud_points_By,
                GRASP_IDX="?",
                object_code="?",
                y_PGS="?",
            )
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
    # TODO: make the numpy torch conversions less bad
    N_FINGERS = 4
    assert x.shape == (
        NUM_GRASP_SAMPLES,
        config.data.grasp_dim,
    ), f"Expected shape ({NUM_GRASP_SAMPLES}, {config.data.grasp_dim}), got {x.shape}"
    trans = x[:, :3].detach().cpu().numpy()
    rot6d = x[:, 3:9].detach().cpu().numpy()
    joint_angles = x[:, 9:25].detach().cpu().numpy()
    grasp_dirs = (
        x[:, 25:37].reshape(NUM_GRASP_SAMPLES, N_FINGERS, 3).detach().cpu().numpy()
    )

    rot = rot6d_to_matrix(rot6d)

    wrist_pose_matrix = (
        torch.eye(4, device=device).unsqueeze(0).repeat(NUM_GRASP_SAMPLES, 1, 1).float()
    )
    wrist_pose_matrix[:, :3, :3] = torch.from_numpy(rot).float().to(device)
    wrist_pose_matrix[:, :3, 3] = torch.from_numpy(trans).float().to(device)

    try:
        wrist_pose = pp.from_matrix(
            wrist_pose_matrix,
            pp.SE3_type,
        ).to(device)
    except ValueError as e:
        print("Error in pp.from_matrix")
        print(e)
        print("rot = ", rot)
        print("Orthogonalization did not work, running with looser tolerances")
        wrist_pose = pp.from_matrix(
            wrist_pose_matrix,
            pp.SE3_type,
            atol=1e-3,
            rtol=1e-3,
        ).to(device)

    assert wrist_pose.lshape == (NUM_GRASP_SAMPLES,)

    grasp_orientations = compute_grasp_orientations(
        grasp_dirs=torch.from_numpy(grasp_dirs).float().to(device),
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
    )

    # Convert to AllegroGraspConfig to dict
    grasp_configs = AllegroGraspConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
        grasp_orientations=grasp_orientations,
    )

    if cfg.filter_less_feasible_grasps:
        wrist_pose_matrix = grasp_configs.wrist_pose.matrix()
        x_dirs = wrist_pose_matrix[:, :, 0]
        z_dirs = wrist_pose_matrix[:, :, 2]

        fingers_forward_cos_theta = math.cos(
            math.radians(cfg.fingers_forward_theta_deg)
        )
        palm_upwards_cos_theta = math.cos(math.radians(cfg.palm_upwards_theta_deg))
        fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
        palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta
        grasp_configs = grasp_configs[fingers_forward & ~palm_upwards]
        print(
            f"Filtered less feasible grasps. New batch size: {grasp_configs.batch_size}"
        )

    if len(grasp_configs) < NUM_GRASPS:
        print(
            f"WARNING: After filtering, only {len(grasp_configs)} grasps remain, less than the requested {NUM_GRASPS} grasps"
        )

    if return_exactly_requested_num_grasps:
        grasp_configs = grasp_configs[:NUM_GRASPS]
    print(f"Returning {grasp_configs.batch_size} grasps")

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
    classifier_config_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_FINAL_LABELED_GRASPS_NOISE_AND_NONOISE_cnn-3d-xyz-global-cnn-cropped_CONTINUE/config.yaml"
    )
    optimizer_type: Literal["sgd", "cem", "random-sampling"] = "random-sampling"
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


def run_dexdiffuser_sim_eval(args: CommandlineArgs) -> None:
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
        nerf_checkpoints_folder = args.output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerf_return_trainer.train_nerf(
            args=train_nerf_return_trainer.Args(
                nerfdata_folder=args.nerfdata_path,
                nerfcheckpoints_folder=nerf_checkpoints_folder,
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
    UNUSED_CLASSIFIER_CONFIG_PATH = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/2024-06-02_nonoise_train_val_test_splits_cnn-3d-xyz-global-cnn-cropped_2024-06-02_16-57-36-630877/config.yaml"
    )
    UNUSED_X_N_Oy = np.eye(4)
    UNUSED_OUTPUT_PATH = pathlib.Path("UNUSED")
    grasp_config_dict = get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=False,  # Not used because causes issues with logging
            init_grasp_config_dict_path=UNUSED_INIT_GRASP_CONFIG_DICT_PATH,
            grasp_metric=GraspMetricConfig(
                nerf_config=nerf_config,
                classifier_config_path=UNUSED_CLASSIFIER_CONFIG_PATH,
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
            print(f"Loading classifier config from {args.classifier_config_path}")
            classifier_config = tyro.extras.from_yaml(
                ClassifierConfig, args.classifier_config_path.open()
            )

            classifier_model = load_classifier(classifier_config=classifier_config)
            grasp_metric = GraspMetric(
                nerf_field=nerf_pipeline.model.field,
                classifier_model=classifier_model,
                fingertip_config=classifier_config.nerfdata_config.fingertip_config,
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
            elif args.optimizer_type == "cem":
                optimizer = CEMOptimizerConfig(
                    num_grasps=args.num_grasps,
                    num_steps=args.num_steps,
                    num_samples=args.num_grasps,
                    num_elite=2,
                    min_cov_std=1e-2,
                )
            elif args.optimizer_type == "random-sampling":
                optimizer = RandomSamplingConfig(
                    num_grasps=args.num_grasps,
                    num_steps=args.num_steps,
                )
            else:
                raise ValueError(f"Invalid args.optimizer_type: {args.optimizer_type}")

            optimized_grasp_config_dict = gg_optimizer.get_optimized_grasps(
                cfg=OptimizationConfig(
                    use_rich=False,  # Not used because causes issues with logging
                    init_grasp_config_dict_path=NEW_init_grasp_config_dict_path,
                    grasp_metric=GraspMetricConfig(
                        nerf_config=nerf_config,
                        classifier_config_path=args.classifier_config_path,
                        X_N_Oy=X_N_Oy,
                    ),  # This is not used because we are passing in a grasp_metric
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
                grasp_metric=grasp_metric,
            )

            grasp_config_dict = optimized_grasp_config_dict
        else:
            print("\n" + "=" * 80)
            print("Step 5: Run ablation")
            print("=" * 80 + "\n")
            EVALUATOR_CKPT_PATH = args.bps_evaluator_ckpt_path

            # B frame is at base of object z up frame
            # By frame is at base of object y up frame

            optimized_grasp_config_dict = ablation_utils.get_optimized_grasps(
                cfg=OptimizationConfig(
                    use_rich=False,  # Not used because causes issues with logging
                    init_grasp_config_dict_path=NEW_init_grasp_config_dict_path,
                    grasp_metric=GraspMetricConfig(
                        nerf_config=nerf_config,
                        classifier_config_path=args.classifier_config_path,
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
    run_dexdiffuser_sim_eval(args)


if __name__ == "__main__":
    main()
