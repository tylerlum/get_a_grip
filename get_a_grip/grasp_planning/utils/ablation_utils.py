from __future__ import annotations

import math
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import trimesh
import tyro
from nerfstudio.pipelines.base_pipeline import Pipeline
from tqdm import tqdm

import wandb
from get_a_grip.grasp_planning.config.nerf_evaluator_wrapper_config import (
    NerfEvaluatorWrapperConfig,
)
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    RandomSamplingConfig,
)
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_bps import nerf_to_bps
from get_a_grip.grasp_planning.utils import (
    ablation_utils,
    train_nerf_return_trainer,
)
from get_a_grip.grasp_planning.utils.ablation_optimizer import RandomSamplingOptimizer
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.grasp_planning.utils.grasp_utils import (
    grasp_to_grasp_config,
)
from get_a_grip.grasp_planning.utils.optimizer_utils import (
    sample_random_rotate_transforms_only_around_y,
)
from get_a_grip.model_training.models.dex_evaluator import DexEvaluator
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.model_training.utils.nerf_utils import (
    compute_centroid_from_nerf,
)


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_pipeline: Pipeline,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    ckpt_path: pathlib.Path,
    optimize: bool,
) -> dict:
    BATCH_SIZE = cfg.eval_batch_size

    N_BASIS_PTS = 4096
    device = torch.device("cuda")
    dex_evaluator = DexEvaluator(in_grasp=3 + 6 + 16 + 12, in_bps=N_BASIS_PTS).to(
        device
    )
    dex_evaluator.eval()

    if ckpt_path.exists():
        dex_evaluator.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"{ckpt_path} does not exist")

    # Get BPS
    bps_values, _, _ = nerf_to_bps(
        nerf_pipeline=nerf_pipeline,
        lb_N=lb_N,
        ub_N=ub_N,
        X_N_By=X_N_By,
    )
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"

    bps_values_repeated = torch.from_numpy(bps_values).float().to(device)
    num_repeats = max(BATCH_SIZE, cfg.optimizer.num_grasps)
    bps_values_repeated = bps_values_repeated.unsqueeze(dim=0).repeat(num_repeats, 1)
    assert bps_values_repeated.shape == (
        num_repeats,
        N_BASIS_PTS,
    ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

    # Load grasp configs
    # TODO: Find a way to load a particular split of the grasp_data.
    init_grasp_config_dict = np.load(
        cfg.init_grasp_config_dict_path, allow_pickle=True
    ).item()

    num_grasps_in_dict = init_grasp_config_dict["trans"].shape[0]
    print(f"Found {num_grasps_in_dict} grasps in grasp config dict dataset")

    if (
        cfg.max_num_grasps_to_eval is not None
        and num_grasps_in_dict > cfg.max_num_grasps_to_eval
    ):
        print(f"Limiting to {cfg.max_num_grasps_to_eval} grasps from dataset.")
        # randomize the order, keep at most max_num_grasps_to_eval
        init_grasp_config_dict = {
            k: v[
                np.random.choice(
                    a=v.shape[0], size=cfg.max_num_grasps_to_eval, replace=False
                )
            ]
            for k, v in init_grasp_config_dict.items()
        }

    init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
        init_grasp_config_dict
    )
    print(f"Loaded {init_grasp_configs.batch_size} initial grasp configs.")

    # Put this here to ensure that the random seed is set before sampling random rotations.
    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    all_success_preds = []
    with torch.no_grad():
        # Sample random rotations
        N_SAMPLES = 1 + cfg.n_random_rotations_per_grasp
        new_grasp_configs_list = []
        for i in range(N_SAMPLES):
            new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                init_grasp_config_dict
            )
            if i != 0:
                random_rotate_transforms = (
                    sample_random_rotate_transforms_only_around_y(
                        new_grasp_configs.batch_size
                    )
                )
                new_grasp_configs.hand_config.set_wrist_pose(
                    random_rotate_transforms @ new_grasp_configs.hand_config.wrist_pose
                )
            new_grasp_configs_list.append(new_grasp_configs)

        new_grasp_config_dicts = defaultdict(list)
        for i in range(N_SAMPLES):
            config_dict = new_grasp_configs_list[i].as_dict()
            for k, v in config_dict.items():
                new_grasp_config_dicts[k].append(v)
        new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            {np.concatenate(v, axis=0) for k, v in new_grasp_config_dicts.items()}
        )
        assert new_grasp_configs.batch_size == init_grasp_configs.batch_size * N_SAMPLES

        # Filter grasps that are less IK feasible
        if cfg.filter_less_feasible_grasps:
            wrist_pose_matrix = new_grasp_configs.wrist_pose.matrix()
            x_dirs = wrist_pose_matrix[:, :, 0]
            z_dirs = wrist_pose_matrix[:, :, 2]

            fingers_forward_cos_theta = math.cos(
                math.radians(cfg.fingers_forward_theta_deg)
            )
            palm_upwards_cos_theta = math.cos(math.radians(cfg.palm_upwards_theta_deg))
            fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
            palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta
            new_grasp_configs = new_grasp_configs[fingers_forward & ~palm_upwards]
            print(
                f"Filtered less feasible grasps. New batch size: {new_grasp_configs.batch_size}"
            )

        # Evaluate grasp metric and collisions
        n_batches = math.ceil(new_grasp_configs.batch_size / BATCH_SIZE)
        for batch_i in tqdm(
            range(n_batches), desc=f"Evaling grasp metric with batch_size={BATCH_SIZE}"
        ):
            start_idx = batch_i * BATCH_SIZE
            end_idx = np.clip(
                (batch_i + 1) * BATCH_SIZE,
                a_min=None,
                a_max=new_grasp_configs.batch_size,
            )
            this_batch_size = end_idx - start_idx

            temp_grasp_configs = new_grasp_configs[start_idx:end_idx].to(device=device)
            wrist_trans_array = temp_grasp_configs.wrist_pose.translation().float()
            wrist_rot_array = temp_grasp_configs.wrist_pose.rotation().matrix().float()
            joint_angles_array = temp_grasp_configs.joint_angles.float()
            grasp_dirs_array = temp_grasp_configs.grasp_dirs.float()
            N_FINGERS = 4
            assert wrist_trans_array.shape == (this_batch_size, 3)
            assert wrist_rot_array.shape == (this_batch_size, 3, 3)
            assert joint_angles_array.shape == (this_batch_size, 16)
            assert grasp_dirs_array.shape == (this_batch_size, N_FINGERS, 3)
            g_O = torch.cat(
                [
                    wrist_trans_array,
                    wrist_rot_array[:, :, :2].reshape(this_batch_size, 6),
                    joint_angles_array,
                    grasp_dirs_array.reshape(this_batch_size, 12),
                ],
                dim=1,
            ).to(device=device)
            assert g_O.shape == (this_batch_size, 3 + 6 + 16 + 12)

            f_O = bps_values_repeated[:this_batch_size]
            assert f_O.shape == (this_batch_size, N_BASIS_PTS)

            success_preds = (
                dex_evaluator(f_O=f_O, g_O=g_O)[:, -1].detach().cpu().numpy()
            )
            assert (
                success_preds.shape == (this_batch_size,)
            ), f"success_preds.shape = {success_preds.shape}, expected ({this_batch_size},)"
            all_success_preds.append(success_preds)

        # Aggregate
        all_success_preds = np.concatenate(all_success_preds)
        assert all_success_preds.shape == (new_grasp_configs.batch_size,)

        # Sort by success_preds
        new_all_success_preds = all_success_preds
        ordered_idxs_best_first = np.argsort(new_all_success_preds)[::-1].copy()

        new_grasp_configs = new_grasp_configs[ordered_idxs_best_first]
        sorted_success_preds = new_all_success_preds[ordered_idxs_best_first][
            : cfg.optimizer.num_grasps
        ]

    init_grasp_configs = new_grasp_configs[: cfg.optimizer.num_grasps]

    if optimize:
        print(f"Optimizing {cfg.optimizer.num_grasps} grasps with random sampling")
        initial_losses_np = 1 - sorted_success_preds

        wrist_trans_array = init_grasp_configs.wrist_pose.translation().float()
        wrist_rot_array = init_grasp_configs.wrist_pose.rotation().matrix().float()
        joint_angles_array = init_grasp_configs.joint_angles.float()
        grasp_dirs_array = init_grasp_configs.grasp_dirs.float()
        N_FINGERS = 4
        assert wrist_trans_array.shape == (cfg.optimizer.num_grasps, 3)
        assert wrist_rot_array.shape == (cfg.optimizer.num_grasps, 3, 3)
        assert joint_angles_array.shape == (cfg.optimizer.num_grasps, 16)
        assert grasp_dirs_array.shape == (cfg.optimizer.num_grasps, N_FINGERS, 3)
        g_O = torch.cat(
            [
                wrist_trans_array,
                wrist_rot_array[:, :, :2].reshape(cfg.optimizer.num_grasps, 6),
                joint_angles_array,
                grasp_dirs_array.reshape(cfg.optimizer.num_grasps, 12),
            ],
            dim=1,
        ).to(device=device)
        assert g_O.shape == (cfg.optimizer.num_grasps, 3 + 6 + 16 + 12)

        random_sampling_optimizer = RandomSamplingOptimizer(
            dex_evaluator=dex_evaluator,
            bps=bps_values_repeated[: cfg.optimizer.num_grasps],
            init_grasps=g_O,
        )
        N_STEPS = cfg.optimizer.num_steps

        if N_STEPS > 0:
            for i in range(N_STEPS):
                losses = random_sampling_optimizer.step()
            losses_np = losses.detach().cpu().numpy()
        else:
            losses_np = initial_losses_np

        diff_losses = losses_np - initial_losses_np

        import sys

        print(
            f"Init Losses:  {[f'{x:.4f}' for x in initial_losses_np.tolist()]}",
            file=sys.stderr,
        )
        print(
            f"Final Losses: {[f'{x:.4f}' for x in losses_np.tolist()]}", file=sys.stderr
        )
        print(
            f"Diff Losses:  {[f'{x:.4f}' for x in diff_losses.tolist()]}",
            file=sys.stderr,
        )
        grasp_config = grasp_to_grasp_config(grasp=random_sampling_optimizer.grasps)
        grasp_config_dict = grasp_config.as_dict()
        grasp_config_dict["loss"] = losses_np
    else:
        print("Skipping optimization of grasps")
        grasp_config_dict = init_grasp_configs.as_dict()
        grasp_config_dict["loss"] = 1 - sorted_success_preds

    print(f"Saving final grasp config dict to {cfg.output_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cfg.output_path), grasp_config_dict, allow_pickle=True)

    if wandb.run is not None:
        wandb.finish()
    return grasp_config_dict


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


def run_ablation_sim_eval(args: CommandlineArgs) -> None:
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
            args=train_nerf_return_trainer.Args(
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
    print("Step 5: Run ablation")
    print("=" * 80 + "\n")
    EVALUATOR_CKPT_PATH = args.bps_evaluator_ckpt_path

    # B frame is at base of object z up frame
    # By frame is at base of object y up frame
    optimized_grasp_config_dict = ablation_utils.get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=False,  # Not used because causes issues with logging
            init_grasp_config_dict_path=args.init_grasp_config_dict_path,
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
        optimize=args.optimize,
    )

    grasp_config_dict = optimized_grasp_config_dict

    print(f"Saving grasp_config_dict to {output_file}")
    np.save(output_file, grasp_config_dict)


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    run_ablation_sim_eval(args)


if __name__ == "__main__":
    main()
