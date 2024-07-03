import time
from typing import Optional, Tuple, List, Literal, Callable
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerf_grasping.run_pipeline import (
    run_curobo,
    MultipleOutputs,
    PipelineConfig,
    CommandlineArgs,
    transform_point,
    add_transform_matrix_traces,
    save_to_file,
    load_from_file,
    visualize,
)
from nerf_grasping.grasp_utils import load_nerf_pipeline
from nerf_grasping.optimizer import get_optimized_grasps
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
    GraspMetric,
    DepthImageGraspMetric,
    load_classifier,
    load_depth_image_classifier,
    is_in_limits,
    clamp_in_limits,
)
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimization_config import OptimizationConfig
from nerf_grasping.config.optimizer_config import (
    SGDOptimizerConfig,
    CEMOptimizerConfig,
    RandomSamplingConfig,
)
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.nerfstudio_train import train_nerfs_return_trainer
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
)
from nerf_grasping.config.classifier_config import ClassifierConfig
import trimesh
import pathlib
import tyro
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
from datetime import datetime

from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_batch import (
    prepare_trajopt_batch,
    solve_prepared_trajopt_batch,
    get_trajectories_from_result,
    compute_over_limit_factors,
)
from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
    # solve_trajopt,
    DEFAULT_Q_FR3,
    DEFAULT_Q_ALGR,
)
from nerf_grasping.curobo_fr3_algr_zed2i.fr3_algr_zed2i_world import (
    get_world_cfg,
)
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
)

from frogger.robots.robot_core import RobotModel

import sys


def compute_nerf_sampler_grasps(
    nerf_model: Model,
    cfg: PipelineConfig,
    ckpt_path: str | pathlib.Path,
    optimize: bool,
    sample_grasps_multiplier: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    trimesh.Trimesh,
    np.ndarray,
    np.ndarray,
]:
    print("=" * 80)
    print("Step 1: Figuring out frames")
    print("=" * 80 + "\n")
    print("Frames are W = world, N = nerf, O = object, Oy = object y-up, H = hand")
    print(
        "W is centered at the robot base. N is centered where origin of NeRF data collection is. O is centered at the object centroid. Oy is centered at the object centroid. H is centered at the base of the middle finger"
    )
    print(
        "W, N, O are z-up frames. Oy is y-up. H has z-up along finger and x-up along palm normal"
    )
    print("X_A_B represents 4x4 transformation matrix of frame B wrt A")
    X_W_N, X_O_Oy = cfg.X_W_N, cfg.X_O_Oy
    lb_N, ub_N = cfg.lb_N, cfg.ub_N

    print("\n" + "=" * 80)
    print("Step 2: Get NERF")
    print("=" * 80 + "\n")
    nerf_field = nerf_model.field
    nerf_config = (
        cfg.nerf_config
        if cfg.nerf_config is not None
        else pathlib.Path("DUMMY_NERF_CONFIG/config.yml")
    )  # Dummy value to put in, not used because nerf_model is passed in

    print("\n" + "=" * 80)
    print("Step 3: Convert NeRF to mesh")
    print("=" * 80 + "\n")
    nerf_to_mesh_folder = cfg.output_folder / "nerf_to_mesh" / cfg.object_code / "coacd"
    nerf_to_mesh_folder.mkdir(parents=True, exist_ok=True)
    mesh_N = nerf_to_mesh(
        field=nerf_field,
        level=cfg.density_levelset_threshold,
        lb=lb_N,
        ub=ub_N,
        save_path=nerf_to_mesh_folder / "decomposed.obj",
    )

    # Save to /tmp/mesh_viz_object.obj as well
    mesh_N.export("/tmp/mesh_viz_object.obj")

    print("\n" + "=" * 80)
    print(
        "Step 4: Compute X_N_Oy (transformation of the object y-up frame wrt the nerf frame)"
    )
    print("=" * 80 + "\n")
    USE_MESH = False
    mesh_centroid_N = mesh_N.centroid
    nerf_centroid_N = compute_centroid_from_nerf(
        nerf_field,
        lb=lb_N,
        ub=ub_N,
        level=cfg.density_levelset_threshold,
        num_pts_x=100,
        num_pts_y=100,
        num_pts_z=100,
    )
    print(f"mesh_centroid_N: {mesh_centroid_N}")
    print(f"nerf_centroid_N: {nerf_centroid_N}")
    centroid_N = mesh_centroid_N if USE_MESH else nerf_centroid_N
    print(f"USE_MESH: {USE_MESH}, centroid_N: {centroid_N}")
    assert centroid_N.shape == (3,), f"centroid_N.shape is {centroid_N.shape}, not (3,)"
    X_N_O = trimesh.transformations.translation_matrix(centroid_N)

    X_N_Oy = X_N_O @ X_O_Oy
    X_Oy_N = np.linalg.inv(X_N_Oy)
    assert X_N_Oy.shape == (4, 4), f"X_N_Oy.shape is {X_N_Oy.shape}, not (4, 4)"

    mesh_W = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_W.apply_transform(X_W_N)

    # For debugging
    mesh_Oy = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_Oy.apply_transform(X_Oy_N)
    nerf_to_mesh_Oy_folder = (
        cfg.output_folder / "nerf_to_mesh_Oy" / cfg.object_code / "coacd"
    )
    nerf_to_mesh_Oy_folder.mkdir(parents=True, exist_ok=True)
    mesh_Oy.export(nerf_to_mesh_Oy_folder / "decomposed.obj")
    mesh_centroid_Oy = transform_point(X_Oy_N, centroid_N)
    nerf_centroid_Oy = transform_point(X_Oy_N, centroid_N)

    if cfg.visualize:
        # Visualize N
        fig_N = go.Figure()
        fig_N.add_trace(
            go.Mesh3d(
                x=mesh_N.vertices[:, 0],
                y=mesh_N.vertices[:, 1],
                z=mesh_N.vertices[:, 2],
                i=mesh_N.faces[:, 0],
                j=mesh_N.faces[:, 1],
                k=mesh_N.faces[:, 2],
                color="lightblue",
                name="Mesh",
                opacity=0.5,
            )
        )
        fig_N.add_trace(
            go.Scatter3d(
                x=[mesh_centroid_N[0]],
                y=[mesh_centroid_N[1]],
                z=[mesh_centroid_N[2]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Mesh centroid",
            )
        )
        fig_N.add_trace(
            go.Scatter3d(
                x=[nerf_centroid_N[0]],
                y=[nerf_centroid_N[1]],
                z=[nerf_centroid_N[2]],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="NeRF centroid",
            )
        )
        fig_N.update_layout(title="Mesh in nerf frame")
        add_transform_matrix_traces(fig=fig_N, transform_matrix=np.eye(4), length=0.1)
        fig_N.show()

        # Visualize Oy
        fig_Oy = go.Figure()
        fig_Oy.add_trace(
            go.Mesh3d(
                x=mesh_Oy.vertices[:, 0],
                y=mesh_Oy.vertices[:, 1],
                z=mesh_Oy.vertices[:, 2],
                i=mesh_Oy.faces[:, 0],
                j=mesh_Oy.faces[:, 1],
                k=mesh_Oy.faces[:, 2],
                color="lightblue",
                name="Mesh",
                opacity=0.5,
            )
        )
        fig_Oy.add_trace(
            go.Scatter3d(
                x=[mesh_centroid_Oy[0]],
                y=[mesh_centroid_Oy[1]],
                z=[mesh_centroid_Oy[2]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Mesh centroid",
            )
        )
        fig_Oy.add_trace(
            go.Scatter3d(
                x=[nerf_centroid_Oy[0]],
                y=[nerf_centroid_Oy[1]],
                z=[nerf_centroid_Oy[2]],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="NeRF centroid",
            )
        )
        fig_Oy.update_layout(title="Mesh in object y-up frame")
        add_transform_matrix_traces(fig=fig_Oy, transform_matrix=np.eye(4), length=0.1)
        fig_Oy.show()

    print("\n" + "=" * 80)
    print("Step 5: Run nerf_sampler")
    print("=" * 80 + "\n")

    from nerf_grasping import nerf_sampler_utils

    return_exactly_requested_num_grasps = True if not optimize else False
    optimized_grasp_config_dict = nerf_sampler_utils.get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=False,  # Not used because causes issues with logging
            init_grasp_config_dict_path=cfg.init_grasp_config_dict_path,  # This is not used
            grasp_metric=GraspMetricConfig(
                nerf_checkpoint_path=nerf_config,
                classifier_config_path=cfg.classifier_config_path,
                X_N_Oy=X_N_Oy,
            ),  # This is not used
            optimizer=SGDOptimizerConfig(
                num_grasps=cfg.num_grasps,
            ),  # This optimizer is not used, but the num_grasps is used
            output_path=pathlib.Path(
                cfg.output_folder
                / "optimized_grasp_config_dicts"
                / f"{cfg.object_code_and_scale_str}.npy"
            ),
            random_seed=cfg.random_seed,
            n_random_rotations_per_grasp=cfg.n_random_rotations_per_grasp,
            eval_batch_size=cfg.eval_batch_size,
            wandb=None,
            filter_less_feasible_grasps=True,
        ),
        nerf_model=nerf_model,
        X_N_Oy=X_N_Oy,
        ckpt_path=ckpt_path,
        return_exactly_requested_num_grasps=return_exactly_requested_num_grasps,
        sample_grasps_multiplier=sample_grasps_multiplier,
    )

    if optimize:
        given_grasp_config_dict = optimized_grasp_config_dict.copy()
        NEW_init_grasp_config_dict_path = (
            cfg.output_folder
            / "NEW_init_grasp_config_dicts.npy"
        )
        np.save(NEW_init_grasp_config_dict_path, given_grasp_config_dict)

        print("\n" + "=" * 80)
        print("Step 5: Load grasp metric")
        print("=" * 80 + "\n")
        print(f"Loading classifier config from {cfg.classifier_config_path}")
        classifier_config = tyro.extras.from_yaml(
            ClassifierConfig, cfg.classifier_config_path.open()
        )

        USE_DEPTH_IMAGES = isinstance(
            classifier_config.nerfdata_config, DepthImageNerfDataConfig
        )
        if USE_DEPTH_IMAGES:
            classifier_model = load_depth_image_classifier(classifier=classifier_config)
            grasp_metric = DepthImageGraspMetric(
                nerf_model=nerf_model,
                classifier_model=classifier_model,
                fingertip_config=classifier_config.nerfdata_config.fingertip_config,
                camera_config=classifier_config.nerfdata_config.fingertip_camera_config,
                X_N_Oy=X_N_Oy,
            )
        else:
            classifier_model = load_classifier(classifier_config=classifier_config)
            grasp_metric = GraspMetric(
                nerf_field=nerf_field,
                classifier_model=classifier_model,
                fingertip_config=classifier_config.nerfdata_config.fingertip_config,
                X_N_Oy=X_N_Oy,
            )

        print("\n" + "=" * 80)
        print("Step 6: Optimize grasps")
        print("=" * 80 + "\n")
        if cfg.optimizer_type == "sgd":
            optimizer = SGDOptimizerConfig(
                num_grasps=cfg.num_grasps,
                num_steps=cfg.num_steps,
                # finger_lr=1e-3,
                finger_lr=0,
                # grasp_dir_lr=1e-4,
                grasp_dir_lr=0,
                wrist_lr=1e-3,
            )
        elif cfg.optimizer_type == "cem":
            optimizer = CEMOptimizerConfig(
                num_grasps=cfg.num_grasps,
                num_steps=cfg.num_steps,
                num_samples=cfg.num_grasps,
                num_elite=2,
                min_cov_std=1e-2,
            )
        elif cfg.optimizer_type == "random-sampling":
            optimizer = RandomSamplingConfig(
                num_grasps=cfg.num_grasps,
                num_steps=cfg.num_steps,
            )
        else:
            raise ValueError(f"Invalid cfg.optimizer_type: {cfg.optimizer_type}")

        optimized_grasp_config_dict = get_optimized_grasps(
            cfg=OptimizationConfig(
                use_rich=False,  # Not used because causes issues with logging
                # init_grasp_config_dict_path=cfg.init_grasp_config_dict_path,
                init_grasp_config_dict_path=NEW_init_grasp_config_dict_path,
                grasp_metric=GraspMetricConfig(
                    nerf_checkpoint_path=nerf_config,
                    classifier_config_path=cfg.classifier_config_path,
                    X_N_Oy=X_N_Oy,
                ),  # This is not used because we are passing in a grasp_metric
                optimizer=optimizer,
                output_path=pathlib.Path(
                    cfg.output_folder
                    / "optimized_grasp_config_dicts"
                    / f"{cfg.object_code_and_scale_str}.npy"
                ),
                random_seed=cfg.random_seed,
                n_random_rotations_per_grasp=cfg.n_random_rotations_per_grasp,
                eval_batch_size=cfg.eval_batch_size,
                wandb=None,
            ),
            grasp_metric=grasp_metric,
        )

    print("\n" + "=" * 80)
    print("Step 7: Convert optimized grasps to joint angles")
    print("=" * 80 + "\n")
    X_Oy_Hs, q_algr_pres, q_algr_posts, q_algr_extra_open, sorted_losses = (
        get_sorted_grasps_from_dict(
            optimized_grasp_config_dict=optimized_grasp_config_dict,
            dist_move_finger=0.06,
            dist_move_finger_backward=-0.03,
            error_if_no_loss=True,
            check=False,
            print_best=False,
        )
    )

    MODE = "EXTRA_OPEN"  # TODO: Compare these
    print("!" * 80)
    print(f"MODE: {MODE}")
    print("!" * 80 + "\n")
    if MODE == "DEFAULT":
        q_algr_pres = q_algr_pres
    elif MODE == "EXTRA_OPEN":
        q_algr_pres = q_algr_extra_open
    elif MODE == "JOINTS_OPEN":
        DELTA = 0.1
        q_algr_pres[:, 1] -= DELTA
        q_algr_pres[:, 2] -= DELTA
        q_algr_pres[:, 3] -= DELTA

        q_algr_pres[:, 5] -= DELTA
        q_algr_pres[:, 6] -= DELTA
        q_algr_pres[:, 7] -= DELTA

        q_algr_pres[:, 9] -= DELTA
        q_algr_pres[:, 10] -= DELTA
        q_algr_pres[:, 11] -= DELTA
    else:
        raise ValueError(f"Invalid MODE: {MODE}")
    q_algr_pres = clamp_in_limits(q_algr_pres)

    num_grasps = X_Oy_Hs.shape[0]
    assert X_Oy_Hs.shape == (num_grasps, 4, 4)
    assert q_algr_pres.shape == (num_grasps, 16)
    assert q_algr_posts.shape == (num_grasps, 16)

    q_algr_pres_is_in_limits = is_in_limits(q_algr_pres)
    assert q_algr_pres_is_in_limits.shape == (num_grasps,)
    pass_idxs = set(np.where(q_algr_pres_is_in_limits)[0])
    print(
        f"Number of grasps in limits: {len(pass_idxs)} / {num_grasps} ({len(pass_idxs) / num_grasps * 100:.2f}%)"
    )
    print(f"pass_idxs: {pass_idxs}")

    X_W_Hs = np.stack([X_W_N @ X_N_Oy @ X_Oy_Hs[i] for i in range(num_grasps)], axis=0)
    assert X_W_Hs.shape == (num_grasps, 4, 4)

    return (
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        sorted_losses,
    )


def run_nerf_sampler_pipeline(
    nerf_model: Model,
    cfg: PipelineConfig,
    q_fr3: np.ndarray,
    q_algr: np.ndarray,
    ckpt_path: str | pathlib.Path,
    optimize: bool,
    sample_grasps_multiplier: int,
    robot_cfg: Optional[RobotConfig] = None,
    ik_solver: Optional[IKSolver] = None,
    ik_solver2: Optional[IKSolver] = None,
    motion_gen: Optional[MotionGen] = None,
    motion_gen_config: Optional[MotionGenConfig] = None,
    lift_robot_cfg: Optional[RobotConfig] = None,
    lift_ik_solver: Optional[IKSolver] = None,
    lift_ik_solver2: Optional[IKSolver] = None,
    lift_motion_gen: Optional[MotionGen] = None,
    lift_motion_gen_config: Optional[MotionGenConfig] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[int], tuple, dict]:
    print(f"Creating a new experiment folder at {cfg.output_folder}")
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    sys.stdout = MultipleOutputs(
        stdout=False, stderr=True, filename=str(cfg.output_folder / "nerf_grasping.log")
    )

    start_time = time.time()
    (
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        sorted_losses,
    ) = compute_nerf_sampler_grasps(nerf_model=nerf_model, cfg=cfg, ckpt_path=ckpt_path, optimize=optimize, sample_grasps_multiplier=sample_grasps_multiplier)
    compute_grasps_time = time.time()
    print("@" * 80)
    print(f"Time to compute_grasps: {compute_grasps_time - start_time:.2f}s")
    print("@" * 80 + "\n")

    start_run_curobo = time.time()
    q_trajs, qd_trajs, T_trajs, success_idxs, DEBUG_TUPLE, log_dict = run_curobo(
        cfg=cfg,
        X_W_Hs=X_W_Hs,
        q_algr_pres=q_algr_pres,
        q_algr_posts=q_algr_posts,
        sorted_losses=sorted_losses,
        q_fr3=q_fr3,
        q_algr=q_algr,
        robot_cfg=robot_cfg,
        ik_solver=ik_solver,
        ik_solver2=ik_solver2,
        motion_gen=motion_gen,
        motion_gen_config=motion_gen_config,
        lift_robot_cfg=lift_robot_cfg,
        lift_ik_solver=lift_ik_solver,
        lift_ik_solver2=lift_ik_solver2,
        lift_motion_gen=lift_motion_gen,
        lift_motion_gen_config=lift_motion_gen_config,
    )
    curobo_time = time.time()
    print("@" * 80)
    print(f"Time to run_curobo: {curobo_time - start_run_curobo:.2f}s")
    print("@" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"Total time: {curobo_time - start_time:.2f}s")
    print("=" * 80 + "\n")

    pipeline_log_dict = {
        "X_W_Hs": X_W_Hs,
        "q_algr_pres": q_algr_pres,
        "q_algr_posts": q_algr_posts,
        "mesh_W": mesh_W,
        "X_N_Oy": X_N_Oy,
        "sorted_losses": sorted_losses,
        "q_trajs": q_trajs,
        "qd_trajs": qd_trajs,
        "T_trajs": T_trajs,
        "success_idxs": success_idxs,
        **log_dict,
    }

    # Print this in green
    print("+" * 80)
    BEST_IDX = success_idxs[0]
    print(
        f"\033[92mFINAL LOSS OF GRASP TO BE EXECUTED: {sorted_losses[BEST_IDX]:.5f} (idx: {BEST_IDX})\033[0m"
    )
    print("+" * 80 + "\n")

    return q_trajs, qd_trajs, T_trajs, success_idxs, DEBUG_TUPLE, pipeline_log_dict


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

    # Prepare nerf model
    if args.nerfdata_path is not None:
        start_time = time.time()
        nerf_checkpoints_folder = args.output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerfs_return_trainer.train_nerf(
            args=train_nerfs_return_trainer.Args(
                nerfdata_folder=args.nerfdata_path,
                nerfcheckpoints_folder=nerf_checkpoints_folder,
                max_num_iterations=args.max_num_iterations,
            )
        )
        nerf_model = nerf_trainer.pipeline.model
        nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
        end_time = time.time()
        print("@" * 80)
        print(f"Time to train_nerf: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    elif args.nerfcheckpoint_path is not None:
        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(
            args.nerfcheckpoint_path, test_mode="test"
        )  # Must be test mode for point cloud gen
        nerf_model = nerf_pipeline.model
        nerf_config = args.nerfcheckpoint_path
        end_time = time.time()
        print("@" * 80)
        print(f"Time to load_nerf_pipeline: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
        )
    args.nerf_config = nerf_config

    # Prepare curobo
    start_prepare_trajopt_batch = time.time()
    # HACK: Need to include a mesh into the world for the motion_gen warmup or else it will not prepare mesh buffers
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh.export("/tmp/DUMMY.obj")
    FAR_AWAY_OBJ_XYZ = (10.0, 0.0, 0.0)
    robot_cfg, ik_solver, ik_solver2, motion_gen, motion_gen_config = (
        prepare_trajopt_batch(
            n_grasps=args.num_grasps,
            collision_check_object=True,
            obj_filepath=pathlib.Path("/tmp/DUMMY.obj"),
            obj_xyz=FAR_AWAY_OBJ_XYZ,
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            collision_check_table=True,
            use_cuda_graph=True,
            collision_sphere_buffer=0.001,
            warmup=False,  # Warmup amortizes the cost of subsequent calls, but takes longer overall, no help in serial program
        )
    )
    (
        lift_robot_cfg,
        lift_ik_solver,
        lift_ik_solver2,
        lift_motion_gen,
        lift_motion_gen_config,
    ) = prepare_trajopt_batch(
        n_grasps=args.num_grasps,
        collision_check_object=True,
        obj_filepath=pathlib.Path("/tmp/DUMMY.obj"),
        obj_xyz=FAR_AWAY_OBJ_XYZ,
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
        use_cuda_graph=True,
        collision_sphere_buffer=0.001,
        warmup=False,  # Warmup amortizes the cost of subsequent calls, but takes longer overall, no help in serial program
    )
    end_prepare_trajopt_batch = time.time()
    print("@" * 80)
    print(
        f"Time to prepare_trajopt_batch: {end_prepare_trajopt_batch - start_prepare_trajopt_batch:.2f}s"
    )
    print("@" * 80 + "\n")


    qs, qds, T_trajs, success_idxs, DEBUG_TUPLE, log_dict = run_nerf_sampler_pipeline(
        nerf_model=nerf_model,
        cfg=args,
        q_fr3=DEFAULT_Q_FR3,
        q_algr=DEFAULT_Q_ALGR,
        ckpt_path="/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_NERF_SAMPLER_V1/ckpt_740900.pth",
        optimize=True,
        sample_grasps_multiplier=100,
        robot_cfg=robot_cfg,
        ik_solver=ik_solver,
        ik_solver2=ik_solver2,
        motion_gen=motion_gen,
        motion_gen_config=motion_gen_config,
        lift_robot_cfg=lift_robot_cfg,
        lift_ik_solver=lift_ik_solver,
        lift_ik_solver2=lift_ik_solver2,
        lift_motion_gen=lift_motion_gen,
        lift_motion_gen_config=lift_motion_gen_config,
    )

    print("Testing save_to_file and load_from_file")
    start_log_time = time.time()
    save_to_file(
        data=log_dict,
        filepath=args.output_folder / "log_dict.pkl",
    )
    end_log_time = time.time()
    print(f"Saving log_dict took {end_log_time - start_log_time:.2f}s")
    loaded_log_dict = load_from_file(args.output_folder / "log_dict.pkl")
    print(f"loaded_log_dict.keys(): {loaded_log_dict.keys()}")

    visualize(
        cfg=args,
        qs=qs,
        T_trajs=T_trajs,
        success_idxs=success_idxs,
        sorted_losses=log_dict["sorted_losses"],
        DEBUG_TUPLE=DEBUG_TUPLE,
    )


if __name__ == "__main__":
    main()
