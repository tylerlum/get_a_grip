import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import transforms3d
import trimesh
import tyro
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
)
from nerfstudio.pipelines.base_pipeline import Pipeline

from get_a_grip.grasp_motion_planning.utils.trajopt import (
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
)
from get_a_grip.grasp_motion_planning.utils.trajopt_batch import (
    compute_over_limit_factors,
    get_trajectories_from_result,
    prepare_trajopt_batch,
    solve_prepared_trajopt_batch,
)
from get_a_grip.grasp_motion_planning.utils.world import (
    get_world_cfg,
)
from get_a_grip.grasp_planning.config.nerf_evaluator_wrapper_config import (
    NerfEvaluatorWrapperConfig,
)
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    RandomSamplingConfig,
    SGDOptimizerConfig,
)
from get_a_grip.grasp_planning.nerf_conversions.nerf_to_mesh import nerf_to_mesh
from get_a_grip.grasp_planning.scripts.optimizer import get_optimized_grasps
from get_a_grip.grasp_planning.utils import (
    train_nerf_return_trainer,
)
from get_a_grip.grasp_planning.utils.joint_limit_utils import (
    clamp_in_limits_np,
    is_in_limits_np,
)
from get_a_grip.grasp_planning.utils.nerf_evaluator_optimizer_utils import (
    get_sorted_grasps_from_dict,
)
from get_a_grip.grasp_planning.utils.nerf_evaluator_wrapper import (
    NerfEvaluatorWrapper,
    load_nerf_evaluator,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)
from get_a_grip.model_training.utils.nerf_load_utils import load_nerf_pipeline
from get_a_grip.model_training.utils.nerf_utils import compute_centroid_from_nerf


class MultipleOutputs:
    def __init__(
        self, stdout: bool = True, stderr: bool = False, filename: Optional[str] = None
    ):
        # Avoid error:
        # UnicodeEncodeError: 'ascii' codec can't encode character '\u2601' in position 0: ordinal not in range(128)
        # *** You may need to add PYTHONIOENCODING=utf-8 to your environment ***
        self.stdout = sys.stdout if stdout else None
        self.stderr = sys.stderr if stderr else None
        self.file = (
            open(filename, "a", encoding="utf-8") if filename is not None else None
        )

    def write(self, message: str) -> None:
        if self.stdout is not None:
            self.stdout.write(message)
        if self.stderr is not None:
            self.stderr.write(message)
        if self.file is not None:
            self.file.write(message)

    def flush(self) -> None:
        if self.stdout is not None:
            self.stdout.flush()
        if self.stderr is not None:
            self.stderr.flush()
        if self.file is not None:
            self.file.flush()


@dataclass
class PipelineConfig:
    init_grasp_config_dict_path: pathlib.Path
    nerf_evaluator_config_path: pathlib.Path
    object_code: str = "unnamed_object"
    output_folder: pathlib.Path = pathlib.Path("experiments") / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    density_levelset_threshold: float = 15.0
    obj_is_z_up: bool = True
    lb_x: float = -0.2
    lb_y: float = -0.2
    lb_z: float = 0.0
    ub_x: float = 0.2
    ub_y: float = 0.2
    ub_z: float = 0.3
    nerf_frame_offset_x: float = 0.65
    visualize: bool = False
    optimizer_type: Literal["sgd", "random-sampling"] = "random-sampling"
    num_grasps: int = 32
    num_steps: int = 0
    random_seed: Optional[int] = None
    n_random_rotations_per_grasp: int = 5
    eval_batch_size: int = 32
    object_scale: float = 0.9999
    nerf_config: Optional[pathlib.Path] = None
    DEBUG_turn_off_object_collision: bool = False

    approach_time: float = 3.0
    stay_open_time: float = 0.2
    # stay_open_time: float = 10  # HACK: Stay open longer
    close_time: float = 0.5
    stay_closed_time: float = 0.2
    lift_time: float = 1.0

    def __post_init__(self) -> None:
        assert (
            self.init_grasp_config_dict_path.exists()
        ), f"{self.init_grasp_config_dict_path} does not exist"
        assert (
            self.init_grasp_config_dict_path.suffix == ".npy"
        ), f"{self.init_grasp_config_dict_path} does not have a .npy suffix"

        assert (
            self.nerf_evaluator_config_path.exists()
        ), f"{self.nerf_evaluator_config_path} does not exist"
        assert self.nerf_evaluator_config_path.suffix in [
            ".yml",
            ".yaml",
        ], f"{self.nerf_evaluator_config_path} does not have a .yml or .yaml suffix"

        if self.nerf_config is not None:
            assert self.nerf_config.exists(), f"{self.nerf_config} does not exist"
            assert (
                self.nerf_config.suffix == ".yml"
            ), f"{self.nerf_config} does not have a .yml suffix"

    @property
    def lb_N(self) -> np.ndarray:
        return np.array([self.lb_x, self.lb_y, self.lb_z])

    @property
    def ub_N(self) -> np.ndarray:
        return np.array([self.ub_x, self.ub_y, self.ub_z])

    @property
    def X_W_N(self) -> np.ndarray:
        return trimesh.transformations.translation_matrix(
            [self.nerf_frame_offset_x, 0, 0]
        )

    @property
    def X_O_Oy(self) -> np.ndarray:
        return (
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            if self.obj_is_z_up
            else np.eye(4)
        )

    @property
    def object_code_and_scale_str(self) -> str:
        return f"{self.object_code}_{self.object_scale:.4f}".replace(".", "_")


def transform_point(transform_matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    assert transform_matrix.shape == (4, 4), f"{transform_matrix.shape} is not (4, 4)"
    assert point.shape == (3,), f"{point.shape} is not (3,)"
    point = np.append(point, 1)
    return np.dot(transform_matrix, point)[:3]


def add_transform_matrix_traces(
    fig: go.Figure, transform_matrix: np.ndarray, length: float = 0.1
) -> None:
    assert transform_matrix.shape == (4, 4), f"{transform_matrix.shape} is not (4, 4)"
    origin = np.array([0, 0, 0])
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])

    origin_transformed = transform_point(transform_matrix, origin)
    x_axis_transformed = transform_point(transform_matrix, x_axis)
    y_axis_transformed = transform_point(transform_matrix, y_axis)
    z_axis_transformed = transform_point(transform_matrix, z_axis)

    for axis, color, name in zip(
        [x_axis_transformed, y_axis_transformed, z_axis_transformed],
        ["red", "green", "blue"],
        ["x", "y", "z"],
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[origin_transformed[0], axis[0]],
                y=[origin_transformed[1], axis[1]],
                z=[origin_transformed[2], axis[2]],
                mode="lines",
                line=dict(color=color, width=5),
                name=name,
            )
        )


def compute_grasps(
    nerf_pipeline: Pipeline,
    cfg: PipelineConfig,
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
    nerf_field = nerf_pipeline.model.field
    nerf_config = (
        cfg.nerf_config
        if cfg.nerf_config is not None
        else pathlib.Path("DUMMY_NERF_CONFIG/config.yml")
    )  # Dummy value to put in, not used because nerf_pipeline is passed in

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
    print("Step 5: Load grasp metric")
    print("=" * 80 + "\n")
    print(f"Loading nerf_evaluator config from {cfg.nerf_evaluator_config_path}")
    nerf_evaluator_config = tyro.extras.from_yaml(
        NerfEvaluatorModelConfig, cfg.nerf_evaluator_config_path.open()
    )

    nerf_evaluator_model = load_nerf_evaluator(
        nerf_evaluator_config=nerf_evaluator_config
    )
    nerf_evaluator_wrapper = NerfEvaluatorWrapper(
        nerf_field=nerf_field,
        nerf_evaluator_model=nerf_evaluator_model,
        fingertip_config=nerf_evaluator_config.nerfdata_config.fingertip_config,
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
            init_grasp_config_dict_path=cfg.init_grasp_config_dict_path,
            nerf_evaluator_wrapper=NerfEvaluatorWrapperConfig(
                nerf_config=nerf_config,
                nerf_evaluator_config_path=cfg.nerf_evaluator_config_path,
                X_N_Oy=X_N_Oy,
            ),  # This is not used because we are passing in a nerf_evaluator_wrapper
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
        nerf_evaluator_wrapper=nerf_evaluator_wrapper,
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
    q_algr_pres = clamp_in_limits_np(q_algr_pres)

    num_grasps = X_Oy_Hs.shape[0]
    assert X_Oy_Hs.shape == (num_grasps, 4, 4)
    assert q_algr_pres.shape == (num_grasps, 16)
    assert q_algr_posts.shape == (num_grasps, 16)

    q_algr_pres_is_in_limits = is_in_limits_np(q_algr_pres)
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


def run_curobo(
    cfg: PipelineConfig,
    X_W_Hs: np.ndarray,
    q_algr_pres: np.ndarray,
    q_algr_posts: np.ndarray,
    q_fr3: np.ndarray,
    q_algr: np.ndarray,
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
    sorted_losses: Optional[np.ndarray] = None,
    X_W_table: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[int], tuple, dict]:
    # Timing
    APPROACH_TIME = cfg.approach_time
    STAY_OPEN_TIME = cfg.stay_open_time
    CLOSE_TIME = cfg.close_time
    STAY_CLOSED_TIME = cfg.stay_closed_time
    LIFT_TIME = cfg.lift_time

    n_grasps = X_W_Hs.shape[0]
    assert X_W_Hs.shape == (n_grasps, 4, 4)
    assert q_algr_pres.shape == (n_grasps, 16)
    assert q_fr3.shape == (7,)
    assert q_algr.shape == (16,)

    # Adjust obj_xyz and quat_wxyz
    X_table_O = np.eye(4)
    X_table_O[:3, 3] = [cfg.nerf_frame_offset_x, 0.0, 0.0]

    if X_W_table is None:
        print("X_W_table is None, using identity matrix")
        X_W_table = np.eye(4)
    else:
        assert X_W_table.shape == (4, 4)
        print(f"X_W_table: {X_W_table}")

    X_W_O = X_W_table @ X_table_O
    obj_xyz = X_W_O[:3, 3]
    obj_quat_wxyz = transforms3d.quaternions.mat2quat(X_W_O[:3, :3])

    if (
        robot_cfg is None
        or ik_solver is None
        or ik_solver2 is None
        or motion_gen is None
        or motion_gen_config is None
    ):
        print("\n" + "=" * 80)
        print(f"robot_cfg is None: {robot_cfg is None}")
        print(f"ik_solver is None: {ik_solver is None}")
        print(f"ik_solver2 is None: {ik_solver2 is None}")
        print(f"motion_gen is None: {motion_gen is None}")
        print(f"motion_gen_config is None: {motion_gen_config is None}")
        print("=" * 80 + "\n")
        print(
            "Creating new robot, ik_solver, ik_solver2, motion_gen, motion_gen_config"
        )
        robot_cfg, ik_solver, ik_solver2, motion_gen, motion_gen_config = (
            prepare_trajopt_batch(
                n_grasps=n_grasps,
                collision_check_object=(
                    True if not cfg.DEBUG_turn_off_object_collision else False
                ),
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=obj_xyz,
                obj_quat_wxyz=obj_quat_wxyz,
                collision_check_table=True,
                use_cuda_graph=True,
                collision_sphere_buffer=0.001,
            )
        )
    if (
        lift_robot_cfg is None
        or lift_ik_solver is None
        or lift_ik_solver2 is None
        or lift_motion_gen is None
        or lift_motion_gen_config is None
    ):
        print("\n" + "=" * 80)
        print(f"lift_robot_cfg is None: {lift_robot_cfg is None}")
        print(f"lift_ik_solver is None: {lift_ik_solver is None}")
        print(f"lift_ik_solver2 is None: {lift_ik_solver2 is None}")
        print(f"lift_motion_gen is None: {lift_motion_gen is None}")
        print(f"lift_motion_gen_config is None: {lift_motion_gen_config is None}")
        print("=" * 80 + "\n")
        print(
            "Creating new lift_robot, lift_ik_solver, lift_ik_solver2, lift_motion_gen, lift_motion_gen_config"
        )
        (
            lift_robot_cfg,
            lift_ik_solver,
            lift_ik_solver2,
            lift_motion_gen,
            lift_motion_gen_config,
        ) = prepare_trajopt_batch(
            n_grasps=n_grasps,
            collision_check_object=(
                True if not cfg.DEBUG_turn_off_object_collision else False
            ),
            obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
            obj_xyz=obj_xyz,
            obj_quat_wxyz=obj_quat_wxyz,
            collision_check_table=True,
            use_cuda_graph=True,
            collision_sphere_buffer=0.001,
        )

    print("\n" + "=" * 80)
    print("Step 9: Solve motion gen for each grasp")
    print("=" * 80 + "\n")
    object_world_cfg = get_world_cfg(
        collision_check_object=(
            True if not cfg.DEBUG_turn_off_object_collision else False
        ),
        obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        collision_check_table=True,
        obj_name="NERF_OBJECT",  # HACK: MUST BE DIFFERENT FROM EXISTING OBJECT NAME "object" OR ELSE COLLISION DETECTION WILL FAIL
    )
    ik_solver.update_world(object_world_cfg)
    ik_solver2.update_world(object_world_cfg)
    motion_gen.update_world(object_world_cfg)
    motion_gen_result, ik_result, ik_result2 = solve_prepared_trajopt_batch(
        X_W_Hs=X_W_Hs,
        q_algrs=q_algr_pres,
        robot_cfg=robot_cfg,
        ik_solver=ik_solver,
        ik_solver2=ik_solver2,
        motion_gen=motion_gen,
        motion_gen_config=motion_gen_config,
        q_fr3_starts=q_fr3[None, ...].repeat(n_grasps, axis=0),
        q_algr_starts=q_algr[None, ...].repeat(n_grasps, axis=0),
        enable_graph=True,
        enable_opt=False,
        timeout=2.0,
    )

    motion_gen_success_idxs = (
        motion_gen_result.success.flatten().nonzero().flatten().tolist()
    )
    ik_success_idxs = ik_result.success.flatten().nonzero().flatten().tolist()
    ik_success_idxs2 = ik_result2.success.flatten().nonzero().flatten().tolist()

    print("\n" + "=" * 80)
    print(
        "Motion generation without trajectory optimization complete, printing results"
    )
    print("=" * 80 + "\n")
    print(
        f"motion_gen_success_idxs: {motion_gen_success_idxs} ({len(motion_gen_success_idxs)} / {n_grasps} = {len(motion_gen_success_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"ik_success_idxs: {ik_success_idxs} ({len(ik_success_idxs)} / {n_grasps} = {len(ik_success_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"ik_success_idxs2: {ik_success_idxs2} ({len(ik_success_idxs2)} / {n_grasps} = {len(ik_success_idxs2) / n_grasps * 100:.2f}%)"
    )

    approach_qs, approach_qds, dts = get_trajectories_from_result(
        result=motion_gen_result,
        desired_trajectory_time=APPROACH_TIME,
    )
    nonzero_q_idxs = [
        i
        for i, approach_q in enumerate(approach_qs)
        if np.absolute(approach_q).sum() > 1e-2
    ]
    overall_success_idxs = sorted(
        list(
            set(motion_gen_success_idxs)
            .intersection(set(ik_success_idxs).intersection(set(ik_success_idxs2)))
            .intersection(set(nonzero_q_idxs))
        )
    )  # All must be successful or else it may be successful for the wrong trajectory

    print(
        f"nonzero_q_idxs: {nonzero_q_idxs} ({len(nonzero_q_idxs)} / {n_grasps} = {len(nonzero_q_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"overall_success_idxs: {overall_success_idxs} ({len(overall_success_idxs)} / {n_grasps} = {len(overall_success_idxs) / n_grasps * 100:.2f}%)"
    )

    # Fix issue with going over limit
    over_limit_factors_approach_qds = compute_over_limit_factors(
        qds=approach_qds, dts=dts
    )
    approach_qds = [
        approach_qd / over_limit_factor
        for approach_qd, over_limit_factor in zip(
            approach_qds, over_limit_factors_approach_qds
        )
    ]
    dts = [
        dt * over_limit_factor
        for dt, over_limit_factor in zip(dts, over_limit_factors_approach_qds)
    ]

    print("\n" + "=" * 80)
    print("Step 10: Add closing motion")
    print("=" * 80 + "\n")
    closing_qs, closing_qds = [], []
    for i, (approach_q, approach_qd, dt) in enumerate(
        zip(approach_qs, approach_qds, dts)
    ):
        # Keep arm joints same, change hand joints
        open_q = approach_q[-1]
        close_q = np.concatenate([open_q[:7], q_algr_posts[i]])

        # Stay open
        N_STAY_OPEN_STEPS = int(STAY_OPEN_TIME / dt)
        interpolated_qs0 = interpolate(start=open_q, end=open_q, N=N_STAY_OPEN_STEPS)
        assert interpolated_qs0.shape == (N_STAY_OPEN_STEPS, 23)

        # Close
        N_CLOSE_STEPS = int(CLOSE_TIME / dt)
        interpolated_qs1 = interpolate(start=open_q, end=close_q, N=N_CLOSE_STEPS)
        assert interpolated_qs1.shape == (N_CLOSE_STEPS, 23)

        # Stay closed
        N_STAY_CLOSED_STEPS = int(STAY_CLOSED_TIME / dt)
        interpolated_qs2 = interpolate(
            start=close_q, end=close_q, N=N_STAY_CLOSED_STEPS
        )
        assert interpolated_qs2.shape == (N_STAY_CLOSED_STEPS, 23)

        closing_q = np.concatenate(
            [interpolated_qs0, interpolated_qs1, interpolated_qs2], axis=0
        )
        assert closing_q.shape == (
            N_STAY_OPEN_STEPS + N_CLOSE_STEPS + N_STAY_CLOSED_STEPS,
            23,
        )

        closing_qd = np.diff(closing_q, axis=0) / dt
        closing_qd = np.concatenate([closing_qd, closing_qd[-1:]], axis=0)

        closing_qs.append(closing_q)
        closing_qds.append(closing_qd)

    print("\n" + "=" * 80)
    print("Step 11: Add lifting motion")
    print("=" * 80 + "\n")
    # Using same approach_qs found from motion gen to ensure they are not starting in collision
    # Not using closing_qs because they potentially could have issues?
    q_start_lifts = np.array([approach_q[-1] for approach_q in approach_qs])
    assert q_start_lifts.shape == (n_grasps, 23)

    X_W_H_lifts = X_W_Hs.copy()
    X_W_H_lifts[:, :3, 3] = np.array([0.440870285, 0.0, 0.563780367])

    # HACK: If motion_gen above fails, then it leaves q as all 0s, which causes next step to fail
    #       So we populate those with another valid one
    assert len(overall_success_idxs) > 0, "overall_success_idxs is empty"
    valid_idx = overall_success_idxs[0]
    for i in range(n_grasps):
        if i in overall_success_idxs:
            continue
        q_start_lifts[i] = q_start_lifts[valid_idx]
        X_W_H_lifts[i] = X_W_H_lifts[valid_idx]

    # Update world to remove object collision check
    no_object_world_cfg = get_world_cfg(
        collision_check_object=False,
        obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        collision_check_table=True,
        obj_name="NO_OBJECT",  # HACK: MUST BE DIFFERENT FROM EXISTING OBJECT NAME "object" OR ELSE COLLISION DETECTION WILL FAIL
    )
    lift_ik_solver.update_world(no_object_world_cfg)
    lift_ik_solver2.update_world(no_object_world_cfg)
    lift_motion_gen.update_world(no_object_world_cfg)
    lift_motion_gen_result, lift_ik_result, lift_ik_result2 = (
        solve_prepared_trajopt_batch(
            X_W_Hs=X_W_H_lifts,
            q_algrs=q_algr_pres,
            robot_cfg=lift_robot_cfg,
            ik_solver=lift_ik_solver,
            ik_solver2=lift_ik_solver2,
            motion_gen=lift_motion_gen,
            motion_gen_config=lift_motion_gen_config,
            q_fr3_starts=q_start_lifts[:, :7],
            q_algr_starts=q_start_lifts[:, 7:],
            enable_graph=True,
            enable_opt=False,
            timeout=1.0,
        )
    )

    lift_motion_gen_success_idxs = (
        lift_motion_gen_result.success.flatten().nonzero().flatten().tolist()
    )
    lift_ik_success_idxs = lift_ik_result.success.flatten().nonzero().flatten().tolist()
    lift_ik_success_idxs2 = (
        lift_ik_result2.success.flatten().nonzero().flatten().tolist()
    )
    print(
        f"lift_motion_gen_success_idxs: {lift_motion_gen_success_idxs} ({len(lift_motion_gen_success_idxs)} / {n_grasps} = {len(lift_motion_gen_success_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"lift_ik_success_idxs: {lift_ik_success_idxs} ({len(lift_ik_success_idxs)} / {n_grasps} = {len(lift_ik_success_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"lift_ik_success_idxs2: {lift_ik_success_idxs2} ({len(lift_ik_success_idxs2)} / {n_grasps} = {len(lift_ik_success_idxs2) / n_grasps * 100:.2f}%)"
    )

    raw_lift_qs, raw_lift_qds, raw_lift_dts = get_trajectories_from_result(
        result=lift_motion_gen_result, desired_trajectory_time=LIFT_TIME
    )
    lift_nonzero_q_idxs = [
        i
        for i, raw_lift_q in enumerate(raw_lift_qs)
        if np.absolute(raw_lift_q).sum() > 1e-2
    ]
    lift_overall_success_idxs = sorted(
        list(
            set(lift_motion_gen_success_idxs)
            .intersection(
                set(lift_ik_success_idxs).intersection(set(lift_ik_success_idxs2))
            )
            .intersection(set(lift_nonzero_q_idxs))
        )
    )  # All must be successful or else it may be successful for the wrong trajectory
    print(
        f"lift_nonzero_q_idxs: {lift_nonzero_q_idxs} ({len(lift_nonzero_q_idxs)} / {n_grasps} = {len(lift_nonzero_q_idxs) / n_grasps * 100:.2f}%"
    )
    print(
        f"lift_overall_success_idxs: {lift_overall_success_idxs} ({len(lift_overall_success_idxs)} / {n_grasps} = {len(lift_overall_success_idxs) / n_grasps * 100:.2f}%)"
    )

    # Need to adjust raw_lift_qs, raw_lift_qds, raw_lift_dts to match dts from trajopt
    new_raw_lift_qs, new_raw_lift_qds, new_raw_lift_dts = [], [], []
    for i, (raw_lift_q, raw_lift_qd, raw_lift_dt, dt) in enumerate(
        zip(raw_lift_qs, raw_lift_qds, raw_lift_dts, dts)
    ):
        n_timepoints = raw_lift_q.shape[0]
        assert raw_lift_q.shape == (n_timepoints, 23)
        total_time = n_timepoints * raw_lift_dt

        # Interpolate with new timepoints
        n_new_timepoints = int(total_time / dt)
        new_raw_lift_q = np.zeros((n_new_timepoints, 23))
        for j in range(23):
            new_raw_lift_q[:, j] = np.interp(
                np.arange(n_new_timepoints) * dt,
                np.linspace(0, total_time, n_timepoints),
                raw_lift_q[:, j],
            )

        new_raw_lift_qd = np.diff(new_raw_lift_q, axis=0) / dt
        new_raw_lift_qd = np.concatenate(
            [new_raw_lift_qd, new_raw_lift_qd[-1:]], axis=0
        )

        new_raw_lift_qs.append(new_raw_lift_q)
        new_raw_lift_qds.append(new_raw_lift_qd)
        new_raw_lift_dts.append(dt)

    raw_lift_qs, raw_lift_qds, raw_lift_dts = (
        new_raw_lift_qs,
        new_raw_lift_qds,
        new_raw_lift_dts,
    )

    # Handle exceeding joint limits
    over_limit_factors_raw_lift_qds = compute_over_limit_factors(
        qds=raw_lift_qds, dts=raw_lift_dts
    )
    new2_raw_lift_qs, new2_raw_lift_qds = [], []
    for i, (raw_lift_q, raw_lift_qd, raw_lift_dt, over_limit_factor) in enumerate(
        zip(raw_lift_qs, raw_lift_qds, raw_lift_dts, over_limit_factors_raw_lift_qds)
    ):
        assert over_limit_factor >= 1.0
        if over_limit_factor > 1.0:
            print(f"Rescaling raw_lift_qs by {over_limit_factor} for grasp {i}")
            n_timepoints = raw_lift_q.shape[0]
            assert raw_lift_q.shape == (n_timepoints, 23)

            previous_total_time = n_timepoints * raw_lift_dt
            new2_total_time = previous_total_time * over_limit_factor
            new2_n_timepoints = int(new2_total_time / raw_lift_dt)

            # Interpolate with new timepoints
            new2_raw_lift_q = np.zeros((new2_n_timepoints, 23))
            for j in range(23):
                new2_raw_lift_q[:, j] = np.interp(
                    np.arange(new2_n_timepoints) * raw_lift_dt,
                    np.linspace(0, new2_total_time, n_timepoints),
                    raw_lift_q[:, j],
                )

            new2_raw_lift_qd = np.diff(new2_raw_lift_q, axis=0) / raw_lift_dt
            new2_raw_lift_qd = np.concatenate(
                [new2_raw_lift_qd, new2_raw_lift_qd[-1:]], axis=0
            )

            new2_raw_lift_qs.append(new2_raw_lift_q)
            new2_raw_lift_qds.append(new2_raw_lift_qd)
        else:
            new2_raw_lift_qs.append(raw_lift_q)
            new2_raw_lift_qds.append(raw_lift_qd)
    raw_lift_qs, raw_lift_qds = new2_raw_lift_qs, new2_raw_lift_qds

    final_success_idxs = sorted(
        list(set(overall_success_idxs).intersection(set(lift_overall_success_idxs)))
    )

    print(
        f"final_success_idxs: {final_success_idxs} ({len(final_success_idxs)} / {n_grasps} = {len(final_success_idxs) / n_grasps * 100:.2f}%)"
    )

    # Adjust the lift qs to have the same hand position as the closing qs
    # We only want the arm position of the lift qs
    adjusted_lift_qs, adjusted_lift_qds = [], []
    for i, (
        closing_q,
        closing_qd,
        dt,
        raw_lift_q,
        raw_lift_qd,
        raw_lift_dt,
    ) in enumerate(
        zip(closing_qs, closing_qds, dts, raw_lift_qs, raw_lift_qds, raw_lift_dts)
    ):
        # TODO: Figure out how to handle if lift_qs has different dt, only a problem if set enable_opt=True
        assert dt == raw_lift_dt, f"dt: {dt}, lift_dt: {raw_lift_dt}"

        # Only want the arm position of the lift closing_q (keep same hand position as before)
        adjusted_lift_q = raw_lift_q.copy()
        last_closing_q = closing_q[-1]
        adjusted_lift_q[:, 7:] = last_closing_q[None, 7:]

        adjusted_lift_qd = raw_lift_qd.copy()
        adjusted_lift_qd[:, 7:] = 0.0

        adjusted_lift_qs.append(adjusted_lift_q)
        adjusted_lift_qds.append(adjusted_lift_qd)

    print("\n" + "=" * 80)
    print("Step 12: Aggregate qs and qds")
    print("=" * 80 + "\n")
    q_trajs, qd_trajs = [], []
    for approach_q, approach_qd, closing_q, closing_qd, lift_q, lift_qd, dt in zip(
        approach_qs,
        approach_qds,
        closing_qs,
        closing_qds,
        adjusted_lift_qs,
        adjusted_lift_qds,
        dts,
    ):
        q_traj = np.concatenate([approach_q, closing_q, lift_q], axis=0)
        qd_traj = np.diff(q_traj, axis=0) / dt
        qd_traj = np.concatenate([qd_traj, qd_traj[-1:]], axis=0)
        q_trajs.append(q_traj)
        qd_trajs.append(qd_traj)

    print("\n" + "=" * 80)
    print("Step 13: Compute T_trajs")
    print("=" * 80 + "\n")
    T_trajs = []
    for q_traj, dt in zip(q_trajs, dts):
        n_timesteps = q_traj.shape[0]
        T_trajs.append(n_timesteps * dt)

    over_limit_factors_qd_trajs = compute_over_limit_factors(
        qds=qd_trajs, dts=raw_lift_dts
    )
    no_crazy_jumps_idxs = [
        i
        for i, over_limit_factor in enumerate(over_limit_factors_qd_trajs)
        if over_limit_factor
        <= 2.0  # Actually just 1.0, but higher to be safe against numerical errors and other weirdness
    ]
    print(
        f"no_crazy_jumps_idxs: {no_crazy_jumps_idxs} ({len(no_crazy_jumps_idxs)} / {n_grasps} = {len(no_crazy_jumps_idxs) / n_grasps * 100:.2f}%)"
    )
    real_final_success_idxs = sorted(
        list(set(final_success_idxs).intersection(set(no_crazy_jumps_idxs)))
    )
    print("\n" + "~" * 80)
    print(
        f"real_final_success_idxs: {real_final_success_idxs} ({len(real_final_success_idxs)} / {n_grasps} = {len(real_final_success_idxs) / n_grasps * 100:.2f}%)"
    )
    if sorted_losses is not None:
        assert sorted_losses.shape == (n_grasps,)
        print(f"sorted_losses = {sorted_losses}")
        print(
            f"sorted_losses of successful grasps: {[sorted_losses[i] for i in real_final_success_idxs]}"
        )
    print("~" * 80 + "\n")

    DEBUG_TUPLE = (
        motion_gen_result,
        ik_result,
        ik_result2,
        lift_motion_gen_result,
        lift_ik_result,
        lift_ik_result2,
    )

    log_dict = {
        "approach_qs": approach_qs,
        "approach_qds": approach_qds,
        "dts": dts,
        "closing_qs": closing_qs,
        "closing_qds": closing_qds,
        "raw_lift_qs": raw_lift_qs,
        "raw_lift_qds": raw_lift_qds,
        "adjusted_lift_qs": adjusted_lift_qs,
        "adjusted_lift_qds": adjusted_lift_qds,
        "motion_gen_success_idxs": motion_gen_success_idxs,
        "ik_success_idxs": ik_success_idxs,
        "ik_success_idxs2": ik_success_idxs2,
        "overall_success_idxs": overall_success_idxs,
        "lift_motion_gen_success_idxs": lift_motion_gen_success_idxs,
        "lift_ik_success_idxs": lift_ik_success_idxs,
        "lift_ik_success_idxs2": lift_ik_success_idxs2,
        "lift_overall_success_idxs": lift_overall_success_idxs,
        "final_success_idxs": final_success_idxs,
        "no_crazy_jumps_idxs": no_crazy_jumps_idxs,
        "real_final_success_idxs": real_final_success_idxs,
        "over_limit_factors_qd_trajs": over_limit_factors_qd_trajs,
        "over_limit_factors_approach_qds": over_limit_factors_approach_qds,
        "over_limit_factors_raw_lift_qds": over_limit_factors_raw_lift_qds,
        "q_start_lifts": q_start_lifts,
    }
    return q_trajs, qd_trajs, T_trajs, real_final_success_idxs, DEBUG_TUPLE, log_dict


def run_pipeline(
    nerf_pipeline: Pipeline,
    cfg: PipelineConfig,
    q_fr3: np.ndarray,
    q_algr: np.ndarray,
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
    X_W_table: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[int], tuple, dict]:
    print(f"Creating a new experiment folder at {cfg.output_folder}")
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    sys.stdout = MultipleOutputs(
        stdout=False, stderr=True, filename=str(cfg.output_folder / "get_a_grip.log")
    )

    start_time = time.time()
    (
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        sorted_losses,
    ) = compute_grasps(nerf_pipeline=nerf_pipeline, cfg=cfg)
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
        X_W_table=X_W_table,
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


def visualize(
    cfg: PipelineConfig,
    qs: List[np.ndarray],
    T_trajs: List[float],
    success_idxs: List[int],
    sorted_losses: np.ndarray,
    DEBUG_TUPLE: tuple,
) -> None:
    # Visualize
    print("\n" + "=" * 80)
    print("Visualizing")
    print("=" * 80 + "\n")
    from get_a_grip.grasp_motion_planning.utils.ik import (
        max_penetration_from_q,
        max_penetration_from_qs,
    )
    from get_a_grip.grasp_motion_planning.utils.visualizer import (
        animate_robot,
        create_urdf,
        draw_collision_spheres_default_config,
        remove_collision_spheres_default_config,
        set_robot_state,
        start_visualizer,
    )

    OBJECT_URDF_PATH = create_urdf(obj_path=pathlib.Path("/tmp/mesh_viz_object.obj"))
    pb_robot = start_visualizer(
        object_urdf_path=OBJECT_URDF_PATH,
        obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    draw_collision_spheres_default_config(pb_robot)
    time.sleep(1.0)

    if len(success_idxs) == 0:
        print("WARNING: No successful trajectories")

    TRAJ_IDX = success_idxs[0] if len(success_idxs) > 0 else 0

    dts = []
    for q, T_traj in zip(qs, T_trajs):
        n_timesteps = q.shape[0]
        dt = T_traj / n_timesteps
        dts.append(dt)

    remove_collision_spheres_default_config()
    q, dt = qs[TRAJ_IDX], dts[TRAJ_IDX]
    print(f"Visualizing trajectory {TRAJ_IDX}")
    animate_robot(robot=pb_robot, qs=q, dt=dt)

    while True:
        input_options = "\n".join(
            [
                "=====================",
                "OPTIONS",
                "b for breakpoint",
                "v to visualize traj",
                "d to print collision distance",
                "i to move hand to exact X_W_H and q_algr_pre IK solution",
                "n to go to next traj",
                "p to go to prev traj",
                "c to draw collision spheres",
                "r to remove collision spheres",
                "q to quit",
                f"success_idxs = {success_idxs}",
                f"sorted_losses = {np.round([sorted_losses[i] for i in success_idxs], 2)}",
                "=====================",
            ]
        )
        x = input("\n" + input_options + "\n\n")
        if x == "b":
            print("Breakpoint")
            breakpoint()
        elif x == "v":
            q, dt = qs[TRAJ_IDX], dts[TRAJ_IDX]
            print(f"Visualizing trajectory {TRAJ_IDX}")
            animate_robot(robot=pb_robot, qs=q, dt=dt)
        elif x == "d":
            print(
                "WARNING: This doesn't make sense when we include the full trajectory of grasping"
            )

            q, dt = qs[TRAJ_IDX], dts[TRAJ_IDX]
            print(f"For trajectory {TRAJ_IDX}")
            d_world, d_self = max_penetration_from_qs(
                qs=q,
                collision_activation_distance=0.0,
                include_object=True,
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                include_table=True,
            )
            print(f"np.max(d_world): {np.max(d_world)}")
            print(f"np.max(d_self): {np.max(d_self)}")
        elif x == "i":
            print(
                f"Moving hand to exact X_W_H and q_algr_pre of trajectory {TRAJ_IDX} with IK collision check"
            )
            ik_result2 = DEBUG_TUPLE[2]  # BRITTLE
            ik_q = ik_result2.solution[TRAJ_IDX].flatten().detach().cpu().numpy()
            assert ik_q.shape == (23,)
            set_robot_state(robot=pb_robot, q=ik_q)
            d_world, d_self = max_penetration_from_q(
                q=ik_q,
                include_object=True,
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                include_table=True,
            )
            print(f"np.max(d_world): {np.max(d_world)}")
            print(f"np.max(d_self): {np.max(d_self)}")
        elif x == "n":
            TRAJ_IDX += 1
            if TRAJ_IDX >= len(qs):
                TRAJ_IDX = 0
            print(f"Updated to trajectory {TRAJ_IDX}")
        elif x == "p":
            TRAJ_IDX -= 1
            if TRAJ_IDX < 0:
                TRAJ_IDX = len(qs) - 1
            print(f"Updated to trajectory {TRAJ_IDX}")
        elif x == "c":
            print("Drawing collision spheres")
            draw_collision_spheres_default_config(robot=pb_robot)
        elif x == "r":
            print("Removing collision spheres")
            remove_collision_spheres_default_config()
        elif x == "q":
            print("Quitting")
            break
        else:
            print(f"Invalid input: {x}")

    breakpoint()


def interpolate(start: np.ndarray, end: np.ndarray, N: int) -> np.ndarray:
    d = start.shape[0]
    assert start.shape == end.shape == (d,)
    interpolated = np.zeros((N, d))
    for i in range(d):
        interpolated[:, i] = np.linspace(start[i], end[i], N)
    return interpolated


def save_to_file(data: dict, filepath: pathlib.Path) -> None:
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_from_file(filepath: pathlib.Path) -> dict:
    import pickle

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


@dataclass
class CommandlineArgs(PipelineConfig):
    nerfdata_path: Optional[pathlib.Path] = None
    nerf_config: Optional[pathlib.Path] = None
    max_num_iterations: int = 400

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


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

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
        nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
        end_time = time.time()
        print("@" * 80)
        print(f"Time to train_nerf: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    elif args.nerf_config is not None:
        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(args.nerf_config)
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

    # Prepare curobo
    start_prepare_trajopt_batch = time.time()
    # HACK: Need to include a mesh into the world for the motion_gen warmup or else it will not prepare mesh buffers
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh.export("/tmp/DUMMY.obj")
    FAR_AWAY_OBJ_XYZ = (10.0, 0.0, 0.0)
    robot_cfg, ik_solver, ik_solver2, motion_gen, motion_gen_config = (
        prepare_trajopt_batch(
            n_grasps=args.num_grasps,
            collision_check_object=(
                True if not args.DEBUG_turn_off_object_collision else False
            ),
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
        collision_check_object=(
            True if not args.DEBUG_turn_off_object_collision else False
        ),
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

    qs, qds, T_trajs, success_idxs, DEBUG_TUPLE, log_dict = run_pipeline(
        nerf_pipeline=nerf_pipeline,
        cfg=args,
        q_fr3=DEFAULT_Q_FR3,
        q_algr=DEFAULT_Q_ALGR,
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
