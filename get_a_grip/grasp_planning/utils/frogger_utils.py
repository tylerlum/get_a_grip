import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import pytorch_kinematics as pk
import torch
import trimesh
import tyro
from frogger import ROOT
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robot_core import RobotModel
from frogger.robots.robots import AlgrModelConfig, FR3AlgrZed2iModelConfig
from frogger.sampling import HeuristicAlgrICSampler, HeuristicFR3AlgrICSampler
from frogger.solvers import FroggerConfig
from pydrake.math import RigidTransform, RotationMatrix
from tqdm import tqdm

from get_a_grip.utils.parse_object_code_and_scale import (
    object_code_and_scale_to_str,
)
from get_a_grip.utils.point_utils import transform_points


@dataclass
class FroggerArgs:
    obj_filepath: pathlib.Path
    obj_scale: float
    obj_code: str
    obj_is_yup: bool = True
    num_grasps: int = 3
    output_grasp_config_dicts_folder: pathlib.Path = pathlib.Path(
        "./output_grasp_config_dicts"
    )
    visualize: bool = False
    grasp_idx_to_visualize: int = 1
    max_time: float = 60.0

    @property
    def object_code_and_scale_str(self) -> str:
        return object_code_and_scale_to_str(
            object_code=self.obj_code, object_scale=self.obj_scale
        )


@dataclass
class RobotConstants:
    robot_model_path: pathlib.Path = (
        pathlib.Path(ROOT) / "models/fr3_algr_zed2i/fr3_algr_zed2i.urdf"
    )
    wrist_body_name: str = "algr_rh_palm"
    fingertip_body_names: List[str] = field(
        default_factory=lambda: [
            "algr_rh_if_ds_tip",
            "algr_rh_mf_ds_tip",
            "algr_rh_rf_ds_tip",
            "algr_rh_th_ds_tip",
        ]
    )


def create_mesh(obj_filepath: pathlib.Path, obj_scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_filepath)
    mesh.apply_transform(trimesh.transformations.scale_matrix(obj_scale))
    return mesh


def compute_X_W_O(mesh: trimesh.Trimesh, obj_is_yup: bool) -> np.ndarray:
    bounds = mesh.bounds
    X_W_O = np.eye(4)
    # # 0.65 is to keep object away from robot base

    if obj_is_yup:
        min_y_O = bounds[0, -2]
        X_W_O[:3, 3] = np.array([0.65, 0.0, -min_y_O])
        X_W_O[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[
            :3, :3
        ]
    else:
        min_z_O = bounds[0, -1]
        X_W_O[:3, 3] = np.array([0.65, 0.0, -min_z_O])
        X_W_O[:3, :3] = np.eye(3)
    return X_W_O


def create_frogger_mesh_object(
    mesh: trimesh.Trimesh, obj_name: str, X_W_O: np.ndarray
) -> MeshObject:
    return MeshObjectConfig(
        X_WO=RigidTransform(
            RotationMatrix(X_W_O[:3, :3]),
            X_W_O[:3, 3],
        ),
        mesh=mesh,
        name=obj_name,
        clean=False,
    ).create()


def create_model(
    mesh_object: MeshObject,
    custom_coll_callback: Optional[Callable[[RobotModel, str, str], float]] = None,
    viz: bool = False,
) -> RobotModel:
    USE_FLOATING_HAND = False
    if USE_FLOATING_HAND:
        return AlgrModelConfig(
            obj=mesh_object,
            ns=4,
            mu=0.7,
            d_min=0.005,
            d_pen=0.005,
            custom_coll_callback=custom_coll_callback,
            viz=viz,
        ).create()
    else:
        return FR3AlgrZed2iModelConfig(
            obj=mesh_object,
            ns=4,
            mu=0.7,
            d_min=0.005,
            d_pen=0.005,
            custom_coll_callback=custom_coll_callback,
            viz=viz,
        ).create()


def zup_mesh_to_q_array(
    mesh_object: MeshObject,
    num_grasps: int,
    custom_coll_callback: Optional[Callable[[RobotModel, str, str], float]] = None,
    max_time: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # loading model
    model = create_model(
        mesh_object=mesh_object, custom_coll_callback=custom_coll_callback, viz=False
    )

    USE_FLOATING_HAND = False
    if USE_FLOATING_HAND:
        sampler = HeuristicAlgrICSampler(model, table=True, z_axis_fwd=True)
    else:
        sampler = HeuristicFR3AlgrICSampler(model, z_axis_fwd=True)

    # creating solver and generating grasp
    frogger = FroggerConfig(
        model=model,
        sampler=sampler,
        tol_surf=1e-3,
        tol_joint=1e-2,
        tol_col=5e-3,
        tol_fclosure=1e-5,
        xtol_rel=1e-6,
        xtol_abs=1e-6,
        maxeval=1000,
    ).create()

    # We want to ensure the total time of doing this takes at most MAX_TIME
    remaining_time = max_time
    q_array = []
    R_O_cf_array = []
    l_array = []
    print(f"max_time={max_time}", file=sys.stderr)
    for i in tqdm(range(num_grasps)):
        print(f"Grasp {i}: remaining_time={remaining_time}", file=sys.stderr)
        start_time = time.time()
        q_star = frogger.generate_grasp(max_time=remaining_time)

        if q_star is None:
            print("&" * 80, file=sys.stderr)
            print(f"Timeout at grasp {i}", file=sys.stderr)
            print("&" * 80, file=sys.stderr)
            # HACK: When timeout, we populate with a bad q that will fail downstream motion planning
            # This assumes q is for full robot arm and hand, careful that it is not pose of wrist
            BAD_Q = np.zeros(23)
            BAD_Q[:7] = np.array([0, 1.5, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
            q_star = BAD_Q

            # Be careful, these might be bad values
            q_array.append(q_star)
            R_O_cf_array.append(np.eye(3).reshape(1, 3, 3).repeat(4, axis=0))
            l_array.append(-1e6)
        else:
            assert q_star is not None
            assert q_star.shape == (23,)

            q_array.append(q_star)
            assert model.R_O_cf is not None
            R_O_cf_array.append(np.copy(model.R_O_cf))
            normalized_l = model.l * model.ns * model.nc
            assert normalized_l < 1.0 + 1e-2
            l_array.append(normalized_l)

        remaining_time = np.clip(
            remaining_time - (time.time() - start_time),
            a_min=1e-6,
            a_max=None,
        )  # Can't have 0 or negative time or timeout may do weird things

    q_array = np.array(q_array)
    R_O_cf_array = np.array(R_O_cf_array)
    l_array = np.array(l_array)
    assert q_array.shape == (num_grasps, 23)
    assert R_O_cf_array.shape == (num_grasps, 4, 3, 3)
    assert l_array.shape == (num_grasps,)
    return q_array, R_O_cf_array, l_array


def visualize_q_with_pydrake_blocking(mesh_object: MeshObject, q: np.ndarray) -> None:
    assert q.shape == (23,)
    # loading model
    model = create_model(mesh_object=mesh_object, viz=True)
    model.viz_config(q)


def add_transform_traces(
    fig: go.Figure, T: np.ndarray, name: str, length: float = 0.02
) -> None:
    assert T.shape == (4, 4)
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    x_axis = np.array([length, 0.0, 0.0, 1.0])
    y_axis = np.array([0.0, length, 0.0, 1.0])
    z_axis = np.array([0.0, 0.0, length, 1.0])

    origin = T @ origin
    x_axis = T @ x_axis
    y_axis = T @ y_axis
    z_axis = T @ z_axis

    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], x_axis[0]],
            y=[origin[1], x_axis[1]],
            z=[origin[2], x_axis[2]],
            mode="lines",
            line=dict(color="red"),
            name=f"{name}_x",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], y_axis[0]],
            y=[origin[1], y_axis[1]],
            z=[origin[2], y_axis[2]],
            mode="lines",
            line=dict(color="green"),
            name=f"{name}_y",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], z_axis[0]],
            y=[origin[1], z_axis[1]],
            z=[origin[2], z_axis[2]],
            mode="lines",
            line=dict(color="blue"),
            name=f"{name}_z",
        )
    )


def get_kinematic_chain(model_path: pathlib.Path) -> pk.Chain:
    with open(model_path) as f:
        chain = pk.build_chain_from_urdf(f.read())
        chain = chain.to(device="cuda", dtype=torch.float32)
    return chain


def q_to_T_W_H_and_joint_angles(
    q: np.ndarray, chain: pk.Chain, wrist_body_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    assert q.shape == (23,)
    hand_joint_angles = q[7:23]

    link_poses_hand_frame = chain.forward_kinematics(q)
    X_W_Wrist = (
        link_poses_hand_frame[wrist_body_name].get_matrix().squeeze(dim=0).cpu().numpy()
    )

    assert X_W_Wrist.shape == (4, 4)
    assert hand_joint_angles.shape == (16,)
    return X_W_Wrist, hand_joint_angles


def q_array_to_grasp_config_dict(
    q_array: np.ndarray,
    chain: pk.Chain,
    X_Oy_W: np.ndarray,
    X_Oy_O: np.ndarray,
    wrist_body_name: str,
    R_O_cf_array: np.ndarray,
    l_array: np.ndarray,
) -> dict:
    # W = world frame z-up
    # O = object frame z-up
    # Oy = object frame y-up
    # H = hand/frame z along finger, x away from palm
    # Assumes q in W frame
    # Assumes grasp_config_dict in Oy frame

    B = q_array.shape[0]
    assert q_array.shape == (B, 23)
    assert X_Oy_W.shape == (4, 4)
    assert X_Oy_O.shape == (4, 4)
    assert R_O_cf_array.shape == (B, 4, 3, 3)
    assert l_array.shape == (B,)

    X_W_H_array, joint_angles_array = [], []
    for i in range(B):
        X_W_H, joint_angles = q_to_T_W_H_and_joint_angles(
            q=q_array[i], chain=chain, wrist_body_name=wrist_body_name
        )
        assert X_W_H.shape == (4, 4)
        assert joint_angles.shape == (16,)
        X_W_H_array.append(X_W_H)
        joint_angles_array.append(joint_angles)

    X_W_H_array = np.array(X_W_H_array)
    joint_angles_array = np.array(joint_angles_array)
    assert X_W_H_array.shape == (B, 4, 4)
    assert joint_angles_array.shape == (B, 16)

    X_Oy_H_array = []
    for i in range(B):
        X_Oy_H = X_Oy_W @ X_W_H_array[i]
        X_Oy_H_array.append(X_Oy_H)
    X_Oy_H_array = np.array(X_Oy_H_array)

    # grasp_orientations should be in Oy frame
    R_Oy_O = X_Oy_O[:3, :3]

    grasp_orientations_array = R_Oy_O @ R_O_cf_array
    assert grasp_orientations_array.shape == (B, 4, 3, 3)

    return {
        "trans": X_Oy_H_array[:, :3, 3],
        "rot": X_Oy_H_array[:, :3, :3],
        "joint_angles": joint_angles_array,
        "grasp_orientations": grasp_orientations_array,
        "loss": -l_array,
    }


def frogger_to_grasp_config_dict(
    args: FroggerArgs,
    X_W_O: Optional[np.ndarray] = None,
    mesh: Optional[trimesh.Trimesh] = None,
    custom_coll_callback: Optional[Callable[[RobotModel, str, str], float]] = None,
) -> dict:
    rc = RobotConstants()

    print("=" * 80)
    print(f"args:\n{tyro.extras.to_yaml(args)}")
    print("=" * 80 + "\n")

    # Prepare mesh object
    if mesh is None:
        mesh = create_mesh(obj_filepath=args.obj_filepath, obj_scale=args.obj_scale)
    if X_W_O is None:
        X_W_O = compute_X_W_O(mesh=mesh, obj_is_yup=args.obj_is_yup)

    mesh_object = create_frogger_mesh_object(
        mesh=mesh, obj_name=args.obj_code, X_W_O=X_W_O
    )

    # Compute grasps
    q_array, R_O_cf_array, l_array = zup_mesh_to_q_array(
        mesh_object=mesh_object,
        num_grasps=args.num_grasps,
        custom_coll_callback=custom_coll_callback,
        max_time=args.max_time,
    )

    # Prepare kinematic chain
    chain = get_kinematic_chain(model_path=rc.robot_model_path)

    # Wrist
    X_W_Wrist, _ = q_to_T_W_H_and_joint_angles(
        q=q_array[args.grasp_idx_to_visualize],
        chain=chain,
        wrist_body_name=rc.wrist_body_name,
    )

    # Fingers
    link_poses_hand_frame = chain.forward_kinematics(
        q_array[args.grasp_idx_to_visualize]
    )
    X_W_fingertip_list = [
        link_poses_hand_frame[ln].get_matrix().squeeze(dim=0).cpu().numpy()
        for ln in rc.fingertip_body_names
    ]

    # Frames
    if args.obj_is_yup:
        X_O_Oy = np.eye(4)
    else:
        X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    X_Oy_O = np.linalg.inv(X_O_Oy)
    X_O_W = np.linalg.inv(X_W_O)
    X_Oy_W = X_Oy_O @ X_O_W

    # Create and save grasp_config_dict
    grasp_config_dict = q_array_to_grasp_config_dict(
        q_array=q_array,
        chain=chain,
        X_Oy_W=X_Oy_W,
        X_Oy_O=X_Oy_O,
        wrist_body_name=rc.wrist_body_name,
        R_O_cf_array=R_O_cf_array,
        l_array=l_array,
    )
    args.output_grasp_config_dicts_folder.mkdir(exist_ok=True)
    output_filepath = (
        args.output_grasp_config_dicts_folder / f"{args.object_code_and_scale_str}.npy"
    )
    np.save(
        file=output_filepath,
        arr=grasp_config_dict,
        allow_pickle=True,
    )

    # Visualize
    if args.visualize:
        # World frame vis
        fig = go.Figure()
        fig.update_layout(scene=dict(aspectmode="data"), title=dict(text="W frame"))
        vertices_O = mesh.vertices
        vertices_W = transform_points(points=vertices_O, T=X_W_O)
        fig.add_trace(
            go.Mesh3d(
                x=vertices_W[:, 0],
                y=vertices_W[:, 1],
                z=vertices_W[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color="lightpink",
                opacity=0.50,
            )
        )
        add_transform_traces(fig=fig, T=np.eye(4), name="T_W")
        add_transform_traces(fig=fig, T=X_W_O, name="T_O")
        add_transform_traces(fig=fig, T=X_W_Wrist, name="T_Wrist")
        for i, X_W_fingertip in enumerate(X_W_fingertip_list):
            add_transform_traces(fig=fig, T=X_W_fingertip, name=f"T_fingertip {i}")
        fig.show()

        # Oy frame vis
        fig = go.Figure()
        fig.update_layout(scene=dict(aspectmode="data"), title=dict(text="Oy frame"))
        vertices_Oy = transform_points(points=vertices_O, T=X_Oy_O)
        fig.add_trace(
            go.Mesh3d(
                x=vertices_Oy[:, 0],
                y=vertices_Oy[:, 1],
                z=vertices_Oy[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color="lightpink",
                opacity=0.50,
            )
        )
        add_transform_traces(fig=fig, T=np.eye(4), name="T_Oy")
        add_transform_traces(fig=fig, T=X_Oy_W @ X_W_Wrist, name="T_Wrist")
        for i, X_W_fingertip in enumerate(X_W_fingertip_list):
            add_transform_traces(
                fig=fig, T=X_Oy_W @ X_W_fingertip, name=f"T_fingertip {i}"
            )
        fig.show()

        # Drake vis
        visualize_q_with_pydrake_blocking(
            mesh_object=mesh_object, q=q_array[args.grasp_idx_to_visualize]
        )
    return grasp_config_dict


def custom_coll_callback(model, name_A: str, name_B: str) -> float:
    """A custom collision callback.

    Given two collision geom names, indicates what the lower bound on separation
    should be between them in meters.

    WARNING: for now, if you overwrite this, you MUST ensure manually that the fingertips
    are allowed some penetration with the object!
    """
    # organizing names
    has_tip = "FROGGERCOL" in name_A or "FROGGERCOL" in name_B  # MUST MANUALLY DO THIS!
    has_palm = "palm" in name_A or "palm" in name_B
    has_ds = (
        "ds_collision" in name_A or "ds_collision" in name_B
    )  # non-tip distal geoms
    has_md = "md" in name_A or "md" in name_B  # medial geoms
    has_px = "px" in name_A or "px" in name_B  # proximal geoms
    has_bs = "bs" in name_A or "bs" in name_B  # base geoms
    has_mp = "mp" in name_A or "mp" in name_B  # metacarpal geoms, thumb only
    has_obj = "obj" in name_A or "obj" in name_B

    # provide custom bounds on different geom pairs
    if has_tip and has_obj:
        # allow tips to penetrate object - MUST MANUALLY DO THIS!
        return -model.d_pen
    elif has_ds and has_obj:
        return 0.002  # ensure at least 2mm separation
    elif has_md and has_obj:  # noqa: SIM114
        return 0.005  # ensure at least 5mm separation
    elif has_px and has_obj:  # noqa: SIM114
        return 0.005  # ensure at least 5mm separation
    elif has_bs and has_obj:  # noqa: SIM114
        return 0.005  # ensure at least 5mm separation
    elif has_mp and has_obj:  # noqa: SIM114
        return 0.005  # ensure at least 5mm separation
    elif has_palm and has_obj:
        return 0.01  # ensure at least 1cm separation
    else:
        return model.d_min  # default case: use d_min


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[FroggerArgs])
    frogger_to_grasp_config_dict(args)


if __name__ == "__main__":
    main()
