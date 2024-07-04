import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
import torch.utils.data as data
import transforms3d
import trimesh
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from get_a_grip.dataset_generation.utils.pose_conversion import hand_config_to_pose
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.models.dex_evaluator import DexEvaluator
from get_a_grip.model_training.models.dex_sampler import DexSampler
from get_a_grip.model_training.utils.bps_grasp_dataset import (
    BpsGraspEvalDataset,
    BpsGraspSampleDataset,
)
from get_a_grip.model_training.utils.diffusion import Diffusion


def main(GRASP_IDX: int = 0, refine: bool = True) -> None:
    # loading dex sampler
    config = DiffusionConfig(
        training=TrainingConfig(
            log_path=Path(
                "logs/dexdiffuser_sampler/stable_jun2"
            )  # [r5ryh0z9] first one trained, tylers arch, not converged, no_noisy
        ),
    )
    model = DexSampler(
        n_pts=config.data.n_pts,
        grasp_dim=config.data.grasp_dim,
        d_model=128,
        virtual_seq_len=4,
    )
    runner = Diffusion(config=config, model=model, load_multigpu_ckpt=True)
    runner.load_checkpoint(config, name="ckpt_final")

    # loading dex evaluator
    evaluator_path = Path(
        "/home/albert/research/nerf_grasping/nerf_grasping/dexdiffuser/dexevaluator_checkpoints/ckpt_yp920sn0_final.pth"
    )  # crappy but the best we have rn
    dex_evaluator = DexEvaluator(in_grasp=37).to(runner.device)
    dex_evaluator.load_state_dict(torch.load(evaluator_path))
    dex_evaluator.eval()

    # loading data
    test_eval_dataset = BpsGraspEvalDataset(
        input_hdf5_filepath=pathlib.Path("TODO"),
    )
    test_loader = data.DataLoader(test_eval_dataset, batch_size=1, shuffle=False)

    # validating the evaluator
    ys_true = []
    ys_pred = []
    for i, (grasps, bpss, y_PGS) in tqdm(
        enumerate(test_loader),
        desc="Iterations",
        total=len(test_loader),
        leave=False,
    ):
        grasps, bpss, y_PGS = (
            grasps.to(runner.device),
            bpss.to(runner.device),
            y_PGS.to(runner.device),
        )
        y_PGS_pred = dex_evaluator(f_O=bpss, g_O=grasps)[0, -1]
        ys_true.append(y_PGS.detach().cpu().numpy())
        ys_pred.append(y_PGS_pred.detach().cpu().numpy())
        if i == 1000:
            break

    plt.scatter(ys_true, ys_pred)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("True PGS")
    plt.ylabel("Predicted PGS")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("PGS True vs Predicted")
    plt.axis("equal")
    plt.show()
    breakpoint()

    # running just the sampler
    test_sample_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=pathlib.Path("TODO"),
    )
    GRASP_IDX = 0
    _, bps, _ = test_sample_dataset[GRASP_IDX]
    xT = torch.randn(1, config.data.grasp_dim, device=runner.device)
    x = runner.sample(xT=xT, cond=bps[None].to(runner.device))  # (1, 37)
    print(f"Sampled grasp shape: {x.shape}")
    print(
        f"Sampled grasp quality: {dex_evaluator(f_O=bps[None].to(runner.device), g_O=x.to(runner.device))[0, -1]}"
    )

    # running the MCMC
    x_refined = dex_evaluator.refine(
        f_O=bps.to(runner.device)[None, ...],
        g_O=x.to(runner.device),
        num_steps=1000,
        stage="all",
    )
    # x_refined = dex_evaluator.refine(
    #     f_O=bps.to(runner.device)[None, ...],
    #     g_O=x.to(runner.device),
    #     num_steps=100,
    #     stage="wrist_pose",
    # )
    # x_refined = dex_evaluator.refine(
    #     f_O=bps.to(runner.device)[None, ...],
    #     g_O=x_refined.to(runner.device),
    #     num_steps=100,
    #     stage="joint_angles",
    # )
    # x_refined = dex_evaluator.refine(
    #     f_O=bps.to(runner.device)[None, ...],
    #     g_O=x_refined.to(runner.device),
    #     num_steps=100,
    #     stage="dirs",
    # )
    print(
        f"Refined grasp quality: {dex_evaluator(f_O=bps[None].to(runner.device), g_O=x_refined.to(runner.device))[0, -1]}"
    )
    breakpoint()
    grasp = x_refined[0].cpu()

    MESHDATA_ROOT = get_data_folder() / "large/meshes"
    print("=" * 79)
    print(f"len(test_sample_dataset): {len(test_sample_dataset)}")

    print("\n" + "=" * 79)
    print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
    print("=" * 79)
    y_PGS = np.array(1)
    print(f"grasp.shape: {grasp.shape}")
    print(f"bps.shape: {bps.shape}")
    print(f"y_PGS.shape: {y_PGS.shape}")

    print("\n" + "=" * 79)
    print("Getting debugging extras")
    print("=" * 79)
    basis_points = test_sample_dataset.get_basis_points()
    object_code = test_sample_dataset.get_object_code(GRASP_IDX)
    object_scale = test_sample_dataset.get_object_scale(GRASP_IDX)
    object_state = test_sample_dataset.get_object_state(GRASP_IDX)
    print(f"basis_points.shape: {basis_points.shape}")

    # Mesh
    mesh_path = pathlib.Path(f"{MESHDATA_ROOT}/{object_code}/coacd/decomposed.obj")
    assert mesh_path.exists(), f"{mesh_path} does not exist"
    print(f"Reading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)

    xyz, quat_xyzw = object_state[:3], object_state[3:7]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    transform = np.eye(4)  # X_W_Oy
    transform[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
    transform[:3, 3] = xyz
    mesh.apply_scale(object_scale)
    mesh.apply_transform(transform)

    # Point cloud
    point_cloud_filepath = test_sample_dataset.get_point_cloud_filepath(GRASP_IDX)
    print(f"Reading point cloud from {point_cloud_filepath}")
    point_cloud = o3d.io.read_point_cloud(point_cloud_filepath)
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    point_cloud_points = np.asarray(point_cloud.points)
    print(f"point_cloud_points.shape: {point_cloud_points.shape}")

    # Grasp
    assert grasp.shape == (
        3 + 6 + 16 + 4 * 3,
    ), f"Expected shape (3 + 6 + 16 + 4 * 3), got {grasp.shape}"
    grasp = grasp.detach().cpu().numpy()
    grasp_trans, grasp_rot6d, grasp_joints, grasp_dirs = (
        grasp[:3],
        grasp[3:9],
        grasp[9:25],
        grasp[25:].reshape(4, 3),
    )
    grasp_rot = np.zeros((3, 3))
    grasp_rot[:3, :2] = grasp_rot6d.reshape(3, 2)
    grasp_rot[:3, 0] = grasp_rot[:3, 0] / np.linalg.norm(grasp_rot[:3, 0])
    # make grasp_rot[:3, 1] orthogonal to grasp_rot[:3, 0]
    grasp_rot[:3, 1] = (
        grasp_rot[:3, 1] - np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) * grasp_rot[:3, 0]
    )
    grasp_rot[:3, 1] = grasp_rot[:3, 1] / np.linalg.norm(grasp_rot[:3, 1])
    assert (
        np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) < 1e-3
    ), f"Expected dot product < 1e-3, got {np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1])}"
    grasp_rot[:3, 2] = np.cross(grasp_rot[:3, 0], grasp_rot[:3, 1])
    grasp_transform = np.eye(4)  # X_Oy_H
    grasp_transform[:3, :3] = grasp_rot
    grasp_transform[:3, 3] = grasp_trans
    print(f"grasp_transform:\n{grasp_transform}")
    grasp_transform = transform @ grasp_transform  # X_W_H = X_W_Oy @ X_Oy_H
    grasp_trans = grasp_transform[:3, 3]
    grasp_rot = grasp_transform[:3, :3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_pose = hand_config_to_pose(
        grasp_trans[None], grasp_rot[None], grasp_joints[None]
    ).to(device)
    grasp_orientations = np.zeros(
        (4, 3, 3)
    )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
    grasp_orientations[:, :, 2] = (
        grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    )
    hand_model = HandModel(device=device)
    hand_model.set_parameters(hand_pose)
    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.8)

    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=torch.from_numpy(grasp_orientations[None]).to(device),
    )
    new_hand_pose = hand_config_to_pose(
        grasp_trans[None],
        grasp_rot[None],
        optimized_joint_angle_targets.detach().cpu().numpy(),
    ).to(device)
    hand_model.set_parameters(new_hand_pose)
    hand_plotly_optimized = hand_model.get_plotly_data(
        i=0, opacity=0.3, color="lightgreen"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=basis_points[:, 0],
            y=basis_points[:, 1],
            z=basis_points[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=bps,
                colorscale="rainbow",
                colorbar=dict(title="Basis points", orientation="h"),
            ),
            name="Basis points",
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name="Object",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud_points[:, 0],
            y=point_cloud_points[:, 1],
            z=point_cloud_points[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="black"),
            name="Point cloud",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, y_PGS: {y_PGS}"
        ),
    )
    VISUALIZE_HAND = True
    if VISUALIZE_HAND:
        for trace in hand_plotly:
            fig.add_trace(trace)
        for trace in hand_plotly_optimized:
            fig.add_trace(trace)
    # fig.write_html("/home/albert/research/nerf_grasping/dex_diffuser_debug.html")  # if headless
    fig.show()


if __name__ == "__main__":
    main()
