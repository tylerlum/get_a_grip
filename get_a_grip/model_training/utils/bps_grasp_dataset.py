import pathlib
from typing import Tuple

import h5py
import torch
from torch.utils import data


class BpsGraspDataset(data.Dataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
    ) -> None:
        self.input_hdf5_filepath = input_hdf5_filepath
        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            # Essentials
            self.grasps = torch.from_numpy(hdf5_file["/grasps"][()]).float()
            self.bpss = torch.from_numpy(hdf5_file["/bpss"][()]).float()
            self.grasp_bps_idxs = torch.from_numpy(hdf5_file["/grasp_bps_idx"][()])
            self.y_PGSs = torch.from_numpy(hdf5_file["/y_PGS"][()]).float()
            self.y_picks = torch.from_numpy(hdf5_file["/y_pick"][()]).float()
            self.y_colls = torch.from_numpy(hdf5_file["/y_coll"][()]).float()

            self.num_grasps = hdf5_file.attrs["num_grasps"]

            assert (
                self.grasps.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} grasps, got {self.grasps.shape[0]}"
            assert (
                self.grasp_bps_idxs.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} grasp_bps_idxs, got {self.grasp_bps_idxs.shape[0]}"
            assert (
                self.y_PGSs.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} y_PGSs, got {self.y_PGSs.shape[0]}"
            assert (
                self.y_picks.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} y_picks, got {self.y_picks.shape[0]}"
            assert (
                self.y_colls.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} y_colls, got {self.y_colls.shape[0]}"

            # Extras
            self.basis_points = torch.from_numpy(hdf5_file["/basis_points"][()]).float()
            self.point_cloud_filepaths = hdf5_file["/point_cloud_filepath"][()]
            self.object_codes = hdf5_file["/object_code"][()]
            self.object_scales = hdf5_file["/object_scale"][()]
            self.object_states = torch.from_numpy(
                hdf5_file["/object_state"][()]
            ).float()
            n_basis_points = self.basis_points.shape[0]
            assert self.basis_points.shape == (
                n_basis_points,
                3,
            ), f"Expected shape ({n_basis_points}, 3), got {self.basis_points.shape}"
            assert (
                self.point_cloud_filepaths.shape[0] == self.bpss.shape[0]
            ), f"Expected {self.bpss.shape[0]} point_cloud_filepaths, got {self.point_cloud_filepaths.shape[0]}"
            assert (
                self.object_codes.shape[0] == self.bpss.shape[0]
            ), f"Expected {self.bpss.shape[0]} object_codes, got {self.object_codes.shape[0]}"
            assert (
                self.object_scales.shape[0] == self.bpss.shape[0]
            ), f"Expected {self.bpss.shape[0]} object_scales, got {self.object_scales.shape[0]}"
            assert (
                self.object_states.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} object_states, got {self.object_states.shape[0]}"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    ###### Extras ######
    def get_basis_points(self) -> torch.Tensor:
        return self.basis_points.clone()


class BpsGraspEvalDataset(BpsGraspDataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
        get_all_labels: bool = False,
        frac_throw_away: float = 0.0,
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)
        self.get_all_labels = get_all_labels
        self.frac_throw_away = frac_throw_away
        print(f"Dataset has {self.num_grasps} grasps")

        if frac_throw_away > 0.0:
            print(f"Throwing away {frac_throw_away * 100}% of grasps")
            print(f"Before: {self.num_grasps}")
            self.num_grasps = int(self.num_grasps * (1 - frac_throw_away))
            print(f"After: {self.num_grasps}")

    def __len__(self) -> int:
        return self.num_grasps

    def __getitem__(
        self, grasp_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bps_idx = self.grasp_bps_idxs[grasp_idx]
        if self.get_all_labels:
            labels = torch.concatenate(
                (
                    self.y_picks[grasp_idx],
                    self.y_colls[grasp_idx],
                    self.y_PGSs[grasp_idx],
                ),
            )  # shape=(3,)
            return self.grasps[grasp_idx], self.bpss[bps_idx], labels
        else:
            return (
                self.grasps[grasp_idx],
                self.bpss[bps_idx],
                self.y_PGSs[grasp_idx],
            )

    ###### Extras ######
    def get_point_cloud_filepath(self, grasp_idx: int) -> pathlib.Path:
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return pathlib.Path(self.point_cloud_filepaths[bpss_idx].decode("utf-8"))

    def get_object_code(self, grasp_idx: int) -> str:
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.object_codes[bpss_idx].decode("utf-8")

    def get_object_scale(self, grasp_idx: int) -> float:
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.object_scales[bpss_idx]

    def get_object_state(self, grasp_idx: int) -> torch.Tensor:
        return self.object_states[grasp_idx]


class BpsGraspSampleDataset(BpsGraspDataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
        get_all_labels: bool = False,
        y_PGS_threshold: float = 0.9,
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)
        self.get_all_labels = get_all_labels

        self.y_PGS_threshold = y_PGS_threshold
        self.successful_grasp_idxs = torch.where(self.y_PGSs >= y_PGS_threshold)[0]
        self.num_successful_grasps = len(self.successful_grasp_idxs)

    def __len__(self) -> int:
        return self.num_successful_grasps

    def __getitem__(
        self, successful_grasp_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        bps_idx = self.grasp_bps_idxs[grasp_idx]
        if self.get_all_labels:
            labels = torch.concatenate(
                (
                    self.y_picks[grasp_idx],
                    self.y_colls[grasp_idx],
                    self.y_PGSs[grasp_idx],
                ),
            )  # shape=(3,)
            return self.grasps[grasp_idx], self.bpss[bps_idx], labels
        else:
            return (
                self.grasps[grasp_idx],
                self.bpss[bps_idx],
                self.y_PGSs[grasp_idx],
            )

    ###### Extras ######
    def get_point_cloud_filepath(self, successful_grasp_idx: int) -> pathlib.Path:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return pathlib.Path(self.point_cloud_filepaths[bpss_idx].decode("utf-8"))

    def get_object_code(self, successful_grasp_idx: int) -> str:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.object_codes[bpss_idx].decode("utf-8")

    def get_object_scale(self, successful_grasp_idx: int) -> float:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.object_scales[bpss_idx]

    def get_object_state(self, successful_grasp_idx: int) -> torch.Tensor:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        return self.object_states[grasp_idx]


def main() -> None:
    import pathlib

    import numpy as np
    import open3d as o3d
    import plotly.graph_objects as go
    import transforms3d
    import trimesh

    from get_a_grip import get_data_folder
    from get_a_grip.dataset_generation.utils.hand_model import HandModel
    from get_a_grip.dataset_generation.utils.joint_angle_targets import (
        compute_optimized_joint_angle_targets_given_grasp_orientations,
    )
    from get_a_grip.dataset_generation.utils.pose_conversion import (
        hand_config_to_pose,
    )

    INPUT_HDF5_FILEPATH = pathlib.Path(
        "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_test.hdf5"
    )
    GRASP_IDX = 2000  # [DEBUG] change this guy for different viz
    MESHDATA_ROOT = get_data_folder() / "large/meshes"
    USE_EVAL_DATASET = True

    print("\n" + "=" * 79)
    print(f"Reading dataset from {INPUT_HDF5_FILEPATH}")
    if USE_EVAL_DATASET:
        dataset = BpsGraspEvalDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    else:
        dataset = BpsGraspSampleDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    print("=" * 79)
    print(f"len(dataset): {len(dataset)}")

    print("\n" + "=" * 79)
    print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
    print("=" * 79)
    grasp, bps, y_PGS = dataset[GRASP_IDX]
    print(f"grasp.shape: {grasp.shape}")
    print(f"bps.shape: {bps.shape}")
    print(f"y_PGS.shape: {y_PGS.shape}")

    print("\n" + "=" * 79)
    print("Getting debugging extras")
    print("=" * 79)
    basis_points = dataset.get_basis_points()
    object_code = dataset.get_object_code(GRASP_IDX)
    object_scale = dataset.get_object_scale(GRASP_IDX)
    object_state = dataset.get_object_state(GRASP_IDX)
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
    point_cloud_filepath = dataset.get_point_cloud_filepath(GRASP_IDX)
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
    assert (
        np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) < 1e-3
    ), f"Expected dot product < 1e-3, got {np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1])}"
    grasp_rot[:3, 2] = np.cross(grasp_rot[:3, 0], grasp_rot[:3, 1])
    grasp_transform = np.eye(4)  # X_Oy_H
    grasp_transform[:3, :3] = grasp_rot
    grasp_transform[:3, 3] = grasp_trans
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
    for trace in hand_plotly:
        fig.add_trace(trace)
    for trace in hand_plotly_optimized:
        fig.add_trace(trace)
    fig.write_html("/home/albert/research/nerf_grasping/bps_debug.html")  # headless
    # fig.show()


if __name__ == "__main__":
    main()
