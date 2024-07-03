from typing import Tuple
import pypose as pp
import numpy as np
import torch
from torch.utils import data
import h5py
from nerf_grasping.dataset.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
    NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
    NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
)
from nerf_grasping.other_utils import (
    get_points_in_grid,
)


def grasp_config_to_grasp(grasp_config: torch.Tensor) -> torch.Tensor:
    B = grasp_config.shape[0]
    N_FINGERS = 4
    assert grasp_config.shape == (
        B,
        N_FINGERS,
        27,
    ), f"Expected shape (B, 4, 27), got {grasp_config.shape}"

    # Extract data from grasp_config
    xyz = grasp_config[:, 0, :3]
    quat_xyzw = grasp_config[:, 0, 3:7]
    joint_angles = grasp_config[:, 0, 7:23]
    grasp_quat_orientations = grasp_config[:, :, 23:]
    assert xyz.shape == (
        B,
        3,
    ), f"Expected shape (3), got {xyz.shape}"
    assert quat_xyzw.shape == (
        B,
        4,
    ), f"Expected shape (4), got {quat_xyzw.shape}"
    assert joint_angles.shape == (
        B,
        16,
    ), f"Expected shape (16), got {joint_angles.shape}"
    assert grasp_quat_orientations.shape == (
        B,
        N_FINGERS,
        4,
    ), f"Expected shape (B, 4, 4), got {grasp_quat_orientations.shape}"

    # Convert rot to matrix
    rot = pp.SO3(quat_xyzw).matrix()
    assert rot.shape == (B, 3, 3), f"Expected shape (3, 3), got {rot.shape}"

    # Convert grasp_quat_orientations to matrix
    grasp_orientations = pp.SO3(grasp_quat_orientations).matrix()
    assert grasp_orientations.shape == (
        B,
        N_FINGERS,
        3,
        3,
    ), f"Expected shape (B, 4, 3, 3), got {grasp_orientations.shape}"

    # Get grasp_dirs from grasp_orientations
    grasp_dirs = grasp_orientations[..., 2]
    assert grasp_dirs.shape == (
        B,
        N_FINGERS,
        3,
    ), f"Expected shape (B, 4, 3), got {grasp_dirs.shape}"

    grasps = torch.cat(
        [
            xyz,
            rot[..., :2].reshape(B, 6),
            joint_angles,
            grasp_dirs.reshape(B, 4 * 3),
        ],
        dim=1,
    )
    assert grasps.shape == (
        B,
        3 + 6 + 16 + 4 * 3,
    ), f"Expected shape (B, 3 + 6 + 16 + 4 * 3), got {grasps.shape}"
    return grasps


def coords_global(device, dtype, batch_size) -> torch.Tensor:
    points = get_points_in_grid(
        lb=lb_Oy,
        ub=ub_Oy,
        num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
        num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
        num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
    )

    assert points.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    )
    points = torch.from_numpy(points).to(device=device, dtype=dtype)
    points = points.permute(3, 0, 1, 2)
    points = points[None, ...].repeat_interleave(batch_size, dim=0)
    assert points.shape == (
        batch_size,
        3,
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
    )
    return points


def coords_global_cropped(device, dtype, batch_size) -> torch.Tensor:
    _coords_global = coords_global(device, dtype, batch_size)
    assert _coords_global.shape == (
        batch_size,
        3,
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
    )

    start_x = (NERF_DENSITIES_GLOBAL_NUM_X - NERF_DENSITIES_GLOBAL_NUM_X_CROPPED) // 2
    start_y = (NERF_DENSITIES_GLOBAL_NUM_Y - NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED) // 2
    start_z = (NERF_DENSITIES_GLOBAL_NUM_Z - NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED) // 2
    end_x = start_x + NERF_DENSITIES_GLOBAL_NUM_X_CROPPED
    end_y = start_y + NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED
    end_z = start_z + NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED
    coords_global_cropped = _coords_global[
        :, :, start_x:end_x, start_y:end_y, start_z:end_z
    ]
    assert coords_global_cropped.shape == (
        batch_size,
        3,
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
    )
    return coords_global_cropped


def add_coords_to_global_grids(global_grids: torch.Tensor) -> torch.Tensor:
    B = global_grids.shape[0]
    assert global_grids.shape == (
        B,
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
    ), f"Expected shape (B, NERF_DENSITIES_GLOBAL_NUM_X_CROPPED, NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED, NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED), got {global_grids.shape}"
    _coords_global_cropped = coords_global_cropped(
        device=global_grids.device,
        dtype=global_grids.dtype,
        batch_size=B,
    )
    assert _coords_global_cropped.shape == (
        B,
        3,
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
    ), f"Expected shape (B, 3, NERF_DENSITIES_GLOBAL_NUM_X_CROPPED, NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED, NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED), got {_coords_global_cropped.shape}"

    global_grids_with_coords = torch.cat(
        (
            global_grids.unsqueeze(dim=1),
            _coords_global_cropped,
        ),
        dim=1,
    )
    assert global_grids_with_coords.shape == (
        B,
        3 + 1,
        NERF_DENSITIES_GLOBAL_NUM_X_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED,
        NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED,
    ), f"Expected shape (B, 3 + 1, NERF_DENSITIES_GLOBAL_NUM_X_CROPPED, NERF_DENSITIES_GLOBAL_NUM_Y_CROPPED, NERF_DENSITIES_GLOBAL_NUM_Z_CROPPED), got {global_grids_with_coords.shape}"
    return global_grids_with_coords


class GraspNerfDataset(data.Dataset):
    def __init__(
        self,
        input_hdf5_filepath: str,
    ) -> None:
        self.input_hdf5_filepath = input_hdf5_filepath
        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            # Essentials
            self.num_grasps = hdf5_file.attrs["num_data_points"]
            grasp_configs = torch.from_numpy(hdf5_file["/grasp_configs"][()]).float()
            self.grasps = grasp_config_to_grasp(grasp_configs).float()
            if self.grasps.shape[0] != self.num_grasps:
                print(
                    f"WARNING: Expected {self.num_grasps} grasps, got {self.grasps.shape[0]}! Truncating data..."
                )

            self.grasps = self.grasps[: self.num_grasps, ...]
            nerf_global_grids = torch.from_numpy(
                hdf5_file["/nerf_densities_global"][()]
            ).float()[: self.num_grasps, ...]
            self.nerf_global_grids_with_coords = add_coords_to_global_grids(
                nerf_global_grids[..., 5:35, 5:35, 5:35]  # TODO(ahl): hardcoded! change later!
            )[: self.num_grasps, ...]
            self.global_grid_idxs = torch.from_numpy(
                hdf5_file["/nerf_densities_global_idx"][()]
            )[: self.num_grasps, ...]
            self.passed_evals = torch.from_numpy(hdf5_file["/passed_eval"][()]).float()[: self.num_grasps, ...]
            self.passed_simulations = torch.from_numpy(
                hdf5_file["/passed_simulation"][()]
            ).float()[: self.num_grasps, ...]
            self.passed_penetration_thresholds = torch.from_numpy(
                hdf5_file["/passed_penetration_threshold"][()]
            ).float()[: self.num_grasps, ...]

            assert (
                self.grasps.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} grasps, got {self.grasps.shape[0]}"
            assert (
                self.global_grid_idxs.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} global_grid_idxs, got {self.global_grid_idxs.shape[0]}"
            assert (
                self.passed_evals.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} passed_evals, got {self.passed_evals.shape[0]}"
            assert (
                self.passed_simulations.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} passed_simulations, got {self.passed_simulations.shape[0]}"
            assert (
                self.passed_penetration_thresholds.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} passed_penetration_thresholds, got {self.passed_penetration_thresholds.shape[0]}"

            # Extras
            # self.object_codes = hdf5_file["/object_code"][()]
            # self.object_scales = hdf5_file["/object_scale"][()]
            # self.object_states = torch.from_numpy(
            #     hdf5_file["/object_state"][()]
            # ).float()
            # assert (
            #     self.object_codes.shape[0] == self.nerf_global_grids.shape[0]
            # ), f"Expected {self.nerf_global_grids.shape[0]} object_codes, got {self.object_codes.shape[0]}"
            # assert (
            #     self.object_scales.shape[0] == self.nerf_global_grids.shape[0]
            # ), f"Expected {self.nerf_global_grids.shape[0]} object_scales, got {self.object_scales.shape[0]}"
            # assert (
            #     self.object_states.shape[0] == self.num_grasps
            # ), f"Expected {self.num_grasps} object_states, got {self.object_states.shape[0]}"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GraspNerfEvalDataset(GraspNerfDataset):
    def __init__(
        self,
        input_hdf5_filepath: str,
        get_all_labels: bool = False,
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)
        self.get_all_labels = get_all_labels

    def __len__(self) -> int:
        return self.num_grasps

    def __getitem__(
        self, grasp_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nerf_global_grid_idx = self.global_grid_idxs[grasp_idx]

        if self.get_all_labels:
            labels = torch.concatenate(
                (
                    self.passed_simulations[grasp_idx],
                    self.passed_penetration_thresholds[grasp_idx],
                    self.passed_evals[grasp_idx],
                ),
            )  # shape=(3,)
            return (
                self.grasps[grasp_idx],
                self.nerf_global_grids_with_coords[nerf_global_grid_idx],
                labels,
            )
        else:
            return (
                self.grasps[grasp_idx],
                self.nerf_global_grids_with_coords[nerf_global_grid_idx],
                self.passed_evals[grasp_idx],
            )

    ###### Extras ######
    # def get_object_code(self, grasp_idx: int) -> str:
    #     nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
    #     return self.object_codes[nerf_global_grids_idx].decode("utf-8")

    # def get_object_scale(self, grasp_idx: int) -> float:
    #     nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
    #     return self.object_scales[nerf_global_grids_idx]

    # def get_object_state(self, grasp_idx: int) -> torch.Tensor:
    #     return self.object_states[grasp_idx]


class GraspNerfSampleDataset(GraspNerfDataset):
    def __init__(
        self,
        input_hdf5_filepath: str,
        get_all_labels: bool = False,
        passed_eval_threshold: float = 0.9,
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)
        self.get_all_labels = get_all_labels

        self.passed_eval_threshold = passed_eval_threshold
        self.successful_grasp_idxs = torch.where(
            self.passed_evals >= passed_eval_threshold
        )[0]
        self.num_successful_grasps = len(self.successful_grasp_idxs)

    def __len__(self) -> int:
        return self.num_successful_grasps

    def __getitem__(
        self, successful_grasp_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        nerf_global_grid_idx = self.global_grid_idxs[grasp_idx]

        if self.get_all_labels:
            labels = torch.concatenate(
                (
                    self.passed_simulations[grasp_idx],
                    self.passed_penetration_thresholds[grasp_idx],
                    self.passed_evals[grasp_idx],
                ),
            )  # shape=(3,)
            return (
                self.grasps[grasp_idx],
                self.nerf_global_grids_with_coords[nerf_global_grid_idx],
                labels,
            )

        else:
            return (
                self.grasps[grasp_idx],
                self.nerf_global_grids_with_coords[nerf_global_grid_idx],
                self.passed_evals[grasp_idx],
            )

    ###### Extras ######
    # def get_object_code(self, successful_grasp_idx: int) -> str:
    #     grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
    #     nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
    #     return self.object_codes[nerf_global_grids_idx].decode("utf-8")

    # def get_object_scale(self, successful_grasp_idx: int) -> float:
    #     grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
    #     nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
    #     return self.object_scales[nerf_global_grids_idx]

    # def get_object_state(self, successful_grasp_idx: int) -> torch.Tensor:
    #     grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
    #     return self.object_states[grasp_idx]


def main() -> None:
    import numpy as np
    import trimesh
    import pathlib
    import transforms3d
    import open3d as o3d
    import plotly.graph_objects as go
    from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
    from nerf_grasping.dexgraspnet_utils.hand_model_type import (
        HandModelType,
    )
    from nerf_grasping.dexgraspnet_utils.pose_conversion import (
        hand_config_to_pose,
    )
    from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
        compute_optimized_joint_angle_targets_given_grasp_orientations,
    )

    INPUT_HDF5_FILEPATH = (
        "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_test.hdf5"
    )
    GRASP_IDX = 2000  # [DEBUG] change this guy for different viz
    MESHDATA_ROOT = (
        "/home/albert/research/nerf_grasping/rsync_meshes/rotated_meshdata_v2"
    )
    USE_EVAL_DATASET = True

    print("\n" + "=" * 79)
    print(f"Reading dataset from {INPUT_HDF5_FILEPATH}")
    if USE_EVAL_DATASET:
        dataset = GraspBPSEvalDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    else:
        dataset = GraspBPSSampleDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    print("=" * 79)
    print(f"len(dataset): {len(dataset)}")

    print("\n" + "=" * 79)
    print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
    print("=" * 79)
    grasp, bps, passed_eval = dataset[GRASP_IDX]
    print(f"grasp.shape: {grasp.shape}")
    print(f"bps.shape: {bps.shape}")
    print(f"passed_eval.shape: {passed_eval.shape}")

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
    hand_model_type = HandModelType.ALLEGRO_HAND
    grasp_orientations = np.zeros(
        (4, 3, 3)
    )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
    grasp_orientations[:, :, 2] = (
        grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    )
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
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
            text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, Passed Eval: {passed_eval}"
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
