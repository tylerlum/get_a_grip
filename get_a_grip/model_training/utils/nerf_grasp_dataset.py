import pathlib
from typing import Tuple

import h5py
import numpy as np
import pypose as pp
import torch
from torch.utils import data

from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from get_a_grip.model_training.utils.point_utils import (
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


def get_coords_global(
    device: torch.device, dtype: torch.dtype, batch_size: int
) -> torch.Tensor:
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


def add_coords_to_global_grids(global_grids: torch.Tensor) -> torch.Tensor:
    B = global_grids.shape[0]
    assert (
        global_grids.shape
        == (
            B,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape ({B}, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {global_grids.shape}"
    coords_global = get_coords_global(
        device=global_grids.device,
        dtype=global_grids.dtype,
        batch_size=B,
    )
    assert (
        coords_global.shape
        == (
            B,
            3,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape (B, 3, NERF_DENSITIES_GLOBAL_NUM_X, NERF_DENSITIES_GLOBAL_NUM_Y, NERF_DENSITIES_GLOBAL_NUM_Z), got {coords_global.shape}"

    global_grids_with_coords = torch.cat(
        (
            global_grids.unsqueeze(dim=1),
            coords_global,
        ),
        dim=1,
    )
    assert (
        global_grids_with_coords.shape
        == (
            B,
            3 + 1,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape (B, 3 + 1, NERF_DENSITIES_GLOBAL_NUM_X, NERF_DENSITIES_GLOBAL_NUM_Y, NERF_DENSITIES_GLOBAL_NUM_Z), got {global_grids_with_coords.shape}"
    return global_grids_with_coords


class NerfGraspDataset(data.Dataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
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
                nerf_global_grids
            )[: self.num_grasps, ...]
            self.global_grid_idxs = torch.from_numpy(
                hdf5_file["/nerf_densities_global_idx"][()]
            )[: self.num_grasps, ...]
            self.y_PGSs = torch.from_numpy(hdf5_file["/y_PGS"][()]).float()[
                : self.num_grasps, ...
            ]
            self.y_picks = torch.from_numpy(hdf5_file["/y_pick"][()]).float()[
                : self.num_grasps, ...
            ]
            self.y_colls = torch.from_numpy(hdf5_file["/y_coll"][()]).float()[
                : self.num_grasps, ...
            ]

            assert (
                self.grasps.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} grasps, got {self.grasps.shape[0]}"
            assert (
                self.global_grid_idxs.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} global_grid_idxs, got {self.global_grid_idxs.shape[0]}"
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
            self.object_codes = hdf5_file["/object_code"][()]
            self.object_scales = hdf5_file["/object_scale"][()]
            self.object_states = torch.from_numpy(
                hdf5_file["/object_state"][()]
            ).float()
            assert (
                self.object_codes.shape[0]
                == self.nerf_global_grids_with_coords.shape[0]
            ), f"Expected {self.nerf_global_grids_with_coords.shape[0]} object_codes, got {self.object_codes.shape[0]}"
            assert (
                self.object_scales.shape[0]
                == self.nerf_global_grids_with_coords.shape[0]
            ), f"Expected {self.nerf_global_grids_with_coords.shape[0]} object_scales, got {self.object_scales.shape[0]}"
            assert (
                self.object_states.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} object_states, got {self.object_states.shape[0]}"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class NerfGraspEvalDataset(NerfGraspDataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
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
                    self.y_picks[grasp_idx],
                    self.y_colls[grasp_idx],
                    self.y_PGSs[grasp_idx],
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
                self.y_PGSs[grasp_idx],
            )

    ###### Extras ######
    def get_object_code(self, grasp_idx: int) -> str:
        nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
        return self.object_codes[nerf_global_grids_idx].decode("utf-8")

    def get_object_scale(self, grasp_idx: int) -> float:
        nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
        return self.object_scales[nerf_global_grids_idx]

    def get_object_state(self, grasp_idx: int) -> torch.Tensor:
        return self.object_states[grasp_idx]


class NerfGraspSampleDataset(NerfGraspDataset):
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
        nerf_global_grid_idx = self.global_grid_idxs[grasp_idx]

        if self.get_all_labels:
            labels = torch.concatenate(
                (
                    self.y_picks[grasp_idx],
                    self.y_colls[grasp_idx],
                    self.y_PGSs[grasp_idx],
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
                self.y_PGSs[grasp_idx],
            )

    ###### Extras ######
    def get_object_code(self, successful_grasp_idx: int) -> str:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
        return self.object_codes[nerf_global_grids_idx].decode("utf-8")

    def get_object_scale(self, successful_grasp_idx: int) -> float:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        nerf_global_grids_idx = self.global_grid_idxs[grasp_idx]
        return self.object_scales[nerf_global_grids_idx]

    def get_object_state(self, successful_grasp_idx: int) -> torch.Tensor:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        return self.object_states[grasp_idx]


def main() -> None:
    import pathlib

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

    INPUT_HDF5_FILEPATH = (
        "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_test.hdf5"
    )
    GRASP_IDX = 2000  # [DEBUG] change this guy for different viz
    MESHDATA_ROOT = get_data_folder() / "large/meshes"
    USE_EVAL_DATASET = True

    print("\n" + "=" * 79)
    print(f"Reading dataset from {INPUT_HDF5_FILEPATH}")
    if USE_EVAL_DATASET:
        dataset = NerfGraspEvalDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    else:
        dataset = NerfGraspSampleDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    print("=" * 79)
    print(f"len(dataset): {len(dataset)}")

    print("\n" + "=" * 79)
    print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
    print("=" * 79)
    grasp, nerf_global_grid_with_coords, y_PGS = dataset[GRASP_IDX]
    print(f"grasp.shape: {grasp.shape}")
    print(f"nerf_global_grid_with_coords.shape: {nerf_global_grid_with_coords.shape}")
    print(f"y_PGS.shape: {y_PGS.shape}")

    assert nerf_global_grid_with_coords.shape == (
        3 + 1,
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
    )
    nerf_densities_flattened = nerf_global_grid_with_coords[3, ...].reshape(-1)
    coords = nerf_global_grid_with_coords[:3, ...].reshape(3, -1).T

    print("\n" + "=" * 79)
    print("Getting debugging extras")
    print("=" * 79)
    object_code = dataset.get_object_code(GRASP_IDX)
    object_scale = dataset.get_object_scale(GRASP_IDX)
    object_state = dataset.get_object_state(GRASP_IDX)

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
    assert hand_model.hand_pose is not None

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
    assert hand_model.hand_pose is not None
    hand_plotly_optimized = hand_model.get_plotly_data(
        i=0, opacity=0.3, color="lightgreen"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=nerf_densities_flattened,
                colorscale="rainbow",
                colorbar=dict(title="NeRF densities", orientation="h"),
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
