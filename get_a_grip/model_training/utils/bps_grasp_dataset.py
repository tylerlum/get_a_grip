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
        frac_throw_away: float = 0.0,
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)
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
        labels = torch.concatenate(
            (
                self.y_picks[grasp_idx].reshape(1),
                self.y_colls[grasp_idx].reshape(1),
                self.y_PGSs[grasp_idx].reshape(1),
            ),
        )
        assert labels.shape == (3,), f"Expected shape (3,), got {labels.shape}"
        return self.grasps[grasp_idx], self.bpss[bps_idx], labels

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
        y_PGS_threshold: float = 0.9,
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)
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
        labels = torch.concatenate(
            (
                self.y_picks[grasp_idx].reshape(1),
                self.y_colls[grasp_idx].reshape(1),
                self.y_PGSs[grasp_idx].reshape(1),
            ),
        )
        assert labels.shape == (3,), f"Expected shape (3,), got {labels.shape}"
        return self.grasps[grasp_idx], self.bpss[bps_idx], labels

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
