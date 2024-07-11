import pathlib
from typing import Tuple

import h5py
import torch
from torch.utils import data

from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    add_coords_to_global_grids,
    get_coords_global,
)


class NerfGraspDataset(data.Dataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
    ) -> None:
        self.input_hdf5_filepath = input_hdf5_filepath
        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            # Essentials
            self.num_grasps: int = hdf5_file.attrs["num_data_points"]
            grasp_configs_tensor = torch.from_numpy(
                hdf5_file["/grasp_configs"][()]
            ).float()
            self.grasps = (
                AllegroGraspConfig.from_tensor(grasp_configs_tensor).as_grasp().detach()
            )  # Need .detach() since we don't want requires_grad=True here
            if self.grasps.shape[0] != self.num_grasps:
                print(
                    f"WARNING: Expected {self.num_grasps} grasps, got {self.grasps.shape[0]}! Truncating data..."
                )

            self.grasps = self.grasps[: self.num_grasps, ...]
            nerf_global_grids = torch.from_numpy(
                hdf5_file["/nerf_densities_global"][()]
            ).float()[: self.num_grasps, ...]

            coords_global = get_coords_global(
                device=nerf_global_grids.device,
                dtype=nerf_global_grids.dtype,
                batch_size=nerf_global_grids.shape[0],
            )
            self.nerf_global_grids_with_coords = add_coords_to_global_grids(
                global_grids=nerf_global_grids, coords_global=coords_global
            )[: self.num_grasps, ...]
            self.global_grid_idxs = torch.from_numpy(
                hdf5_file["/nerf_densities_global_idx"][()]
            ).long()[: self.num_grasps, ...]
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
                self.object_codes.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} object_codes, got {self.object_codes.shape[0]}"
            assert (
                self.object_scales.shape[0] == self.num_grasps
            ), f"Expected {self.num_grasps} object_scales, got {self.object_scales.shape[0]}"
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
    ) -> None:
        super().__init__(input_hdf5_filepath=input_hdf5_filepath)

    def __len__(self) -> int:
        return self.num_grasps

    def __getitem__(
        self, grasp_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nerf_global_grid_idx = self.global_grid_idxs[grasp_idx]

        labels = torch.concatenate(
            (
                self.y_picks[grasp_idx].reshape(1),
                self.y_colls[grasp_idx].reshape(1),
                self.y_PGSs[grasp_idx].reshape(1),
            ),
        )
        assert labels.shape == (3,), f"Expected shape (3,), got {labels.shape}"
        return (
            self.grasps[grasp_idx],
            self.nerf_global_grids_with_coords[nerf_global_grid_idx],
            labels,
        )

    ###### Extras ######
    def get_object_code(self, grasp_idx: int) -> str:
        return self.object_codes[grasp_idx].decode("utf-8")

    def get_object_scale(self, grasp_idx: int) -> float:
        return self.object_scales[grasp_idx]

    def get_object_state(self, grasp_idx: int) -> torch.Tensor:
        return self.object_states[grasp_idx]


class NerfGraspSampleDataset(NerfGraspDataset):
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
        nerf_global_grid_idx = self.global_grid_idxs[grasp_idx]

        labels = torch.concatenate(
            (
                self.y_picks[grasp_idx].reshape(1),
                self.y_colls[grasp_idx].reshape(1),
                self.y_PGSs[grasp_idx].reshape(1),
            ),
        )
        assert labels.shape == (3,), f"Expected shape (3,), got {labels.shape}"
        return (
            self.grasps[grasp_idx],
            self.nerf_global_grids_with_coords[nerf_global_grid_idx],
            labels,
        )

    ###### Extras ######
    def get_object_code(self, successful_grasp_idx: int) -> str:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        return self.object_codes[grasp_idx].decode("utf-8")

    def get_object_scale(self, successful_grasp_idx: int) -> float:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        return self.object_scales[grasp_idx]

    def get_object_state(self, successful_grasp_idx: int) -> torch.Tensor:
        grasp_idx = self.successful_grasp_idxs[successful_grasp_idx]
        return self.object_states[grasp_idx]
