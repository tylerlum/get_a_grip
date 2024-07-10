import pathlib
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)

# Make atol and rtol larger than default to avoid errors due to floating point precision.
# Otherwise we get errors about invalid rotation matrices
PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class NerfGraspEvalDataset(Dataset):
    def __init__(
        self,
        input_hdf5_filepath: pathlib.Path,
        fingertip_config: EvenlySpacedFingertipConfig,
        max_num_data_points: Optional[int] = None,
        load_nerf_densities_in_ram: bool = False,  # Too big for RAM
        load_nerf_densities_global_in_ram: bool = False,  # Too big for RAM
        load_nerf_densities_global_idx_in_ram: bool = True,
        load_grasp_labels_in_ram: bool = True,
        load_grasp_transforms_in_ram: bool = True,
        load_nerf_configs_in_ram: bool = True,
        load_grasp_configs_in_ram: bool = True,
    ) -> None:
        super().__init__()
        self.input_hdf5_filepath = input_hdf5_filepath
        self.fingertip_config = fingertip_config

        # Recommended in https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/13
        self.hdf5_file = None

        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            self.len = self._set_length(
                hdf5_file=hdf5_file, max_num_data_points=max_num_data_points
            )

            # Check that the data is in the expected format
            assert_equals(len(hdf5_file["/y_pick"].shape), 1)
            assert_equals(len(hdf5_file["/y_coll"].shape), 1)
            assert_equals(len(hdf5_file["/y_PGS"].shape), 1)
            assert_equals(
                hdf5_file["/nerf_densities"].shape[1:],
                (
                    self.NUM_FINGERS,
                    self.NUM_PTS_X,
                    self.NUM_PTS_Y,
                    self.NUM_PTS_Z,
                ),
            )
            assert_equals(
                hdf5_file["/grasp_transforms"].shape[1:],
                (
                    self.NUM_FINGERS,
                    4,
                    4,
                ),
            )
            assert_equals(
                hdf5_file["/grasp_configs"].shape[1:], (self.NUM_FINGERS, 7 + 16 + 4)
            )

            # This is usually too big for RAM
            self.nerf_densities = (
                torch.from_numpy(hdf5_file["/nerf_densities"][()]).float()
                if load_nerf_densities_in_ram
                else None
            )
            self.nerf_densities_global = (
                torch.from_numpy(hdf5_file["/nerf_densities_global"][()]).float()
                if load_nerf_densities_global_in_ram
                and "nerf_densities_global" in hdf5_file
                else None
            )
            self.nerf_densities_global_idx = (
                torch.from_numpy(hdf5_file["/nerf_densities_global_idx"][()]).long()
                if load_nerf_densities_global_idx_in_ram
                and "nerf_densities_global_idx" in hdf5_file
                else None
            )

            # This is small enough to fit in RAM
            self.y_picks = (
                torch.from_numpy(hdf5_file["/y_pick"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.y_colls = (
                torch.from_numpy(hdf5_file["/y_coll"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.y_PGSs = (
                torch.from_numpy(hdf5_file["/y_PGS"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float()
                if load_grasp_transforms_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.nerf_configs = (
                hdf5_file["/nerf_config"][()] if load_nerf_configs_in_ram else None
            )

            self.grasp_configs = (
                torch.from_numpy(hdf5_file["/grasp_configs"][()]).float()
                if load_grasp_configs_in_ram
                else None
            )

    def _set_length(
        self, hdf5_file: h5py.File, max_num_data_points: Optional[int]
    ) -> int:
        length = (
            hdf5_file.attrs["num_data_points"]
            if "num_data_points" in hdf5_file.attrs
            else hdf5_file["/y_pick"].shape[0]
        )
        if length != hdf5_file["/y_pick"].shape[0]:
            print(
                f"WARNING: num_data_points = {length} != y_pick.shape[0] = {hdf5_file['/y_pick'].shape[0]}"
            )

        # Constrain length of dataset if max_num_data_points is set
        if max_num_data_points is not None:
            print(f"Constraining dataset length to {max_num_data_points}")
            length = max_num_data_points

        return length

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        if self.hdf5_file is None:
            # Hope to speed up with rdcc params
            self.hdf5_file = h5py.File(
                self.input_hdf5_filepath,
                "r",
                rdcc_nbytes=1024**2 * 4_000,
                rdcc_w0=0.75,
                rdcc_nslots=4_000,
            )

        nerf_densities = (
            torch.from_numpy(self.hdf5_file["/nerf_densities"][idx]).float()
            if self.nerf_densities is None
            else self.nerf_densities[idx]
        )
        nerf_densities_global_idx = (
            self.hdf5_file["/nerf_densities_global_idx"][idx]
            if self.nerf_densities_global_idx is None
            and "nerf_densities_global_idx" in self.hdf5_file
            else (
                self.nerf_densities_global_idx[idx]
                if self.nerf_densities_global_idx is not None
                else None
            )
        )
        nerf_densities_global = (
            torch.from_numpy(
                self.hdf5_file["/nerf_densities_global"][nerf_densities_global_idx]
            ).float()
            if self.nerf_densities_global is None
            and "nerf_densities_global" in self.hdf5_file
            and nerf_densities_global_idx is not None
            else (
                self.nerf_densities_global[nerf_densities_global_idx]
                if self.nerf_densities_global is not None
                and nerf_densities_global_idx is not None
                else None
            )
        )

        y_pick = (
            torch.from_numpy(np.array(self.hdf5_file["/y_pick"][idx])).float()
            if self.y_picks is None
            else self.y_picks[idx]
        )
        y_coll = (
            torch.from_numpy(np.array(self.hdf5_file["/y_coll"][idx])).float()
            if self.y_colls is None
            else self.y_colls[idx]
        )
        y_PGS = (
            torch.from_numpy(np.array(self.hdf5_file["/y_PGS"][idx])).float()
            if self.y_PGSs is None
            else self.y_PGSs[idx]
        )
        assert_equals(y_pick.shape, ())
        assert_equals(y_coll.shape, ())
        assert_equals(y_PGS.shape, ())

        # Consider thresholding y_* labels to be 0 or 1
        # Convert to float classes (N,) -> (N, 2)
        y_pick = torch.stack([1 - y_pick, y_pick], dim=-1)
        y_coll = torch.stack([1 - y_coll, y_coll], dim=-1)
        y_PGS = torch.stack([1 - y_PGS, y_PGS], dim=-1)

        grasp_transforms = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_transforms"][idx])).float()
            if self.grasp_transforms is None
            else self.grasp_transforms[idx]
        )

        nerf_config = (
            self.hdf5_file["/nerf_config"][idx]
            if self.nerf_configs is None
            else self.nerf_configs[idx]
        ).decode("utf-8")

        grasp_configs = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_configs"][idx])).float()
            if self.grasp_configs is None
            else self.grasp_configs[idx]
        )

        assert_equals(
            nerf_densities.shape,
            (self.NUM_FINGERS, self.NUM_PTS_X, self.NUM_PTS_Y, self.NUM_PTS_Z),
        )
        if nerf_densities_global is not None:
            assert_equals(
                nerf_densities_global.shape,
                (
                    NERF_DENSITIES_GLOBAL_NUM_X,
                    NERF_DENSITIES_GLOBAL_NUM_Y,
                    NERF_DENSITIES_GLOBAL_NUM_Z,
                ),
            )
        NUM_CLASSES = 2
        assert_equals(y_pick.shape, (NUM_CLASSES,))
        assert_equals(y_coll.shape, (NUM_CLASSES,))
        assert_equals(y_PGS.shape, (NUM_CLASSES,))
        assert_equals(grasp_transforms.shape, (self.NUM_FINGERS, 4, 4))
        assert_equals(grasp_configs.shape, (self.NUM_FINGERS, 7 + 16 + 4))

        return (
            nerf_densities,
            nerf_densities_global,
            y_pick,
            y_coll,
            y_PGS,
            grasp_transforms,
            nerf_config,
            grasp_configs,
        )

    @property
    def NUM_FINGERS(self) -> int:
        return self.fingertip_config.n_fingers

    @property
    def NUM_PTS_X(self) -> int:
        assert self.fingertip_config.num_pts_x is not None
        return self.fingertip_config.num_pts_x

    @property
    def NUM_PTS_Y(self) -> int:
        assert self.fingertip_config.num_pts_y is not None
        return self.fingertip_config.num_pts_y

    @property
    def NUM_PTS_Z(self) -> int:
        assert self.fingertip_config.num_pts_z is not None
        return self.fingertip_config.num_pts_z
