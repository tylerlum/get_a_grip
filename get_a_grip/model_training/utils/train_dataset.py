from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pypose as pp
from nerf_grasping.config.fingertip_config import BaseFingertipConfig
from nerf_grasping.config.camera_config import CameraConfig
from nerf_grasping.dataset.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)


# Make atol and rtol larger than default to avoid errors due to floating point precision.
# Otherwise we get errors about invalid rotation matrices
PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class NeRFGrid_To_GraspSuccess_HDF5_Dataset(Dataset):
    def __init__(
        self,
        input_hdf5_filepath: str,
        fingertip_config: BaseFingertipConfig,
        max_num_data_points: Optional[int] = None,
        load_nerf_densities_in_ram: bool = False,
        load_nerf_densities_global_in_ram: bool = False,
        load_nerf_densities_global_idx_in_ram: bool = False,
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
            assert_equals(len(hdf5_file["/passed_simulation"].shape), 1)
            assert_equals(len(hdf5_file["/passed_penetration_threshold"].shape), 1)
            assert_equals(len(hdf5_file["/passed_eval"].shape), 1)
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
            self.passed_simulations = (
                torch.from_numpy(hdf5_file["/passed_simulation"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_penetration_thresholds = (
                torch.from_numpy(hdf5_file["/passed_penetration_threshold"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_evals = (
                torch.from_numpy(hdf5_file["/passed_eval"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                pp.from_matrix(
                    torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float(),
                    pp.SE3_type,
                    atol=PP_MATRIX_ATOL,
                    rtol=PP_MATRIX_RTOL,
                )
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

            self.object_scale = (
                torch.from_numpy(hdf5_file["/object_scale"][()]).float()
                if False
                else None
            )
            self.object_state = (
                torch.from_numpy(hdf5_file["/object_state"][()]).float()
                if False
                else None
            )

    def _set_length(
        self, hdf5_file: h5py.File, max_num_data_points: Optional[int]
    ) -> int:
        length = (
            hdf5_file.attrs["num_data_points"]
            if "num_data_points" in hdf5_file.attrs
            else hdf5_file["/passed_simulation"].shape[0]
        )
        if length != hdf5_file["/passed_simulation"].shape[0]:
            print(
                f"WARNING: num_data_points = {length} != passed_simulation.shape[0] = {hdf5_file['/passed_simulation'].shape[0]}"
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
            torch.from_numpy(self.hdf5_file["/nerf_densities_global"][nerf_densities_global_idx]).float()
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

        passed_simulation = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_simulation"][idx])
            ).float()
            if self.passed_simulations is None
            else self.passed_simulations[idx]
        )
        passed_penetration_threshold = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_penetration_threshold"][idx])
            ).float()
            if self.passed_penetration_thresholds is None
            else self.passed_penetration_thresholds[idx]
        )
        passed_eval = (
            torch.from_numpy(np.array(self.hdf5_file["/passed_eval"][idx])).float()
            if self.passed_evals is None
            else self.passed_evals[idx]
        )
        assert_equals(passed_simulation.shape, ())
        assert_equals(passed_penetration_threshold.shape, ())
        assert_equals(passed_eval.shape, ())

        # TODO: Consider thresholding passed_X labels to be 0 or 1
        # Convert to float classes (N,) -> (N, 2)
        passed_simulation = torch.stack(
            [1 - passed_simulation, passed_simulation], dim=-1
        )
        passed_penetration_threshold = torch.stack(
            [1 - passed_penetration_threshold, passed_penetration_threshold], dim=-1
        )
        passed_eval = torch.stack([1 - passed_eval, passed_eval], dim=-1)

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
        object_scale = (
            torch.from_numpy(self.hdf5_file["/object_scale"][idx : idx + 1]).float()
            if self.object_scale is None
            else self.object_scale[idx : idx + 1]
        )[0]
        object_state = (
            torch.from_numpy(self.hdf5_file["/object_state"][idx]).float()
            if self.object_state is None
            else self.object_state[idx]
        )

        assert_equals(
            nerf_densities.shape,
            (self.NUM_FINGERS, self.NUM_PTS_X, self.NUM_PTS_Y, self.NUM_PTS_Z),
        )
        if nerf_densities_global is not None:
            assert_equals(
                nerf_densities_global.shape,
                (NERF_DENSITIES_GLOBAL_NUM_X, NERF_DENSITIES_GLOBAL_NUM_Y, NERF_DENSITIES_GLOBAL_NUM_Z),
            )
        NUM_CLASSES = 2
        assert_equals(passed_simulation.shape, (NUM_CLASSES,))
        assert_equals(passed_penetration_threshold.shape, (NUM_CLASSES,))
        assert_equals(passed_eval.shape, (NUM_CLASSES,))
        assert_equals(grasp_transforms.shape, (self.NUM_FINGERS, 4, 4))
        assert_equals(grasp_configs.shape, (self.NUM_FINGERS, 7 + 16 + 4))
        assert_equals(object_scale.shape, ())
        assert_equals(object_state.shape, (13,))

        # BRITTLE: Assumes that object_state is given in a frame such that y=0 is the table surface
        object_y = object_state[1]
        object_y_wrt_table = object_y
        # assert object_y_wrt_table >= 0,  # May still be negative if object is below table surface, fell off
        if object_y_wrt_table < 0:
            if not hasattr(self, "num_warnings"):
                self.num_warnings = 0
            self.num_warnings += 1
            MAX_WARNINGS = 10
            if self.num_warnings < MAX_WARNINGS:
                print("!" * 80)
                print(f"WARNING: object_y_wrt_table = {object_y_wrt_table} < 0")
                print(f"object_state = {object_state}")
                print("!" * 80)
            elif self.num_warnings == MAX_WARNINGS:
                print("!" * 80)
                print(f"WARNING: Suppressing further warnings about object_y_wrt_table < 0")
                print("!" * 80)


        return (
            nerf_densities,
            nerf_densities_global,
            passed_simulation,
            passed_penetration_threshold,
            passed_eval,
            grasp_transforms,
            nerf_config,
            grasp_configs,
            object_y_wrt_table,
        )

    @property
    def NUM_FINGERS(self) -> int:
        return self.fingertip_config.n_fingers

    @property
    def NUM_PTS_X(self) -> int:
        return self.fingertip_config.num_pts_x

    @property
    def NUM_PTS_Y(self) -> int:
        return self.fingertip_config.num_pts_y

    @property
    def NUM_PTS_Z(self) -> int:
        return self.fingertip_config.num_pts_z


# %%
class DepthImage_To_GraspSuccess_HDF5_Dataset(Dataset):
    def __init__(
        self,
        input_hdf5_filepath: str,
        fingertip_config: BaseFingertipConfig,
        fingertip_camera_config: CameraConfig,
        max_num_data_points: Optional[int] = None,
        load_depth_images_in_ram: bool = False,
        load_grasp_labels_in_ram: bool = True,
        load_grasp_transforms_in_ram: bool = True,
        load_nerf_configs_in_ram: bool = True,
        load_grasp_configs_in_ram: bool = True,
    ) -> None:
        super().__init__()
        self.input_hdf5_filepath = input_hdf5_filepath
        self.fingertip_config = fingertip_config
        self.fingertip_camera_config = fingertip_camera_config

        # Recommended in https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/13
        self.hdf5_file = None

        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            self.len = self._set_length(
                hdf5_file=hdf5_file, max_num_data_points=max_num_data_points
            )

            # Check that the data is in the expected format
            assert_equals(len(hdf5_file["/passed_simulation"].shape), 1)
            assert_equals(len(hdf5_file["/passed_penetration_threshold"].shape), 1)
            assert_equals(len(hdf5_file["/passed_eval"].shape), 1)
            assert_equals(
                hdf5_file["/depth_images"].shape[1:],
                (
                    self.NUM_FINGERS,
                    self.DEPTH_IMAGE_HEIGHT,
                    self.DEPTH_IMAGE_WIDTH,
                ),
            )
            assert_equals(
                hdf5_file["/uncertainty_images"].shape[1:],
                (
                    self.NUM_FINGERS,
                    self.DEPTH_IMAGE_HEIGHT,
                    self.DEPTH_IMAGE_WIDTH,
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
            self.depth_uncertainty_images = (
                torch.stack(
                    [
                        torch.from_numpy(hdf5_file["/depth_images"][()]).float(),
                        torch.from_numpy(hdf5_file["/uncertainty_images"][()]).float(),
                    ],
                    dim=-3,
                )
                if load_depth_images_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.passed_simulations = (
                torch.from_numpy(hdf5_file["/passed_simulation"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_penetration_thresholds = (
                torch.from_numpy(hdf5_file["/passed_penetration_threshold"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_evals = (
                torch.from_numpy(hdf5_file["/passed_eval"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                pp.from_matrix(
                    torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float(),
                    pp.SE3_type,
                    atol=PP_MATRIX_ATOL,
                    rtol=PP_MATRIX_RTOL,
                )
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

            self.object_scale = (
                torch.from_numpy(hdf5_file["/object_scale"][()]).float()
                if False
                else None
            )
            self.object_state = (
                torch.from_numpy(hdf5_file["/object_state"][()]).float()
                if False
                else None
            )

    def _set_length(
        self, hdf5_file: h5py.File, max_num_data_points: Optional[int]
    ) -> int:
        length = (
            hdf5_file.attrs["num_data_points"]
            if "num_data_points" in hdf5_file.attrs
            else hdf5_file["/passed_simulation"].shape[0]
        )
        if length != hdf5_file["/passed_simulation"].shape[0]:
            print(
                f"WARNING: num_data_points = {length} != passed_simulation.shape[0] = {hdf5_file['/passed_simulation'].shape[0]}"
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

        depth_uncertainty_images = (
            torch.stack(
                [
                    torch.from_numpy(self.hdf5_file["/depth_images"][idx]).float(),
                    torch.from_numpy(
                        self.hdf5_file["/uncertainty_images"][idx]
                    ).float(),
                ],
                dim=-3,
            )
            if self.depth_uncertainty_images is None
            else self.depth_uncertainty_images[idx]
        )

        passed_simulation = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_simulation"][idx])
            ).float()
            if self.passed_simulations is None
            else self.passed_simulations[idx]
        )
        passed_penetration_threshold = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_penetration_threshold"][idx])
            ).float()
            if self.passed_penetration_thresholds is None
            else self.passed_penetration_thresholds[idx]
        )
        passed_eval = (
            torch.from_numpy(np.array(self.hdf5_file["/passed_eval"][idx])).float()
            if self.passed_evals is None
            else self.passed_evals[idx]
        )
        assert_equals(passed_simulation.shape, ())
        assert_equals(passed_penetration_threshold.shape, ())
        assert_equals(passed_eval.shape, ())

        # TODO: Consider thresholding passed_X labels to be 0 or 1
        # Convert to float classes (N,) -> (N, 2)
        passed_simulation = torch.stack(
            [1 - passed_simulation, passed_simulation], dim=-1
        )
        passed_penetration_threshold = torch.stack(
            [1 - passed_penetration_threshold, passed_penetration_threshold], dim=-1
        )
        passed_eval = torch.stack([1 - passed_eval, passed_eval], dim=-1)

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
        object_scale = (
            torch.from_numpy(self.hdf5_file["/object_scale"][idx : idx + 1]).float()
            if self.object_scale is None
            else self.object_scale[idx : idx + 1]
        )[0]
        object_state = (
            torch.from_numpy(self.hdf5_file["/object_state"][idx]).float()
            if self.object_state is None
            else self.object_state[idx]
        )

        assert_equals(
            depth_uncertainty_images.shape,
            (
                self.NUM_FINGERS,
                self.DEPTH_IMAGE_N_CHANNELS,
                self.DEPTH_IMAGE_HEIGHT,
                self.DEPTH_IMAGE_WIDTH,
            ),
        )
        NUM_CLASSES = 2
        assert_equals(passed_simulation.shape, (NUM_CLASSES,))
        assert_equals(passed_penetration_threshold.shape, (NUM_CLASSES,))
        assert_equals(passed_eval.shape, (NUM_CLASSES,))
        assert_equals(grasp_transforms.shape, (self.NUM_FINGERS, 4, 4))
        assert_equals(grasp_configs.shape, (self.NUM_FINGERS, 7 + 16 + 4))
        assert_equals(object_scale.shape, ())

        # BRITTLE: hardcoding buffer, exclude table thickness, so this is wrt table surface
        BUFFER = 1.2
        OBJ_MAX_EXTENT_FROM_ORIGIN = 1.0 * object_scale * BUFFER
        y_offset = OBJ_MAX_EXTENT_FROM_ORIGIN
        object_y = object_state[1]
        object_y_wrt_table = y_offset - object_y
        assert object_y_wrt_table >= 0

        return (
            depth_uncertainty_images,
            passed_simulation,
            passed_penetration_threshold,
            passed_eval,
            grasp_transforms,
            nerf_config,
            grasp_configs,
            object_y_wrt_table,
        )

    @property
    def NUM_FINGERS(self) -> int:
        return self.fingertip_config.n_fingers

    @property
    def DEPTH_IMAGE_HEIGHT(self) -> int:
        return self.fingertip_camera_config.H

    @property
    def DEPTH_IMAGE_WIDTH(self) -> int:
        return self.fingertip_camera_config.W

    @property
    def DEPTH_IMAGE_N_CHANNELS(self) -> int:
        return 2
