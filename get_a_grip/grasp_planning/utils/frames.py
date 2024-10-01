from dataclasses import dataclass
from functools import cached_property

import numpy as np
import trimesh


@dataclass
class GraspPlanningFrames:
    """
    Frames:
      * N: Nerf frame, can be either z-up or y-up (depending on nerf_is_z_up), that is at the surface of the table
      * O: Object frame, can be either z-up or y-up (depending on nerf_is_z_up), that is at the centroid of the object
      * B: Base frame, can be either z-up or y-up (depending on nerf_is_z_up), that is at the horizontal position of the object, but vertical position of the table surface
      * Ny: Nerf frame with y-up
      * Oy: Object frame with y-up
      * By: Object frame with y-up

    Transforms: X_A_B = X_A_C @ X_C_B, p_A = X_A_B @ p_B
      * X_N_O: O frame wrt N frame. These have the same 0 orientation. The translation is simply the centroid of the object in N frame
      * X_N_B: B frame wrt N frame. These have the same 0 orientation. The translation is simply the horizontal position of the object in N frame with no vertical component
      * X_O_Oy, X_N_Ny, X_B_By: These have the same 0 translation. The orientation is either a 90 degree rotation about x axis or identity depending on whether the frame is z-up or y-up
      * X_By_Oy: This only has a vertical component along y, which is the distance between the centroid of the object and the table surface
      * X_B_O: This only has a vertical component along either z or y (depending on nerf_is_z_up), which is the distance between the centroid of the object and the table surface

    Information:
      * N frame is the frame used to captured the NeRF
      * By frame is the frame that is used for computing the BPS (N frame during data_generation is same as By)
      * Oy frame is the frame that is used for computing the NeRF densities and the grasp wrt the object

    With 6 unique frames, there are C(6, 2) = 5 + 4 + 3 + 2 + 1 = 15 unique pairs of frames, so 30 unique transforms
    To figure out how many transforms we need to define, we can view this as a graph with n = 6 nodes
    The minimum number of edges needed to connect all nodes is n-1 = 5 nodes

    To easily compute the transform from any frame A to frame B, we first compute all transforms from O to all other frames
    Then provide a function to compute the transform from arbitrary frames A to B
    """

    nerf_is_z_up: bool
    object_centroid_N_x: float
    object_centroid_N_y: float
    object_centroid_N_z: float

    @property
    def object_centroid_N(self) -> np.ndarray:
        return np.array(
            [
                self.object_centroid_N_x,
                self.object_centroid_N_y,
                self.object_centroid_N_z,
            ]
        )

    ############### ROTATION START ###############
    @cached_property
    def X_O_Oy(self) -> np.ndarray:
        return (
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            if self.nerf_is_z_up
            else np.eye(4)
        )

    @cached_property
    def X_B_By(self) -> np.ndarray:
        return self.X_O_Oy

    @cached_property
    def X_N_Ny(self) -> np.ndarray:
        return self.X_O_Oy

    ############### ROTATION END ###############

    ############### TRANSLATION START ###############
    @cached_property
    def X_N_O(self) -> np.ndarray:
        X_N_O = trimesh.transformations.translation_matrix(self.object_centroid_N)
        return X_N_O

    @cached_property
    def X_By_Oy(self) -> np.ndarray:
        centroid_vertical_component = (
            self.object_centroid_N[2]
            if self.nerf_is_z_up
            else self.object_centroid_N[1]
        )

        return trimesh.transformations.translation_matrix(
            [0, centroid_vertical_component, 0]
        )

    ############### TRANSLATION END ###############

    ############### FROM O TRANSFORMS START ###############
    @cached_property
    def X_O_N(self) -> np.ndarray:
        return np.linalg.inv(self.X_N_O)

    @cached_property
    def X_O_Ny(self) -> np.ndarray:
        return self.X_O_N @ self.X_N_Ny

    @cached_property
    def X_O_By(self) -> np.ndarray:
        X_Oy_By = np.linalg.inv(self.X_By_Oy)
        return self.X_O_Oy @ X_Oy_By

    @cached_property
    def X_O_B(self) -> np.ndarray:
        X_By_B = np.linalg.inv(self.X_B_By)
        return self.X_O_By @ X_By_B

    @cached_property
    def X_O_O(self) -> np.ndarray:
        return np.eye(4)

    ############### FROM O TRANSFORMS END ###############

    def X_(self, A: str, B: str) -> np.ndarray:
        X_O_A = getattr(self, f"X_O_{A}")
        X_O_B = getattr(self, f"X_O_{B}")
        X_A_O = np.linalg.inv(X_O_A)
        X_A_B = X_A_O @ X_O_B
        return X_A_B

    @cached_property
    def available_frames(self) -> list:
        return ["N", "O", "B", "Ny", "Oy", "By"]


@dataclass
class GraspMotionPlanningFrames(GraspPlanningFrames):
    """
    grasp_motion_planning
    * nerf_is_z_up = True

    Frames
    * W: World frame, only used for grasp_motion_planning. The origin is at the robot base with z-up and x-forward from robot.
    * Nz: Nerf frame with z-up. the origin is defined at <nerf_frame_offset_W> wrt W with z-up and x-forward from robot.

    Transforms: X_A_B = X_A_C @ X_C_B, p_A = X_A_B @ p_B
    * X_W_Nz: Nz frame wrt W frame. These have the same 0 orientation. The translation is simply the <nerf_frame_offset_W>
    """

    nerf_frame_offset_W_x: float
    nerf_frame_offset_W_y: float
    nerf_frame_offset_W_z: float

    @property
    def nerf_frame_offset_W(self) -> np.ndarray:
        return np.array(
            [
                self.nerf_frame_offset_W_x,
                self.nerf_frame_offset_W_y,
                self.nerf_frame_offset_W_z,
            ]
        )

    ############### TRANSLATION START ###############
    @cached_property
    def X_W_Nz(self) -> np.ndarray:
        X_W_Nz = trimesh.transformations.translation_matrix(self.nerf_frame_offset_W)
        return X_W_Nz

    @cached_property
    def X_Nz_N(self) -> np.ndarray:
        return (
            np.eye(4)
            if self.nerf_is_z_up
            else trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )

    ############### TRANSLATION END ###############

    ############### FROM O TRANSFORMS START ###############
    @cached_property
    def X_O_Nz(self) -> np.ndarray:
        X_N_Nz = np.linalg.inv(self.X_Nz_N)
        return self.X_O_N @ X_N_Nz

    @cached_property
    def X_O_W(self) -> np.ndarray:
        X_Nz_W = np.linalg.inv(self.X_W_Nz)
        return self.X_O_Nz @ X_Nz_W

    ############### FROM O TRANSFORMS END ###############

    @cached_property
    def available_frames(self) -> list:
        return ["W", "Nz"] + super().available_frames
