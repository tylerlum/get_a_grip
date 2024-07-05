from __future__ import annotations

import numpy as np
import pypose as pp
import torch

from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_grasp_orientations_from_z_dirs,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
    AllegroHandConfig,
    hand_config_to_hand_model,
)


def normalize_with_warning(v: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    B = v.shape[0]
    assert v.shape == (B, 3), f"Expected shape ({B}, 3), got {v.shape}"
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norm < atol):
        print("^" * 80)
        print(
            f"Warning: Found {np.sum(norm < atol)} vectors with norm less than {atol}"
        )
        print("^" * 80)
    return v / (norm + atol)


def rot6d_to_matrix(rot6d: np.ndarray, check: bool = True) -> np.ndarray:
    B = rot6d.shape[0]
    assert rot6d.shape == (B, 6), f"Expected shape ({B}, 6), got {rot6d.shape}"

    # Step 1: Reshape to (B, 3, 2)
    rot3x2 = rot6d.reshape(B, 3, 2)

    # Step 2: Normalize the first column
    col1 = rot3x2[:, :, 0]
    col1_normalized = normalize_with_warning(col1)

    # Step 3: Orthogonalize the second column with respect to the first column
    col2 = rot3x2[:, :, 1]
    dot_product = np.sum(col1_normalized * col2, axis=1, keepdims=True)
    col2_orthogonal = col2 - dot_product * col1_normalized

    # Step 4: Normalize the second column
    col2_normalized = normalize_with_warning(col2_orthogonal)

    # Step 5: Compute the cross product to obtain the third column
    col3 = np.cross(col1_normalized, col2_normalized)

    # Combine the columns to form the rotation matrix
    rotation_matrices = np.stack((col1_normalized, col2_normalized, col3), axis=-1)

    # Step 6: Check orthogonality and determinant
    if check:
        for i in range(B):
            mat = rotation_matrices[i]
            assert np.allclose(
                np.dot(mat.T, mat), np.eye(3), atol=1e-3
            ), f"Matrix {i} is not orthogonal, got {np.dot(mat.T, mat)}"
            assert np.allclose(
                np.linalg.det(mat), 1.0, atol=1e-3
            ), f"Matrix {i} does not have determinant 1, got {np.linalg.det(mat)}"

    assert rotation_matrices.shape == (
        B,
        3,
        3,
    ), f"Expected shape ({B}, 3, 3), got {rotation_matrices.shape}"
    return rotation_matrices


def compute_grasp_orientations(
    grasp_dirs: torch.Tensor,
    wrist_pose: pp.LieTensor,
    joint_angles: torch.Tensor,
) -> pp.LieTensor:
    B = grasp_dirs.shape[0]
    N_FINGERS = 4
    assert grasp_dirs.shape == (
        B,
        N_FINGERS,
        3,
    ), f"Expected shape ({B}, {N_FINGERS}, 3), got {grasp_dirs.shape}"
    assert wrist_pose.lshape == (B,), f"Expected shape ({B},), got {wrist_pose.lshape}"

    # Normalize
    z_dirs = grasp_dirs
    z_dirs = z_dirs / z_dirs.norm(dim=-1, keepdim=True)

    # Get hand model
    hand_config = AllegroHandConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=joint_angles,
    )
    hand_model = hand_config_to_hand_model(
        hand_config=hand_config,
    )

    grasp_orientations = compute_grasp_orientations_from_z_dirs(
        joint_angles_start=joint_angles,
        hand_model=hand_model,
        z_dirs=z_dirs,
    )
    grasp_orientations = pp.from_matrix(
        grasp_orientations,
        pp.SO3_type,
    )
    assert grasp_orientations.lshape == (
        B,
        N_FINGERS,
    ), f"Expected shape ({B}, {N_FINGERS}), got {grasp_orientations.lshape}"

    return grasp_orientations


def grasp_to_grasp_config(grasp: torch.Tensor) -> AllegroGraspConfig:
    device = grasp.device

    N_FINGERS = 4
    GRASP_DIM = 3 + 6 + 16 + 4 * 3
    x = grasp
    B = x.shape[0]

    assert x.shape == (
        B,
        GRASP_DIM,
    ), f"Expected shape ({B}, {GRASP_DIM}), got {x.shape}"
    trans = x[:, :3].detach().cpu().numpy()
    rot6d = x[:, 3:9].detach().cpu().numpy()
    joint_angles = x[:, 9:25].detach().cpu().numpy()
    grasp_dirs = x[:, 25:37].reshape(B, N_FINGERS, 3).detach().cpu().numpy()

    rot = rot6d_to_matrix(rot6d)

    wrist_pose_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1).float()
    wrist_pose_matrix[:, :3, :3] = torch.from_numpy(rot).float().to(device)
    wrist_pose_matrix[:, :3, 3] = torch.from_numpy(trans).float().to(device)

    wrist_pose = pp.from_matrix(
        wrist_pose_matrix.to(device),
        pp.SE3_type,
        atol=1e-3,
        rtol=1e-3,
    )

    assert wrist_pose.lshape == (B,)

    grasp_orientations = compute_grasp_orientations(
        grasp_dirs=torch.from_numpy(grasp_dirs).float().to(device),
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
    )

    # Convert to AllegroGraspConfig to dict
    grasp_configs = AllegroGraspConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
        grasp_orientations=grasp_orientations,
    )
    return grasp_configs


def grasp_config_to_grasp(grasp_config: AllegroGraspConfig) -> torch.Tensor:
    B = grasp_config.wrist_pose.lshape[0]
    N_FINGERS = 4
    GRASP_DIM = 3 + 6 + 16 + N_FINGERS * 3

    trans = grasp_config.wrist_pose.translation()
    rot = grasp_config.wrist_pose.rotation().matrix()
    joint_angles = grasp_config.joint_angles
    grasp_orientations = grasp_config.grasp_orientations.matrix()

    # Shape check
    B = trans.shape[0]
    assert trans.shape == (B, 3), f"Expected shape ({B}, 3), got {trans.shape}"
    assert rot.shape == (B, 3, 3), f"Expected shape ({B}, 3, 3), got {rot.shape}"
    assert joint_angles.shape == (
        B,
        16,
    ), f"Expected shape ({B}, 16), got {joint_angles.shape}"
    assert grasp_orientations.shape == (
        B,
        N_FINGERS,
        3,
        3,
    ), f"Expected shape ({B}, 3, 3), got {grasp_orientations.shape}"
    grasp_dirs = grasp_orientations[..., 2]
    grasps = torch.cat(
        [
            trans,
            rot[..., :2].reshape(B, 6),
            joint_angles,
            grasp_dirs.reshape(B, -1),
        ],
        dim=1,
    )
    assert grasps.shape == (
        B,
        GRASP_DIM,
    ), f"Expected shape ({B}, {GRASP_DIM}), got {grasps.shape}"
    return grasps
