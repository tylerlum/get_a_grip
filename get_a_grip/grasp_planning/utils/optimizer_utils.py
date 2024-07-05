from __future__ import annotations

import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import pypose as pp
import torch

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig


def sample_random_rotate_transforms_only_around_y(N: int) -> pp.LieTensor:
    PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # TODO: Figure out better uniform sampling on SO(3).
    x_rotations = torch.zeros(N)
    y_rotations = 4 * torch.pi * (2 * torch.rand(N) - 1)
    z_rotations = torch.zeros(N)
    xyz_rotations = torch.stack([x_rotations, y_rotations, z_rotations], dim=-1)
    log_random_rotations = pp.so3(xyz_rotations)

    # Return exponentiated rotations.
    random_SO3_rotations = log_random_rotations.Exp()

    # A bit annoying -- need to cast SO(3) -> SE(3).
    random_rotate_transforms = pp.from_matrix(
        random_SO3_rotations.matrix(),
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    return random_rotate_transforms


def get_sorted_grasps_from_file(
    optimized_grasp_config_dict_filepath: pathlib.Path,
    dist_move_finger: Optional[float] = None,
    dist_move_finger_backward: Optional[float] = None,
    error_if_no_loss: bool = True,
    check: bool = True,
    print_best: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function processes optimized grasping configurations in preparation for hardware tests.

    It reads a given .npy file containing optimized grasps, computes target and pre joint angles for each grasp, and sorts these grasps based on a pre-computed grasp metric, with the most favorable grasp appearing first in the batch dimension.

    Parameters:
    optimized_grasp_config_dict_filepath (pathlib.Path): The file path to the optimized grasp .npy file. This file should contain wrist poses, joint angles, grasp orientations, and loss from grasp metric.
    dist_move_finger (Optional[float]): The distance to move fingers for target joint angles. Defaults to None, which means default distance.
    dist_move_finger_backward (Optional[float]): The distance to move fingers backwards for pre joint angles. Defaults to None, which means default distance.
    error_if_no_loss (bool): Whether to raise an error if the loss is not found in the grasp config dict. Defaults to True.
    check (bool): Whether to check the validity of the grasp configurations (sometimes sensitive or off manifold from optimization?). Defaults to True.
    print_best (bool): Whether to print the best grasp configurations. Defaults to True.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    - A batch of wrist transformations of object yup frame wrt nerf frame in a numpy array of shape (B, 4, 4), representing pose in nerf frame (avoid quat to be less ambiguous about order)
    - A batch of joint angles in a numpy array of shape (B, 16)
    - A batch of target joint angles in a numpy array of shape (B, 16)
    - A batch of pre joint angles in a numpy array of shape (B, 16)
    - A batch of sorted losses in a numpy array of shape (B,)

    Example:
    >>> X_Oy_H_array, joint_angles_array, target_joint_angles_array, pre_joint_angles_array, sorted_losses = get_sorted_grasps_from_file(pathlib.Path("path/to/optimized_grasp_config.npy"))
    >>> B = X_Oy_H_array.shape[0]
    >>> assert X_Oy_H_array.shape == (B, 4, 4)
    >>> assert joint_angles_array.shape == (B, 16)
    >>> assert target_joint_angles_array.shape == (B, 16)
    >>> assert pre_joint_angles_array.shape == (B, 16)
    >>> assert sorted_losses.shape == (B,)
    """
    # Read in
    grasp_config_dict = np.load(
        optimized_grasp_config_dict_filepath, allow_pickle=True
    ).item()
    return get_sorted_grasps_from_dict(
        grasp_config_dict,
        dist_move_finger=dist_move_finger,
        dist_move_finger_backward=dist_move_finger_backward,
        error_if_no_loss=error_if_no_loss,
        check=check,
        print_best=print_best,
    )


def get_sorted_grasps_from_dict(
    optimized_grasp_config_dict: Dict[str, np.ndarray],
    dist_move_finger: Optional[float] = None,
    dist_move_finger_backward: Optional[float] = None,
    error_if_no_loss: bool = True,
    check: bool = True,
    print_best: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
        optimized_grasp_config_dict, check=check
    )
    B = len(grasp_configs)

    # Look for loss or y_PGS
    if "loss" not in optimized_grasp_config_dict:
        if error_if_no_loss:
            raise ValueError(
                f"loss not found in grasp config dict keys: {optimized_grasp_config_dict.keys()}, if you want to skip this error, set error_if_no_loss=False"
            )
        print("=" * 80)
        print(
            f"loss not found in grasp config dict keys: {optimized_grasp_config_dict.keys()}"
        )
        print("Looking for y_PGS...")
        print("=" * 80 + "\n")
        if "y_PGS" in optimized_grasp_config_dict:
            print("~" * 80)
            print("y_PGS found! Using 1 - y_PGS as loss.")
            print("~" * 80 + "\n")
            failed_eval = 1 - optimized_grasp_config_dict["y_PGS"]
            losses = failed_eval
        else:
            print("~" * 80)
            print("y_PGS not found! Using dummy losses.")
            print("~" * 80 + "\n")
            dummy_losses = np.arange(B)
            losses = dummy_losses
    else:
        losses = optimized_grasp_config_dict["loss"]

    # Sort by loss
    sorted_idxs = np.argsort(losses)
    sorted_losses = losses[sorted_idxs]
    sorted_grasp_configs = grasp_configs[sorted_idxs]

    if print_best:
        BEST_K = 2
        print(f"Best grasp configs: {sorted_grasp_configs[:BEST_K]}")
        print(f"Best grasp losses: {sorted_losses[:BEST_K]}")

    wrist_trans_array = (
        sorted_grasp_configs.wrist_pose.translation().detach().cpu().numpy()
    )
    wrist_rot_array = (
        sorted_grasp_configs.wrist_pose.rotation().matrix().detach().cpu().numpy()
    )
    joint_angles_array = sorted_grasp_configs.joint_angles.detach().cpu().numpy()
    target_joint_angles_array = (
        sorted_grasp_configs.target_joint_angles(dist_move_finger=dist_move_finger)
        .detach()
        .cpu()
        .numpy()
    )
    pre_joint_angles_array = (
        sorted_grasp_configs.pre_joint_angles(
            dist_move_finger_backwards=dist_move_finger_backward
        )
        .detach()
        .cpu()
        .numpy()
    )

    assert wrist_trans_array.shape == (B, 3)
    assert wrist_rot_array.shape == (B, 3, 3)
    assert joint_angles_array.shape == (B, 16)
    assert target_joint_angles_array.shape == (B, 16)
    assert pre_joint_angles_array.shape == (B, 16)
    assert sorted_losses.shape == (B,)

    # Put into transforms X_Oy_H_array
    X_Oy_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
    assert X_Oy_H_array.shape == (B, 4, 4)
    X_Oy_H_array[:, :3, :3] = wrist_rot_array
    X_Oy_H_array[:, :3, 3] = wrist_trans_array

    return (
        X_Oy_H_array,
        joint_angles_array,
        target_joint_angles_array,
        pre_joint_angles_array,
        sorted_losses,
    )


def main() -> None:
    FILEPATH = pathlib.Path("OUTPUT.npy")
    assert FILEPATH.exists(), f"Filepath {FILEPATH} does not exist"

    print(f"Processing {FILEPATH}")

    try:
        wrist_trans, wrist_rot, joint_angles, target_joint_angles, pre_joint_angles = (
            get_sorted_grasps_from_file(FILEPATH)
        )
    except ValueError as e:
        print(f"Error processing {FILEPATH}: {e}")
        print("Try again skipping check")
        wrist_trans, wrist_rot, joint_angles, target_joint_angles, pre_joint_angles = (
            get_sorted_grasps_from_file(FILEPATH, check=False)
        )
    print(
        f"Found wrist_trans.shape = {wrist_trans.shape}, wrist_rot.shape = {wrist_rot.shape}, joint_angles.shape = {joint_angles.shape}, target_joint_angles.shape = {target_joint_angles.shape}"
    )


if __name__ == "__main__":
    main()
