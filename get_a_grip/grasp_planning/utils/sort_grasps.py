from __future__ import annotations

import pathlib
from typing import Dict, Optional, Tuple

import numpy as np

from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)


def get_sorted_grasps_from_file(
    optimized_grasp_config_dict_filepath: pathlib.Path,
    dist_move_finger: Optional[float] = None,
    dist_move_finger_backward: Optional[float] = None,
    error_if_no_loss: bool = True,
    check: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, AllegroGraspConfig
]:
    """
    This function processes optimized grasping configurations in preparation for hardware tests.

    It reads a given .npy file containing optimized grasps, computes target and pre joint angles for each grasp, and sorts these grasps based on a pre-computed grasp metric, with the most favorable grasp appearing first in the batch dimension.

    Parameters:
    optimized_grasp_config_dict_filepath (pathlib.Path): The file path to the optimized grasp .npy file. This file should contain wrist poses, joint angles, grasp orientations, and loss from grasp metric.
    dist_move_finger (Optional[float]): The distance to move fingers for target joint angles. Defaults to None, which means default distance.
    dist_move_finger_backward (Optional[float]): The distance to move fingers backwards for pre joint angles. Defaults to None, which means default distance.
    error_if_no_loss (bool): Whether to raise an error if the loss is not found in the grasp config dict. Defaults to True.
    check (bool): Whether to check the validity of the grasp configurations (sometimes sensitive or off manifold from optimization?). Defaults to True.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, AllegroGraspConfig]:
    - A batch of wrist transformations of object yup frame wrt nerf frame in a numpy array of shape (B, 4, 4), representing pose in nerf frame (avoid quat to be less ambiguous about order)
    - A batch of joint angles in a numpy array of shape (B, 16)
    - A batch of target joint angles in a numpy array of shape (B, 16)
    - A batch of pre joint angles in a numpy array of shape (B, 16)
    - A batch of sorted losses in a numpy array of shape (B,)
    - A batch of grasp configs of batch size B

    Example:
    >>> X_Oy_H_array, joint_angles_array, target_joint_angles_array, pre_joint_angles_array, sorted_losses, sorted_grasp_configs = get_sorted_grasps_from_file(pathlib.Path("path/to/optimized_grasp_config.npy"))
    >>> B = X_Oy_H_array.shape[0]
    >>> assert X_Oy_H_array.shape == (B, 4, 4)
    >>> assert joint_angles_array.shape == (B, 16)
    >>> assert target_joint_angles_array.shape == (B, 16)
    >>> assert pre_joint_angles_array.shape == (B, 16)
    >>> assert sorted_losses.shape == (B,)
    >>> assert len(sorted_grasp_configs) == B
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
    )


def get_sorted_grasps_from_dict(
    optimized_grasp_config_dict: Dict[str, np.ndarray],
    dist_move_finger: Optional[float] = None,
    dist_move_finger_backward: Optional[float] = None,
    error_if_no_loss: bool = True,
    check: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, AllegroGraspConfig
]:
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
        sorted_grasp_configs,
    )
