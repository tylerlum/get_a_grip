from __future__ import annotations

from typing import Tuple

import numpy as np

from get_a_grip.dataset_generation.utils.hand_model import HandModel


def get_joint_limits() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the joint limits for the hand model.
    """
    device = "cuda"
    hand_model = HandModel(device=device, n_surface_points=1000)
    joint_lower_limits, joint_upper_limits = (
        hand_model.joints_lower,
        hand_model.joints_upper,
    )
    assert joint_lower_limits.shape == joint_upper_limits.shape == (16,)

    return (
        joint_lower_limits.detach().cpu().numpy(),
        joint_upper_limits.detach().cpu().numpy(),
    )


def is_in_limits(joint_angles: np.ndarray) -> np.ndarray:
    N = joint_angles.shape[0]
    assert joint_angles.shape == (N, 16)

    joints_lower, joints_upper = get_joint_limits()
    assert joints_upper.shape == (16,)
    assert joints_lower.shape == (16,)

    in_limits = np.all(
        np.logical_and(
            joint_angles >= joints_lower[None, ...],
            joint_angles <= joints_upper[None, ...],
        ),
        axis=1,
    )
    assert in_limits.shape == (N,)
    return in_limits


def clamp_in_limits(joint_angles: np.ndarray) -> np.ndarray:
    N = joint_angles.shape[0]
    assert joint_angles.shape == (N, 16)

    joints_lower, joints_upper = get_joint_limits()
    assert joints_upper.shape == (16,)
    assert joints_lower.shape == (16,)

    joint_angles = np.clip(
        joint_angles, joints_lower[None, ...], joints_upper[None, ...]
    )
    return joint_angles
