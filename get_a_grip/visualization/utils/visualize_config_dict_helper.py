from typing import Any, Dict, Optional, Union

import numpy as np
import plotly.graph_objects as go
import torch

from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_fingertip_mean_contact_positions,
    compute_fingertip_targets,
    compute_optimized_joint_angle_targets_given_fingertip_targets,
)
from get_a_grip.dataset_generation.utils.object_model import ObjectModel
from get_a_grip.dataset_generation.utils.pose_conversion import hand_config_np_to_pose


def get_scene_dict() -> Dict[str, Any]:
    return dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    )


def get_yup_camera() -> Dict[str, Any]:
    return dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.0, y=1.0, z=0.0),
    )


def get_hand_config_dict_plotly_data_list(
    hand_model: HandModel,
    hand_pose: torch.Tensor,
    hand_pose_start: Optional[torch.Tensor],
) -> list:
    if hand_pose_start is not None:
        hand_model.set_parameters(hand_pose_start)
        hand_start_plotly = hand_model.get_plotly_data(
            i=0,
            opacity=0.5,
            color="lightblue",
            with_contact_points=False,
            with_contact_candidates=True,
            with_surface_points=True,
            with_penetration_keypoints=True,
        )
    else:
        hand_start_plotly = []

    hand_model.set_parameters(hand_pose)
    hand_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=1,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=True,
        with_surface_points=True,
        with_penetration_keypoints=False,
    )

    return hand_start_plotly + hand_plotly


def get_grasp_config_dict_plotly_data_list(
    hand_model: HandModel,
    hand_pose: torch.Tensor,
    config_dict: Dict[str, np.ndarray],
    idx_to_visualize: int,
    device: Union[str, torch.device],
) -> list:
    if "grasp_orientations" not in config_dict:
        print(
            "This is not a grasp_config_dict, skipping get_grasp_config_dict_plotly_data_list"
        )
        return []

    wrist_pose = hand_pose[:, :9]
    joint_angles = hand_pose[:, 9:]

    # fingertips
    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (1, hand_model.num_fingers, 3)
    fingertips_plotly = [
        go.Scatter3d(
            x=fingertip_mean_positions[0, :, 0],
            y=fingertip_mean_positions[0, :, 1],
            z=fingertip_mean_positions[0, :, 2],
            mode="markers",
            marker=dict(size=7, color="goldenrod"),
            name="fingertip mean positions",
        ),
    ]

    # fingertip targets
    grasp_orientations = torch.tensor(
        config_dict["grasp_orientations"], dtype=torch.float, device=device
    )[idx_to_visualize]
    assert grasp_orientations.shape == (hand_model.num_fingers, 3, 3)
    fingertip_targets = compute_fingertip_targets(
        joint_angles_start=joint_angles,
        hand_model=hand_model,
        grasp_orientations=grasp_orientations.unsqueeze(dim=0),
    )
    assert fingertip_targets.shape == (1, hand_model.num_fingers, 3)
    fingertip_targets_plotly = [
        go.Scatter3d(
            x=fingertip_targets[0, :, 0],
            y=fingertip_targets[0, :, 1],
            z=fingertip_targets[0, :, 2],
            mode="markers",
            marker=dict(size=10, color="magenta"),
            name="fingertip targets",
        ),
    ]

    # grasp_orientations
    grasp_orientations_plotly = []
    for i in range(hand_model.num_fingers):
        origin = fingertip_mean_positions[0, i]
        line_length = 0.01
        x_dir = grasp_orientations[i, :, 0] * line_length
        y_dir = grasp_orientations[i, :, 1] * line_length
        z_dir = grasp_orientations[i, :, 2] * line_length
        grasp_orientations_plotly += [
            go.Scatter3d(
                x=[origin[0], origin[0] + x_dir[0]],
                y=[origin[1], origin[1] + x_dir[1]],
                z=[origin[2], origin[2] + x_dir[2]],
                mode="lines",
                marker=dict(size=5, color="red"),
                name=f"x_dir for finger {i}",
            ),
            go.Scatter3d(
                x=[origin[0], origin[0] + y_dir[0]],
                y=[origin[1], origin[1] + y_dir[1]],
                z=[origin[2], origin[2] + y_dir[2]],
                mode="lines",
                marker=dict(size=5, color="green"),
                name=f"y_dir for finger {i}",
            ),
            go.Scatter3d(
                x=[origin[0], origin[0] + z_dir[0]],
                y=[origin[1], origin[1] + z_dir[1]],
                z=[origin[2], origin[2] + z_dir[2]],
                mode="lines",
                marker=dict(size=5, color="blue"),
                name=f"z_dir for finger {i}",
            ),
        ]

    # joint angle targets
    (
        joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_fingertip_targets(
        joint_angles_start=joint_angles,
        hand_model=hand_model,
        fingertip_targets=fingertip_targets,
    )
    hand_pose_target = torch.cat(
        [
            wrist_pose,
            joint_angle_targets,
        ],
        dim=1,
    )
    hand_model.set_parameters(hand_pose_target)
    hand_target_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=0.5,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=False,
        with_surface_points=False,
        with_penetration_keypoints=False,
    )
    return (
        fingertips_plotly
        + fingertip_targets_plotly
        + grasp_orientations_plotly
        + hand_target_plotly
    )


def create_config_dict_fig(
    config_dict: Dict[str, Any],
    hand_model: HandModel,
    object_model: Optional[ObjectModel],
    skip_visualize_qpos_start: bool,
    skip_visualize_grasp_config_dict: bool,
    title: str,
    idx_to_visualize: int,
) -> go.Figure:
    if object_model is not None:
        object_plotly = object_model.get_plotly_data(
            i=0,
            color="lightgreen",
            opacity=0.5,
            with_surface_points=True,
            with_table=True,
        )
    else:
        object_plotly = []

    # hand pose
    hand_pose = (
        hand_config_np_to_pose(
            trans=config_dict["trans"],
            rot=config_dict["rot"],
            joint_angles=config_dict["joint_angles"],
        )[idx_to_visualize]
        .to(hand_model.device)
        .unsqueeze(0)
    )

    # hand pose start
    if "joint_angles_start" in config_dict and not skip_visualize_qpos_start:
        hand_pose_start = hand_config_np_to_pose(
            trans=config_dict["trans_start"],
            rot=config_dict["rot_start"],
            joint_angles=config_dict["joint_angles_start"],
        ).to(hand_model.device)
    else:
        hand_pose_start = None

    # hand config dict
    hand_config_dict_plotly_data_list = get_hand_config_dict_plotly_data_list(
        hand_model=hand_model,
        hand_pose=hand_pose,
        hand_pose_start=hand_pose_start,
    )

    # grasp config dict
    if not skip_visualize_grasp_config_dict:
        # Slowest part of this function
        grasp_config_dict_plotly_data_list = get_grasp_config_dict_plotly_data_list(
            hand_model=hand_model,
            hand_pose=hand_pose,
            config_dict=config_dict,
            idx_to_visualize=idx_to_visualize,
            device=hand_model.device,
        )
    else:
        grasp_config_dict_plotly_data_list = []

    # Create fig
    fig = go.Figure(
        data=(
            object_plotly
            + hand_config_dict_plotly_data_list
            + grasp_config_dict_plotly_data_list
        )
    )

    # y_PGS
    if "y_PGS" in config_dict:
        y_PGS = config_dict["y_PGS"][idx_to_visualize]
        y_PGS_str = f"y_PGS: {y_PGS:.2f}"
        fig.add_annotation(text=y_PGS_str, x=0.5, y=0.05, xref="paper", yref="paper")
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {y_PGS_str}"

    if "y_coll" in config_dict:
        y_coll = config_dict["y_coll"][idx_to_visualize]
        y_coll_str = f"y_coll: {y_coll:.2f}"
        fig.add_annotation(
            text=y_coll_str,
            x=0.5,
            y=0.1,
            xref="paper",
            yref="paper",
        )
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {y_coll_str}"

    if "y_pick" in config_dict:
        y_pick = config_dict["y_pick"][idx_to_visualize]
        y_pick_str = f"y_pick: {y_pick:.2f}"
        fig.add_annotation(text=y_pick_str, x=0.5, y=0.2, xref="paper", yref="paper")
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {y_pick_str}"

    # loss
    if "loss" in config_dict:
        loss = round(config_dict["loss"][idx_to_visualize], 3)
        y_PGS_pred = 1 - loss
        predicted_score_str = f"y_PGS_pred: {y_PGS_pred:.2f}"
        fig.add_annotation(
            text=predicted_score_str, x=0.5, y=0.25, xref="paper", yref="paper"
        )
        title += f" | {predicted_score_str}"

    fig.update_layout(
        title=title,
        scene=get_scene_dict(),
        scene_camera=get_yup_camera(),
    )
    return fig
