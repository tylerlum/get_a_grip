from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pypose as pp
import pytorch_kinematics as pk
import torch
from pytorch_kinematics.chain import Chain

from get_a_grip import get_assets_folder
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_FILE,
)
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_grasp_orientations_from_z_dirs,
    compute_init_joint_angles_given_grasp_orientations,
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from get_a_grip.dataset_generation.utils.pose_conversion import (
    hand_config_to_pose,
)
from get_a_grip.grasp_planning.utils.joint_limit_utils import (
    clamp_in_limits,
    get_joint_limits_np,
)

Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float32)

FINGERTIP_LINK_NAMES = [
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
]


def load_allegro() -> Chain:
    allegro_path = str(get_assets_folder() / ALLEGRO_HAND_FILE)
    return pk.build_chain_from_urdf(open(allegro_path).read())


class AllegroHandConfig(torch.nn.Module):
    """
    A container specifying a batch of configurations for an Allegro hand, i.e., the
    wrist pose and the joint configurations.
    """

    def __init__(
        self,
        batch_size: int = 1,
        chain: Chain = load_allegro(),
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        self.chain = chain
        self.wrist_pose = pp.Parameter(
            pp.randn_SE3(batch_size), requires_grad=requires_grad
        )
        self.joint_angles = torch.nn.Parameter(
            torch.zeros(batch_size, 16), requires_grad=requires_grad
        )
        self.batch_size = batch_size

    @classmethod
    def from_values(
        cls,
        wrist_pose: pp.LieTensor,
        joint_angles: torch.Tensor,
        chain: Chain = load_allegro(),
        requires_grad: bool = True,
    ) -> AllegroHandConfig:
        """
        Factory method to create an AllegroHandConfig from a wrist pose and joint angles.
        """
        batch_size = wrist_pose.shape[0]
        assert wrist_pose.shape == (batch_size, 7)
        assert joint_angles.shape == (batch_size, 16)

        hand_config = cls(batch_size, chain, requires_grad).to(
            device=wrist_pose.device, dtype=wrist_pose.dtype
        )
        hand_config.set_wrist_pose(wrist_pose)
        hand_config.set_joint_angles(joint_angles)
        return hand_config

    @classmethod
    def from_hand_config_dict(
        cls, hand_config_dict: Dict[str, Any], check: bool = True
    ) -> AllegroHandConfig:
        trans = torch.from_numpy(hand_config_dict["trans"]).float()
        rot = torch.from_numpy(hand_config_dict["rot"]).float()
        joint_angles = torch.from_numpy(hand_config_dict["joint_angles"]).float()
        batch_size = trans.shape[0]
        assert trans.shape == (batch_size, 3)
        assert rot.shape == (batch_size, 3, 3)
        assert joint_angles.shape == (batch_size, 16)

        wrist_translation = trans

        wrist_quat = pp.from_matrix(
            rot,
            pp.SO3_type,
            check=check,
            # Set atol and rtol to be a bit larger than default to handle large matrices
            # (numerical errors larger affect the sanity checking)
            atol=1e-3,
            rtol=1e-3,
        )
        wrist_pose = pp.SE3(torch.cat([wrist_translation, wrist_quat], dim=1))

        return cls.from_values(wrist_pose=wrist_pose, joint_angles=joint_angles)

    @classmethod
    def from_tensor(cls, hand_config_tensor: torch.Tensor) -> AllegroHandConfig:
        """
        Expects a tensor of shape [batch_size, 23]
        with all config parameters.
        """
        B = hand_config_tensor.shape[0]
        assert hand_config_tensor.shape == (
            B,
            23,
        ), f"Expected shape ({B}, 23), got {hand_config_tensor.shape}"
        wrist_pose = pp.SE3(hand_config_tensor[:, :7]).to(
            device=hand_config_tensor.device
        )
        joint_angles = hand_config_tensor[:, 7:]
        return cls.from_values(
            wrist_pose=wrist_pose,
            joint_angles=joint_angles,
        )

    def set_wrist_pose(self, wrist_pose: pp.LieTensor) -> None:
        assert (
            wrist_pose.shape == self.wrist_pose.shape
        ), f"New wrist pose, shape {wrist_pose.shape} does not match current wrist pose shape {self.wrist_pose.shape}"
        self.wrist_pose.data = wrist_pose.data.clone()

    def set_joint_angles(self, joint_angles: torch.Tensor) -> None:
        assert (
            joint_angles.shape == self.joint_angles.shape
        ), f"New hand config, shape {joint_angles.shape}, does not match shape of current hand config, {self.joint_angles.shape}."
        self.joint_angles.data = joint_angles

        # Clamp
        joint_lower_limits, joint_upper_limits = get_joint_limits_np(
            hand_model_type=self.hand_model_type
        )
        self.joint_lower_limits, self.joint_upper_limits = (
            torch.from_numpy(joint_lower_limits).float().to(self.device),
            torch.from_numpy(joint_upper_limits).float().to(self.device),
        )

        assert self.joint_lower_limits.shape == (16,)
        assert self.joint_upper_limits.shape == (16,)

        self.joint_angles.data = torch.clamp(
            self.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )

    def get_fingertip_transforms(self) -> pp.LieTensor:
        self.chain = self.chain.to(device=self.device)

        # Run batched FK from current hand config.
        link_poses_hand_frame = self.chain.forward_kinematics(self.joint_angles)

        # Pull out fingertip poses + cast to PyPose.
        fingertip_poses = [link_poses_hand_frame[ln] for ln in FINGERTIP_LINK_NAMES]
        fingertip_pyposes = [
            pp.from_matrix(fp.get_matrix(), pp.SE3_type) for fp in fingertip_poses
        ]

        # Apply wrist transformation to get world-frame fingertip poses.
        return torch.stack(
            [self.wrist_pose @ fp for fp in fingertip_pyposes], dim=1
        )  # shape [B, batch_size, 7]

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns a hand config dict
        """
        trans = self.wrist_pose.translation().detach().cpu().numpy()
        rot = self.wrist_pose.rotation().matrix().detach().cpu().numpy()
        joint_angles = self.joint_angles.detach().cpu().numpy()
        return {
            "trans": trans,
            "rot": rot,
            "joint_angles": joint_angles,
        }

    def as_tensor(self) -> torch.Tensor:
        """
        Returns a tensor of shape [batch_size, 23]
        with all config parameters.
        """
        return torch.cat((self.wrist_pose.tensor(), self.joint_angles), dim=-1)

    def as_hand_model(
        self,
        n_surface_points: int = 0,
    ) -> HandModel:
        """
        Convert an AllegroHandConfig to a HandModel.
        """
        device = self.device
        translation = self.wrist_pose.translation()
        rotation = self.wrist_pose.rotation().matrix()
        joint_angles = self.joint_angles
        hand_model = HandModel(
            hand_model_type=self.hand_model_type,
            device=device,
            n_surface_points=n_surface_points,
        )
        hand_pose = hand_config_to_pose(
            trans=translation, rot=rotation, joint_angles=joint_angles
        ).to(device)
        hand_model.set_parameters(hand_pose)
        return hand_model

    def __repr__(self) -> str:
        wrist_pose_repr = np.array2string(
            self.wrist_pose.data.cpu().numpy(), separator=", "
        )
        joint_angles_repr = np.array2string(
            self.joint_angles.data.cpu().numpy(), separator=", "
        )
        repr_parts = [
            "AllegroHandConfig(",
            f"  batch_size={self.batch_size},",
            "  wrist_pose=(",
            f"{wrist_pose_repr}",
            "  ),",
            "  joint_angles=(",
            f"{joint_angles_repr}",
            "  ),",
            ")",
        ]
        return "\n".join(repr_parts)

    def __len__(self) -> int:
        return self.batch_size

    @property
    def device(self) -> torch.device:
        return self.wrist_pose.device

    @property
    def hand_model_type(self) -> HandModelType:
        return HandModelType.ALLEGRO


class AllegroGraspConfig(torch.nn.Module):
    """Container defining a batch of grasps -- both pre-grasps
    and grasping directions -- for use in grasp optimization."""

    def __init__(
        self,
        batch_size: int = 1,
        chain: Chain = load_allegro(),
        requires_grad: bool = True,
        num_fingers: int = 4,
    ) -> None:
        self.batch_size = batch_size
        super().__init__()
        self.hand_config = AllegroHandConfig(batch_size, chain, requires_grad)

        # NOTE: grasp orientations has a batch dim for fingers,
        # since we choose one grasp dir / finger.
        # grasp_orientations refers to the orientation of each finger in world frame
        # (i.e. the third column of grasp_orientations rotation matrix is the finger approach direction in world frame)
        self.grasp_orientations = pp.Parameter(
            pp.identity_SO3(batch_size, num_fingers),
            requires_grad=requires_grad,
        )
        self.num_fingers = num_fingers

    @classmethod
    def from_values(
        cls,
        wrist_pose: pp.LieTensor,
        joint_angles: torch.Tensor,
        grasp_orientations: pp.LieTensor,
        num_fingers: int = 4,
    ) -> AllegroGraspConfig:
        """
        Factory method to create an AllegroGraspConfig from values
        for the wrist pose, joint angles, and grasp orientations.
        """
        batch_size = wrist_pose.shape[0]

        # Check shapes.
        assert joint_angles.shape == (batch_size, 16)
        assert wrist_pose.shape == (batch_size, 7)
        assert grasp_orientations.shape == (batch_size, num_fingers, 4)

        grasp_config = cls(batch_size, num_fingers=num_fingers).to(
            device=wrist_pose.device, dtype=wrist_pose.dtype
        )
        grasp_config.hand_config.set_wrist_pose(wrist_pose)
        grasp_config.hand_config.set_joint_angles(joint_angles)
        grasp_config.set_grasp_orientations(grasp_orientations)
        return grasp_config

    @classmethod
    def from_grasp_config_dict(
        cls,
        grasp_config_dict: Dict[str, Any],
        num_fingers: int = 4,
        check: bool = True,
    ) -> AllegroGraspConfig:
        """
        Factory method get grasp configs from grasp config_dict
        """
        # Load grasp data + instantiate correctly-sized config object.
        batch_size = grasp_config_dict["trans"].shape[0]
        grasp_config = cls(batch_size, num_fingers=num_fingers)
        device = grasp_config.grasp_orientations.device
        dtype = grasp_config.grasp_orientations.dtype

        # Load hand config
        grasp_config.hand_config = AllegroHandConfig.from_hand_config_dict(
            grasp_config_dict, check=check
        )

        grasp_orientations = (
            torch.from_numpy(grasp_config_dict["grasp_orientations"])
            .to(device)
            .to(dtype)
        )
        assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

        # Set the grasp config's data.
        grasp_config.set_grasp_orientations(
            # Set atol and rtol to be a bit larger than default to handle large matrices
            # (numerical errors larger affect the sanity checking)
            pp.from_matrix(
                grasp_orientations, pp.SO3_type, atol=1e-3, rtol=1e-3, check=check
            )
        )

        return grasp_config

    @classmethod
    def from_tensor(cls, grasp_config_tensor: torch.Tensor) -> AllegroGraspConfig:
        """
        Expects a tensor of shape [batch_size, num_fingers, 7 + 16 + 4]
        with all config parameters.
        """
        B = grasp_config_tensor.shape[0]
        N_FINGERS = 4
        assert (
            grasp_config_tensor.shape == (B, N_FINGERS, 7 + 16 + 4)
        ), f"Expected shape ({B}, {N_FINGERS}, 7 + 16 + 4), got {grasp_config_tensor.shape}"

        hand_config_tensor = grasp_config_tensor[:, 0, :23]
        hand_config = AllegroHandConfig.from_tensor(hand_config_tensor)

        grasp_orientations = grasp_config_tensor[:, :, 23:]
        assert grasp_orientations.shape == (
            B,
            N_FINGERS,
            4,
        ), f"Expected shape ({B}, {N_FINGERS}, 4), got {grasp_orientations.shape}"
        grasp_orientations = pp.SO3(grasp_orientations).to(grasp_config_tensor.device)

        grasp_configs = AllegroGraspConfig.from_values(
            wrist_pose=hand_config.wrist_pose,
            joint_angles=hand_config.joint_angles,
            grasp_orientations=grasp_orientations,
        )
        return grasp_configs

    @classmethod
    def from_grasp(cls, grasp: torch.Tensor) -> AllegroGraspConfig:
        device = grasp.device

        N_FINGERS = 4
        GRASP_DIM = 3 + 6 + 16 + 4 * 3
        x = grasp
        B = x.shape[0]

        assert x.shape == (
            B,
            GRASP_DIM,
        ), f"Expected shape ({B}, {GRASP_DIM}), got {x.shape}"
        trans = x[:, :3]
        rot6d = x[:, 3:9]
        joint_angles = x[:, 9:25]
        grasp_dirs = x[:, 25:37].reshape(B, N_FINGERS, 3)
        rot = rot6d_to_matrix(rot6d)

        wrist_pose_matrix = (
            torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1).float()
        )
        wrist_pose_matrix[:, :3, :3] = rot
        wrist_pose_matrix[:, :3, 3] = trans

        wrist_pose = pp.from_matrix(
            wrist_pose_matrix.to(device),
            pp.SE3_type,
            atol=1e-3,
            rtol=1e-3,
        )

        assert wrist_pose.lshape == (B,)

        grasp_orientations = compute_grasp_orientations(
            grasp_dirs=grasp_dirs,
            wrist_pose=wrist_pose,
            joint_angles=joint_angles,
        )

        # Convert to AllegroGraspConfig
        grasp_configs = AllegroGraspConfig.from_values(
            wrist_pose=wrist_pose,
            joint_angles=joint_angles,
            grasp_orientations=grasp_orientations,
        )
        return grasp_configs

    @classmethod
    def from_multiple_grasp_configs(
        cls, grasp_configs: List[AllegroGraspConfig]
    ) -> AllegroGraspConfig:
        return cls.from_values(
            wrist_pose=torch.cat([gc.wrist_pose for gc in grasp_configs], dim=0),
            joint_angles=torch.cat([gc.joint_angles for gc in grasp_configs], dim=0),
            grasp_orientations=torch.cat(
                [gc.grasp_orientations for gc in grasp_configs], dim=0
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        grasp_config_dict = self.hand_config.as_dict()
        batch_size = grasp_config_dict["trans"].shape[0]
        assert (
            batch_size == self.batch_size
        ), f"Batch size {self.batch_size} does not match hand_config_dict batch_size of {batch_size}"

        grasp_config_dict["grasp_orientations"] = (
            self.grasp_orientations.matrix().detach().cpu().numpy()
        )
        return grasp_config_dict

    def as_tensor(self) -> torch.Tensor:
        """
        Returns a tensor of shape [batch_size, num_fingers, 7 + 16 + 4]
        with all config parameters.
        """
        return torch.cat(
            (
                self.hand_config.as_tensor()
                .unsqueeze(-2)
                .expand(-1, self.num_fingers, -1),
                self.grasp_orientations.tensor(),
            ),
            dim=-1,
        )

    def as_grasp(self) -> torch.Tensor:
        B = self.wrist_pose.lshape[0]
        N_FINGERS = 4
        GRASP_DIM = 3 + 6 + 16 + N_FINGERS * 3

        trans = self.wrist_pose.translation()
        rot = self.wrist_pose.rotation().matrix()
        joint_angles = self.joint_angles
        grasp_orientations = self.grasp_orientations.matrix()

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

    def set_grasp_orientations(self, grasp_orientations: pp.LieTensor) -> None:
        assert (
            grasp_orientations.shape == self.grasp_orientations.shape
        ), f"New grasp orientations, shape {grasp_orientations.shape}, do not match current grasp orientations shape {self.grasp_orientations.shape}"
        self.grasp_orientations.data = grasp_orientations.data.clone()

    def __getitem__(self, idxs) -> AllegroGraspConfig:
        """
        Enables indexing/slicing into a batch of grasp configs.
        """
        return type(self).from_values(
            self.wrist_pose[idxs],
            self.joint_angles[idxs],
            self.grasp_orientations[idxs],
        )

    @property
    def wrist_pose(self) -> pp.LieTensor:
        return self.hand_config.wrist_pose

    @property
    def joint_angles(self) -> torch.Tensor:
        return self.hand_config.joint_angles

    @property
    def fingertip_transforms(self) -> pp.LieTensor:
        """Returns finger-to-world transforms."""
        return self.hand_config.get_fingertip_transforms()

    @property
    def grasp_frame_transforms(self) -> pp.LieTensor:
        """Returns SE(3) transforms for ``grasp frame'', i.e.,
        z-axis pointing along grasp direction."""
        fingertip_positions = self.fingertip_transforms.translation()
        assert fingertip_positions.shape == (
            self.batch_size,
            self.num_fingers,
            3,
        )

        grasp_orientations = self.grasp_orientations
        assert grasp_orientations.lshape == (self.batch_size, self.num_fingers)

        transforms = pp.SE3(
            torch.cat(
                [
                    fingertip_positions,
                    grasp_orientations,
                ],
                dim=-1,
            )
        )
        assert transforms.lshape == (self.batch_size, self.num_fingers)
        return transforms

    @property
    def grasp_dirs(self) -> torch.Tensor:  # shape [B, 4, 3].
        return self.grasp_frame_transforms.rotation() @ Z_AXIS.to(
            device=self.grasp_orientations.device, dtype=self.grasp_orientations.dtype
        ).unsqueeze(0).unsqueeze(0)

    def filter_less_feasible(
        self, fingers_forward_theta_deg: float, palm_upwards_theta_deg: float
    ) -> AllegroGraspConfig:
        wrist_pose_matrix = self.wrist_pose.matrix()
        x_dirs = wrist_pose_matrix[:, :, 0]
        z_dirs = wrist_pose_matrix[:, :, 2]

        fingers_forward_cos_theta = math.cos(math.radians(fingers_forward_theta_deg))
        palm_upwards_cos_theta = math.cos(math.radians(palm_upwards_theta_deg))
        fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
        palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta

        before_batch_size = len(self)
        grasp_configs = self[fingers_forward & ~palm_upwards]
        after_batch_size = len(grasp_configs)
        print(
            f"Filtered less feasible grasps. {before_batch_size} -> {after_batch_size}"
        )
        return grasp_configs

    def sample_random_rotations_only_around_y(self) -> AllegroGraspConfig:
        new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(self.as_dict())
        random_rotate_transforms = sample_random_rotate_transforms_only_around_y(
            len(new_grasp_configs)
        )
        new_grasp_configs.hand_config.set_wrist_pose(
            random_rotate_transforms @ new_grasp_configs.hand_config.wrist_pose
        )
        return new_grasp_configs

    def target_joint_angles(
        self,
        dist_move_finger: Optional[float] = None,
        clamp_to_limits: bool = True,
    ) -> torch.Tensor:
        device = self.device

        hand_model = self.hand_config.as_hand_model()
        assert hand_model.hand_pose is not None
        grasp_orientations = self.grasp_orientations.matrix().to(device)
        (
            optimized_joint_angle_targets,
            _,
        ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
            joint_angles_start=hand_model.hand_pose[:, 9:],
            hand_model=hand_model,
            grasp_orientations=grasp_orientations,
            dist_move_finger=dist_move_finger,
        )

        num_joints = hand_model.num_joints
        assert optimized_joint_angle_targets.shape == (
            hand_model.batch_size,
            num_joints,
        )

        optimized_joint_angle_targets = optimized_joint_angle_targets.to(device)
        if clamp_to_limits:
            optimized_joint_angle_targets = clamp_in_limits(
                optimized_joint_angle_targets,
                hand_model_type=self.hand_model_type,
            )

        return optimized_joint_angle_targets

    def pre_joint_angles(
        self,
        dist_move_finger_backwards: Optional[float] = None,
        clamp_to_limits: bool = True,
    ) -> torch.Tensor:
        device = self.device

        hand_model = self.hand_config.as_hand_model()
        assert hand_model.hand_pose is not None
        grasp_orientations = self.grasp_orientations.matrix().to(device)
        (
            pre_joint_angle_targets,
            _,
        ) = compute_init_joint_angles_given_grasp_orientations(
            joint_angles_start=hand_model.hand_pose[:, 9:],
            hand_model=hand_model,
            grasp_orientations=grasp_orientations,
            dist_move_finger_backwards=dist_move_finger_backwards,
        )

        num_joints = hand_model.num_joints
        assert pre_joint_angle_targets.shape == (hand_model.batch_size, num_joints)

        pre_joint_angle_targets = pre_joint_angle_targets.to(device)
        if clamp_to_limits:
            pre_joint_angle_targets = clamp_in_limits(
                pre_joint_angle_targets, hand_model_type=self.hand_model_type
            )

        return pre_joint_angle_targets

    def get_target_hand_config(
        self, dist_move_finger: Optional[float] = None
    ) -> AllegroHandConfig:
        target_joint_angles = self.target_joint_angles(dist_move_finger)
        target_hand_config = AllegroHandConfig.from_values(
            wrist_pose=self.wrist_pose,
            joint_angles=target_joint_angles,
        )
        return target_hand_config

    def get_pre_hand_config(
        self, dist_move_finger_backwards: Optional[float] = None
    ) -> AllegroHandConfig:
        pre_joint_angles = self.pre_joint_angles(dist_move_finger_backwards)
        pre_hand_config = AllegroHandConfig.from_values(
            wrist_pose=self.wrist_pose,
            joint_angles=pre_joint_angles,
        )
        return pre_hand_config

    def __repr__(self) -> str:
        hand_config_repr = self.hand_config.__repr__()
        grasp_orientations_repr = np.array2string(
            self.grasp_orientations.matrix().data.cpu().numpy(), separator=", "
        )
        repr_parts = [
            "AllegroGraspConfig(",
            f"  batch_size={self.batch_size},",
            f"  hand_config={hand_config_repr},",
            "  grasp_orientations=(",
            f"{grasp_orientations_repr}",
            "),",
            f"  num_fingers={self.num_fingers}",
            ")",
        ]
        return "\n".join(repr_parts)

    def __len__(self) -> int:
        return self.batch_size

    @property
    def device(self) -> torch.device:
        return self.hand_config.device

    @property
    def hand_model_type(self) -> HandModelType:
        return self.hand_config.hand_model_type


def normalize_with_warning(v: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    B = v.shape[0]
    assert v.shape == (B, 3), f"Expected shape ({B}, 3), got {v.shape}"
    norm = torch.norm(v, dim=1, keepdim=True)
    if torch.any(norm < atol):
        print("^" * 80)
        print(
            f"Warning: Found {torch.sum(norm < atol)} vectors with norm less than {atol}"
        )
        print("^" * 80)
    return v / (norm + atol)


def validate_rotation_matrices(
    rotation_matrices: torch.Tensor, atol: float = 1e-3
) -> None:
    """
    Validate that a batch of rotation matrices are orthogonal and have determinant 1.

    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (B, 3, 3) containing B rotation matrices.
        atol (float): Absolute tolerance for the checks. Default is 1e-3.

    Raises:
        AssertionError: If any of the matrices fail the orthogonality or determinant check.
    """
    B = rotation_matrices.shape[0]
    assert rotation_matrices.shape == (
        B,
        3,
        3,
    ), f"Expected shape ({B}, 3, 3), got {rotation_matrices.shape}"

    # Check orthogonality: mat.T @ mat should be close to identity
    identity_matrices: torch.Tensor = (
        torch.eye(3, device=rotation_matrices.device, dtype=rotation_matrices.dtype)
        .unsqueeze(0)
        .repeat(B, 1, 1)
    )
    orthogonality_check: bool = torch.allclose(
        torch.einsum(
            "bij,bjk->bik", rotation_matrices.transpose(1, 2), rotation_matrices
        ),
        identity_matrices,
        atol=atol,
    )

    # Check determinant: det(mat) should be close to 1
    determinants: torch.Tensor = torch.linalg.det(rotation_matrices)
    determinant_check: bool = torch.allclose(
        determinants,
        torch.ones(B, device=determinants.device, dtype=determinants.dtype),
        atol=atol,
    )

    assert orthogonality_check, "One or more matrices are not orthogonal."
    assert determinant_check, "One or more matrices do not have determinant 1."


def rot6d_to_matrix(rot6d: torch.Tensor, check: bool = True) -> torch.Tensor:
    B = rot6d.shape[0]
    assert rot6d.shape == (B, 6), f"Expected shape ({B}, 6), got {rot6d.shape}"

    # Step 1: Reshape to (B, 3, 2)
    rot3x2 = rot6d.reshape(B, 3, 2)

    # Step 2: Normalize the first column
    col1 = rot3x2[:, :, 0]
    col1_normalized = normalize_with_warning(col1)

    # Step 3: Orthogonalize the second column with respect to the first column
    col2 = rot3x2[:, :, 1]
    dot_product = torch.sum(col1_normalized * col2, dim=1, keepdim=True)
    col2_orthogonal = col2 - dot_product * col1_normalized

    # Step 4: Normalize the second column
    col2_normalized = normalize_with_warning(col2_orthogonal)

    # Step 5: Compute the cross product to obtain the third column
    col3 = torch.linalg.cross(col1_normalized, col2_normalized, dim=-1)

    # Combine the columns to form the rotation matrix
    rotation_matrices = torch.stack((col1_normalized, col2_normalized, col3), dim=-1)

    # Step 6: Check orthogonality and determinant
    if check:
        validate_rotation_matrices(rotation_matrices=rotation_matrices, atol=1e-3)

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
    PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4
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
    hand_model = AllegroHandConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=joint_angles,
    ).as_hand_model()

    grasp_orientations = compute_grasp_orientations_from_z_dirs(
        joint_angles_start=joint_angles,
        hand_model=hand_model,
        z_dirs=z_dirs,
    )
    grasp_orientations = pp.from_matrix(
        grasp_orientations,
        pp.SO3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )
    assert grasp_orientations.lshape == (
        B,
        N_FINGERS,
    ), f"Expected shape ({B}, {N_FINGERS}), got {grasp_orientations.lshape}"

    return grasp_orientations


def sample_random_rotate_transforms_only_around_y(N: int) -> pp.LieTensor:
    PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # Consider finding a better uniform sampling on SO(3).
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
