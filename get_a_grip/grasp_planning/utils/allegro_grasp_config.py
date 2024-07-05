from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import numpy as np
import pypose as pp
import pytorch_kinematics as pk
import torch
from pytorch_kinematics.chain import Chain

from get_a_grip import get_assets_folder
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_JOINT_NAMES,
    ALLEGRO_HAND_ROOT_HAND_FILE,
)
from get_a_grip.dataset_generation.utils.hand_model import HandModel
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_init_joint_angles_given_grasp_orientations,
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from get_a_grip.dataset_generation.utils.pose_conversion import (
    hand_config_to_pose,
)
from get_a_grip.grasp_planning.utils.joint_limit_utils import (
    get_joint_limits,
)

Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float32)

FINGERTIP_LINK_NAMES = [
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
]


def load_allegro() -> Chain:
    allegro_path = str(get_assets_folder() / ALLEGRO_HAND_ROOT_HAND_FILE)
    return pk.build_chain_from_urdf(open(allegro_path).read())


class AllegroHandConfig(torch.nn.Module):
    """
    A container specifying a batch of configurations for an Allegro hand, i.e., the
    wrist pose and the joint configurations.
    """

    def __init__(
        self,
        batch_size: int = 1,  # TODO(pculbert): refactor for arbitrary batch sizes.
        chain: Chain = load_allegro(),
        requires_grad: bool = True,
    ) -> None:
        # TODO(pculbert): add device/dtype kwargs.
        super().__init__()
        self.chain = chain
        self.wrist_pose = pp.Parameter(
            pp.randn_SE3(batch_size), requires_grad=requires_grad
        )
        self.joint_angles = torch.nn.Parameter(
            torch.zeros(batch_size, 16), requires_grad=requires_grad
        )
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.batch_size

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
            atol=1e-4,
            rtol=1e-4,
        )
        wrist_pose = pp.SE3(torch.cat([wrist_translation, wrist_quat], dim=1))

        return cls.from_values(wrist_pose=wrist_pose, joint_angles=joint_angles)

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
        joint_lower_limits, joint_upper_limits = get_joint_limits()
        self.joint_lower_limits, self.joint_upper_limits = (
            torch.from_numpy(joint_lower_limits).float().to(self.wrist_pose.device),
            torch.from_numpy(joint_upper_limits).float().to(self.wrist_pose.device),
        )

        assert self.joint_lower_limits.shape == (16,)
        assert self.joint_upper_limits.shape == (16,)

        self.joint_angles.data = torch.clamp(
            self.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )

    def get_fingertip_transforms(self) -> pp.LieTensor:
        # Pretty hacky -- need to cast chain to the same device as the wrist pose.
        self.chain = self.chain.to(device=self.wrist_pose.device)

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

    def mean(self) -> AllegroHandConfig:
        """
        Returns the mean of the batch of hand configs.
        A bit hacky -- just works in the Lie algebra, which
        is hopefully ok.
        """
        mean_joint_angles = self.joint_angles.mean(dim=0, keepdim=True)
        mean_wrist_pose = pp.se3(self.wrist_pose.Log().mean(dim=0, keepdim=True)).Exp()

        return AllegroHandConfig.from_values(
            wrist_pose=mean_wrist_pose,
            joint_angles=mean_joint_angles,
            chain=self.chain,
        )

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
        # TODO(pculbert): refactor for arbitrary batch sizes.
        # TODO(pculbert): add device/dtype kwargs.

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

    def __len__(self) -> int:
        return self.batch_size

    @classmethod
    def from_path(cls, path: pathlib.Path) -> AllegroGraspConfig:
        """
        Factory method to create an AllegroGraspConfig from a path to a saved state dict.
        """
        state_dict = torch.load(str(path))
        batch_size = state_dict["hand_config.wrist_pose"].shape[0]
        grasp_config = cls(batch_size)
        grasp_config.load_state_dict(state_dict)
        return grasp_config

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
        # TODO (pculbert): refactor for arbitrary batch sizes via lshape.

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
    def randn(
        cls,
        batch_size: int = 1,
        std_orientation: float = 0.1,
        std_wrist_pose: float = 0.1,
        std_joint_angles: float = 0.1,
        num_fingers: int = 4,
    ) -> AllegroGraspConfig:
        """
        Factory method to create a random AllegroGraspConfig.
        """
        grasp_config = cls(batch_size)

        # TODO(pculbert): think about setting a mean pose that's
        # reasonable, tune the default stds.

        grasp_orientations = pp.so3(
            std_orientation
            * torch.randn(
                batch_size,
                num_fingers,
                3,
                device=grasp_config.grasp_orientations.device,
                dtype=grasp_config.grasp_orientations.dtype,
            )
        ).Exp()

        wrist_pose = pp.se3(
            std_wrist_pose
            * torch.randn(
                batch_size,
                6,
                dtype=grasp_config.grasp_orientations.dtype,
                device=grasp_config.grasp_orientations.device,
            )
        ).Exp()

        joint_angles = std_joint_angles * torch.randn(
            batch_size,
            16,
            dtype=grasp_config.grasp_orientations.dtype,
            device=grasp_config.grasp_orientations.device,
        )

        return grasp_config.from_values(wrist_pose, joint_angles, grasp_orientations)

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
                grasp_orientations, pp.SO3_type, atol=1e-4, rtol=1e-4, check=check
            )
        )

        return grasp_config

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

    def mean(self) -> AllegroGraspConfig:
        """
        Returns the mean of the batch of grasp configs.
        """
        mean_hand_config = self.hand_config.mean()
        mean_grasp_orientations = pp.so3(
            self.grasp_orientations.Log().mean(dim=0, keepdim=True)
        ).Exp()

        return AllegroGraspConfig.from_values(
            wrist_pose=mean_hand_config.wrist_pose,
            joint_angles=mean_hand_config.joint_angles,
            grasp_orientations=mean_grasp_orientations,
        )

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

    def target_joint_angles(
        self, dist_move_finger: Optional[float] = None
    ) -> torch.Tensor:
        device = self.wrist_pose.device
        target_joint_angles = compute_joint_angle_targets(
            trans=self.wrist_pose.translation().detach().cpu().numpy(),
            rot=self.wrist_pose.rotation().matrix().detach().cpu().numpy(),
            joint_angles=self.joint_angles.detach().cpu().numpy(),
            grasp_orientations=self.grasp_orientations.matrix(),
            device=device,
            dist_move_finger=dist_move_finger,
        )
        return torch.from_numpy(target_joint_angles).to(device)

    def pre_joint_angles(
        self, dist_move_finger_backward: Optional[float] = None
    ) -> torch.Tensor:
        device = self.wrist_pose.device
        pre_joint_angles = compute_joint_angle_pre(
            trans=self.wrist_pose.translation().detach().cpu().numpy(),
            rot=self.wrist_pose.rotation().matrix().detach().cpu().numpy(),
            joint_angles=self.joint_angles.detach().cpu().numpy(),
            grasp_orientations=self.grasp_orientations.matrix(),
            device=device,
            dist_move_finger_backwards=dist_move_finger_backward,
        )
        return torch.from_numpy(pre_joint_angles).to(device)

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


def hand_config_to_hand_model(
    hand_config: AllegroHandConfig,
    n_surface_points: int = 0,
) -> HandModel:
    """
    Convert an AllegroHandConfig to a HandModel.
    """
    device = hand_config.wrist_pose.device
    translation = hand_config.wrist_pose.translation().detach().cpu().numpy()
    rotation = hand_config.wrist_pose.rotation().matrix().detach().cpu().numpy()
    joint_angles = hand_config.joint_angles.detach().cpu().numpy()
    hand_model = HandModel(
        device=device,
        n_surface_points=n_surface_points,
    )
    hand_pose = hand_config_to_pose(translation, rotation, joint_angles).to(device)
    hand_model.set_parameters(hand_pose)
    return hand_model


def compute_joint_angle_targets(
    trans: np.ndarray,
    rot: np.ndarray,
    joint_angles: np.ndarray,
    grasp_orientations: torch.Tensor,
    device: torch.device,
    dist_move_finger: Optional[float] = None,
) -> np.ndarray:
    hand_pose = hand_config_to_pose(trans, rot, joint_angles).to(device)
    grasp_orientations = grasp_orientations.to(device)

    # hand model
    hand_model = HandModel(device=device)
    hand_model.set_parameters(hand_pose)
    assert hand_model.hand_pose is not None

    # Optimization
    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
        dist_move_finger=dist_move_finger,
    )

    num_joints = len(ALLEGRO_HAND_JOINT_NAMES)
    assert optimized_joint_angle_targets.shape == (hand_model.batch_size, num_joints)

    return optimized_joint_angle_targets.detach().cpu().numpy()


def compute_joint_angle_pre(
    trans: np.ndarray,
    rot: np.ndarray,
    joint_angles: np.ndarray,
    grasp_orientations: torch.Tensor,
    device: torch.device,
    dist_move_finger_backwards: Optional[float] = None,
) -> np.ndarray:
    hand_pose = hand_config_to_pose(trans, rot, joint_angles).to(device)
    grasp_orientations = grasp_orientations.to(device)

    # hand model
    hand_model = HandModel(device=device)
    hand_model.set_parameters(hand_pose)
    assert hand_model.hand_pose is not None

    # Optimization
    (
        pre_joint_angle_targets,
        _,
    ) = compute_init_joint_angles_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
        dist_move_finger_backwards=dist_move_finger_backwards,
    )

    num_joints = len(ALLEGRO_HAND_JOINT_NAMES)
    assert pre_joint_angle_targets.shape == (hand_model.batch_size, num_joints)

    return pre_joint_angle_targets.detach().cpu().numpy()
