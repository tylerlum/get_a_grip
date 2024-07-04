from __future__ import annotations

import pathlib
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import pypose as pp
import pytorch_kinematics as pk
import torch
from nerfstudio.fields.base_field import Field
from pytorch_kinematics.chain import Chain
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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
from get_a_grip.grasp_planning.utils.visualize_utils import (
    plot_nerf_densities,
    plot_mesh,
)
from get_a_grip.grasp_planning.config.grasp_metric_config import GraspMetricConfig
from get_a_grip.model_training.config.classifier_config import ClassifierConfig
from get_a_grip.model_training.config.fingertip_config import UnionFingertipConfig
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from get_a_grip.model_training.models.classifier import (
    Classifier,
)
from get_a_grip.model_training.utils.nerf_grasp_evaluator_batch_data import (
    BatchDataInput,
)
from get_a_grip.model_training.utils.nerf_load_utils import (
    load_nerf_field,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
    get_ray_samples,
)
from get_a_grip.model_training.utils.nerf_utils import (
    get_densities_in_grid,
    get_density,
)
from get_a_grip.model_training.utils.point_utils import (
    transform_point,
)

Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float32)

FINGERTIP_LINK_NAMES = [
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
]


def load_allegro() -> Chain:
    allegro_path = ALLEGRO_HAND_ROOT_HAND_FILE
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

    def cov(self) -> Tuple[pp.LieTensor, torch.Tensor]:
        """
        Returns the covariance of the batch of hand configs.
        A bit hacky -- just works in the Lie algebra, which
        is hopefully ok.

        Returns a tuple of covariance tensors for the wrist pose and joint angles.
        """
        cov_wrist_pose = batch_cov(
            self.wrist_pose.Log(), dim=0
        )  # Leave in tangent space.
        cov_joint_angles = batch_cov(self.joint_angles, dim=0)

        return (cov_wrist_pose, cov_joint_angles)

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
        hand_config_dict = self.hand_config.as_dict()
        hand_config_dict_batch_size = hand_config_dict["trans"].shape[0]
        assert (
            hand_config_dict_batch_size == self.batch_size
        ), f"Batch size {self.batch_size} does not match hand_config_dict_batch_size of {hand_config_dict_batch_size}"

        hand_config_dict["grasp_orientations"] = (
            self.grasp_orientations.matrix().detach().cpu().numpy()
        )
        return hand_config_dict

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

    def cov(self) -> Tuple[pp.LieTensor, torch.Tensor, torch.Tensor]:
        """
        Returns the covariance of the batch of grasp configs.
        """
        cov_wrist_pose, cov_joint_angles = self.hand_config.cov()
        cov_grasp_orientations = batch_cov(self.grasp_orientations.Log(), dim=0)

        return (
            cov_wrist_pose,
            cov_joint_angles,
            cov_grasp_orientations,
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


class GraspMetric(torch.nn.Module):
    """
    Wrapper for NeRF + grasp classifier to evaluate
    a particular AllegroGraspConfig.
    """

    def __init__(
        self,
        nerf_field: Field,
        classifier_model: Classifier,
        fingertip_config: UnionFingertipConfig,
        X_N_Oy: np.ndarray,
        return_type: str = "failure_probability",
    ) -> None:
        super().__init__()
        self.nerf_field = nerf_field
        self.classifier_model = classifier_model
        self.fingertip_config = fingertip_config
        self.X_N_Oy = X_N_Oy
        self.ray_origins_finger_frame = get_ray_origins_finger_frame(fingertip_config)
        self.return_type = return_type

    def forward(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        ray_samples = self.compute_ray_samples(grasp_config)

        # Query NeRF at RaySamples.
        densities = self.compute_nerf_densities(
            ray_samples,
        )
        assert densities.shape == (
            grasp_config.batch_size,
            4,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        # Query NeRF in grid
        # BRITTLE: Require that the classifier_model has the word "Global" in it if needed
        need_to_query_global = (
            "global" in self.classifier_model.__class__.__name__.lower()
        )
        if need_to_query_global:
            lb_N = transform_point(T=self.X_N_Oy, point=lb_Oy)
            ub_N = transform_point(T=self.X_N_Oy, point=ub_Oy)
            nerf_densities_global, query_points_global_N = get_densities_in_grid(
                field=self.nerf_field,
                lb=lb_N,
                ub=ub_N,
                num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
                num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
                num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
            )
            nerf_densities_global = (
                torch.from_numpy(nerf_densities_global)
                .float()[None, ...]
                .repeat_interleave(grasp_config.batch_size, dim=0)
            )
        else:
            nerf_densities_global, query_points_global_N = None, None

        # DEBUG PLOT
        PLOT = False
        if PLOT:
            from pathlib import Path

            import trimesh

            if Path("/tmp/mesh_viz_object.obj").exists():
                mesh = trimesh.load("/tmp/mesh_viz_object.obj")
            else:
                mesh = None

            # Do not need this, just for debugging
            query_points_global_N = (
                torch.from_numpy(query_points_global_N)
                .float()[None, ...]
                .repeat_interleave(grasp_config.batch_size, dim=0)
            )
            all_query_points = ray_samples.frustums.get_positions()

            for batch_idx in range(2):
                fig = go.Figure()
                N_FINGERS = 4
                for i in range(N_FINGERS):
                    plot_nerf_densities(
                        fig=fig,
                        densities=densities[batch_idx, i].reshape(-1),
                        query_points=all_query_points[batch_idx, i].reshape(-1, 3),
                        name=f"finger_{i}",
                        opacity=0.2,
                    )
                if need_to_query_global:
                    nerf_densities_global_flattened = nerf_densities_global[
                        batch_idx
                    ].reshape(-1)
                    query_points_global_N_flattened = query_points_global_N[
                        batch_idx
                    ].reshape(-1, 3)
                    plot_nerf_densities(
                        fig=fig,
                        densities=nerf_densities_global_flattened[
                            nerf_densities_global_flattened > 15
                        ],
                        query_points=query_points_global_N_flattened[
                            nerf_densities_global_flattened > 15
                        ],
                        name="global",
                    )

                if mesh is not None:
                    plot_mesh(fig=fig, mesh=mesh)

                wrist_trans_array = (
                    grasp_config.wrist_pose.translation().detach().cpu().numpy()
                )
                wrist_rot_array = (
                    grasp_config.wrist_pose.rotation().matrix().detach().cpu().numpy()
                )
                joint_angles_array = grasp_config.joint_angles.detach().cpu().numpy()

                # Put into transforms X_Oy_H_array
                B = grasp_config.batch_size
                X_Oy_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
                assert X_Oy_H_array.shape == (B, 4, 4)
                X_Oy_H_array[:, :3, :3] = wrist_rot_array
                X_Oy_H_array[:, :3, 3] = wrist_trans_array

                X_N_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
                for i in range(B):
                    X_N_H_array[i] = self.X_N_Oy @ X_Oy_H_array[i]

                device = "cuda"
                hand_model = HandModel(device=device)

                # Compute pregrasp and target hand poses
                trans_array = X_N_H_array[:, :3, 3]
                rot_array = X_N_H_array[:, :3, :3]

                pregrasp_hand_pose = hand_config_to_pose(
                    trans_array, rot_array, joint_angles_array
                ).to(device)

                # Get plotly data
                hand_model.set_parameters(pregrasp_hand_pose)
                pregrasp_plot_data = hand_model.get_plotly_data(
                    i=batch_idx, opacity=1.0
                )

                for x in pregrasp_plot_data:
                    fig.add_trace(x)
                # Add title with idx
                fig.update_layout(title_text=f"Batch idx {batch_idx}")
                fig.show()

        # HACK: NOT SURE HOW TO FILL THIS
        # raise NotImplementedError("Need to implement this object scale")
        batch_data_input = BatchDataInput(
            nerf_densities=densities,
            grasp_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=self.fingertip_config,
            grasp_configs=grasp_config.as_tensor(),
            nerf_densities_global=(
                nerf_densities_global if nerf_densities_global is not None else None
            ),
        ).to(grasp_config.hand_config.wrist_pose.device)

        # Pass grasp transforms, densities into classifier.
        if self.return_type == "failure_probability":
            return self.classifier_model.get_failure_probability(batch_data_input)
        elif self.return_type == "failure_logits":
            return self.classifier_model(batch_data_input)[:, -1]
        elif self.return_type == "all_logits":
            return self.classifier_model(batch_data_input)
        else:
            raise ValueError(f"return_type {self.return_type} not recognized")

    def compute_ray_samples(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        # Let Oy be object yup frame (centroid of object)
        # Let N be nerf frame (where the nerf is defined)
        # For NeRFs trained from sim data, Oy and N are the same.
        # But for real-world data, Oy and N are different (N is a point on the table, used as NeRF origin)
        # When sampling from the NeRF, we must give ray samples in N frame
        # But classifier is trained on Oy frame
        # Thus, we must transform grasp_frame_transforms from Oy frame to N frame
        # Let Fi be the finger frame (origin at each fingertip i)
        # Let p_Fi be points in Fi frame
        # self.ray_origins_finger_frame = p_Fi
        # grasp_frame_transforms = T_{Oy <- Fi}
        # X_N_Oy = T_{N <- Oy}
        # TODO: Batch this to avoid OOM (refer to create_nerf_grasp_dataset.py)

        # Prepare transforms
        T_Oy_Fi = grasp_config.grasp_frame_transforms
        assert T_Oy_Fi.lshape == (grasp_config.batch_size, grasp_config.num_fingers)

        assert self.X_N_Oy.shape == (
            4,
            4,
        )
        X_N_Oy_repeated = (
            torch.from_numpy(self.X_N_Oy)
            .float()
            .unsqueeze(dim=0)
            .repeat_interleave(
                grasp_config.batch_size * grasp_config.num_fingers, dim=0
            )
            .reshape(grasp_config.batch_size, grasp_config.num_fingers, 4, 4)
        )

        T_N_Oy = pp.from_matrix(
            X_N_Oy_repeated,
            pp.SE3_type,
        ).to(T_Oy_Fi.device)

        # Transform grasp_frame_transforms to nerf frame
        T_N_Fi = T_N_Oy @ T_Oy_Fi

        # Generate RaySamples.
        ray_samples = get_ray_samples(
            self.ray_origins_finger_frame,
            T_N_Fi,
            self.fingertip_config,
        )
        return ray_samples

    def compute_nerf_densities(
        self,
        ray_samples,
    ) -> torch.Tensor:
        # Query NeRF at RaySamples.
        densities = self.nerf_field.get_density(ray_samples.to("cuda"))[0][
            ..., 0
        ]  # Shape [B, 4, n_x, n_y, n_z]
        return densities

    def get_failure_probability(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        return self(grasp_config)

    @classmethod
    def from_config(
        cls,
        grasp_metric_config: GraspMetricConfig,
        console: Optional[Console] = None,
    ) -> GraspMetric:
        assert grasp_metric_config.X_N_Oy is not None
        return cls.from_configs(
            nerf_config=grasp_metric_config.nerf_config,
            classifier_config=grasp_metric_config.classifier_config,
            X_N_Oy=grasp_metric_config.X_N_Oy,
            classifier_checkpoint=grasp_metric_config.classifier_checkpoint,
            console=console,
        )

    @classmethod
    def from_configs(
        cls,
        nerf_config: pathlib.Path,
        classifier_config: ClassifierConfig,
        X_N_Oy: np.ndarray,
        classifier_checkpoint: int = -1,
        console: Optional[Console] = None,
    ) -> GraspMetric:
        # Load nerf
        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description} "),
                TimeElapsedColumn(),
                console=console,
            )
            if console is not None
            else nullcontext()
        ) as progress:
            task = (
                progress.add_task("Loading NeRF", total=1)
                if progress is not None
                else None
            )

            nerf_field = load_nerf_field(nerf_config)

            if progress is not None and task is not None:
                progress.update(task, advance=1)

        # Load classifier
        classifier = load_classifier(
            classifier_config=classifier_config,
            classifier_checkpoint=classifier_checkpoint,
            console=console,
        )

        return cls(
            nerf_field,
            classifier,
            classifier_config.nerfdata_config.fingertip_config,
            X_N_Oy,
        )


def load_classifier(
    classifier_config: ClassifierConfig,
    classifier_checkpoint: int = -1,
    console: Optional[Console] = None,
) -> Classifier:
    # Load classifier
    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        )
        if console is not None
        else nullcontext()
    ) as progress:
        task = (
            progress.add_task("Loading classifier", total=1)
            if progress is not None
            else None
        )

        # (should device thing be here? probably since saved on gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = (
            classifier_config.model_config.get_classifier_from_fingertip_config(
                fingertip_config=classifier_config.nerfdata_config.fingertip_config,
                n_tasks=classifier_config.task_type.n_tasks,
            )
        ).to(device)

        # Load classifier weights
        assert classifier_config.checkpoint_workspace.output_dir.exists(), f"checkpoint_workspace.output_dir does not exist at {classifier_config.checkpoint_workspace.output_dir}"
        print(
            f"Loading checkpoint ({classifier_config.checkpoint_workspace.output_dir})..."
        )

        output_checkpoint_paths = (
            classifier_config.checkpoint_workspace.output_checkpoint_paths
        )
        assert (
            len(output_checkpoint_paths) > 0
        ), f"No checkpoints found in {classifier_config.checkpoint_workspace.output_checkpoint_paths}"
        assert (
            classifier_checkpoint < len(output_checkpoint_paths)
        ), f"Requested checkpoint {classifier_checkpoint} does not exist in {classifier_config.checkpoint_workspace.output_checkpoint_paths}"
        checkpoint_path = output_checkpoint_paths[classifier_checkpoint]

        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.load_state_dict(torch.load(checkpoint_path)["classifier"])

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    return classifier


def predict_in_collision_with_object(
    nerf_field: Field,
    hand_surface_points_Oy: torch.Tensor,
    max_density_threshold: float = 8.5,
) -> torch.Tensor:
    # TODO: Move elsewhere
    surface_points = hand_surface_points_Oy
    assert surface_points.shape[-1] == 3
    num_grasps, num_points_per_grasp, _ = surface_points.shape

    densities = (
        get_density(
            field=nerf_field,
            positions=surface_points,
        )[0]
        .squeeze(dim=-1)
        .detach()
        .cpu()
        .numpy()
    )
    assert densities.shape == (num_grasps, num_points_per_grasp)
    max_densities = densities.max(axis=-1)

    predict_penetrations = max_densities > max_density_threshold
    return predict_penetrations


def predict_in_collision_with_table(
    table_y_Oy: float,
    hand_surface_points_Oy: torch.Tensor,
    buffer: float = 0.02,
) -> np.ndarray:
    # TODO: Move elsewhere
    assert hand_surface_points_Oy.shape[-1] == 3
    num_grasps, num_points_per_grasp, _ = hand_surface_points_Oy.shape

    hand_surface_points_Oy = hand_surface_points_Oy.detach().cpu().numpy()

    predict_penetrations = (
        hand_surface_points_Oy[:, :, 1].min(axis=-1) < table_y_Oy + buffer
    )
    assert predict_penetrations.shape == (num_grasps,)
    return predict_penetrations


def get_hand_surface_points_Oy(
    grasp_config: AllegroGraspConfig,
    n_surface_points: int = 1000,
) -> torch.Tensor:
    # TODO: Move elsewhere
    device = grasp_config.hand_config.wrist_pose.device

    translation = grasp_config.wrist_pose.translation().detach().cpu().numpy()
    rotation = grasp_config.wrist_pose.rotation().matrix().detach().cpu().numpy()
    joint_angles = grasp_config.joint_angles.detach().cpu().numpy()
    hand_model = HandModel(
        device=device,
        n_surface_points=n_surface_points,
    )
    hand_pose = hand_config_to_pose(translation, rotation, joint_angles).to(device)
    hand_model.set_parameters(hand_pose)
    surface_points = hand_model.get_surface_points()
    assert surface_points.shape == (grasp_config.batch_size, n_surface_points, 3)
    return surface_points


def get_joint_limits() -> Tuple[np.ndarray, np.ndarray]:
    # TODO: Move elsewhere
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
    # TODO: Move elsewhere
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
    # TODO: Move elsewhere
    N = joint_angles.shape[0]
    assert joint_angles.shape == (N, 16)

    joints_lower, joints_upper = get_joint_limits()
    assert joints_upper.shape == (16,)
    assert joints_lower.shape == (16,)

    joint_angles = np.clip(
        joint_angles, joints_lower[None, ...], joints_upper[None, ...]
    )
    return joint_angles


class IndexingDataset(torch.utils.data.Dataset):
    def __init__(self, num_datapoints: int):
        self.num_datapoints = num_datapoints

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_datapoints


def get_split_inds(
    num_datapoints: int, split: Iterable[Union[int, float]], random_seed: int
):
    indexing_dataset = IndexingDataset(num_datapoints)
    splits = torch.utils.data.random_split(
        indexing_dataset, split, generator=torch.Generator().manual_seed(random_seed)
    )

    return [split.indices for split in splits]


def SO3_to_SE3(R: pp.LieTensor):
    assert R.ltype == pp.SO3_type, f"R must be an SO3, not {R.ltype}"

    return pp.SE3(torch.cat((torch.zeros_like(R[..., :3]), R.tensor()), dim=-1))


def batch_cov(x: torch.Tensor, dim: int = 0, keepdim=False):
    n_dim = x.shape[dim]
    x_mean = x.mean(dim, keepdim=True)
    x_centered = x - x_mean

    return (x_centered.unsqueeze(-2) * x_centered.unsqueeze(-1)).sum(
        dim=dim, keepdim=keepdim
    ) / (n_dim - 1)


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
    B = grasp_configs.batch_size

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
            dist_move_finger_backward=dist_move_finger_backward
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
