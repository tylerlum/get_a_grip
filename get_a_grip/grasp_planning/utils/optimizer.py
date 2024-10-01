from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
from tqdm import tqdm

from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.grasp_planning.utils.evaluator import (
    BpsEvaluator,
    NerfEvaluator,
)
from get_a_grip.grasp_planning.utils.joint_limit_utils import get_joint_limits


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self, grasp_config: AllegroGraspConfig
    ) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_optimized_grasps(self) -> int:
        raise NotImplementedError


class BpsRandomSamplingOptimizer(Optimizer):
    def __init__(
        self,
        bps_evaluator: BpsEvaluator,
        num_grasps: int,
        num_steps: int,
        trans_noise: float,
        rot_noise: float,
        joint_angle_noise: float,
        grasp_orientation_noise: float,
    ) -> None:
        self.bps_evaluator = bps_evaluator

        self.num_grasps = num_grasps
        self.num_steps = num_steps
        self.trans_noise = trans_noise
        self.rot_noise = rot_noise
        self.joint_angle_noise = joint_angle_noise
        self.grasp_orientation_noise = grasp_orientation_noise

    @property
    def n_optimized_grasps(self) -> int:
        return self.num_grasps

    @torch.no_grad()
    def optimize(
        self, grasp_config: AllegroGraspConfig
    ) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        assert (
            len(grasp_config) >= self.n_optimized_grasps
        ), f"{len(grasp_config)} < {self.n_optimized_grasps}"
        grasp_config = grasp_config[: self.n_optimized_grasps]

        self.joint_lower_limits, self.joint_upper_limits = get_joint_limits(
            hand_model_type=grasp_config.hand_model_type, device=grasp_config.device
        )

        g_O = grasp_config.as_grasp()
        initial_grasps = g_O.detach().clone()
        initial_losses_np = (
            self.bps_evaluator.evaluate_grasps(initial_grasps).detach().cpu().numpy()
        )
        assert g_O.shape == (self.n_optimized_grasps, 3 + 6 + 16 + 4 * 3)

        if self.num_steps > 0:
            for i in tqdm(range(self.num_steps), desc="Optimizing grasps"):
                g_O, losses = self._step(g_O)
            losses_np = losses.detach().cpu().numpy()
        else:
            losses_np = initial_losses_np

        diff_losses = losses_np - initial_losses_np

        print(f"Init Losses:  {[f'{x:.4f}' for x in initial_losses_np.tolist()]}")
        print(f"Final Losses: {[f'{x:.4f}' for x in losses_np.tolist()]}")
        print(f"Diff Losses:  {[f'{x:.4f}' for x in diff_losses.tolist()]}")
        grasp_config = AllegroGraspConfig.from_grasp(
            grasp=g_O,
        )

        assert (
            len(grasp_config) == self.n_optimized_grasps
        ), f"{len(grasp_config)} != {self.n_optimized_grasps}"
        return grasp_config, losses

    @torch.no_grad()
    def _step(self, grasps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = grasps.shape[0]
        assert grasps.shape == (B, 3 + 6 + 16 + 4 * 3)

        with torch.no_grad():
            old_losses = self.bps_evaluator.evaluate_grasps(grasps)
            new_grasps = self.add_noise(grasps)
            new_losses = self.bps_evaluator.evaluate_grasps(grasps)

            # Update grasp config
            improved_idxs = new_losses < old_losses

            grasps[improved_idxs] = new_grasps[improved_idxs]
            updated_losses = torch.where(improved_idxs, new_losses, old_losses)

            print(f"Old: {old_losses}")
            print(f"Noise: {new_losses}")
            print(f"New: {updated_losses}")
            print()
        return grasps, updated_losses

    def add_noise(self, grasps: torch.Tensor) -> torch.Tensor:
        B = grasps.shape[0]
        N_FINGERS = 4
        xyz_noise = (
            torch.randn_like(grasps[:, :3], device=grasps.device) * self.trans_noise
        )
        rot_noise = (pp.randn_so3(B, device=grasps.device) * self.rot_noise).Exp()
        joint_noise = (
            torch.randn_like(grasps[:, 9:25], device=grasps.device)
            * self.joint_angle_noise
        )
        grasp_orientation_perturbations = (
            pp.randn_so3(B, N_FINGERS, device=grasps.device)
            * self.grasp_orientation_noise
        ).Exp()

        new_xyz = grasps[:, :3] + xyz_noise
        new_rot6d = self.add_noise_to_rot6d(grasps[:, 3:9].reshape(B, 3, 2), rot_noise)
        new_joint = self.clip_joint_angles(grasps[:, 9:25] + joint_noise)
        new_grasp_dirs = self.add_noise_to_grasp_dirs(
            grasps[:, 25:37].reshape(B, N_FINGERS, 3), grasp_orientation_perturbations
        )

        new_grasp = torch.cat(
            [
                new_xyz,
                new_rot6d.reshape(B, 6),
                new_joint,
                new_grasp_dirs.reshape(B, N_FINGERS * 3),
            ],
            dim=-1,
        )
        assert new_grasp.shape == grasps.shape, f"{new_grasp.shape} != {grasps.shape}"

        return new_grasp

    def add_noise_to_rot6d(
        self, rot6d: torch.Tensor, rot_noise: pp.SO3
    ) -> torch.Tensor:
        B = rot6d.shape[0]
        assert rot6d.shape == (B, 3, 2)
        assert rot_noise.lshape == (B,)

        rot = self.rot6d_to_rot(rot6d)
        rot_pp = pp.from_matrix(rot, pp.SO3_type)
        new_rot = (rot_noise @ rot_pp).matrix()

        new_rot6d = new_rot[..., :2]
        assert new_rot6d.shape == (B, 3, 2)
        return new_rot6d

    def add_noise_to_grasp_dirs(
        self, grasp_dirs: torch.Tensor, grasp_orientation_perturbations: pp.SO3
    ) -> torch.Tensor:
        B = grasp_dirs.shape[0]
        N_FINGERS = 4
        assert grasp_dirs.shape == (B, N_FINGERS, 3)
        assert grasp_orientation_perturbations.lshape == (B, N_FINGERS)

        new_grasp_dirs = grasp_orientation_perturbations @ grasp_dirs
        new_grasp_dirs = new_grasp_dirs / torch.norm(
            new_grasp_dirs, dim=-1, keepdim=True
        )
        return new_grasp_dirs

    def rot6d_to_rot(self, rot6d: torch.Tensor) -> torch.Tensor:
        B = rot6d.shape[0]
        assert rot6d.shape == (B, 3, 2)

        rot6d = self.orthogonalize_rot6d(rot6d)

        x_col = rot6d[..., 0]
        y_col = rot6d[..., 1]
        z_col = torch.cross(x_col, y_col, dim=-1)
        rot = torch.stack([x_col, y_col, z_col], dim=-1)
        assert rot.shape == (B, 3, 3)
        return rot

    def orthogonalize_rot6d(self, rot6d: torch.Tensor) -> torch.Tensor:
        B = rot6d.shape[0]
        assert rot6d.shape == (B, 3, 2)
        _x_col = rot6d[..., 0]  # Shape: (B, 3)
        x_col = _x_col / torch.norm(_x_col, dim=-1, keepdim=True)
        _y_col = rot6d[..., 1]  # Shape: (B, 3)
        y_col = _y_col - torch.sum(_y_col * x_col, dim=-1, keepdim=True) * x_col
        y_col = y_col / torch.norm(y_col, dim=-1, keepdim=True)

        new_rot6d = torch.stack([x_col, y_col], dim=-1)
        assert new_rot6d.shape == (B, 3, 2)
        return new_rot6d

    def clip_joint_angles(self, joint_angles: torch.Tensor) -> torch.Tensor:
        B = joint_angles.shape[0]
        assert self.joint_lower_limits.shape == (16,)
        assert self.joint_upper_limits.shape == (16,)
        assert joint_angles.shape == (B, 16)
        joint_angles = torch.clamp(
            joint_angles, self.joint_lower_limits, self.joint_upper_limits
        )
        return joint_angles


class NerfRandomSamplingOptimizer(Optimizer):
    def __init__(
        self,
        nerf_evaluator: NerfEvaluator,
        num_grasps: int,
        num_steps: int,
        trans_noise: float,
        rot_noise: float,
        joint_angle_noise: float,
        grasp_orientation_noise: float,
        print_freq: int = 10,
    ) -> None:
        self.nerf_evaluator = nerf_evaluator

        self.num_grasps = num_grasps
        self.num_steps = num_steps
        self.trans_noise = trans_noise
        self.rot_noise = rot_noise
        self.joint_angle_noise = joint_angle_noise
        self.grasp_orientation_noise = grasp_orientation_noise
        self.print_freq = print_freq

    @property
    def n_optimized_grasps(self) -> int:
        return self.num_grasps

    @torch.no_grad()
    def optimize(
        self, grasp_config: AllegroGraspConfig
    ) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        assert (
            len(grasp_config) >= self.n_optimized_grasps
        ), f"{len(grasp_config)} < {self.n_optimized_grasps}"
        grasp_config = grasp_config[: self.n_optimized_grasps]

        self.joint_lower_limits, self.joint_upper_limits = get_joint_limits(
            hand_model_type=grasp_config.hand_model_type, device=grasp_config.device
        )

        for iter in tqdm(range(self.num_steps), desc="Optimizing grasps"):
            losses_np = (
                self.nerf_evaluator.evaluate(grasp_config).detach().cpu().numpy()
            )

            if iter % self.print_freq == 0:
                print(
                    f"Iter: {iter} | Losses: {np.round(losses_np.tolist(), decimals=3)} | Min loss: {losses_np.min():.3f} | Max loss: {losses_np.max():.3f} | Mean loss: {losses_np.mean():.3f} | Std dev: {losses_np.std():.3f}"
                )

            grasp_config, losses = self._step(grasp_config)

        assert (
            len(grasp_config) == self.n_optimized_grasps
        ), f"{len(grasp_config)} != {self.n_optimized_grasps}"
        return grasp_config, losses

    @torch.no_grad()
    def _step(
        self, grasp_config: AllegroGraspConfig
    ) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        with torch.no_grad():
            # Eval old
            old_losses = (
                self.nerf_evaluator.evaluate(grasp_config).detach().cpu().numpy()
            )

            # Sample new
            new_grasp_config = AllegroGraspConfig.from_grasp_config_dict(
                grasp_config.as_dict(),
                check=False,  # Check causing issues, probably just numerical issues
            ).to(device=grasp_config.device)

            wrist_trans_perturbations = (
                torch.randn((*new_grasp_config.wrist_pose.lshape, 3)) * self.trans_noise
            ).to(device=grasp_config.device)
            wrist_rot_perturbations = (
                (pp.randn_so3(new_grasp_config.wrist_pose.lshape) * self.rot_noise)
                .Exp()
                .to(device=grasp_config.device)
            )
            wrist_pose_perturbations = pp.SE3(
                torch.cat([wrist_trans_perturbations, wrist_rot_perturbations], dim=-1)
            ).to(device=grasp_config.device)

            joint_angle_perturbations = (
                torch.randn_like(new_grasp_config.joint_angles) * self.joint_angle_noise
            ).to(device=grasp_config.device)
            grasp_orientation_perturbations = (
                (
                    pp.randn_so3(new_grasp_config.grasp_orientations.lshape)
                    * self.grasp_orientation_noise
                )
                .Exp()
                .to(device=grasp_config.device)
            )

            new_grasp_config.hand_config.set_wrist_pose(
                wrist_pose_perturbations @ new_grasp_config.hand_config.wrist_pose
            )
            new_grasp_config.hand_config.set_joint_angles(
                new_grasp_config.hand_config.joint_angles + joint_angle_perturbations
            )
            new_grasp_config.set_grasp_orientations(
                grasp_orientation_perturbations @ new_grasp_config.grasp_orientations
            )

            # Clip joint angles to feasible range.
            new_grasp_config.joint_angles.data = torch.clamp(
                new_grasp_config.joint_angles,
                min=self.joint_lower_limits,
                max=self.joint_upper_limits,
            )

            # Eval new
            new_losses = (
                self.nerf_evaluator.evaluate(new_grasp_config).detach().cpu().numpy()
            )

            # Update grasp config
            old_dict = grasp_config.as_dict()
            new_dict = new_grasp_config.as_dict()
            improved_idxs = new_losses < old_losses

            improved_dict = {}
            for key in old_dict.keys():
                assert old_dict[key].shape == new_dict[key].shape
                improved_dict[key] = old_dict[key].copy()
                improved_dict[key][improved_idxs] = new_dict[key][improved_idxs]

            grasp_config = AllegroGraspConfig.from_grasp_config_dict(
                improved_dict,
                check=False,  # Check causing issues, probably just numerical issues
            ).to(device=grasp_config.device)
            updated_losses = np.where(improved_idxs, new_losses, old_losses)

        return grasp_config, torch.from_numpy(updated_losses).float().to(
            grasp_config.device
        )


class NerfGradientOptimizer(nn.Module, Optimizer):
    def __init__(
        self,
        nerf_evaluator: NerfEvaluator,
        num_grasps: int,
        num_steps: int,
        finger_lr: float,
        grasp_dir_lr: float,
        wrist_lr: float,
        momentum: float,
        opt_fingers: bool,
        opt_grasp_dirs: bool,
        opt_wrist_pose: bool,
        use_adamw: bool,
        print_freq: int = 10,
    ) -> None:
        super().__init__()
        self.nerf_evaluator = nerf_evaluator
        self.num_grasps = num_grasps
        self.num_steps = num_steps
        self.finger_lr = finger_lr
        self.grasp_dir_lr = grasp_dir_lr
        self.wrist_lr = wrist_lr
        self.momentum = momentum
        self.opt_fingers = opt_fingers
        self.opt_grasp_dirs = opt_grasp_dirs
        self.opt_wrist_pose = opt_wrist_pose
        self.use_adamw = use_adamw
        self.print_freq = print_freq

    @property
    def n_optimized_grasps(self) -> int:
        return self.num_grasps

    def optimize(
        self, grasp_config: AllegroGraspConfig
    ) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        assert (
            len(grasp_config) >= self.n_optimized_grasps
        ), f"{len(grasp_config)} < {self.n_optimized_grasps}"
        grasp_config = grasp_config[: self.n_optimized_grasps]

        self.grasp_config = grasp_config

        if self.opt_fingers:
            self.joint_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.joint_angles],
                    lr=self.finger_lr,
                    momentum=self.momentum,
                )
                if not self.use_adamw
                else torch.optim.AdamW(
                    [self.grasp_config.joint_angles],
                    lr=self.finger_lr,
                )
            )

        if self.opt_wrist_pose:
            self.wrist_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.wrist_pose],
                    lr=self.wrist_lr,
                    momentum=self.momentum,
                )
                if not self.use_adamw
                else torch.optim.AdamW(
                    [self.grasp_config.wrist_pose],
                    lr=self.wrist_lr,
                )
            )

        if self.opt_grasp_dirs:
            self.grasp_dir_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.grasp_orientations],
                    lr=self.grasp_dir_lr,
                    momentum=self.momentum,
                )
                if not self.use_adamw
                else torch.optim.AdamW(
                    [self.grasp_config.grasp_orientations],
                    lr=self.grasp_dir_lr,
                )
            )

        self.joint_lower_limits, self.joint_upper_limits = get_joint_limits(
            hand_model_type=grasp_config.hand_model_type,
            device=self.grasp_config.device,
        )
        assert self.joint_lower_limits.shape == (16,)
        assert self.joint_upper_limits.shape == (16,)

        self.grasp_config.joint_angles.data = torch.clamp(
            self.grasp_config.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )

        for iter in tqdm(range(self.num_steps), desc="Optimizing grasps"):
            if iter % self.print_freq == 0:
                losses_np = (
                    self.nerf_evaluator.evaluate(self.grasp_config)
                    .detach()
                    .cpu()
                    .numpy()
                )
                print(
                    f"Iter: {iter} | Losses: {np.round(losses_np.tolist(), decimals=3)} | Min loss: {losses_np.min():.3f} | Max loss: {losses_np.max():.3f} | Mean loss: {losses_np.mean():.3f} | Std dev: {losses_np.std():.3f}"
                )

            losses = self._step()
        assert (
            len(self.grasp_config) == self.n_optimized_grasps
        ), f"{len(self.grasp_config)} != {self.n_optimized_grasps}"
        return self.grasp_config, losses

    def _step(self) -> torch.Tensor:
        if self.opt_fingers:
            self.joint_optimizer.zero_grad()
        if self.opt_wrist_pose:
            self.wrist_optimizer.zero_grad()
        if self.opt_grasp_dirs:
            self.grasp_dir_optimizer.zero_grad()

        losses = self.nerf_evaluator.evaluate(self.grasp_config)
        assert losses.shape == (len(self.grasp_config),)

        losses.sum().backward()  # Should be sum so gradient magnitude per parameter is invariant to batch size.

        if self.opt_fingers:
            self.joint_optimizer.step()
        if self.opt_wrist_pose:
            self.wrist_optimizer.step()
        if self.opt_grasp_dirs:
            self.grasp_dir_optimizer.step()

        # Clip joint angles to feasible range.
        self.grasp_config.joint_angles.data = torch.clamp(
            self.grasp_config.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )
        return losses


class NoOptimizer(Optimizer):
    def __init__(self, num_grasps: int) -> None:
        self.num_grasps = num_grasps

    def optimize(
        self, grasp_config: AllegroGraspConfig
    ) -> Tuple[AllegroGraspConfig, torch.Tensor]:
        return grasp_config[: self.n_optimized_grasps], torch.linspace(
            0, 0.001, self.n_optimized_grasps
        )

    @property
    def n_optimized_grasps(self) -> int:
        return self.num_grasps
