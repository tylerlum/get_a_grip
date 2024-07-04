import pypose as pp
import torch

from get_a_grip.grasp_planning.utils.optimizer_utils import get_joint_limits
from get_a_grip.model_training.models.dex_evaluator import DexEvaluator


class RandomSamplingOptimizer:
    def __init__(
        self, dex_evaluator: DexEvaluator, bps: torch.Tensor, init_grasps: torch.Tensor
    ):
        self.dex_evaluator = dex_evaluator
        self.bps = bps
        self.grasps = init_grasps

        self.trans_noise = 0.005
        self.rot_noise = 0.05
        self.joint_angle_noise = 0.01
        self.grasp_orientation_noise = 0.05

        joint_lower_limits, joint_upper_limits = get_joint_limits()
        self.joint_lower_limits, self.joint_upper_limits = (
            torch.from_numpy(joint_lower_limits).float().to(self.grasps.device),
            torch.from_numpy(joint_upper_limits).float().to(self.grasps.device),
        )

    def step(self) -> torch.Tensor:
        with torch.no_grad():
            old_losses = 1 - self.dex_evaluator(f_O=self.bps, g_O=self.grasps)[:, -1]
            # print(f"Old grasps: {self.grasps}")

            new_grasps = self.add_noise(self.grasps)

            new_losses = 1 - self.dex_evaluator(f_O=self.bps, g_O=new_grasps)[:, -1]

            # Update grasp config
            improved_idxs = new_losses < old_losses

            self.grasps[improved_idxs] = new_grasps[improved_idxs]
            updated_losses = torch.where(improved_idxs, new_losses, old_losses)
            # print(f"new_grasps: {new_grasps}")
            # print(f"improved_idxs: {improved_idxs}")
            # print(f"self.grasps: {self.grasps}")
            print(f"Old: {old_losses}")
            print(f"Noise: {new_losses}")
            print(f"New: {updated_losses}")
            print()
            return updated_losses

    def add_noise(self, grasps):
        B = grasps.shape[0]
        N_FINGERS = 4
        xyz_noise = torch.randn_like(grasps[:, :3]) * self.trans_noise
        rot_noise = (pp.randn_so3(B) * self.rot_noise).Exp().to(device=grasps.device)
        joint_noise = torch.randn_like(grasps[:, 9:25]) * self.joint_angle_noise
        grasp_orientation_perturbations = (
            (pp.randn_so3((B, N_FINGERS)) * self.grasp_orientation_noise)
            .Exp()
            .to(device=grasps.device)
        )

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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dex_evaluator = DexEvaluator(in_grasp=37, in_bps=4096).to(device)
    dex_evaluator.eval()
    bps = torch.zeros(2, 4096).to(device)
    init_grasps = torch.randn(2, 37).to(device)

    random_samping_optimizer = RandomSamplingOptimizer(
        dex_evaluator=dex_evaluator, bps=bps, init_grasps=init_grasps
    )

    losses_list = []
    for i in range(10):
        losses_list.append(random_samping_optimizer.step().tolist())

    random_samping_optimizer.grasps
    for losses in losses_list:
        print(f"{losses}")


if __name__ == "__main__":
    main()
