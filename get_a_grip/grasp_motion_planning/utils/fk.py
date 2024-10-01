import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
)
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)

from get_a_grip.dataset_generation.utils.torch_quat_utils import (
    quat_wxyz_to_matrix,
)


def solve_fk(
    q_fr3: np.ndarray,
    q_algr: np.ndarray,
) -> np.ndarray:
    assert q_fr3.shape == (7,)
    assert q_algr.shape == (16,)

    X_W_H = solve_fks(
        q_fr3s=q_fr3[None, ...],
        q_algrs=q_algr[None, ...],
    ).squeeze(axis=0)
    assert X_W_H.shape == (4, 4)

    return X_W_H


def solve_fks(
    q_fr3s: np.ndarray,
    q_algrs: np.ndarray,
) -> np.ndarray:
    N = q_fr3s.shape[0]
    assert q_fr3s.shape == (
        N,
        7,
    )
    assert q_algrs.shape == (
        N,
        16,
    )

    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    q = torch.from_numpy(np.concatenate([q_fr3s, q_algrs], axis=1)).float().to(device)
    assert q.shape == (N, 23)

    state = kin_model.get_state(q)
    trans = state.ee_position.detach().cpu().numpy()
    rot_matrix = quat_wxyz_to_matrix(state.ee_quaternion).detach().cpu().numpy()

    X_W_H = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    X_W_H[:, :3, :3] = rot_matrix
    X_W_H[:, :3, 3] = trans
    return X_W_H


def main() -> None:
    DEFAULT_Q_FR3 = np.array([0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
    DEFAULT_Q_ALGR = np.array(
        [
            2.90945620e-01,
            7.37109400e-01,
            5.10859200e-01,
            1.22637060e-01,
            1.20125350e-01,
            5.84513500e-01,
            3.43829930e-01,
            6.05035000e-01,
            -2.68431900e-01,
            8.78457900e-01,
            8.49713500e-01,
            8.97218400e-01,
            1.33282830e00,
            3.47787830e-01,
            2.09215670e-01,
            -6.50969000e-03,
        ]
    )

    X_W_H = solve_fk(q_fr3=DEFAULT_Q_FR3, q_algr=DEFAULT_Q_ALGR)
    print(f"X_W_H: {X_W_H}")


if __name__ == "__main__":
    main()
