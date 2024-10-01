import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_grasp_orientations_from_z_dirs,
)
from get_a_grip.dataset_generation.utils.pose_conversion import (
    hand_config_np_to_pose,
)


@dataclass
class AugmentGraspConfigDictsArgs:
    input_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        get_data_folder() / "dataset/NEW/evaled_grasp_config_dicts"
    )
    output_augmented_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        get_data_folder() / "dataset/NEW/augmented_grasp_config_dicts"
    )

    trans_max_noise = 0.005
    rot_deg_max_noise = 2.5
    joint_pos_rad_max_noise = 0.05
    grasp_orientation_deg_max_noise = 10

    num_noisy_samples_per_grasp_same_object: int = 10
    num_noisy_samples_per_grasp_different_object: int = 2

    hand_model_type: HandModelType = HandModelType.ALLEGRO
    all_mid_optimization_steps: bool = False
    noise_type: Literal["uniform", "normal", "halton"] = "halton"
    debug_plot: bool = False


def add_noise_to_rot_matrices(
    rot_matrices: np.ndarray,
    rpy_noise: np.ndarray,
) -> np.ndarray:
    N = rot_matrices.shape[0]
    assert rot_matrices.shape == (N, 3, 3)
    assert rpy_noise.shape == (N, 3)

    R_x = np.eye(3)[None, ...].repeat(N, axis=0)
    R_y = np.eye(3)[None, ...].repeat(N, axis=0)
    R_z = np.eye(3)[None, ...].repeat(N, axis=0)

    R_x[:, 1, 1] = np.cos(rpy_noise[:, 0])
    R_x[:, 1, 2] = -np.sin(rpy_noise[:, 0])
    R_x[:, 2, 1] = np.sin(rpy_noise[:, 0])
    R_x[:, 2, 2] = np.cos(rpy_noise[:, 0])

    R_y[:, 0, 0] = np.cos(rpy_noise[:, 1])
    R_y[:, 0, 2] = np.sin(rpy_noise[:, 1])
    R_y[:, 2, 0] = -np.sin(rpy_noise[:, 1])
    R_y[:, 2, 2] = np.cos(rpy_noise[:, 1])

    R_z[:, 0, 0] = np.cos(rpy_noise[:, 2])
    R_z[:, 0, 1] = -np.sin(rpy_noise[:, 2])
    R_z[:, 1, 0] = np.sin(rpy_noise[:, 2])
    R_z[:, 1, 1] = np.cos(rpy_noise[:, 2])

    R_zy = np.einsum("ijk,ikl->ijl", R_z, R_y)
    R_zyx = np.einsum("ijk,ikl->ijl", R_zy, R_x)

    new_rot_matrices = np.einsum("ijk,ikl->ijl", rot_matrices, R_zyx)
    return new_rot_matrices


def add_noise_to_dirs(dirs: np.ndarray, theta_phi_noise: np.ndarray) -> np.ndarray:
    N = dirs.shape[0]
    assert dirs.shape == (N, 3)
    assert theta_phi_noise.shape == (N, 2)

    cos_thetas = np.cos(theta_phi_noise[:, 0])
    sin_thetas = np.sin(theta_phi_noise[:, 0])
    cos_phis = np.cos(theta_phi_noise[:, 1])
    sin_phis = np.sin(theta_phi_noise[:, 1])

    # Rotation around z-axis
    RRz = np.eye(3)[None, ...].repeat(N, axis=0)
    RRz[:, 0, 0] = cos_thetas
    RRz[:, 0, 1] = -sin_thetas
    RRz[:, 1, 0] = sin_thetas
    RRz[:, 1, 1] = cos_thetas

    RRy = np.eye(3)[None, ...].repeat(N, axis=0)
    RRy[:, 0, 0] = cos_phis
    RRy[:, 0, 2] = sin_phis
    RRy[:, 2, 0] = -sin_phis
    RRy[:, 2, 2] = cos_phis

    RRyz = np.einsum("ijk,ikl->ijl", RRy, RRz)
    new_z_dirs = np.einsum("ik,ikl->il", dirs, RRyz)
    return new_z_dirs


def clamp_joint_angles(
    joint_angles: np.ndarray,
    hand_model: HandModel,
) -> np.ndarray:
    N = joint_angles.shape[0]
    assert joint_angles.shape == (N, 16)

    joint_lowers = hand_model.joints_lower.detach().cpu().numpy()
    joint_uppers = hand_model.joints_upper.detach().cpu().numpy()
    new_joint_angles = np.clip(joint_angles, joint_lowers[None], joint_uppers[None])
    return new_joint_angles


def sample_noise(
    shape: Tuple[int, ...],
    scale: float,
    noise_type: Literal["uniform", "normal", "halton"],
) -> np.ndarray:
    batch_dims = shape[:-1]
    d = shape[-1]

    if noise_type == "halton":
        from scipy.stats.qmc import Halton

        N = np.prod(batch_dims)
        noise = (Halton(d=d, scramble=True).random(n=N) * 2 - 1) * scale
        noise = noise.reshape(*batch_dims, d)
    elif noise_type == "uniform":
        noise = np.random.uniform(low=-scale, high=scale, size=(*batch_dims, d))
    elif noise_type == "normal":
        noise = np.random.normal(loc=0, scale=scale / 2, size=(*batch_dims, d))
    else:
        raise ValueError(f"Invalid noise_type: {noise_type}")

    assert noise.shape == (*batch_dims, d)
    return noise


def add_noise(
    data_dict: dict,
    N_noisy: int,
    hand_model: HandModel,
    trans_max_noise: float,
    rot_deg_max_noise: float,
    joint_pos_max_noise: float,
    grasp_orientation_deg_max_noise: float,
    noise_type: Literal["uniform", "normal", "halton"],
) -> dict:
    N_FINGERS = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if N_noisy == 0:
        return {}

    B = data_dict["trans"].shape[0]
    if B == 0:
        return {}

    xyz_noise = sample_noise(
        shape=(B, N_noisy, 3), scale=trans_max_noise, noise_type=noise_type
    )
    rpy_noise = sample_noise(
        shape=(B, N_noisy, 3),
        scale=np.deg2rad(rot_deg_max_noise),
        noise_type=noise_type,
    )
    joint_angles_noise = sample_noise(
        shape=(B, N_noisy, 16), scale=joint_pos_max_noise, noise_type=noise_type
    )
    grasp_orientation_noise = sample_noise(
        shape=(B, N_noisy, N_FINGERS, 2),
        scale=np.deg2rad(grasp_orientation_deg_max_noise),
        noise_type=noise_type,
    )

    # 0 noise for the first noisy sample of each batch dim
    xyz_noise[:, 0] = 0
    rpy_noise[:, 0] = 0
    joint_angles_noise[:, 0] = 0
    grasp_orientation_noise[:, 0] = 0

    orig_trans = data_dict["trans"]
    orig_rot = data_dict["rot"]
    orig_joint_angles = data_dict["joint_angles"]
    orig_grasp_orientations = data_dict["grasp_orientations"]

    assert orig_trans.shape == (B, 3)
    assert orig_rot.shape == (B, 3, 3)
    assert orig_joint_angles.shape == (B, 16)
    assert orig_grasp_orientations.shape == (B, N_FINGERS, 3, 3)

    new_data_dict = {}

    # trans
    new_trans = orig_trans[:, None, ...].repeat(N_noisy, axis=1)
    new_trans = (new_trans + xyz_noise).reshape(N_noisy * B, 3)
    new_data_dict["trans"] = new_trans

    # rot
    new_rot = orig_rot[:, None, ...].repeat(N_noisy, axis=1)
    new_rot = add_noise_to_rot_matrices(
        rot_matrices=new_rot.reshape(B * N_noisy, 3, 3),
        rpy_noise=rpy_noise.reshape(B * N_noisy, 3),
    ).reshape(N_noisy * B, 3, 3)
    new_data_dict["rot"] = new_rot

    # joint_angles
    new_joint_angles = orig_joint_angles[:, None, ...].repeat(N_noisy, axis=1)
    new_joint_angles += joint_angles_noise
    new_joint_angles = clamp_joint_angles(
        joint_angles=new_joint_angles.reshape(N_noisy * B, 16), hand_model=hand_model
    )
    new_data_dict["joint_angles"] = new_joint_angles

    # hand_model
    hand_pose = hand_config_np_to_pose(
        trans=new_trans, rot=new_rot, joint_angles=new_joint_angles
    ).to(hand_model.device)
    hand_model.set_parameters(hand_pose)

    # grasp_orientations
    orig_z_dirs = orig_grasp_orientations[:, :, :, 2]
    new_z_dirs = orig_z_dirs[:, None, ...].repeat(N_noisy, axis=1)
    new_z_dirs = add_noise_to_dirs(
        dirs=new_z_dirs.reshape(B * N_noisy * N_FINGERS, 3),
        theta_phi_noise=grasp_orientation_noise.reshape(B * N_noisy * N_FINGERS, 2),
    ).reshape(N_noisy * B, N_FINGERS, 3)
    new_grasp_orientations = compute_grasp_orientations_from_z_dirs(
        joint_angles_start=torch.from_numpy(new_joint_angles).float().to(device),
        hand_model=hand_model,
        z_dirs=torch.from_numpy(new_z_dirs).float().to(device),
    )
    new_data_dict["grasp_orientations"] = new_grasp_orientations.detach().cpu().numpy()

    # Do not copy other keys to be safe, just the essentials
    # for k, v in data_dict.items():
    #     if k in ["trans", "rot", "joint_angles", "grasp_orientations"]:
    #         continue

    #     new_data_dict[k] = (
    #         v[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B, *v.shape[1:])
    #     )
    return new_data_dict


def augment_grasp_config_dicts(
    args: AugmentGraspConfigDictsArgs,
    input_evaled_grasp_config_dicts_path: pathlib.Path,
    output_augmented_grasp_config_dicts_path: pathlib.Path,
) -> None:
    OUTPUT_PATH = output_augmented_grasp_config_dicts_path
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)

    # Step 0: Get all data paths
    all_data_paths = sorted(list(input_evaled_grasp_config_dicts_path.glob("*.npy")))
    print(f"Found {len(all_data_paths)} data_paths")

    # Step 1: Get obj_to_all_grasps
    obj_to_all_grasps = {
        data_path.stem: np.load(data_path, allow_pickle=True).item()
        for data_path in tqdm(all_data_paths, desc="Loading data")
    }

    objs = list(obj_to_all_grasps.keys())

    # Step 2: Get obj_to_good_grasps
    obj_to_good_grasps = {}
    for obj, all_grasps_dict in tqdm(obj_to_all_grasps.items(), desc="Filtering data"):
        good_idxs = all_grasps_dict["y_PGS"] > 0.5
        good_data_dict = {k: v[good_idxs] for k, v in all_grasps_dict.items()}
        if good_data_dict["y_PGS"].shape[0] > 0:
            obj_to_good_grasps[obj] = good_data_dict

    # Debug print
    print(f"len(objs): {len(objs)}")
    print(f"len(obj_to_good_grasps): {len(obj_to_good_grasps)}")

    DEBUG = False
    if DEBUG:
        for obj, good_grasps_dict in obj_to_good_grasps.items():
            print(f"{obj}: {good_grasps_dict['y_PGS'].shape}")
        print()
        for obj, all_grasps_dict in obj_to_all_grasps.items():
            print(f"{obj}: {all_grasps_dict['y_PGS'].shape}")

    num_good_grasps = 0
    for good_grasps_dict in obj_to_good_grasps.values():
        num_good_grasps += good_grasps_dict["trans"].shape[0]
    num_all_grasps = 0
    for all_grasps_dict in obj_to_all_grasps.values():
        num_all_grasps += all_grasps_dict["trans"].shape[0]
    print(
        f"num_good_grasps/num_all_grasps: {num_good_grasps}/{num_all_grasps} = {num_good_grasps/num_all_grasps:.2f}"
    )

    # Debug plot
    if args.debug_plot:
        # Test adding noise
        good_objs = list(obj_to_good_grasps.keys())
        test_input = obj_to_good_grasps[good_objs[1]]
        test_n_noisy = 3
        test_output = add_noise(
            data_dict=test_input,
            N_noisy=test_n_noisy,
            hand_model=hand_model,
            trans_max_noise=args.trans_max_noise,
            rot_deg_max_noise=args.rot_deg_max_noise,
            joint_pos_max_noise=args.joint_pos_rad_max_noise,
            grasp_orientation_deg_max_noise=args.grasp_orientation_deg_max_noise,
            noise_type=args.noise_type,
        )

        print(f"test_input trans: {test_input['trans'].shape}")
        print(f"test_output trans: {test_output['trans'].shape}")

        object_idx = 1
        noise_idx = 1
        test_input_trans = test_input["trans"][object_idx]
        test_output_trans = test_output["trans"][test_n_noisy * object_idx + noise_idx]
        trans_diff = np.linalg.norm(test_input_trans - test_output_trans)
        print(
            f"for object_idx: {object_idx}, noise_idx: {noise_idx}, trans diff: {trans_diff}"
        )

        # hand_model
        input_hand_pose = hand_config_np_to_pose(
            trans=test_input["trans"],
            rot=test_input["rot"],
            joint_angles=test_input["joint_angles"],
        ).to(device)
        hand_model.set_parameters(input_hand_pose)
        input_plotly = hand_model.get_plotly_data(
            i=object_idx, color="lightblue", opacity=1.0
        )

        output_hand_pose = hand_config_np_to_pose(
            trans=test_output["trans"],
            rot=test_output["rot"],
            joint_angles=test_output["joint_angles"],
        ).to(device)
        hand_model.set_parameters(output_hand_pose)
        output_plotly = hand_model.get_plotly_data(
            i=test_n_noisy * object_idx + noise_idx,
            color="lightgreen",
            opacity=0.5,
        )

        fig = go.Figure(data=input_plotly + output_plotly)
        fig.show()

    # Step 3: Get obj_to_noisy_good_grasps, obj_to_noisy_other_grasps
    obj_to_noisy_good_grasps = defaultdict(list)
    obj_to_noisy_other_grasps = defaultdict(list)

    N_noisy = args.num_noisy_samples_per_grasp_same_object
    N_other = args.num_noisy_samples_per_grasp_different_object
    for obj, good_grasps_dict in tqdm(obj_to_good_grasps.items(), desc="Adding noise"):
        # Add noise
        noisy_good_data_dict = add_noise(
            data_dict=good_grasps_dict,
            N_noisy=N_noisy,
            hand_model=hand_model,
            trans_max_noise=args.trans_max_noise,
            rot_deg_max_noise=args.rot_deg_max_noise,
            joint_pos_max_noise=args.joint_pos_rad_max_noise,
            grasp_orientation_deg_max_noise=args.grasp_orientation_deg_max_noise,
            noise_type=args.noise_type,
        )
        noisy_other_data_dict = add_noise(
            data_dict=good_grasps_dict,
            N_noisy=N_other,
            hand_model=hand_model,
            trans_max_noise=args.trans_max_noise,
            rot_deg_max_noise=args.rot_deg_max_noise,
            joint_pos_max_noise=args.joint_pos_rad_max_noise,
            grasp_orientation_deg_max_noise=args.grasp_orientation_deg_max_noise,
            noise_type=args.noise_type,
        )

        # Get other object
        other_obj = np.random.choice([o for o in objs if o != obj])
        assert other_obj != obj, f"other_obj == obj: {other_obj} == {obj}, objs: {objs}"

        if len(noisy_good_data_dict) > 0:
            obj_to_noisy_good_grasps[obj].append(noisy_good_data_dict)
        if len(noisy_other_data_dict) > 0:
            obj_to_noisy_other_grasps[other_obj].append(noisy_other_data_dict)

    # Step 4: Aggregate
    obj_to_new_grasps = {}
    for obj in tqdm(objs, desc="Aggregating"):
        new_dicts = obj_to_noisy_good_grasps[obj] + obj_to_noisy_other_grasps[obj]
        if len(new_dicts) == 0:
            continue

        new_dict = {
            k: np.concatenate([d[k] for d in new_dicts if k in d], axis=0)
            for k in new_dicts[0].keys()
        }
        if len(new_dict) == 0:
            continue
        obj_to_new_grasps[obj] = new_dict

    # Step 5: Save
    for obj, new_dict in tqdm(obj_to_new_grasps.items(), desc="Saving"):
        np.save(file=OUTPUT_PATH / f"{obj}.npy", arr=new_dict)

    # Step 6: Count
    n_grasps = 0
    for obj, new_dict in tqdm(obj_to_new_grasps.items(), desc="Counting"):
        n_grasps += new_dict["trans"].shape[0]
    print(
        f"n_grasps: {n_grasps} across {len(obj_to_new_grasps)} objects in {OUTPUT_PATH}"
    )


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[AugmentGraspConfigDictsArgs])

    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    augment_grasp_config_dicts(
        args=args,
        input_evaled_grasp_config_dicts_path=args.input_evaled_grasp_config_dicts_path,
        output_augmented_grasp_config_dicts_path=args.output_augmented_grasp_config_dicts_path,
    )

    if not args.all_mid_optimization_steps:
        return

    mid_optimization_steps = (
        sorted(
            [
                int(pp.name)
                for pp in args.input_evaled_grasp_config_dicts_path.glob(
                    "mid_optimization/*"
                )
            ]
        )
        if (args.input_evaled_grasp_config_dicts_path / "mid_optimization").exists()
        else []
    )
    print(f"mid_optimization_steps: {mid_optimization_steps}")

    for mid_optimization_step in mid_optimization_steps:
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_path = (
            args.input_evaled_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_path = (
            args.output_augmented_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        augment_grasp_config_dicts(
            args=args,
            input_evaled_grasp_config_dicts_path=mid_optimization_input_path,
            output_augmented_grasp_config_dicts_path=mid_optimization_output_path,
        )


if __name__ == "__main__":
    main()
