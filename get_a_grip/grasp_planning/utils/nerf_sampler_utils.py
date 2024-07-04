from __future__ import annotations

import math
import pathlib

import numpy as np
import plotly.graph_objects as go
import pypose as pp
import torch
import trimesh
from nerfstudio.models.base_model import Model
from tqdm import tqdm

from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.grasp_planning.utils.grasp_utils import (
    compute_grasp_orientations,
    rot6d_to_matrix,
)
from get_a_grip.grasp_planning.utils.visualize_utils import (
    plot_mesh_and_grasp,
    plot_nerf_densities,
)
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from get_a_grip.model_training.models.nerf_sampler import NerfSampler
from get_a_grip.model_training.utils.diffusion import Diffusion
from get_a_grip.model_training.utils.nerf_utils import (
    get_density,
)
from get_a_grip.model_training.utils.point_utils import (
    get_points_in_grid,
    transform_points,
)


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_model: Model,
    X_N_Oy: np.ndarray,
    ckpt_path: pathlib.Path,
    return_exactly_requested_num_grasps: bool = True,
    sample_grasps_multiplier: int = 10,
) -> dict:
    NUM_GRASPS = cfg.optimizer.num_grasps

    config = DiffusionConfig(
        training=TrainingConfig(
            log_path=ckpt_path.parent,
        ),
    )
    model = NerfSampler(
        global_grid_shape=(
            4,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        ),
        grasp_dim=config.data.grasp_dim,
        d_model=128,
        virtual_seq_len=4,
        conv_channels=(32, 64, 128),
    )
    model.HACK_MODE_FOR_PERFORMANCE = (
        True  # Big hack to speed up from sampling wasting dumb compute
    )

    runner = Diffusion(config=config, model=model, load_multigpu_ckpt=True)
    runner.load_checkpoint(config, name=ckpt_path.stem)
    device = runner.device

    # Get nerf densities global
    # Must sample points in grid in Oy before transforming to Nz
    query_points_Oy = get_points_in_grid(
        lb=lb_Oy,
        ub=ub_Oy,
        num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
        num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
        num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
    )
    assert query_points_Oy.shape == (
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    )
    query_points_Nz = transform_points(
        T=X_N_Oy,
        points=query_points_Oy.reshape(-1, 3),
    ).reshape(
        NERF_DENSITIES_GLOBAL_NUM_X,
        NERF_DENSITIES_GLOBAL_NUM_Y,
        NERF_DENSITIES_GLOBAL_NUM_Z,
        3,
    )
    nerf_densities_global = (
        get_density(
            field=nerf_model.field,
            positions=torch.from_numpy(query_points_Nz).cuda(),
        )[0]
        .squeeze(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .reshape(
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    )

    assert (
        nerf_densities_global.shape
        == (
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape ({NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global.shape}"
    assert (
        query_points_Nz.shape
        == (
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
            3,
        )
    ), f"Expected shape ({NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}, 3), got {query_points_Nz.shape}"

    X_Oy_N = np.linalg.inv(X_N_Oy)
    nerf_densities_global_with_coords = np.concatenate(
        [nerf_densities_global[..., None], query_points_Oy], axis=-1
    )
    assert (
        nerf_densities_global_with_coords.shape
        == (
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
            4,
        )
    ), f"Expected shape ({NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}, 4), got {nerf_densities_global_with_coords.shape}"
    nerf_densities_global_with_coords = nerf_densities_global_with_coords.transpose(
        3, 0, 1, 2
    )
    assert (
        nerf_densities_global_with_coords.shape
        == (
            4,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
    ), f"Expected shape (4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords.shape}"

    nerf_densities_global_with_coords = (
        torch.from_numpy(nerf_densities_global_with_coords).float().to(device)
    )

    # We sample more grasps than needed to account for filtering
    NUM_GRASP_SAMPLES = sample_grasps_multiplier * NUM_GRASPS
    MAX_BATCH_SIZE = 400
    if NUM_GRASP_SAMPLES > MAX_BATCH_SIZE:
        nerf_densities_global_with_coords_repeated = (
            nerf_densities_global_with_coords.unsqueeze(0).repeat(
                MAX_BATCH_SIZE, 1, 1, 1, 1
            )
        )

        assert (
            nerf_densities_global_with_coords_repeated.shape
            == (
                MAX_BATCH_SIZE,
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape ({MAX_BATCH_SIZE}, 4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords_repeated.shape}"

        n_batches = int(math.ceil(NUM_GRASP_SAMPLES / MAX_BATCH_SIZE))
        x_list = []
        for i in tqdm(range(n_batches), desc="Sampling grasps in batches"):
            start_idx = i * MAX_BATCH_SIZE
            end_idx = min((i + 1) * MAX_BATCH_SIZE, NUM_GRASP_SAMPLES)
            num_samples = end_idx - start_idx

            xT = torch.randn(num_samples, config.data.grasp_dim, device=runner.device)
            x = runner.sample(
                xT=xT,
                cond=nerf_densities_global_with_coords_repeated[:num_samples],
            ).cpu()
            x_list.append(x)
        x = torch.cat(x_list, dim=0)
    else:
        nerf_densities_global_with_coords_repeated = (
            nerf_densities_global_with_coords.unsqueeze(0).repeat(
                NUM_GRASP_SAMPLES, 1, 1, 1, 1
            )
        )

        assert (
            nerf_densities_global_with_coords_repeated.shape
            == (
                NUM_GRASP_SAMPLES,
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape ({NUM_GRASP_SAMPLES}, 4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords_repeated.shape}"

        # Sample grasps
        xT = torch.randn(NUM_GRASP_SAMPLES, config.data.grasp_dim, device=runner.device)
        x = runner.sample(xT=xT, cond=nerf_densities_global_with_coords_repeated)

    PLOT = False
    if PLOT:
        fig = go.Figure()
        nerf_densities_global_flattened = nerf_densities_global.reshape(-1)
        query_points_global_N_flattened = query_points_Oy.reshape(-1, 3)
        THRESHOLD = 15
        plot_nerf_densities(
            fig=fig,
            densities=torch.from_numpy(
                nerf_densities_global_flattened[
                    nerf_densities_global_flattened > THRESHOLD
                ]
            ),
            query_points=torch.from_numpy(
                query_points_global_N_flattened[
                    nerf_densities_global_flattened > THRESHOLD
                ]
            ),
            name="global",
        )

        X_Oy_N = np.linalg.inv(X_N_Oy)
        _mesh_N = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_Oy = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_Oy.apply_transform(X_Oy_N)

        GRASP_IDX = 0
        breakpoint()
        plot_mesh_and_grasp(
            fig=fig,
            mesh_Oy=mesh_Oy,
            grasp=x[GRASP_IDX],
        )
        fig.show()

    # grasp to AllegroGraspConfig
    # TODO: make the numpy torch conversions less bad
    N_FINGERS = 4
    assert x.shape == (
        NUM_GRASP_SAMPLES,
        config.data.grasp_dim,
    ), f"Expected shape ({NUM_GRASP_SAMPLES}, {config.data.grasp_dim}), got {x.shape}"
    trans = x[:, :3].detach().cpu().numpy()
    rot6d = x[:, 3:9].detach().cpu().numpy()
    joint_angles = x[:, 9:25].detach().cpu().numpy()
    grasp_dirs = (
        x[:, 25:37].reshape(NUM_GRASP_SAMPLES, N_FINGERS, 3).detach().cpu().numpy()
    )

    rot = rot6d_to_matrix(rot6d)

    wrist_pose_matrix = (
        torch.eye(4, device=device).unsqueeze(0).repeat(NUM_GRASP_SAMPLES, 1, 1).float()
    )
    wrist_pose_matrix[:, :3, :3] = torch.from_numpy(rot).float().to(device)
    wrist_pose_matrix[:, :3, 3] = torch.from_numpy(trans).float().to(device)

    wrist_pose = pp.from_matrix(
        wrist_pose_matrix.to(device),
        pp.SE3_type,
        atol=1e-3,
        rtol=1e-3,
    )

    assert wrist_pose.lshape == (NUM_GRASP_SAMPLES,)

    grasp_orientations = compute_grasp_orientations(
        grasp_dirs=torch.from_numpy(grasp_dirs).float().to(device),
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
    )

    # Convert to AllegroGraspConfig to dict
    grasp_configs = AllegroGraspConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
        grasp_orientations=grasp_orientations,
    )

    if cfg.filter_less_feasible_grasps:
        wrist_pose_matrix = grasp_configs.wrist_pose.matrix()
        x_dirs = wrist_pose_matrix[:, :, 0]
        z_dirs = wrist_pose_matrix[:, :, 2]

        fingers_forward_cos_theta = math.cos(
            math.radians(cfg.fingers_forward_theta_deg)
        )
        palm_upwards_cos_theta = math.cos(math.radians(cfg.palm_upwards_theta_deg))
        fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
        palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta
        grasp_configs = grasp_configs[fingers_forward & ~palm_upwards]
        print(
            f"Filtered less feasible grasps. New batch size: {grasp_configs.batch_size}"
        )

    if len(grasp_configs) < NUM_GRASPS:
        print(
            f"WARNING: After filtering, only {len(grasp_configs)} grasps remain, less than the requested {NUM_GRASPS} grasps"
        )

    if return_exactly_requested_num_grasps:
        grasp_configs = grasp_configs[:NUM_GRASPS]
    print(f"Returning {grasp_configs.batch_size} grasps")

    grasp_config_dicts = grasp_configs.as_dict()
    grasp_config_dicts["loss"] = np.linspace(
        0, 0.001, len(grasp_configs)
    )  # HACK: Currently don't have a loss, but need something here to sort

    return grasp_config_dicts
