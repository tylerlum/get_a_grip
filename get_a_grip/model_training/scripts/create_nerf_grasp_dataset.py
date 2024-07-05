"""
The purpose of this script is to iterate through each NeRF object and evaled grasp config,
sample densities in a fixed grid and along the fingertip trajectories,
and store the data to an HDF5 file for training.
"""

import os
import pathlib
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pypose as pp
import torch
import trimesh
import tyro
from clean_loop_timer import LoopTimer
from tqdm import tqdm as std_tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.model_training.config.base import CONFIG_DATETIME_STR
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from get_a_grip.model_training.config.nerfdata_config import (
    GridNerfDataConfig,
)
from get_a_grip.model_training.utils.nerf_load_utils import (
    get_nerf_configs,
    load_nerf_pipeline,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
    get_ray_samples,
)
from get_a_grip.model_training.utils.nerf_utils import (
    get_densities_in_grid,
)
from get_a_grip.model_training.utils.point_utils import (
    transform_point,
)

tqdm = partial(std_tqdm, dynamic_ncols=True)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def nerf_config_to_object_code_and_scale_str(nerf_config: pathlib.Path) -> str:
    # Input: PosixPath('2023-08-25_nerfcheckpoints/sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_1000/nerfacto/2023-08-25_132225/config.yml')
    # Return sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_1000
    parts = nerf_config.parts
    object_code_and_scale_str = parts[-4]
    return object_code_and_scale_str


def get_matching_nerf_config(
    target_object_code_and_scale_str: str, nerf_configs: List[pathlib.Path]
) -> pathlib.Path:
    # Prepare data for comparisons
    nerf_object_code_and_scale_strs = [
        nerf_config_to_object_code_and_scale_str(config) for config in nerf_configs
    ]

    # Check for exact match
    exact_matches = [
        nerf_config
        for object_code_and_scale_str, nerf_config in zip(
            nerf_object_code_and_scale_strs, nerf_configs
        )
        if target_object_code_and_scale_str == object_code_and_scale_str
    ]
    if exact_matches:
        if len(exact_matches) > 1:
            print(
                f"Multiple exact matches found for {target_object_code_and_scale_str}, {exact_matches}"
            )
        return exact_matches[0]

    raise ValueError(f"No exact matches found for {target_object_code_and_scale_str}")


def count_total_num_grasps(
    evaled_grasp_config_dict_filepaths: List[pathlib.Path],
    actually_count_all: bool = True,
) -> int:
    total_num_grasps = 0

    for evaled_grasp_config_dict_filepath in tqdm(
        evaled_grasp_config_dict_filepaths,
        desc="counting num grasps",
        dynamic_ncols=True,
    ):
        # Read in grasp data
        assert evaled_grasp_config_dict_filepath.exists(), f"evaled_grasp_config_dict_filepath {evaled_grasp_config_dict_filepath} does not exist"
        evaled_grasp_config_dict: Dict[str, Any] = np.load(
            evaled_grasp_config_dict_filepath, allow_pickle=True
        ).item()

        num_grasps = evaled_grasp_config_dict["trans"].shape[0]
        assert_equals(
            evaled_grasp_config_dict["trans"].shape,
            (
                num_grasps,
                3,
            ),
        )  # Sanity check

        # Count num_grasps
        if not actually_count_all:
            print(
                f"assuming all {len(evaled_grasp_config_dict_filepaths)} evaled grasp config dicts have {num_grasps} grasps"
            )
            return num_grasps * len(evaled_grasp_config_dict_filepaths)

        total_num_grasps += num_grasps
    return total_num_grasps


def create_mesh(object_code: str, object_scale: float) -> trimesh.Trimesh:
    MESHDATA_ROOT = get_data_folder() / "large/meshes"
    mesh_path = MESHDATA_ROOT / object_code / "coacd" / "decomposed.obj"
    assert mesh_path.exists(), f"mesh_path {mesh_path} does not exist"
    mesh_Oy = trimesh.load(mesh_path, force="mesh")
    mesh_Oy.apply_transform(trimesh.transformations.scale_matrix(object_scale))
    return mesh_Oy


def compute_X_N_Oy(mesh_Oy: trimesh.Trimesh) -> np.ndarray:
    N_FRAME_ORIGIN = "table"
    if N_FRAME_ORIGIN == "table":
        # N frame has origin at surface of table. Object is dropped onto table with x=0, y>0, z=0
        # It settles on the table, assume x=z=0, y>0. Can use the object's bound and scale to compute X_N_Oy
        bounds = mesh_Oy.bounds
        assert bounds.shape == (2, 3)
        min_y = bounds[0, 1]  # min_y is bottom of the object in Oy
        X_N_Oy = np.eye(4)
        X_N_Oy[
            1, 3
        ] = (
            -min_y
        )  # Eg. min_y=-0.1, then this means that the object is 0.1m above the table
    elif N_FRAME_ORIGIN == "object":
        X_N_Oy = np.eye(4)
    else:
        raise ValueError(f"Unknown N_FRAME_ORIGIN {N_FRAME_ORIGIN}")
    return X_N_Oy


def create_grid_dataset(
    cfg: GridNerfDataConfig,
    hdf5_file: h5py.File,
    max_num_datapoints: int,
    num_objects: int,
) -> Tuple[h5py.Dataset, ...]:
    assert cfg.fingertip_config is not None

    nerf_densities_dataset = hdf5_file.create_dataset(
        "/nerf_densities",
        shape=(
            max_num_datapoints,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
        ),
        dtype="f",
        chunks=(
            1,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
        ),
    )
    # Only need one global density field per object, not per grasp
    nerf_densities_global_dataset = hdf5_file.create_dataset(
        "/nerf_densities_global",
        shape=(
            num_objects,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        ),
        dtype="f",
        chunks=(
            1,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        ),
    )
    # Map from grasp idx to global density idx
    nerf_densities_global_idx_dataset = hdf5_file.create_dataset(
        "/nerf_densities_global_idx",
        shape=(max_num_datapoints,),
        dtype="i",
    )

    y_PGS_dataset = hdf5_file.create_dataset(
        "/y_PGS", shape=(max_num_datapoints,), dtype="f"
    )
    y_pick_dataset = hdf5_file.create_dataset(
        "/y_pick",
        shape=(max_num_datapoints,),
        dtype="f",
    )
    y_coll_dataset = hdf5_file.create_dataset(
        "/y_coll",
        shape=(max_num_datapoints,),
        dtype="f",
    )
    nerf_config_dataset = hdf5_file.create_dataset(
        "/nerf_config", shape=(max_num_datapoints,), dtype=h5py.string_dtype()
    )
    object_code_dataset = hdf5_file.create_dataset(
        "/object_code", shape=(max_num_datapoints,), dtype=h5py.string_dtype()
    )
    object_scale_dataset = hdf5_file.create_dataset(
        "/object_scale", shape=(max_num_datapoints,), dtype="f"
    )
    object_state_dataset = hdf5_file.create_dataset(
        "/object_state",
        shape=(
            max_num_datapoints,
            13,
        ),
        dtype="f",
    )
    grasp_idx_dataset = hdf5_file.create_dataset(
        "/grasp_idx", shape=(max_num_datapoints,), dtype="i"
    )
    grasp_transforms_dataset = hdf5_file.create_dataset(
        "/grasp_transforms",
        shape=(max_num_datapoints, cfg.fingertip_config.n_fingers, 4, 4),
        dtype="f",
    )
    grasp_configs_dataset = hdf5_file.create_dataset(
        "/grasp_configs",
        shape=(
            max_num_datapoints,
            cfg.fingertip_config.n_fingers,
            7 + 16 + 4,
        ),  # 7 for pose, 16 for joint angles, 4 for grasp orientation, for each finger
        dtype="f",
    )

    return (
        nerf_densities_dataset,
        nerf_densities_global_dataset,
        nerf_densities_global_idx_dataset,
        y_PGS_dataset,
        y_pick_dataset,
        y_coll_dataset,
        nerf_config_dataset,
        object_code_dataset,
        object_scale_dataset,
        object_state_dataset,
        grasp_idx_dataset,
        grasp_transforms_dataset,
        grasp_configs_dataset,
    )


@torch.no_grad()
def get_nerf_densities(
    loop_timer: LoopTimer,
    fingertip_config: EvenlySpacedFingertipConfig,
    ray_samples_chunk_size: int,
    grasp_frame_transforms: pp.LieTensor,
    ray_origins_finger_frame: torch.Tensor,
    nerf_config: pathlib.Path,
    X_N_Oy: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    # Shape check grasp_frame_transforms
    batch_size = grasp_frame_transforms.shape[0]
    assert_equals(
        grasp_frame_transforms.lshape,
        (
            batch_size,
            fingertip_config.n_fingers,
        ),
    )

    # Load nerf pipeline
    # Note: I don't like that we are mixing IO and computation here, but it seems to avoid OOM better
    #       if IO in main code, I believe it causes weird memory leak having nerf_pipeline in global scope
    # Note: for some reason this seems to help avoid GPU memory leak
    #       whereas loading nerf_model or nerf_field directly causes GPU memory leak
    nerf_pipeline = load_nerf_pipeline(nerf_config)

    # Prepare transforms
    # grasp_frame_transforms are in Oy frame
    # Need to convert to NeRF frame N
    T_Oy_Fi = grasp_frame_transforms
    assert T_Oy_Fi.lshape == (batch_size, fingertip_config.n_fingers)

    assert X_N_Oy.shape == (
        4,
        4,
    )
    X_N_Oy_repeated = (
        torch.from_numpy(X_N_Oy)
        .float()
        .unsqueeze(dim=0)
        .repeat_interleave(batch_size * fingertip_config.n_fingers, dim=0)
        .reshape(batch_size, fingertip_config.n_fingers, 4, 4)
    )

    T_N_Oy = pp.from_matrix(
        X_N_Oy_repeated.to(T_Oy_Fi.device),
        pp.SE3_type,
    )

    # Transform grasp_frame_transforms to nerf frame
    T_N_Fi = T_N_Oy @ T_Oy_Fi

    # Transform query points
    with loop_timer.add_section_timer("get_ray_samples"):
        ray_samples = get_ray_samples(
            ray_origins_finger_frame,
            T_N_Fi,
            fingertip_config,
        )

    with loop_timer.add_section_timer("get_density"):
        # Split ray_samples into chunks so everything fits on the gpu
        split_inds = torch.arange(0, batch_size, ray_samples_chunk_size)
        split_inds = torch.cat(
            [split_inds, torch.tensor([batch_size]).to(split_inds.device)]
        )
        # Preallocate to avoid OOM (instead of making a list and concatenating at the end)
        nerf_densities = torch.zeros(
            (
                batch_size,
                fingertip_config.n_fingers,
                fingertip_config.num_pts_x,
                fingertip_config.num_pts_y,
                fingertip_config.num_pts_z,
            ),
            dtype=torch.float,
            device="cpu",
        )
        for curr_ind, next_ind in tqdm(
            zip(split_inds[:-1], split_inds[1:]),
            total=len(split_inds) - 1,
            desc="get_density",
            dynamic_ncols=True,
        ):
            curr_ray_samples = ray_samples[curr_ind:next_ind].to("cuda")
            curr_nerf_densities = (
                nerf_pipeline.model.field.get_density(curr_ray_samples)[0]
                .reshape(
                    -1,
                    fingertip_config.n_fingers,
                    fingertip_config.num_pts_x,
                    fingertip_config.num_pts_y,
                    fingertip_config.num_pts_z,
                )
                .cpu()
            )
            curr_ray_samples.to("cpu")
            nerf_densities[curr_ind:next_ind] = curr_nerf_densities

    with loop_timer.add_section_timer("frustums.get_positions"):
        query_points_N = ray_samples.frustums.get_positions().reshape(
            batch_size,
            fingertip_config.n_fingers,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
            3,
        )

    with loop_timer.add_section_timer("get_densities_in_grid"):
        lb_N = transform_point(T=X_N_Oy, point=lb_Oy)
        ub_N = transform_point(T=X_N_Oy, point=ub_Oy)
        nerf_densities_global, query_points_global_N = get_densities_in_grid(
            field=nerf_pipeline.model.field,
            lb=lb_N,
            ub=ub_N,
            num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
            num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
            num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        assert nerf_densities_global.shape == (
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        )
        assert query_points_global_N.shape == (
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
            3,
        )

    return nerf_densities, query_points_N, nerf_densities_global, query_points_global_N


def get_data(
    evaled_grasp_config_dict_filepath: pathlib.Path,
    fingertip_config: EvenlySpacedFingertipConfig,
    ray_samples_chunk_size: int,
    max_num_data_points_per_file: Optional[int],
    nerf_configs: List[pathlib.Path],
    ray_origins_finger_frame: torch.Tensor,
    loop_timer: LoopTimer,
) -> tuple:
    with loop_timer.add_section_timer("prepare to read in data"):
        object_code_and_scale_str = evaled_grasp_config_dict_filepath.stem
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        # Get nerf config
        nerf_config = get_matching_nerf_config(
            target_object_code_and_scale_str=object_code_and_scale_str,
            nerf_configs=nerf_configs,
        )

        # Check that mesh and grasp dataset exist
        assert os.path.exists(
            evaled_grasp_config_dict_filepath
        ), f"evaled_grasp_config_dict_filepath {evaled_grasp_config_dict_filepath} does not exist"

    with loop_timer.add_section_timer("load grasp data"):
        evaled_grasp_config_dict: Dict[str, Any] = np.load(
            evaled_grasp_config_dict_filepath, allow_pickle=True
        ).item()

    with loop_timer.add_section_timer("load mesh"):
        mesh_Oy = create_mesh(object_code=object_code, object_scale=object_scale)

    # Extract useful parts of grasp data
    grasp_configs = AllegroGraspConfig.from_grasp_config_dict(evaled_grasp_config_dict)
    y_PGSs = evaled_grasp_config_dict["y_PGS"]
    y_picks = evaled_grasp_config_dict["y_pick"]
    y_colls = evaled_grasp_config_dict["y_coll"]
    object_state = evaled_grasp_config_dict["object_states_before_grasp"]

    print(f"grasp_configs.batch_size = {grasp_configs.batch_size}")
    if (
        max_num_data_points_per_file is not None
        and grasp_configs.batch_size > max_num_data_points_per_file
    ):
        print(
            "WARNING: Too many grasp configs, dropping some datapoints from NeRF dataset."
        )
        print(
            f"batch_size = {grasp_configs.batch_size}, max_num_data_points_per_file = {max_num_data_points_per_file}"
        )

        grasp_configs = grasp_configs[:max_num_data_points_per_file]

        y_PGSs = y_PGSs[:max_num_data_points_per_file]
        y_picks = y_picks[:max_num_data_points_per_file]
        y_colls = y_colls[:max_num_data_points_per_file]
        object_state = object_state[:max_num_data_points_per_file]

    grasp_frame_transforms = grasp_configs.grasp_frame_transforms
    grasp_config_tensors = grasp_configs.as_tensor().detach().cpu().numpy()

    assert_equals(y_PGSs.shape, (grasp_configs.batch_size,))
    assert_equals(y_picks.shape, (grasp_configs.batch_size,))
    assert_equals(y_colls.shape, (grasp_configs.batch_size,))
    assert_equals(
        grasp_frame_transforms.lshape,
        (
            grasp_configs.batch_size,
            fingertip_config.n_fingers,
        ),
    )
    assert_equals(
        grasp_config_tensors.shape,
        (
            grasp_configs.batch_size,
            fingertip_config.n_fingers,
            7 + 16 + 4,  # wrist pose, joint angles, grasp orientations (as quats)
        ),
    )
    # Annoying hack because I stored all the object states for multiple noise runs, only need 1
    assert len(object_state.shape) == 3
    n_runs = object_state.shape[1]
    assert_equals(object_state.shape, (grasp_configs.batch_size, n_runs, 13))
    object_state = object_state[:, 0]

    X_N_Oy = compute_X_N_Oy(mesh_Oy)
    assert np.allclose(X_N_Oy[:3, :3], np.eye(3)), f"X_N_Oy = {X_N_Oy}"
    mesh_N = mesh_Oy.copy()
    mesh_N.apply_transform(X_N_Oy)

    # Process batch of grasp data.
    (
        nerf_densities,
        query_points_N,
        nerf_densities_global,
        query_points_global_N,
    ) = get_nerf_densities(
        loop_timer=loop_timer,
        fingertip_config=fingertip_config,
        ray_samples_chunk_size=ray_samples_chunk_size,
        grasp_frame_transforms=grasp_frame_transforms,
        ray_origins_finger_frame=ray_origins_finger_frame,
        nerf_config=nerf_config,
        X_N_Oy=X_N_Oy,
    )
    return (
        nerf_densities,
        nerf_densities_global,
        grasp_frame_transforms,
        grasp_config_tensors,
        y_PGSs,
        y_picks,
        y_colls,
        nerf_config,
        object_code,
        object_scale,
        object_state,
        query_points_N,
        query_points_global_N,
    )


def main() -> None:
    cfg = tyro.cli(GridNerfDataConfig)

    print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
    assert cfg.fingertip_config is not None

    # Prepare output filepath
    if cfg.output_filepath is None:
        cfg.output_filepath = (
            cfg.input_evaled_grasp_config_dicts_path.parent
            / "learned_metric_dataset"
            / f"{CONFIG_DATETIME_STR}_learned_metric_dataset.h5"
        )
    assert cfg.output_filepath is not None
    if not cfg.output_filepath.parent.exists():
        print(f"Creating output folder {cfg.output_filepath.parent}")
        cfg.output_filepath.parent.mkdir(parents=True)
    else:
        print(f"Output folder {cfg.output_filepath.parent} already exists")

    if cfg.output_filepath.exists():
        raise ValueError(f"Output file {cfg.output_filepath} already exists")

    # Find all nerf configs
    assert (
        cfg.input_nerfcheckpoints_path.exists()
    ), f"{cfg.input_nerfcheckpoints_path} does not exist"
    nerf_configs = get_nerf_configs(
        nerfcheckpoints_path=str(cfg.input_nerfcheckpoints_path),
    )
    assert (
        len(nerf_configs) > 0
    ), f"Did not find any nerf configs in {cfg.input_nerfcheckpoints_path}"
    print(f"Found {len(nerf_configs)} nerf configs")

    # Find all evaled grasp config dicts
    assert (
        cfg.input_evaled_grasp_config_dicts_path.exists()
    ), f"{cfg.input_evaled_grasp_config_dicts_path} does not exist"
    evaled_grasp_config_dict_filepaths = sorted(
        list(cfg.input_evaled_grasp_config_dicts_path.glob("*.npy"))
    )
    assert (
        len(evaled_grasp_config_dict_filepaths) > 0
    ), f"Did not find any evaled grasp config dicts in {cfg.input_evaled_grasp_config_dicts_path}"
    print(f"Found {len(evaled_grasp_config_dict_filepaths)} evaled grasp config dicts")

    # Precompute ray origins in finger frame
    ray_origins_finger_frame = get_ray_origins_finger_frame(cfg.fingertip_config)

    # Restrict data size
    if cfg.limit_num_objects is not None:
        print(f"Limiting number of objects to {cfg.limit_num_objects}")
        evaled_grasp_config_dict_filepaths = evaled_grasp_config_dict_filepaths[
            : cfg.limit_num_objects
        ]

    if cfg.max_num_data_points_per_file is not None:
        max_num_datapoints = (
            len(evaled_grasp_config_dict_filepaths) * cfg.max_num_data_points_per_file
        )
    else:
        max_num_datapoints = count_total_num_grasps(
            evaled_grasp_config_dict_filepaths=evaled_grasp_config_dict_filepaths,
        )
    print(f"max num datapoints: {max_num_datapoints}")

    # Save config
    print(f"Saving config to {cfg.config_filepath}")
    cfg_yaml = tyro.extras.to_yaml(cfg)
    with open(cfg.config_filepath, "w") as f:
        f.write(cfg_yaml)

    # Create HDF5 file
    hdf5_file = h5py.File(cfg.output_filepath, "w")
    current_idx = 0
    (
        nerf_densities_dataset,
        nerf_densities_global_dataset,
        nerf_densities_global_idx_dataset,
        y_PGS_dataset,
        y_pick_dataset,
        y_coll_dataset,
        nerf_config_dataset,
        object_code_dataset,
        object_scale_dataset,
        object_state_dataset,
        grasp_idx_dataset,
        grasp_transforms_dataset,
        grasp_configs_dataset,
    ) = create_grid_dataset(
        cfg=cfg,
        hdf5_file=hdf5_file,
        max_num_datapoints=max_num_datapoints,
        num_objects=len(evaled_grasp_config_dict_filepaths),
    )

    # Iterate through all
    loop_timer = LoopTimer()
    pbar = tqdm(
        enumerate(evaled_grasp_config_dict_filepaths),
        dynamic_ncols=True,
        total=len(evaled_grasp_config_dict_filepaths),
    )
    for object_i, evaled_grasp_config_dict_filepath in pbar:
        torch.cuda.empty_cache()

        pbar.set_description(f"Processing {evaled_grasp_config_dict_filepath}")
        (
            nerf_densities,
            nerf_densities_global,
            grasp_frame_transforms,
            grasp_config_tensors,
            y_PGSs,
            y_picks,
            y_colls,
            nerf_config,
            object_code,
            object_scale,
            object_state,
            _,
            _,
        ) = get_data(
            evaled_grasp_config_dict_filepath=evaled_grasp_config_dict_filepath,
            fingertip_config=cfg.fingertip_config,
            ray_samples_chunk_size=cfg.ray_samples_chunk_size,
            max_num_data_points_per_file=cfg.max_num_data_points_per_file,
            nerf_configs=nerf_configs,
            ray_origins_finger_frame=ray_origins_finger_frame,
            loop_timer=loop_timer,
        )

        # Ensure no nans (most likely come from weird grasp transforms)
        if grasp_frame_transforms.isnan().any():
            print("\n" + "-" * 80)
            print(
                f"WARNING: Found {grasp_frame_transforms.isnan().sum()} transform nans in {evaled_grasp_config_dict_filepath}"
            )
            print("Skipping this one...")
            print("-" * 80 + "\n")
            continue
        if nerf_densities.isnan().any():
            print("\n" + "-" * 80)
            print(
                f"WARNING: Found {nerf_densities.isnan().sum()} nerf density nans in {nerf_config}"
            )
            print("Skipping this one...")
            print("-" * 80 + "\n")
            continue

        # Save values
        with loop_timer.add_section_timer("save values"):
            prev_idx = current_idx
            current_idx += y_PGSs.shape[0]
            y_PGS_dataset[prev_idx:current_idx] = y_PGSs
            y_pick_dataset[prev_idx:current_idx] = y_picks
            y_coll_dataset[prev_idx:current_idx] = y_colls
            nerf_config_dataset[prev_idx:current_idx] = [str(nerf_config)] * (
                current_idx - prev_idx
            )
            object_code_dataset[prev_idx:current_idx] = [object_code] * (
                current_idx - prev_idx
            )
            object_scale_dataset[prev_idx:current_idx] = object_scale
            object_state_dataset[prev_idx:current_idx] = object_state
            grasp_idx_dataset[prev_idx:current_idx] = np.arange(
                0, current_idx - prev_idx
            )
            grasp_transforms_dataset[prev_idx:current_idx] = (
                grasp_frame_transforms.matrix().cpu().detach().numpy()
            )

            nerf_densities_dataset[prev_idx:current_idx] = (
                nerf_densities.detach().cpu().numpy()
            )
            del nerf_densities
            nerf_densities_global_dataset[object_i] = nerf_densities_global
            del nerf_densities_global
            nerf_densities_global_idx_dataset[prev_idx:current_idx] = object_i

            grasp_configs_dataset[prev_idx:current_idx] = grasp_config_tensors

            # May not be max_num_data_points if missing grasps
            hdf5_file.attrs["num_data_points"] = current_idx

        print(f"End of loop for {evaled_grasp_config_dict_filepath} ({object_i})")

        if cfg.print_timing:
            loop_timer.pretty_print_section_times()
        print()

    hdf5_file.close()


if __name__ == "__main__":
    main()
