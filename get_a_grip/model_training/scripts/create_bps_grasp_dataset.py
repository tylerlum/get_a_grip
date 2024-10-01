import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
import open3d as o3d
import tyro
from bps import bps
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)

N_FINGERS = 4
GRASP_DIM = 3 + 6 + 16 + N_FINGERS * 3


@dataclass
class BpsGraspDatasetConfig:
    input_point_clouds_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/point_clouds"
    )
    input_evaled_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/final_evaled_grasp_config_dicts_train"
    )
    output_filepath: pathlib.Path = (
        get_data_folder() / "dataset/NEW/bps_grasp_dataset/train_dataset.h5"
    )
    overwrite: bool = False

    def __post_init__(self):
        assert (
            self.input_point_clouds_path.exists()
        ), f"Path {self.input_point_clouds_path} does not exist"
        assert (
            self.input_evaled_grasp_config_dicts_path.exists()
        ), f"Path {self.input_evaled_grasp_config_dicts_path} does not exist"

        if not self.overwrite:
            assert (
                not self.output_filepath.exists()
            ), f"Path {self.output_filepath} already exists"
        elif self.output_filepath.exists():
            print("\033[93m" + "WARNING: OVERWRITING FILE" + "\033[0m")
            SLEEP_TIME = 10
            print(f"\033[93mOverwriting will begin in {SLEEP_TIME} seconds\033[0m")
            time.sleep(SLEEP_TIME)
            print(f"\033[93mOverwriting {self.output_filepath} will happen\033[0m")
        else:
            print(f"Creating {self.output_filepath} at end of script")


def construct_graph(points: np.ndarray, distance_threshold: float = 0.01) -> csr_matrix:
    kdtree = KDTree(points)
    rows, cols = [], []

    for i, point in enumerate(points):
        neighbors = kdtree.query_ball_point(point, distance_threshold)
        for neighbor in neighbors:
            if neighbor != i:
                rows.append(i)
                cols.append(neighbor)

    data = np.ones(len(rows), dtype=np.int8)
    adjacency_matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(points), len(points))
    )
    return adjacency_matrix


def get_largest_connected_component(adjacency_matrix: csr_matrix) -> np.ndarray:
    n_components, labels = connected_components(
        csgraph=adjacency_matrix, directed=False, return_labels=True
    )
    largest_cc_label = np.bincount(labels).argmax()
    largest_cc_indices = np.where(labels == largest_cc_label)[0]
    return largest_cc_indices


def get_point_cloud_largest_connected_component(
    points: np.ndarray, distance_threshold: float = 0.01
) -> np.ndarray:
    adjacency_matrix = construct_graph(points, distance_threshold)
    largest_cc_indices = get_largest_connected_component(adjacency_matrix)
    return points[largest_cc_indices]


def get_grasp_data(
    all_config_dict_paths: List[pathlib.Path],
    object_code_and_scale_str_to_bps_idx: Dict[str, int],
) -> tuple:
    # Per grasp
    (
        all_grasps,
        all_grasp_bps_idxs,
        all_grasp_idxs,
        all_y_PGSs,
        all_y_picks,
        all_y_colls,
        all_object_states,
    ) = ([], [], [], [], [], [], [])
    num_data_points = 0
    for grasp_data_path in tqdm(
        all_config_dict_paths,
        desc="Getting grasps",
        total=len(all_config_dict_paths),
    ):
        # Load data
        grasp_config_dict = np.load(grasp_data_path, allow_pickle=True).item()

        # Extract object info
        object_code_and_scale_str = grasp_data_path.stem
        if object_code_and_scale_str not in object_code_and_scale_str_to_bps_idx:
            print(
                f"Object code and scale {object_code_and_scale_str} does not have BPS, skipping"
            )
            continue
        bps_idx = object_code_and_scale_str_to_bps_idx[object_code_and_scale_str]

        # Extract grasp info
        grasp_config = AllegroGraspConfig.from_grasp_config_dict(grasp_config_dict)
        B = len(grasp_config)
        grasps = grasp_config.as_grasp().detach().cpu().numpy()
        assert grasps.shape == (
            B,
            GRASP_DIM,
        ), f"Expected shape ({B}, {GRASP_DIM}), got {grasps.shape}"

        # Extract grasp evaluation info
        y_PGSs = grasp_config_dict["y_PGS"]
        y_picks = grasp_config_dict["y_pick"]
        y_colls = grasp_config_dict["y_coll"]
        object_state = grasp_config_dict["object_states_before_grasp"]
        assert y_PGSs.shape == (B,), f"Expected shape ({B},), got {y_PGSs.shape}"
        assert y_picks.shape == (B,), f"Expected shape ({B},), got {y_picks.shape}"
        assert y_colls.shape == (B,), f"Expected shape ({B},), got {y_colls.shape}"
        N_NOISY_GRASPS = 6  # 1 for original grasp, 5 for noisy grasps
        assert object_state.shape == (
            B,
            N_NOISY_GRASPS,
            13,
        ), f"Expected shape ({B}, {N_NOISY_GRASPS}, 13), got {object_state.shape}"
        object_state = object_state[:, 0, :]

        all_grasps.append(grasps)
        all_grasp_bps_idxs.append(np.repeat(bps_idx, B))
        all_grasp_idxs.append(np.arange(B))
        all_y_PGSs.append(y_PGSs)
        all_y_picks.append(y_picks)
        all_y_colls.append(y_colls)
        all_object_states.append(object_state)
        num_data_points += B

    all_grasps = np.concatenate(all_grasps, axis=0)
    all_grasp_bps_idxs = np.concatenate(all_grasp_bps_idxs, axis=0)
    all_grasp_idxs = np.concatenate(all_grasp_idxs, axis=0)
    all_y_PGSs = np.concatenate(all_y_PGSs, axis=0)
    all_y_picks = np.concatenate(all_y_picks, axis=0)
    all_y_colls = np.concatenate(all_y_colls, axis=0)
    all_object_states = np.concatenate(all_object_states, axis=0)

    NUM_GRASPS = all_grasps.shape[0]
    assert (
        all_grasps.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_grasps.shape[0]}"
    assert (
        all_grasp_bps_idxs.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_grasp_bps_idxs.shape[0]}"
    assert (
        all_grasp_idxs.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_grasp_idxs.shape[0]}"
    assert (
        all_y_PGSs.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_y_PGSs.shape[0]}"
    assert (
        all_y_picks.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_y_picks.shape[0]}"
    assert (
        all_y_colls.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_y_colls.shape[0]}"
    assert (
        all_object_states.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_object_states.shape[0]}"
    assert all_y_colls.shape == (
        NUM_GRASPS,
    ), f"Expected shape ({NUM_GRASPS},), got {all_y_colls.shape}"
    print(f"Got {NUM_GRASPS} grasps with {num_data_points} data points")
    return (
        all_grasps,
        all_grasp_bps_idxs,
        all_grasp_idxs,
        all_y_PGSs,
        all_y_picks,
        all_y_colls,
        all_object_states,
    )


def get_fixed_basis_points() -> np.ndarray:
    # Intentionally hardcoded for getting fixed basis points every time
    HARDCODED_N_PTS = 4096
    HARDCODED_RADIUS = 0.3
    HARDCODED_RANDOM_SEED = 13
    basis_points = bps.generate_random_basis(
        n_points=HARDCODED_N_PTS,
        radius=HARDCODED_RADIUS,
        random_seed=HARDCODED_RANDOM_SEED,
    ) + np.array(
        [0.0, HARDCODED_RADIUS / 2, 0.0]
    )  # Shift up to get less under the table
    assert basis_points.shape == (
        HARDCODED_N_PTS,
        3,
    ), f"Expected shape ({HARDCODED_N_PTS}, 3), got {basis_points.shape}"
    return basis_points


def get_bps(
    all_points: np.ndarray,
    basis_points: np.ndarray,
) -> np.ndarray:
    n_point_clouds, n_points_per_point_cloud = all_points.shape[:2]
    n_basis_points = basis_points.shape[0]
    assert (
        all_points.shape
        == (
            n_point_clouds,
            n_points_per_point_cloud,
            3,
        )
    ), f"Expected shape ({n_point_clouds}, {n_points_per_point_cloud}, 3), got {all_points.shape}"
    assert basis_points.shape == (
        n_basis_points,
        3,
    ), f"Expected shape ({n_basis_points}, 3), got {basis_points.shape}"

    bps_values = bps.encode(
        all_points,
        bps_arrangement="custom",
        bps_cell_type="dists",
        custom_basis=basis_points,
        verbose=0,
    )
    assert bps_values.shape == (
        n_point_clouds,
        n_basis_points,
    ), f"Expected shape ({n_point_clouds}, {n_basis_points}), got {bps_values.shape}"
    return bps_values


def read_raw_single_point_cloud(point_cloud_path: pathlib.Path) -> np.ndarray:
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    return np.asarray(point_cloud.points)


def read_and_process_single_point_cloud(point_cloud_path: pathlib.Path) -> np.ndarray:
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    return process_single_point_cloud(point_cloud)


def process_single_point_cloud(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    points = np.asarray(point_cloud.points)
    inlier_points = get_point_cloud_largest_connected_component(
        points, distance_threshold=0.01
    )
    return inlier_points


def crop_single_point_cloud(
    points: np.ndarray, n_pts: int = 3000
) -> Optional[np.ndarray]:
    input_n_pts = points.shape[0]
    assert points.shape == (
        input_n_pts,
        3,
    ), f"Expected shape ({input_n_pts}, 3), got {points.shape}"

    if input_n_pts < n_pts:
        print(f"WARNING: {input_n_pts} points is less than {n_pts}, skipping")
        return None

    return points[:n_pts]


def main() -> None:
    cfg = tyro.cli(tyro.conf.FlagConversionOff[BpsGraspDatasetConfig])
    print("=" * 80)
    print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
    print("=" * 80 + "\n")

    all_point_cloud_paths_unfiltered = sorted(
        list(cfg.input_point_clouds_path.rglob("*.ply"))
    )
    all_config_dict_paths = sorted(
        list(cfg.input_evaled_grasp_config_dicts_path.rglob("*.npy"))
    )
    assert len(all_point_cloud_paths_unfiltered) > 0, "No point cloud paths found"
    assert len(all_config_dict_paths) > 0, "No config dict paths found"

    print(f"Found {len(all_point_cloud_paths_unfiltered)} point cloud paths")
    print(f"Found {len(all_config_dict_paths)} config dict paths")

    # BPS selection (done once)
    basis_points = get_fixed_basis_points()
    assert basis_points.shape == (
        4096,
        3,
    ), f"Expected shape (4096, 3), got {basis_points.shape}"

    # Parallel processing of point clouds
    all_points_uncropped_unfiltered: List[np.ndarray] = Parallel(n_jobs=-1)(
        delayed(read_and_process_single_point_cloud)(data_path)
        for data_path in tqdm(
            all_point_cloud_paths_unfiltered, desc="Processing point clouds"
        )
    )

    # Ensure all point clouds have the same number of points
    all_points_cropped_unfiltered = [
        crop_single_point_cloud(x) for x in all_points_uncropped_unfiltered
    ]

    all_points = np.stack(
        [x for x in all_points_cropped_unfiltered if x is not None], axis=0
    )
    all_point_cloud_paths = [
        x
        for x, y in zip(all_point_cloud_paths_unfiltered, all_points_cropped_unfiltered)
        if y is not None
    ]
    print(
        f"Rejected {len(all_point_cloud_paths_unfiltered) - len(all_point_cloud_paths)} point clouds with insufficient points after processing"
    )

    NUM_POINT_CLOUDS = all_points.shape[0]
    n_pts = all_points.shape[1]
    assert all_points.shape == (
        NUM_POINT_CLOUDS,
        n_pts,
        3,
    ), f"Expected shape ({NUM_POINT_CLOUDS}, {n_pts}, 3), got {all_points.shape}"

    # BPS values per object
    bps_values = get_bps(
        all_points=all_points,
        basis_points=basis_points,
    )
    assert bps_values.shape == (
        NUM_POINT_CLOUDS,
        4096,
    ), f"Expected shape ({NUM_POINT_CLOUDS}, 4096), got {bps_values.shape}"

    # Extract the object_code_and_scale_strs we want
    point_cloud_object_code_and_scale_strs = [
        x.parent.name for x in all_point_cloud_paths
    ]
    point_cloud_object_codes, point_cloud_object_scales = [], []
    for object_code_and_scale_str in point_cloud_object_code_and_scale_strs:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        point_cloud_object_codes.append(object_code)
        point_cloud_object_scales.append(object_scale)

    object_code_and_scale_str_to_bps_idx = {
        object_code_and_scale_str: i
        for i, object_code_and_scale_str in enumerate(
            point_cloud_object_code_and_scale_strs
        )
    }

    # Grasp data per grasp
    (
        all_grasps,
        all_grasp_bps_idxs,
        all_grasp_idxs,
        all_y_PGSs,
        all_y_picks,
        all_y_colls,
        all_object_states,
    ) = get_grasp_data(
        all_config_dict_paths=all_config_dict_paths,
        object_code_and_scale_str_to_bps_idx=object_code_and_scale_str_to_bps_idx,
    )

    # Important shapes
    NUM_GRASPS = all_grasps.shape[0]

    # Save to HDF5
    cfg.output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.output_filepath, "w") as hdf5_file:
        # Just one
        basis_points_dataset = hdf5_file.create_dataset(
            "/basis_points", shape=(4096, 3), dtype="f"
        )

        # Per object
        bpss_dataset = hdf5_file.create_dataset(
            "/bpss",
            shape=(
                NUM_POINT_CLOUDS,
                4096,
            ),
            dtype="f",
        )
        point_cloud_filepath_dataset = hdf5_file.create_dataset(
            "/point_cloud_filepath",
            shape=(NUM_POINT_CLOUDS,),
            dtype=h5py.string_dtype(),
        )
        object_code_dataset = hdf5_file.create_dataset(
            "/object_code", shape=(NUM_POINT_CLOUDS,), dtype=h5py.string_dtype()
        )
        object_scale_dataset = hdf5_file.create_dataset(
            "/object_scale", shape=(NUM_POINT_CLOUDS,), dtype="f"
        )

        # Per grasp
        grasps_dataset = hdf5_file.create_dataset(
            "/grasps", shape=(NUM_GRASPS, GRASP_DIM), dtype="f"
        )
        grasp_bps_idx_dataset = hdf5_file.create_dataset(
            "/grasp_bps_idx", shape=(NUM_GRASPS,), dtype="i"
        )
        y_PGS_dataset = hdf5_file.create_dataset(
            "/y_PGS",
            shape=(NUM_GRASPS,),
            dtype="f",
        )
        y_pick_dataset = hdf5_file.create_dataset(
            "/y_pick",
            shape=(NUM_GRASPS,),
            dtype="f",
        )
        y_coll_dataset = hdf5_file.create_dataset(
            "/y_coll",
            shape=(NUM_GRASPS,),
            dtype="f",
        )
        object_state_dataset = hdf5_file.create_dataset(
            "/object_state",
            shape=(
                NUM_GRASPS,
                13,
            ),
            dtype="f",
        )
        grasp_idx_dataset = hdf5_file.create_dataset(
            "/grasp_idx", shape=(NUM_GRASPS,), dtype="i"
        )

        # Just one
        basis_points_dataset[:] = basis_points

        # Per object
        bpss_dataset[:NUM_POINT_CLOUDS] = bps_values
        point_cloud_filepath_dataset[:NUM_POINT_CLOUDS] = [
            str(x) for x in all_point_cloud_paths
        ]
        object_code_dataset[:NUM_POINT_CLOUDS] = point_cloud_object_codes
        object_scale_dataset[:NUM_POINT_CLOUDS] = point_cloud_object_scales

        # Per grasp
        grasps_dataset[:NUM_GRASPS] = all_grasps
        grasp_bps_idx_dataset[:NUM_GRASPS] = all_grasp_bps_idxs
        y_PGS_dataset[:NUM_GRASPS] = all_y_PGSs
        y_pick_dataset[:NUM_GRASPS] = all_y_picks
        y_coll_dataset[:NUM_GRASPS] = all_y_colls
        object_state_dataset[:NUM_GRASPS] = all_object_states
        grasp_idx_dataset[:NUM_GRASPS] = all_grasp_idxs
        hdf5_file.attrs["num_grasps"] = NUM_GRASPS


if __name__ == "__main__":
    main()
