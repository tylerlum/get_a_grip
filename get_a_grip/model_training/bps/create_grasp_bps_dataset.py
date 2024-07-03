from joblib import Parallel, delayed

from tqdm import tqdm
import time
import h5py
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from bps import bps
import pathlib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    parse_object_code_and_scale,
)

from dataclasses import dataclass

N_FINGERS = 4
GRASP_DIM = 3 + 6 + 16 + N_FINGERS * 3


@dataclass
class BpsGraspConfig:
    point_cloud_folder: pathlib.Path
    config_dict_folder: pathlib.Path
    output_filepath: pathlib.Path
    N_BASIS_PTS: int = 4096
    BASIS_RADIUS: float = 0.3
    overwrite: bool = False

    def __post_init__(self):
        assert (
            self.point_cloud_folder.exists()
        ), f"Path {self.point_cloud_folder} does not exist"
        assert (
            self.config_dict_folder.exists()
        ), f"Path {self.config_dict_folder} does not exist"

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

def construct_graph(points, distance_threshold=0.01):
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


def get_largest_connected_component(adjacency_matrix):
    n_components, labels = connected_components(
        csgraph=adjacency_matrix, directed=False, return_labels=True
    )
    largest_cc_label = np.bincount(labels).argmax()
    largest_cc_indices = np.where(labels == largest_cc_label)[0]
    return largest_cc_indices


def process_point_cloud(points, distance_threshold=0.01):
    adjacency_matrix = construct_graph(points, distance_threshold)
    largest_cc_indices = get_largest_connected_component(adjacency_matrix)
    return points[largest_cc_indices]


def get_grasp_data(all_config_dict_paths, object_code_and_scale_str_to_idx) -> tuple:
    # Per grasp
    (
        all_grasps,
        all_grasp_bps_idxs,
        all_passed_evals,
        all_passed_simulations,
        all_passed_penetration_thresholds,
        all_object_states,
    ) = ([], [], [], [], [], [])
    num_data_points = 0
    for i, grasp_data_path in tqdm(
        enumerate(all_config_dict_paths),
        desc="Getting grasps",
        total=len(all_config_dict_paths),
    ):
        # Load data
        grasp_config_dict = np.load(grasp_data_path, allow_pickle=True).item()

        # Extract object info
        object_code_and_scale_str = grasp_data_path.stem
        if object_code_and_scale_str not in object_code_and_scale_str_to_idx:
            print(f"Object code and scale {object_code_and_scale_str} not found")
            continue
        bps_idx = object_code_and_scale_str_to_idx[object_code_and_scale_str]

        # Extract grasp info
        trans = grasp_config_dict["trans"]
        rot = grasp_config_dict["rot"]
        joint_angles = grasp_config_dict["joint_angles"]
        grasp_orientations = grasp_config_dict["grasp_orientations"]

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
        grasps = np.concatenate(
            [
                trans,
                rot[..., :2].reshape(B, 6),
                joint_angles,
                grasp_dirs.reshape(B, -1),
            ],
            axis=1,
        )
        assert grasps.shape == (
            B,
            GRASP_DIM,
        ), f"Expected shape ({B}, {GRASP_DIM}), got {grasps.shape}"

        # Extract grasp evaluation info
        passed_evals = grasp_config_dict["passed_eval"]
        passed_simulations = grasp_config_dict["passed_simulation"]
        passed_penetration_thresholds = grasp_config_dict["passed_new_penetration_test"]
        object_state = grasp_config_dict["object_states_before_grasp"]
        assert passed_evals.shape == (
            B,
        ), f"Expected shape ({B},), got {passed_evals.shape}"
        assert passed_simulations.shape == (
            B,
        ), f"Expected shape ({B},), got {passed_simulations.shape}"
        assert passed_penetration_thresholds.shape == (
            B,
        ), f"Expected shape ({B},), got {passed_penetration_thresholds.shape}"
        N_NOISY_GRASPS = 6
        assert object_state.shape == (
            B,
            N_NOISY_GRASPS,
            13,
        ), f"Expected shape ({B}, {N_NOISY_GRASPS}, 13), got {object_state.shape}"
        object_state = object_state[:, 0, :]

        all_grasps.append(grasps)
        all_grasp_bps_idxs.append(np.repeat(bps_idx, B))
        all_passed_evals.append(passed_evals)
        all_passed_simulations.append(passed_simulations)
        all_passed_penetration_thresholds.append(passed_penetration_thresholds)
        all_object_states.append(object_state)
        num_data_points += B

    all_grasps = np.concatenate(all_grasps, axis=0)
    all_grasp_bps_idxs = np.concatenate(all_grasp_bps_idxs, axis=0)
    all_passed_evals = np.concatenate(all_passed_evals, axis=0)
    all_passed_simulations = np.concatenate(all_passed_simulations, axis=0)
    all_passed_penetration_thresholds = np.concatenate(
        all_passed_penetration_thresholds, axis=0
    )
    all_object_states = np.concatenate(all_object_states, axis=0)

    NUM_GRASPS = all_grasps.shape[0]
    assert (
        all_grasps.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_grasps.shape[0]}"
    assert (
        all_grasp_bps_idxs.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_grasp_bps_idxs.shape[0]}"
    assert (
        all_passed_evals.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_passed_evals.shape[0]}"
    assert (
        all_passed_simulations.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_passed_simulations.shape[0]}"
    assert (
        all_passed_penetration_thresholds.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_passed_penetration_thresholds.shape[0]}"
    assert (
        all_object_states.shape[0] == NUM_GRASPS
    ), f"Expected shape ({NUM_GRASPS},), got {all_object_states.shape[0]}"
    assert all_passed_penetration_thresholds.shape == (
        NUM_GRASPS,
    ), f"Expected shape ({NUM_GRASPS},), got {all_passed_penetration_thresholds.shape}"
    print(f"Got {NUM_GRASPS} grasps with {num_data_points} data points")
    return (
        all_grasps,
        all_grasp_bps_idxs,
        all_passed_evals,
        all_passed_simulations,
        all_passed_penetration_thresholds,
        all_object_states,
    )


def main(args) -> None:
    cfg = BpsGraspConfig(
        point_cloud_folder=pathlib.Path(
            "/home/albert/research/nerf_grasping/rsync_point_clouds/point_clouds"
        ),
        config_dict_folder=pathlib.Path(
            # "/home/albert/research/nerf_grasping/rsync_grasps/grasps/all"
            # "/home/albert/research/nerf_grasping/rsync_final_labeled_grasps_noise_and_nonoise/evaled_grasp_config_dicts_train",
            # "/home/albert/research/nerf_grasping/rsync_final_labeled_grasps_noise_and_nonoise/evaled_grasp_config_dicts_val",
            # "/home/albert/research/nerf_grasping/rsync_final_labeled_grasps_noise_and_nonoise/evaled_grasp_config_dicts_test",
            args.config_dict_folder,
        ),
        output_filepath=pathlib.Path(
            # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset.hdf5"
            # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_train.hdf5",
            # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_val.hdf5",
            # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_test.hdf5",
            args.output_filepath,
        ),
    )
    all_point_cloud_paths = sorted(list(cfg.point_cloud_folder.rglob("*.ply")))
    all_config_dict_paths = sorted(list(cfg.config_dict_folder.rglob("*.npy")))
    assert len(all_point_cloud_paths) > 0, "No point cloud paths found"
    assert len(all_config_dict_paths) > 0, "No config dict paths found"

    print(f"Found {len(all_point_cloud_paths)} point cloud paths")
    print(f"Found {len(all_config_dict_paths)} config dict paths")

    # BPS selection (done once)
    basis_points = bps.generate_random_basis(
        n_points=cfg.N_BASIS_PTS, radius=cfg.BASIS_RADIUS, random_seed=13
    ) + np.array(
        [0.0, cfg.BASIS_RADIUS / 2, 0.0]
    )  # Shift up to get less under the table
    assert basis_points.shape == (
        cfg.N_BASIS_PTS,
        3,
    ), f"Expected shape ({cfg.N_BASIS_PTS}, 3), got {basis_points.shape}"

    # Point cloud per object
    # all_points = []
    # for i, data_path in tqdm(
    #     enumerate(all_point_cloud_paths),
    #     desc="Getting point clouds",
    #     total=len(all_point_cloud_paths),
    # ):
    #     point_cloud = o3d.io.read_point_cloud(str(data_path))
    #     point_cloud, _ = point_cloud.remove_statistical_outlier(
    #         nb_neighbors=20, std_ratio=2.0
    #     )
    #     point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    #     points = np.asarray(point_cloud.points)

    #     # inlier_points = process_point_cloud(points)
    #     # all_points.append(inlier_points)
    #     all_points.append(points)

    def process_single_point_cloud(data_path):
        point_cloud = o3d.io.read_point_cloud(str(data_path))
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
        points = np.asarray(point_cloud.points)
        inlier_points = process_point_cloud(points, distance_threshold=0.01)
        return inlier_points

    all_points = Parallel(n_jobs=-1)(
        delayed(process_single_point_cloud)(data_path) for data_path in tqdm(all_point_cloud_paths, desc="Processing point clouds")
    )

    min_n_pts = min([x.shape[0] for x in all_points])
    all_points = np.stack([x[:min_n_pts] for x in all_points])
    n_point_clouds, n_point_cloud_pts = all_points.shape[:2]

    # BPS values per object
    bps_values = bps.encode(
        all_points,
        bps_arrangement="custom",
        bps_cell_type="dists",
        custom_basis=basis_points,
        verbose=0,
    )
    assert bps_values.shape == (
        n_point_clouds,
        cfg.N_BASIS_PTS,
    ), f"Expected shape ({n_point_clouds}, {cfg.N_BASIS_PTS}), got {bps_values.shape}"

    # Extract the object_code_and_scale_strs we want
    object_code_and_scale_strs = [x.parents[0].name for x in all_point_cloud_paths]
    object_codes, object_scales = [], []
    for object_code_and_scale_str in object_code_and_scale_strs:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        object_codes.append(object_code)
        object_scales.append(object_scale)

    object_code_and_scale_str_to_idx = {
        object_code_and_scale_str: i
        for i, object_code_and_scale_str in enumerate(object_code_and_scale_strs)
    }

    # Grasp data per grasp
    (
        all_grasps,
        all_grasp_bps_idxs,
        all_passed_evals,
        all_passed_simulations,
        all_passed_penetration_thresholds,
        all_object_states,
    ) = get_grasp_data(
        all_config_dict_paths=all_config_dict_paths,
        object_code_and_scale_str_to_idx=object_code_and_scale_str_to_idx,
    )

    # Important shapes
    MAX_NUM_POINT_CLOUDS = n_point_clouds
    NUM_GRASPS = all_grasps.shape[0]
    MAX_NUM_GRASPS = NUM_GRASPS

    # Save to HDF5
    cfg.output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.output_filepath, "w") as hdf5_file:
        # Just one
        basis_points_dataset = hdf5_file.create_dataset(
            "/basis_points", shape=(cfg.N_BASIS_PTS, 3), dtype="f"
        )

        # Per object
        bpss_dataset = hdf5_file.create_dataset(
            "/bpss",
            shape=(
                MAX_NUM_POINT_CLOUDS,
                cfg.N_BASIS_PTS,
            ),
            dtype="f",
        )
        point_cloud_filepath_dataset = hdf5_file.create_dataset(
            "/point_cloud_filepath",
            shape=(MAX_NUM_POINT_CLOUDS,),
            dtype=h5py.string_dtype(),
        )
        object_code_dataset = hdf5_file.create_dataset(
            "/object_code", shape=(MAX_NUM_POINT_CLOUDS,), dtype=h5py.string_dtype()
        )
        object_scale_dataset = hdf5_file.create_dataset(
            "/object_scale", shape=(MAX_NUM_POINT_CLOUDS,), dtype="f"
        )

        # Per grasp
        grasps_dataset = hdf5_file.create_dataset(
            "/grasps", shape=(MAX_NUM_GRASPS, GRASP_DIM), dtype="f"
        )
        grasp_bps_idx_dataset = hdf5_file.create_dataset(
            "/grasp_bps_idx", shape=(MAX_NUM_GRASPS,), dtype="i"
        )
        passed_eval_dataset = hdf5_file.create_dataset(
            "/passed_eval",
            shape=(MAX_NUM_GRASPS,),
            dtype="f",
        )
        passed_simulation_dataset = hdf5_file.create_dataset(
            "/passed_simulation",
            shape=(MAX_NUM_GRASPS,),
            dtype="f",
        )
        passed_penetration_threshold_dataset = hdf5_file.create_dataset(
            "/passed_penetration_threshold",
            shape=(MAX_NUM_GRASPS,),
            dtype="f",
        )
        object_state_dataset = hdf5_file.create_dataset(
            "/object_state",
            shape=(
                MAX_NUM_GRASPS,
                13,
            ),
            dtype="f",
        )
        grasp_idx_dataset = hdf5_file.create_dataset(
            "/grasp_idx", shape=(MAX_NUM_GRASPS,), dtype="i"
        )

        # Just one
        basis_points_dataset[:] = basis_points

        # Per object
        bpss_dataset[:n_point_clouds] = bps_values
        point_cloud_filepath_dataset[:n_point_clouds] = [
            str(x) for x in all_point_cloud_paths
        ]
        object_code_dataset[:n_point_clouds] = object_codes
        object_scale_dataset[:n_point_clouds] = object_scales

        # Per grasp
        grasps_dataset[:NUM_GRASPS] = all_grasps
        grasp_bps_idx_dataset[:NUM_GRASPS] = all_grasp_bps_idxs
        passed_eval_dataset[:NUM_GRASPS] = all_passed_evals
        passed_simulation_dataset[:NUM_GRASPS] = all_passed_simulations
        passed_penetration_threshold_dataset[:NUM_GRASPS] = (
            all_passed_penetration_thresholds
        )
        object_state_dataset[:NUM_GRASPS] = all_object_states
        hdf5_file.attrs["num_grasps"] = NUM_GRASPS

if __name__ == "__main__":
    import argparse

    # [config-dict-folder paths]
    # "/home/albert/research/nerf_grasping/rsync_final_labeled_grasps_noise_and_nonoise/evaled_grasp_config_dicts_train",
    # "/home/albert/research/nerf_grasping/rsync_final_labeled_grasps_noise_and_nonoise/evaled_grasp_config_dicts_val",
    # "/home/albert/research/nerf_grasping/rsync_final_labeled_grasps_noise_and_nonoise/evaled_grasp_config_dicts_test",

    # [output-folder paths]
    # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_train.hdf5",
    # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_val.hdf5",
    # "/home/albert/research/nerf_grasping/bps_data/grasp_bps_dataset_final_test.hdf5",

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dict-folder", type=str, required=True)
    parser.add_argument("--output-filepath", type=str, required=True)
    args = parser.parse_args()

    main(args)