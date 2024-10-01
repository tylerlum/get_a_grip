# %%
"""
Visualization script for nerf grasp dataset
Useful to interactively understand the data
Percent script can be run like a Jupyter notebook or as a script
"""

# %%
import os
import pathlib
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import transforms3d
import trimesh
import tyro

from get_a_grip import get_data_folder, get_repo_folder
from get_a_grip.model_training.utils.nerf_grasp_dataset import (
    NerfGraspEvalDataset,
    NerfGraspSampleDataset,
)
from get_a_grip.model_training.utils.plot_utils import (
    plot_grasp_and_mesh_and_more,
)


# %%
@dataclass
class VisualizeNerfGraspDatasetV2Config:
    dataset_path: Path = (
        get_data_folder() / "SMALL_DATASET/nerf_grasp_dataset/train_dataset.h5"
    )
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    successful_grasps_only: bool = False
    grasp_idx: int = 0


# %%
os.chdir(get_repo_folder())


# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# %%
# Setup tqdm progress bar
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


# %%
# Setup config
if is_notebook():
    # Manually insert arguments here
    arguments = []
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")


# %%
cfg = tyro.cli(
    tyro.conf.FlagConversionOff[VisualizeNerfGraspDatasetV2Config], args=arguments
)
print("=" * 80)
print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
print("=" * 80 + "\n")

# %%
if cfg.successful_grasps_only:
    print("Loading successful grasps only")
    dataset = NerfGraspEvalDataset(
        input_hdf5_filepath=cfg.dataset_path,
    )
else:
    print("Loading all grasps")
    dataset = NerfGraspSampleDataset(
        input_hdf5_filepath=cfg.dataset_path,
    )
print(f"len(dataset): {len(dataset)}")

# %%
GRASP_IDX = cfg.grasp_idx
print(f"Getting data for grasp_idx={GRASP_IDX} from dataset")

# %%
grasp, nerf_global_grids_with_coords, y = dataset[GRASP_IDX]
print(f"grasp.shape: {grasp.shape}")
print(f"nerf_global_grids_with_coords.shape: {nerf_global_grids_with_coords.shape}")
print(f"y.shape: {y.shape}")

# %%
object_code = dataset.get_object_code(GRASP_IDX)
object_scale = dataset.get_object_scale(GRASP_IDX)
object_state = dataset.get_object_state(GRASP_IDX)
print(f"object_code: {object_code}")
print(f"object_scale: {object_scale}")
print(f"object_state.shape: {object_state.shape}")

# %%
# Mesh
mesh_path = cfg.meshdata_root_path / f"{object_code}/coacd/decomposed.obj"
assert mesh_path.exists(), f"{mesh_path} does not exist"
print(f"Reading mesh from {mesh_path}")
mesh = trimesh.load(mesh_path)

xyz, quat_xyzw = object_state[:3], object_state[3:7]
quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
X_N_Oy = np.eye(4)
X_N_Oy[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
X_N_Oy[:3, 3] = xyz
mesh.apply_scale(object_scale)
mesh.apply_transform(X_N_Oy)
print(f"Applying scale {object_scale} and transform {X_N_Oy} to mesh")

# %%

# %%
fig = plot_grasp_and_mesh_and_more(
    grasp=grasp,
    X_N_Oy=X_N_Oy,
    visualize_target_hand=True,
    visualize_pre_hand=False,
    mesh=mesh,
    basis_points=None,
    bps=None,
    raw_point_cloud_points=None,
    processed_point_cloud_points=None,
    nerf_global_grids_with_coords=nerf_global_grids_with_coords.detach().cpu().numpy(),
    title=f"{object_code}, idx: {GRASP_IDX}, y: {y}",
)


fig.show()

# %%

X_N_Oy[:3, 3]
# %%
