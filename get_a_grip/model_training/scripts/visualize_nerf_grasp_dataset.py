# %%
"""
Visualization script for nerf_grasp_dataset
Useful to interactively understand the data
Percent script can be run like a Jupyter notebook or as a script
"""

# %%
import os
import sys
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tyro
from clean_loop_timer import LoopTimer

from get_a_grip import get_repo_folder
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.config.nerf_grasp_dataset_config import (
    NerfGraspDatasetConfig,
)
from get_a_grip.model_training.scripts.create_nerf_grasp_dataset import (
    compute_X_N_Oy,
    create_mesh,
    get_data,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
)
from get_a_grip.model_training.utils.plot_utils import (
    plot_mesh_and_high_density_points,
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
)
from get_a_grip.utils.nerf_load_utils import (
    get_nerf_configs,
)
from get_a_grip.utils.point_utils import (
    transform_points,
)

# %%
os.chdir(get_repo_folder())


# %%
@dataclass
class VisualizeNerfGraspDatasetConfig(NerfGraspDatasetConfig):
    object_idx: int = 0
    grasp_idx: int = 0


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
    arguments = [
        "--input_nerfcheckpoints_path",
        "data/dataset/NEW/nerfcheckpoints",
        "--input_evaled_grasp_config_dicts_path",
        "data/dataset/NEW/final_evaled_grasp_config_dicts_train",
        "--output_filepath",
        "data/dataset/NEW/nerf_grasp_dataset/train_dataset.h5",
    ]
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")

# %%

cfg = tyro.cli(
    tyro.conf.FlagConversionOff[VisualizeNerfGraspDatasetConfig], args=arguments
)
print("=" * 80)
print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
print("=" * 80 + "\n")

# %%
# Find all nerf configs
assert (
    cfg.input_nerfcheckpoints_path.exists()
), f"{cfg.input_nerfcheckpoints_path} does not exist"
nerf_configs = get_nerf_configs(
    nerfcheckpoints_path=cfg.input_nerfcheckpoints_path,
)
assert (
    len(nerf_configs) > 0
), f"Did not find any nerf configs in {cfg.input_nerfcheckpoints_path}"
print(f"Found {len(nerf_configs)} nerf configs")

# %%
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

# %%
# Precompute ray origins in finger frame
ray_origins_finger_frame = get_ray_origins_finger_frame(cfg.fingertip_config)

# %%
# Get data
loop_timer = LoopTimer()
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
    query_points_N,
    query_points_global_N,
) = get_data(
    evaled_grasp_config_dict_filepath=evaled_grasp_config_dict_filepaths[
        cfg.object_idx
    ],
    fingertip_config=cfg.fingertip_config,
    ray_samples_chunk_size=cfg.ray_samples_chunk_size,
    max_num_data_points_per_file=cfg.max_num_data_points_per_file,
    nerf_configs=nerf_configs,
    ray_origins_finger_frame=ray_origins_finger_frame,
    loop_timer=loop_timer,
)

# %%
mesh_Oy = create_mesh(object_code=object_code, object_scale=object_scale)
X_N_Oy = compute_X_N_Oy(mesh_Oy=mesh_Oy)
assert np.allclose(X_N_Oy[:3, :3], np.eye(3)), f"X_N_Oy = {X_N_Oy}"
mesh_N = mesh_Oy.copy()
mesh_N.apply_transform(X_N_Oy)

# %%
grasp_frame_transforms = (
    grasp_frame_transforms.matrix().cpu().detach().numpy()[cfg.grasp_idx]
)
assert grasp_frame_transforms.shape == (cfg.fingertip_config.n_fingers, 4, 4)
nerf_densities = nerf_densities.detach().cpu().numpy()[cfg.grasp_idx]
assert nerf_densities.shape == (
    cfg.fingertip_config.n_fingers,
    cfg.fingertip_config.num_pts_x,
    cfg.fingertip_config.num_pts_y,
    cfg.fingertip_config.num_pts_z,
)

query_points_N = query_points_N.detach().cpu().numpy()[cfg.grasp_idx]
assert query_points_N.shape == (
    cfg.fingertip_config.n_fingers,
    cfg.fingertip_config.num_pts_x,
    cfg.fingertip_config.num_pts_y,
    cfg.fingertip_config.num_pts_z,
    3,
)

nerf_densities_global = nerf_densities_global.detach().cpu().numpy()
assert nerf_densities_global.shape == (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)

query_points_global_N = query_points_global_N.detach().cpu().numpy()
assert query_points_global_N.shape == (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    3,
)

# %%
# Compute alpha values
delta = cfg.fingertip_config.distance_between_pts_mm
nerf_alphas = [1 - np.exp(-delta * dd) for dd in nerf_densities]

# %%
# Plot 1
query_points_N = np.stack([qq.reshape(-1, 3) for qq in query_points_N], axis=0)
query_points_colors = np.stack([x.reshape(-1) for x in nerf_alphas], axis=0)
fig = plot_mesh_and_query_points(
    mesh=mesh_N,
    all_query_points=query_points_N,
    all_query_points_colors=np.stack([x.reshape(-1) for x in nerf_alphas], axis=0),
    num_fingers=cfg.fingertip_config.n_fingers,
    title=f"Mesh and Query Points, Success: {y_PGSs[cfg.grasp_idx]}",
)
fig.show()

# %%
# Plot 2
fig2 = plot_mesh_and_transforms(
    mesh=mesh_Oy,
    transforms=[
        grasp_frame_transforms[i] for i in range(cfg.fingertip_config.n_fingers)
    ],
    num_fingers=cfg.fingertip_config.n_fingers,
    title=f"Mesh and Transforms, Success: {y_PGSs[cfg.grasp_idx]}",
)
fig2.show()

# %%
# Plot 3
nerf_alphas_global = 1 - np.exp(-delta * nerf_densities_global)

X_Oy_N = np.linalg.inv(X_N_Oy)
query_points_global_Oy = transform_points(
    T=X_Oy_N, points=query_points_global_N.reshape(-1, 3)
).reshape(*query_points_global_N.shape)
fig3 = plot_mesh_and_high_density_points(
    mesh=mesh_Oy,
    all_query_points=query_points_global_Oy.reshape(-1, 3),
    all_query_points_colors=nerf_alphas_global.reshape(-1),
    density_threshold=0.01,
)
fig3.show()

# %%
# Plot 4
nrows, ncols = cfg.fingertip_config.n_fingers, 1
fig4, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
axes = axes.flatten()
for i in range(cfg.fingertip_config.n_fingers):
    ax = axes[i]
    finger_alphas = nerf_alphas[i].reshape(
        cfg.fingertip_config.num_pts_x,
        cfg.fingertip_config.num_pts_y,
        cfg.fingertip_config.num_pts_z,
    )
    finger_alphas_maxes = np.max(finger_alphas, axis=(0, 1))
    finger_alphas_means = np.mean(finger_alphas, axis=(0, 1))
    ax.plot(finger_alphas_maxes, label="max")
    ax.plot(finger_alphas_means, label="mean")
    ax.legend()
    ax.set_xlabel("z")
    ax.set_ylabel("alpha")
    ax.set_title(f"finger {i}")
    ax.set_ylim([0, 1])
fig4.tight_layout()
fig4.show()

# %%
# Plot 5
num_images = 5
nrows, ncols = cfg.fingertip_config.n_fingers, num_images
fig5, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
alpha_images = [
    x.reshape(
        cfg.fingertip_config.num_pts_x,
        cfg.fingertip_config.num_pts_y,
        cfg.fingertip_config.num_pts_z,
    )
    for x in nerf_alphas
]

for finger_i in range(cfg.fingertip_config.n_fingers):
    for image_i in range(num_images):
        ax = axes[finger_i, image_i]
        image = alpha_images[finger_i][
            :,
            :,
            int(image_i * cfg.fingertip_config.num_pts_z / num_images),
        ]
        ax.imshow(
            image,
            vmin=nerf_alphas[i].min(),
            vmax=nerf_alphas[i].max(),
        )
        ax.set_title(f"finger {finger_i}, image {image_i}")
fig5.tight_layout()
fig5.show()
plt.show(block=True)

# %%
