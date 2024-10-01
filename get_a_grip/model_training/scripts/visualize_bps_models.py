# %%
"""
Visualization script for bps models
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import transforms3d
import trimesh
import tyro

from get_a_grip import get_data_folder, get_repo_folder
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.models.bps_evaluator_model import BpsEvaluatorModel
from get_a_grip.model_training.models.bps_sampler_model import BpsSamplerModel
from get_a_grip.model_training.scripts.create_bps_grasp_dataset import (
    crop_single_point_cloud,
    read_and_process_single_point_cloud,
    read_raw_single_point_cloud,
)
from get_a_grip.model_training.utils.bps_grasp_dataset import (
    BpsGraspEvalDataset,
    BpsGraspSampleDataset,
)
from get_a_grip.model_training.utils.diffusion import Diffusion
from get_a_grip.model_training.utils.plot_utils import (
    plot_grasp_and_mesh_and_more,
)
from get_a_grip.visualization.utils.visualize_optimization_helper import (
    create_figure_with_buttons_and_slider,
)


# %%
@dataclass
class VisualizeBpsModelsConfig:
    sampler_ckpt_path: pathlib.Path = (
        get_data_folder()
        / "trained_models/bps_sampler_model/20240705003721/ckpt_100.pth"
    )
    evaluator_ckpt_path: pathlib.Path = (
        get_data_folder()
        / "trained_models/bps_grasp_evaluator/20240705_002442/ckpt-cyzuv30d-step-0.pth"
    )
    dataset_path: Path = (
        get_data_folder() / "SMALL_DATASET/bps_grasp_dataset/train_dataset.h5"
    )
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
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
cfg = tyro.cli(tyro.conf.FlagConversionOff[VisualizeBpsModelsConfig], args=arguments)
print("=" * 80)
print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
print("=" * 80 + "\n")

# %%
# loading bps sampler
diffusion_cfg = DiffusionConfig(
    training=TrainingConfig(
        output_dir=cfg.sampler_ckpt_path.parent,
    )
)
model = BpsSamplerModel(
    n_pts=diffusion_cfg.data.n_pts,
    grasp_dim=diffusion_cfg.data.grasp_dim,
)
runner = Diffusion(config=diffusion_cfg, model=model, load_multigpu_ckpt=True)
runner.load_checkpoint(diffusion_cfg, filename=cfg.sampler_ckpt_path.name)

# %%
# loading bps evaluator
bps_evaluator = BpsEvaluatorModel(in_grasp=37).to(runner.device)
bps_evaluator.load_state_dict(torch.load(cfg.evaluator_ckpt_path))
bps_evaluator.eval()

# %%
# running just the sampler
sample_dataset = BpsGraspSampleDataset(
    input_hdf5_filepath=cfg.dataset_path,
)
print(f"len(sample_dataset): {len(sample_dataset)}")

# %%
# loading eval data
eval_dataset = BpsGraspEvalDataset(
    input_hdf5_filepath=cfg.dataset_path,
)
print(f"len(eval_dataset): {len(eval_dataset)}")

eval_loader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

# %%
# validating the evaluator
ys_PGS_true = []
ys_PGS_pred = []
for i, (grasps, bpss, y_true) in tqdm(
    enumerate(eval_loader),
    desc="Iterations",
    total=len(eval_loader),
    leave=False,
):
    grasps, bpss, y_true = (
        grasps.to(runner.device),
        bpss.to(runner.device),
        y_true.to(runner.device),
    )
    y_pred = bps_evaluator(f_O=bpss, g_O=grasps)

    y_PGS_pred = y_pred[0, -1].detach().cpu().numpy()
    y_PGS_true = y_true[0, -1].detach().cpu().numpy()

    ys_PGS_true.append(y_PGS_true)
    ys_PGS_pred.append(y_PGS_pred)
    if i == 10_000:
        break

plt.scatter(ys_PGS_true, ys_PGS_pred, alpha=0.5, s=0.1)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("True PGS")
plt.ylabel("Predicted PGS")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("PGS True vs Predicted")
plt.axis("equal")
plt.show()

# %%
# sampling a grasp
GRASP_IDX = cfg.grasp_idx

_, bps, y = sample_dataset[GRASP_IDX]
print(f"bps.shape: {bps.shape}")
print(f"y.shape: {y.shape}")
xT = torch.randn(1, diffusion_cfg.data.grasp_dim, device=runner.device)
x = runner.sample(xT=xT, cond=bps[None].to(runner.device))  # (1, 37)
grasp = x

print(f"Sampled grasp shape: {grasp.shape}")
assert grasp.shape == (1, 37), f"Expected shape (1, 37), got {grasp.shape}"

# %%
# evaluating a grasp
y_pred = bps_evaluator(f_O=bps[None].to(runner.device), g_O=grasp.to(runner.device))
print(f"y_pred.shape: {y_pred.shape}")
assert y_pred.shape == (
    1,
    3,
), f"Expected shape (1, 3), got {y_pred.shape}"
print(f"Sampled grasp predicted quality: {y_pred}")

# %%
basis_points = sample_dataset.get_basis_points()
object_code = sample_dataset.get_object_code(GRASP_IDX)
object_scale = sample_dataset.get_object_scale(GRASP_IDX)
object_state = sample_dataset.get_object_state(GRASP_IDX)
point_cloud_filepath = sample_dataset.get_point_cloud_filepath(GRASP_IDX)
print(f"basis_points.shape: {basis_points.shape}")
print(f"object_code: {object_code}")
print(f"object_scale: {object_scale}")
print(f"object_state.shape: {object_state.shape}")
print(f"point_cloud_filepath: {point_cloud_filepath}")

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
# Point cloud
print(f"Reading point cloud from {point_cloud_filepath}")

raw_point_cloud_points = read_raw_single_point_cloud(
    point_cloud_filepath,
)
print(f"raw_point_cloud_points.shape = {raw_point_cloud_points.shape}")

processed_point_cloud_points_pre = read_and_process_single_point_cloud(
    point_cloud_filepath
)
processed_point_cloud_points = crop_single_point_cloud(processed_point_cloud_points_pre)
if processed_point_cloud_points is None:
    print(
        f"Skipping point cloud processing for {point_cloud_filepath}, processed_point_cloud_points_pre.shape: {processed_point_cloud_points_pre.shape}"
    )
print(f"processed_point_cloud_points.shape = {processed_point_cloud_points.shape}")


# %%
if grasp.abs().max() > 10:
    print(f"WARNING: Sampled grasp is large: {grasp.abs().max()}, replacing with None")
    grasp = None

fig = plot_grasp_and_mesh_and_more(
    grasp=grasp,
    X_N_Oy=X_N_Oy,
    visualize_target_hand=True,
    visualize_pre_hand=False,
    mesh=mesh,
    basis_points=basis_points.detach().cpu().numpy(),
    bps=bps.detach().cpu().numpy(),
    raw_point_cloud_points=raw_point_cloud_points,
    processed_point_cloud_points=processed_point_cloud_points,
    title=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, y_pred: {y_pred}",
)
fig.show()

# %%
xs, x0_preds = runner.sample_and_return_all_steps(
    xT=xT, cond=bps[None].to(runner.device)
)

N_STEPS, N_GRASPS, _ = xs.shape
assert xs.shape == (
    N_STEPS,
    N_GRASPS,
    37,
), f"Expected shape ({N_STEPS}, {N_GRASPS}, 37), got {xs.shape}"

# %%
# Visualize diffusion
individual_figs = []
for i in range(N_STEPS):
    x = xs[i][GRASP_IDX]
    assert x.shape == (37,), f"Expected shape (37,), got {x.shape}"

    if x.abs().max() > 10:
        print(f"WARNING: Sampled x is large: {x.abs().max()}, replacing with None")
        x = None

    individual_fig = plot_grasp_and_mesh_and_more(
        grasp=x,
        X_N_Oy=X_N_Oy,
        visualize_target_hand=True,
        visualize_pre_hand=False,
        mesh=mesh,
        basis_points=basis_points.detach().cpu().numpy(),
        bps=bps.detach().cpu().numpy(),
        raw_point_cloud_points=raw_point_cloud_points,
        processed_point_cloud_points=processed_point_cloud_points,
        title=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, y_pred: {y_pred}",
    )
    individual_figs.append(individual_fig)

fig_diffusion = create_figure_with_buttons_and_slider(
    input_figs=individual_figs,
    optimization_steps=[i for i in range(len(individual_figs))],
    frame_duration=200,
    transition_duration=100,
)
fig_diffusion.show()
