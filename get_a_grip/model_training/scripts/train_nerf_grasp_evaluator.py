# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [markdown]
# # Imports

# %%
from __future__ import annotations

import os
import pathlib
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pypose as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import tyro
import wandb
from clean_loop_timer import LoopTimer
from localscope import localscope
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import (
    DataLoader,
    Subset,
    random_split,
)
from torchinfo import summary
from torchviz import make_dot
from wandb.util import generate_id

from get_a_grip import get_data_folder, get_repo_folder
from get_a_grip.dataset_generation.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.model_training.config.classifier_config import (
    ClassifierConfig,
    ClassifierTrainingConfig,
    TaskType,
    UnionClassifierConfig,
)
from get_a_grip.model_training.config.fingertip_config import BaseFingertipConfig
from get_a_grip.model_training.models.classifier import Classifier
from get_a_grip.model_training.utils.nerf_grasp_evaluator_batch_data import (
    BatchData,
    BatchDataInput,
    BatchDataOutput,
)
from get_a_grip.model_training.utils.nerf_grasp_evaluator_dataset import (
    NeRFGrid_To_GraspSuccess_HDF5_Dataset,
)
from get_a_grip.model_training.utils.plot_utils import (
    plot_mesh_and_query_points,
)
from get_a_grip.model_training.utils.scheduler import get_scheduler

# %%
os.chdir(get_repo_folder())


# %%
def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


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
# Make atol and rtol larger than default to avoid errors due to floating point precision.
# Otherwise we get errors about invalid rotation matrices
PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4

NUM_XYZ = 3


class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    EVAL_TRAIN = auto()


# %%
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

# %% [markdown]
# # Setup Config for Static Type-Checking


# %% [markdown]
# # Load Config

# %%
if is_notebook():
    arguments = [
        "grasp-cond-simple-cnn-2d-1d",
        "--task-type",
        "Y_PICK",
        "--train-dataset-filepath",
        "data/2024-02-07_softball_0-051_5random/grid_dataset/train_dataset.h5",
        "--dataloader.batch-size",
        "32",
        "--wandb.name",
        "probe_DEBUG_softball9k-5random_grid_grasp-cond-simple-cnn-2d-1d_weighted-l1",
        "--checkpoint-workspace.input_leaf_dir_name",
        "DEBUG_softball9k-5random_grid_grasp-cond-simple-cnn-2d-1d_weighted-l1_2024-02-07_15-39-08-912606",
    ]
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")

# %%
cfg: ClassifierConfig = tyro.cli(UnionClassifierConfig, args=arguments)
assert cfg.nerfdata_config.fingertip_config is not None

# A relatively dirty hack: create script globals from the config vars.
NUM_FINGERS = cfg.nerfdata_config.fingertip_config.n_fingers
NUM_PTS_X = cfg.nerfdata_config.fingertip_config.num_pts_x
NUM_PTS_Y = cfg.nerfdata_config.fingertip_config.num_pts_y
NUM_PTS_Z = cfg.nerfdata_config.fingertip_config.num_pts_z


# %%
print("=" * 80)
print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
print("=" * 80 + "\n")

# %% [markdown]
# # Set Random Seed


# %%
@localscope.mfc
def set_seed(seed) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # TODO: Is this slowing things down?

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Set random seed to {seed}")


set_seed(cfg.random_seed)

# %% [markdown]
# # Setup Checkpoint Workspace and Maybe Resume Previous Run


# %%
# Set up checkpoint_workspace
cfg.checkpoint_workspace.root_dir.mkdir(parents=True, exist_ok=True)

# If input_dir != output_dir, then we create a new output_dir and wandb_run_id
if cfg.checkpoint_workspace.input_dir != cfg.checkpoint_workspace.output_dir:
    assert not cfg.checkpoint_workspace.output_dir.exists(), f"checkpoint_workspace.output_dir already exists at {cfg.checkpoint_workspace.output_dir}"
    print(
        f"input {cfg.checkpoint_workspace.input_dir} != output {cfg.checkpoint_workspace.output_dir}. Creating new wandb_run_id"
    )

    cfg.checkpoint_workspace.output_dir.mkdir()
    print(
        f"Done creating cfg.checkpoint_workspace.output_dir {cfg.checkpoint_workspace.output_dir}"
    )

    wandb_run_id = generate_id()
    wandb_run_id_filepath = cfg.checkpoint_workspace.output_dir / "wandb_run_id.txt"
    print(f"Saving wandb_run_id = {wandb_run_id} to {wandb_run_id_filepath}")
    with open(wandb_run_id_filepath, "w") as f:
        f.write(wandb_run_id)
    print("Done saving wandb_run_id")
# If input_dir == output_dir, then we must resume from checkpoint (else weird behavior in checkpoint dir)
else:
    assert (
        cfg.checkpoint_workspace.input_dir is not None
        and cfg.checkpoint_workspace.input_dir.exists()
    ), f"checkpoint_workspace.input_dir does not exist at {cfg.checkpoint_workspace.input_dir}"
    assert (
        cfg.wandb.resume != "never"
    ), f"checkpoint_workspace.input_dir is {cfg.checkpoint_workspace.input_dir}, but cfg.wandb.resume is {cfg.wandb.resume}"

    wandb_run_id_filepath = cfg.checkpoint_workspace.output_dir / "wandb_run_id.txt"
    print(f"Loading wandb_run_id from {wandb_run_id_filepath}")
    with open(wandb_run_id_filepath, "r") as f:
        wandb_run_id = f.read()
    print(f"Done loading wandb_run_id = {wandb_run_id}")

# %% [markdown]
# # Setup Wandb Logging

# %%
# Add to config

run = wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    name=cfg.wandb.name,
    group=cfg.wandb.group,
    job_type=cfg.wandb.job_type,
    config=asdict(cfg),
    id=wandb_run_id,
    resume=cfg.wandb.resume,
    reinit=True,
)

# %% [markdown]
# # Dataset and Dataloader


# %%
@localscope.mfc
def create_grid_dataset(
    input_hdf5_filepath: str, cfg: ClassifierConfig
) -> NeRFGrid_To_GraspSuccess_HDF5_Dataset:
    assert cfg.nerfdata_config.fingertip_config is not None
    return NeRFGrid_To_GraspSuccess_HDF5_Dataset(
        input_hdf5_filepath=input_hdf5_filepath,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        max_num_data_points=cfg.data.max_num_data_points,
        load_nerf_densities_in_ram=cfg.dataloader.load_nerf_grid_inputs_in_ram,
        load_grasp_labels_in_ram=cfg.dataloader.load_grasp_labels_in_ram,
        load_grasp_transforms_in_ram=cfg.dataloader.load_grasp_transforms_in_ram,
        load_nerf_configs_in_ram=cfg.dataloader.load_nerf_configs_in_ram,
    )


# %%
input_dataset_full_path = str(cfg.actual_train_dataset_filepath)
if cfg.create_val_test_from_train:
    print(
        f"Creating val and test datasets from train dataset: {input_dataset_full_path}"
    )
    full_dataset = create_grid_dataset(
        input_hdf5_filepath=input_dataset_full_path, cfg=cfg
    )

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [cfg.data.frac_train, cfg.data.frac_val, cfg.data.frac_test],
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
    assert_equals(
        len(set.intersection(set(train_dataset.indices), set(val_dataset.indices))), 0
    )
    assert_equals(
        len(set.intersection(set(train_dataset.indices), set(test_dataset.indices))), 0
    )
    assert_equals(
        len(set.intersection(set(val_dataset.indices), set(test_dataset.indices))), 0
    )
else:
    print(
        f"Using actual val and test datasets: input_dataset_full_path = {input_dataset_full_path}, cfg.actual_val_dataset_filepath = {cfg.actual_val_dataset_filepath}, cfg.actual_test_dataset_filepath = {cfg.actual_test_dataset_filepath}"
    )
    train_dataset = create_grid_dataset(
        input_hdf5_filepath=input_dataset_full_path, cfg=cfg
    )
    val_dataset = create_grid_dataset(
        input_hdf5_filepath=str(cfg.actual_val_dataset_filepath), cfg=cfg
    )
    test_dataset = create_grid_dataset(
        input_hdf5_filepath=str(cfg.actual_test_dataset_filepath), cfg=cfg
    )

# %%
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %%


@localscope.mfc(allowed=["PP_MATRIX_ATOL", "PP_MATRIX_RTOL"])
def sample_random_rotate_transforms(N: int) -> pp.LieTensor:
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # TODO(pculbert): Figure out better uniform sampling on SO(3).
    log_random_rotations = pp.so3(4 * torch.pi * (2 * torch.rand(N, 3) - 1))

    # Return exponentiated rotations.
    random_SO3_rotations = log_random_rotations.Exp()

    # A bit annoying -- need to cast SO(3) -> SE(3).
    random_rotate_transforms = pp.from_matrix(
        random_SO3_rotations.matrix(),
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    return random_rotate_transforms


@localscope.mfc(allowed=["PP_MATRIX_ATOL", "PP_MATRIX_RTOL"])
def sample_random_rotate_transforms_only_around_y(N: int) -> pp.LieTensor:
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # TODO(pculbert): Figure out better uniform sampling on SO(3).
    x_rotations = torch.zeros(N)
    y_rotations = 4 * torch.pi * (2 * torch.rand(N) - 1)
    z_rotations = torch.zeros(N)
    xyz_rotations = torch.stack([x_rotations, y_rotations, z_rotations], dim=-1)
    log_random_rotations = pp.so3(xyz_rotations)

    # Return exponentiated rotations.
    random_SO3_rotations = log_random_rotations.Exp()

    # A bit annoying -- need to cast SO(3) -> SE(3).
    random_rotate_transforms = pp.from_matrix(
        random_SO3_rotations.matrix(),
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    return random_rotate_transforms


@localscope.mfc(allowed=["PP_MATRIX_ATOL", "PP_MATRIX_RTOL"])
def custom_collate_fn(
    batch,
    fingertip_config: BaseFingertipConfig,
    use_random_rotations: bool = True,
    debug_shuffle_labels: bool = False,
    nerf_density_threshold_value: Optional[float] = None,
) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)
    (
        nerf_densities,
        nerf_densities_global,
        y_pick,
        y_coll,
        y_PGS,
        grasp_transforms,
        nerf_configs,
        grasp_configs,
    ) = batch

    if debug_shuffle_labels:
        shuffle_inds = torch.randperm(y_pick.shape[0])
        y_pick = y_pick[shuffle_inds]
        y_coll = y_coll[shuffle_inds]
        y_PGS = y_PGS[shuffle_inds]

    grasp_transforms = pp.from_matrix(
        grasp_transforms,
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    batch_size = nerf_densities.shape[0]
    if use_random_rotations:
        random_rotate_transform = sample_random_rotate_transforms_only_around_y(
            N=batch_size
        )
    else:
        random_rotate_transform = None

    return BatchData(
        input=BatchDataInput(
            nerf_densities=nerf_densities,
            grasp_transforms=grasp_transforms,
            random_rotate_transform=random_rotate_transform,
            fingertip_config=fingertip_config,
            nerf_density_threshold_value=nerf_density_threshold_value,
            grasp_configs=grasp_configs,
            nerf_densities_global=nerf_densities_global,
        ),
        output=BatchDataOutput(
            y_pick=y_pick,
            y_coll=y_coll,
            y_PGS=y_PGS,
        ),
        nerf_config=nerf_configs,
    )


# %%
train_collate_fn = partial(
    custom_collate_fn,
    fingertip_config=cfg.nerfdata_config.fingertip_config,
    use_random_rotations=cfg.data.use_random_rotations,
    debug_shuffle_labels=cfg.data.debug_shuffle_labels,
    nerf_density_threshold_value=cfg.data.nerf_density_threshold_value,
)
val_test_collate_fn = partial(
    custom_collate_fn,
    use_random_rotations=False,
    fingertip_config=cfg.nerfdata_config.fingertip_config,
    nerf_density_threshold_value=cfg.data.nerf_density_threshold_value,
)  # Run test over actual test transforms.

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=True,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=train_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=val_test_collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=val_test_collate_fn,
)

if cfg.data.use_random_rotations:
    print("Using random rotations for training")
else:
    print("Not using random rotations for training")


# %%
@localscope.mfc
def print_shapes(batch_data: BatchData) -> None:
    print(f"nerf_alphas.shape: {batch_data.input.nerf_alphas.shape}")
    print(f"coords.shape = {batch_data.input.coords.shape}")
    print(
        f"nerf_alphas_with_coords.shape = {batch_data.input.nerf_alphas_with_coords.shape}"
    )
    print(
        f"augmented_grasp_transforms.shape = {batch_data.input.augmented_grasp_transforms.shape}"
    )
    print(f"grasp_transforms.shape: {batch_data.input.grasp_transforms.shape}")
    print(f"y_pick.shape: {batch_data.output.y_pick.shape}")
    print(f"y_coll.shape: {batch_data.output.y_coll.shape}")
    print(f"y_PGS.shape: {batch_data.output.y_PGS.shape}")
    print(f"len(nerf_config): {len(batch_data.nerf_config)}")


EXAMPLE_BATCH_DATA: BatchData = next(iter(train_loader))
print_shapes(batch_data=EXAMPLE_BATCH_DATA)

# %% [markdown]
# # Visualize Data


# %%
@localscope.mfc(
    allowed=["NUM_FINGERS", "NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z", "NUM_XYZ"]
)
def nerf_densities_plot_example(
    batch_data: BatchData, idx_to_visualize: int = 0, augmented: bool = False
) -> go.Figure:
    assert isinstance(batch_data.input, BatchDataInput)

    if augmented:
        query_points_list = batch_data.input.augmented_coords[idx_to_visualize]
        additional_mesh_transform = (
            batch_data.input.random_rotate_transform[idx_to_visualize]
            .matrix()
            .cpu()
            .numpy()
            if batch_data.input.random_rotate_transform is not None
            else None
        )
    else:
        query_points_list = batch_data.input.coords[idx_to_visualize]
        additional_mesh_transform = None

    # Extract data
    colors = batch_data.input.nerf_alphas[idx_to_visualize]
    y_pick = batch_data.output.y_pick[idx_to_visualize].tolist()
    y_coll = batch_data.output.y_coll[idx_to_visualize].tolist()
    y_PGS = batch_data.output.y_PGS[idx_to_visualize].tolist()
    NUM_CLASSES = 2
    assert_equals(len(y_pick), NUM_CLASSES)
    assert_equals(len(y_coll), NUM_CLASSES)
    assert_equals(len(y_PGS), NUM_CLASSES)

    # Get probabilities of passing
    y_pick = y_pick[1]
    y_coll = y_coll[1]
    y_PGS = y_PGS[1]
    assert 0 <= y_pick <= 1, y_pick
    assert 0 <= y_coll <= 1, y_coll
    assert 0 <= y_PGS <= 1, y_PGS

    assert_equals(colors.shape, (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z))

    nerf_config = pathlib.Path(batch_data.nerf_config[idx_to_visualize])
    object_code_and_scale_str = nerf_config.parents[2].name
    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)

    # Path to meshes
    MESHDATA_ROOT = get_data_folder() / "large/meshes"
    mesh_path = MESHDATA_ROOT / object_code / "coacd" / "decomposed.obj"

    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))
    if additional_mesh_transform is not None:
        mesh.apply_transform(additional_mesh_transform)

    # Get query points from grasp_transforms
    assert_equals(
        query_points_list.shape,
        (
            NUM_FINGERS,
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        ),
    )
    query_points_list = query_points_list.permute((0, 2, 3, 4, 1))
    assert_equals(
        query_points_list.shape,
        (
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
            NUM_XYZ,
        ),
    )
    query_points_list = [
        query_points_list[finger_idx].reshape(-1, NUM_XYZ).cpu().numpy()
        for finger_idx in range(NUM_FINGERS)
    ]
    query_point_colors_list = [
        colors[finger_idx].reshape(-1).cpu().numpy()
        for finger_idx in range(NUM_FINGERS)
    ]
    fig = plot_mesh_and_query_points(
        mesh=mesh,
        query_points_list=query_points_list,
        query_points_colors_list=query_point_colors_list,
        num_fingers=NUM_FINGERS,
    )
    # Set title to label
    fig.update_layout(
        title_text=f"y_pick = {y_pick}, y_coll = {y_coll}, y_PGS = {y_PGS}"
    )
    return fig


# Add config var to enable / disable plotting.
# %%
PLOT_EXAMPLES = False
if PLOT_EXAMPLES:
    fig = nerf_densities_plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=1)
    fig.show()

# %%
if PLOT_EXAMPLES:
    fig = nerf_densities_plot_example(
        batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=2, augmented=True
    )
    fig.show()

# %%
if PLOT_EXAMPLES:
    fig = nerf_densities_plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=2)
    fig.show()

# %%
print(f"y_pick = {EXAMPLE_BATCH_DATA.output.y_pick}")
print(f"y_coll = {EXAMPLE_BATCH_DATA.output.y_coll}")


# %% [markdown]
# # Create Neural Network Model

# %%
# TODO(pculbert): double-check the specific instantiate call here is needed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pull out just the CNN (without wrapping for LieTorch) for training.
assert cfg.model_config is not None
classifier: Classifier = cfg.model_config.get_classifier_from_fingertip_config(
    fingertip_config=cfg.nerfdata_config.fingertip_config,
    n_tasks=cfg.task_type.n_tasks,
).to(device)

# %%
start_epoch = 0
optimizer = torch.optim.AdamW(
    params=classifier.parameters(),
    lr=cfg.training.lr,
    betas=cfg.training.betas,
    weight_decay=cfg.training.weight_decay,
)
lr_scheduler = get_scheduler(
    name=cfg.training.lr_scheduler_name,
    optimizer=optimizer,
    num_warmup_steps=cfg.training.lr_scheduler_num_warmup_steps,
    num_training_steps=(len(train_loader) * cfg.training.n_epochs),
    last_epoch=start_epoch - 1,
)

# %% [markdown]
# # Load Checkpoint

# %%
if cfg.checkpoint_workspace.input_dir is not None:
    assert cfg.checkpoint_workspace.input_dir.exists(), f"checkpoint_workspace.input_dir does not exist at {cfg.checkpoint_workspace.input_dir}"
    print(f"Loading checkpoint ({cfg.checkpoint_workspace.input_dir})...")
    latest_checkpoint_path = cfg.checkpoint_workspace.latest_input_checkpoint_path
    assert (
        latest_checkpoint_path is not None and latest_checkpoint_path.exists()
    ), f"latest_checkpoint_path does not exist at {latest_checkpoint_path}"

    checkpoint = torch.load(latest_checkpoint_path)
    classifier.load_state_dict(checkpoint["classifier"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    print("Done loading checkpoint")

# %% [markdown]
# # Visualize Model

# %%
print(f"classifier = {classifier}")
print(f"optimizer = {optimizer}")
print(f"lr_scheduler = {lr_scheduler}")

# %%
try:
    summary(
        model=classifier,
        input_size=(
            cfg.dataloader.batch_size,
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        ),
        device=device,
    )
except Exception as e:
    print(f"Exception: {e}")
    print("Skipping summary")

# %%
try:
    example_input = (
        torch.zeros(
            (
                cfg.dataloader.batch_size,
                NUM_FINGERS,
                NUM_PTS_X,
                NUM_PTS_Y,
                NUM_PTS_Z,
            )
        )
        .to(device)
        .requires_grad_(True)
    )
    example_output = classifier(example_input)
    dot = make_dot(
        example_output,
        params={
            **dict(classifier.named_parameters()),
            **{"NERF_INPUT": example_input},
            **{"GRASP_LABELS": example_output},
        },
    )
    model_graph_filename = "model_graph.png"
    model_graph_filename_no_ext, model_graph_file_ext = model_graph_filename.split(".")
    print(f"Saving to {model_graph_filename}...")
    dot.render(model_graph_filename_no_ext, format=model_graph_file_ext)
    print(f"Done saving to {model_graph_filename}")
except Exception as e:
    print(f"Exception: {e}")
    print("Skipping make_dot")

SHOW_DOT = False
if SHOW_DOT:
    dot

# %% [markdown]
# # Training Setup


# %%
@localscope.mfc
def save_checkpoint(
    checkpoint_output_dir: pathlib.Path,
    epoch: int,
    classifier: Classifier,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    checkpoint_filepath = checkpoint_output_dir / f"checkpoint_{epoch:04}.pt"
    print(f"Saving checkpoint to {checkpoint_filepath}")
    torch.save(
        {
            "epoch": epoch,
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


@localscope.mfc
def save_checkpoint_batch(
    checkpoint_output_dir: pathlib.Path,
    batch_idx: int,
    classifier: Classifier,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    checkpoint_filepath = checkpoint_output_dir / f"checkpoint_batch_{batch_idx:04}.pt"
    print(f"Saving checkpoint to {checkpoint_filepath}")
    torch.save(
        {
            "batch_idx": batch_idx,
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


# %%
GLOBAL_BATCH_IDX = 0


@localscope.mfc(allowed=["tqdm", "GLOBAL_BATCH_IDX"])
def _iterate_through_dataloader(
    loop_timer: LoopTimer,
    phase: Phase,
    dataloader: DataLoader,
    classifier: Classifier,
    device: torch.device,
    loss_fns: List[nn.Module],
    task_type: TaskType,
    checkpoint_output_dir: pathlib.Path,
    training_cfg: Optional[ClassifierTrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    max_num_batches: Optional[int] = None,
    log_every_n_batches: int = 5,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    losses_dict = defaultdict(list)  # loss name => list of losses (one per datapoint)
    predictions_dict, ground_truths_dict = (
        defaultdict(list),
        defaultdict(list),
    )  # task name => list of predictions / ground truths (one per datapoint)

    assert phase in [Phase.TRAIN, Phase.VAL, Phase.TEST, Phase.EVAL_TRAIN]
    if phase == Phase.TRAIN:
        classifier.train()
        assert training_cfg is not None and optimizer is not None
    else:
        classifier.eval()
        assert training_cfg is None and optimizer is None

    assert_equals(len(loss_fns), classifier.n_tasks)
    assert_equals(len(task_type.task_names), classifier.n_tasks)

    with torch.set_grad_enabled(phase == Phase.TRAIN):
        dataload_section_timer = loop_timer.add_section_timer("Data").start()
        for batch_idx, batch_data in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            dataload_section_timer.stop()

            batch_idx = int(batch_idx)

            if batch_idx % 1000 == 0 and phase == Phase.TRAIN:
                global GLOBAL_BATCH_IDX
                save_checkpoint_batch(
                    checkpoint_output_dir=checkpoint_output_dir,
                    batch_idx=GLOBAL_BATCH_IDX,
                    classifier=classifier,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
                GLOBAL_BATCH_IDX += 1

            if max_num_batches is not None and batch_idx >= max_num_batches:
                break

            batch_data: BatchData = batch_data.to(device)
            if torch.isnan(batch_data.input.nerf_alphas_with_augmented_coords).any():
                print("!" * 80)
                print(
                    f"Found {torch.isnan(batch_data.input.nerf_alphas_with_augmented_coords).sum()} NANs in batch_data.input.nerf_alphas_with_augmented_coords"
                )
                print("Skipping batch...")
                print("!" * 80)
                print()
                continue

            # Forward pass
            with loop_timer.add_section_timer("Fwd"):
                all_logits = classifier.get_all_logits(batch_data.input)
                assert_equals(
                    all_logits.shape,
                    (
                        batch_data.batch_size,
                        classifier.n_tasks,
                        classifier.n_classes,
                    ),
                )

                if task_type == TaskType.Y_PICK:
                    task_targets = [batch_data.output.y_pick]
                elif task_type == TaskType.Y_COLL:
                    task_targets = [batch_data.output.y_coll]
                elif task_type == TaskType.Y_PGS:
                    task_targets = [batch_data.output.y_PGS]
                elif task_type == TaskType.Y_PICK_AND_Y_COLL:
                    task_targets = [
                        batch_data.output.y_pick,
                        batch_data.output.y_coll,
                    ]
                elif task_type == TaskType.Y_PICK_AND_Y_COLL_AND_Y_PGS:
                    task_targets = [
                        batch_data.output.y_pick,
                        batch_data.output.y_coll,
                        batch_data.output.y_PGS,
                    ]
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")

                assert_equals(len(task_targets), classifier.n_tasks)

                task_losses = []
                for task_i, (loss_fn, task_target, task_name) in enumerate(
                    zip(loss_fns, task_targets, task_type.task_names)
                ):
                    task_logits = all_logits[:, task_i, :]
                    task_loss = loss_fn(
                        input=task_logits,
                        target=task_target,
                    )
                    assert task_loss.shape == (batch_data.batch_size,)
                    losses_dict[f"{task_name}_loss"].extend(task_loss.tolist())
                    task_losses.append(task_loss)
                task_losses = torch.stack(task_losses, dim=0)
                assert_equals(
                    task_losses.shape, (classifier.n_tasks, batch_data.batch_size)
                )
                total_losses = torch.mean(
                    task_losses, dim=0
                )  # TODO: Consider weighting losses.
                assert_equals(total_losses.shape, (batch_data.batch_size,))
                total_loss = torch.mean(total_losses)

            # Gradient step
            with loop_timer.add_section_timer("Bwd"):
                if phase == Phase.TRAIN and optimizer is not None:
                    optimizer.zero_grad()
                    total_loss.backward()

                    if (
                        training_cfg is not None
                        and training_cfg.grad_clip_val is not None
                    ):
                        torch.nn.utils.clip_grad_value_(
                            classifier.parameters(),
                            training_cfg.grad_clip_val,
                        )

                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

            # Loss logging
            with loop_timer.add_section_timer("Loss"):
                losses_dict["loss"].extend(total_losses.tolist())

            # Gather predictions
            with loop_timer.add_section_timer("Gather"):
                for task_i, (task_target, task_name) in enumerate(
                    zip(task_targets, task_type.task_names)
                ):
                    task_logits = all_logits[:, task_i, :]
                    assert_equals(task_logits.shape, (batch_data.batch_size, 2))
                    assert_equals(task_target.shape, (batch_data.batch_size, 2))

                    predictions = F.softmax(task_logits, dim=-1)[..., -1]
                    ground_truths = task_target[..., -1]
                    assert_equals(predictions.shape, (batch_data.batch_size,))
                    assert_equals(ground_truths.shape, (batch_data.batch_size,))

                    predictions_dict[f"{task_name}"] += predictions.tolist()
                    ground_truths_dict[f"{task_name}"] += ground_truths.tolist()

            if (
                wandb.run is not None
                and log_every_n_batches is not None
                and batch_idx % log_every_n_batches == 0
                and batch_idx != 0
            ):
                with loop_timer.add_section_timer("Log"):
                    mid_epoch_log_dict = {}
                    for loss_name, losses in losses_dict.items():
                        mid_epoch_log_dict.update(
                            {
                                f"mid_epoch/{phase.name.lower()}_{loss_name}": np.mean(
                                    losses
                                ),
                                f"mid_epoch/{phase.name.lower()}_{loss_name}_min": np.min(
                                    losses
                                ),
                                f"mid_epoch/{phase.name.lower()}_{loss_name}_quartile_25": np.quantile(
                                    losses, 0.25
                                ),
                                f"mid_epoch/{phase.name.lower()}_{loss_name}_median": np.median(
                                    losses
                                ),
                                f"mid_epoch/{phase.name.lower()}_{loss_name}_quartile_75": np.quantile(
                                    losses, 0.75
                                ),
                                f"mid_epoch/{phase.name.lower()}_{loss_name}_max": np.max(
                                    losses
                                ),
                            }
                        )
                    wandb.log(mid_epoch_log_dict)

                # loop_timer.pretty_print_section_times()

            # Set description
            if len(losses_dict["loss"]) > 0:
                loss_log_strs = [
                    f"{loss_name}: ".replace("_loss", "")
                    + f"{np.mean(losses):.3f}, "
                    + f"{np.median(losses):.3f}"
                    # + f"({np.min(losses):.3f}, "
                    # + f"{np.quantile(losses, 0.25):.3f}, "
                    # + f"{np.median(losses):.3f}, "
                    # + f"{np.quantile(losses, 0.75):.3f}, "
                    # + f"{np.max(losses):.3f})"
                    for loss_name, losses in losses_dict.items()
                ]

            else:
                loss_log_strs = ["loss: N/A"]

            description = " | ".join(
                [
                    f"{phase.name.lower()}",
                ]
                + loss_log_strs
            )
            pbar.set_description(description)

            if batch_idx < len(dataloader) - 1:
                # Avoid starting timer at end of last batch
                dataload_section_timer = loop_timer.add_section_timer("Data").start()

    return losses_dict, predictions_dict, ground_truths_dict


@localscope.mfc
def _create_wandb_scatter_plot(
    ground_truths: List[float],
    predictions: List[float],
    title: str,
) -> wandb.plots.Plot:
    # data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
    # table = wandb.Table(data=data, columns=["class_x", "class_y"])
    # wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
    data = [[x, y] for (x, y) in zip(ground_truths, predictions)]
    table = wandb.Table(data=data, columns=["ground_truth", "prediction"])
    return wandb.plot.scatter(
        table=table,
        x="ground_truth",
        y="prediction",
        title=title,
    )


@localscope.mfc
def _create_wandb_histogram_plot(
    ground_truths: List[float],
    predictions: List[float],
    title: str,
    match_ylims: bool = False,
) -> wandb.Image:
    unique_labels = np.unique(ground_truths)
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(10, 10))
    axes = axes.flatten()

    # Get predictions per label
    unique_labels_to_preds = {}
    for i, unique_label in enumerate(unique_labels):
        preds = np.array(predictions)
        idxs = np.array(ground_truths) == unique_label
        unique_labels_to_preds[unique_label] = preds[idxs]

    # Plot histogram per label
    for i, (unique_label, preds) in enumerate(sorted(unique_labels_to_preds.items())):
        axes[i].hist(preds, bins=50, alpha=0.7, color="blue")
        axes[i].set_title(f"Ground Truth: {unique_label}")
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel("Prediction")
        axes[i].set_ylabel("Count")

    # Matching ylims
    if match_ylims:
        max_y_val = max(ax.get_ylim()[1] for ax in axes)
        for i in range(len(axes)):
            axes[i].set_ylim(0, max_y_val)

    fig.suptitle(title)
    fig.tight_layout()
    return wandb.Image(fig)


@localscope.mfc
def create_log_dict(
    phase: Phase,
    loop_timer: LoopTimer,
    task_type: TaskType,
    losses_dict: Dict[str, List[float]],
    predictions_dict: Dict[str, List[float]],
    ground_truths_dict: Dict[str, List[float]],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    assert_equals(set(predictions_dict.keys()), set(ground_truths_dict.keys()))
    assert_equals(set(predictions_dict.keys()), set(task_type.task_names))

    temp_log_dict = {}  # Make code cleaner by excluding phase name until the end
    if optimizer is not None:
        temp_log_dict["lr"] = optimizer.param_groups[0]["lr"]

    with loop_timer.add_section_timer("Agg Loss"):
        for loss_name, losses in losses_dict.items():
            temp_log_dict[f"{loss_name}"] = np.mean(losses)
            temp_log_dict[f"{loss_name}_min"] = np.min(losses)
            temp_log_dict[f"{loss_name}_quartile_25"] = np.quantile(losses, 0.25)
            temp_log_dict[f"{loss_name}_median"] = np.median(losses)
            temp_log_dict[f"{loss_name}_quartile_75"] = np.quantile(losses, 0.75)
            temp_log_dict[f"{loss_name}_max"] = np.max(losses)

    with loop_timer.add_section_timer("Scatter"):
        pass  # Skip for now
        # Make scatter plot of predicted vs ground truth
        # for task_name in task_type.task_names:
        #     predictions = predictions_dict[task_name]
        #     ground_truths = ground_truths_dict[task_name]
        #     temp_log_dict[f"{task_name}_scatter"] = _create_wandb_scatter_plot(
        #         ground_truths=ground_truths,
        #         predictions=predictions,
        #         title=f"{phase.name.title()} {task_name} Scatter Plot",
        #     )

    with loop_timer.add_section_timer("Histogram"):
        pass  # Skip for now
        # For each task, make multiple histograms of prediction (one per label)
        # for task_name in task_type.task_names:
        #     temp_log_dict[f"{task_name}_histogram"] = _create_wandb_histogram_plot(
        #         ground_truths=ground_truths_dict[task_name],
        #         predictions=predictions_dict[task_name],
        #         title=f"{phase.name.title()} {task_name} Histogram",
        #     )

    with loop_timer.add_section_timer("Metrics"):
        for task_name in task_type.task_names:
            predictions = (
                np.array(predictions_dict[task_name]).round().astype(int).tolist()
            )
            ground_truths = (
                np.array(ground_truths_dict[task_name]).round().astype(int).tolist()
            )
            for metric_name, function in [
                ("accuracy", accuracy_score),
                ("precision", precision_score),
                ("recall", recall_score),
                ("f1", f1_score),
            ]:
                temp_log_dict[f"{task_name}_{metric_name}"] = function(
                    y_true=ground_truths, y_pred=predictions
                )

    with loop_timer.add_section_timer("Confusion Matrix"):
        pass  # Skip for now
        # for task_name in task_type.task_names:
        #     predictions = (
        #         np.array(predictions_dict[task_name]).round().astype(int).tolist()
        #     )
        #     ground_truths = (
        #         np.array(ground_truths_dict[task_name]).round().astype(int).tolist()
        #     )
        #     temp_log_dict[
        #         f"{task_name}_confusion_matrix"
        #     ] = wandb.plot.confusion_matrix(
        #         preds=predictions,
        #         y_true=ground_truths,
        #         class_names=["failure", "success"],
        #         title=f"{phase.name.title()} {task_name} Confusion Matrix",
        #     )

    log_dict = {}
    for key, value in temp_log_dict.items():
        log_dict[f"{phase.name.lower()}_{key}"] = value
    return log_dict


@localscope.mfc(allowed=["tqdm"])
def iterate_through_dataloader(
    phase: Phase,
    dataloader: DataLoader,
    classifier: Classifier,
    device: torch.device,
    loss_fns: List[nn.Module],
    task_type: TaskType,
    checkpoint_output_dir: pathlib.Path,
    training_cfg: Optional[ClassifierTrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    assert_equals(len(loss_fns), classifier.n_tasks)
    assert_equals(len(task_type.task_names), classifier.n_tasks)

    loop_timer = LoopTimer()

    # Iterate through dataloader and get logged results
    losses_dict, predictions_dict, ground_truths_dict = _iterate_through_dataloader(
        loop_timer=loop_timer,
        phase=phase,
        dataloader=dataloader,
        classifier=classifier,
        device=device,
        loss_fns=loss_fns,
        task_type=task_type,
        checkpoint_output_dir=checkpoint_output_dir,
        training_cfg=training_cfg,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Log
    log_dict = create_log_dict(
        loop_timer=loop_timer,
        phase=phase,
        task_type=task_type,
        losses_dict=losses_dict,
        predictions_dict=predictions_dict,
        ground_truths_dict=ground_truths_dict,
        optimizer=optimizer,
    )

    loop_timer.pretty_print_section_times()
    print()
    print()

    return log_dict


@localscope.mfc(allowed=["tqdm"])
def run_training_loop(
    training_cfg: ClassifierTrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classifier: Classifier,
    device: torch.device,
    loss_fns: List[nn.Module],
    optimizer: torch.optim.Optimizer,
    checkpoint_output_dir: pathlib.Path,
    task_type: TaskType,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    start_epoch: int = 0,
) -> None:
    training_loop_base_description = "Training Loop"
    for epoch in (
        pbar := tqdm(
            range(start_epoch, training_cfg.n_epochs),
            desc=training_loop_base_description,
        )
    ):
        epoch = int(epoch)
        wandb_log_dict = {}
        wandb_log_dict["epoch"] = epoch

        # Save checkpoint
        start_save_checkpoint_time = time.time()
        if epoch % training_cfg.save_checkpoint_freq == 0 and (
            epoch != 0 or training_cfg.save_checkpoint_on_epoch_0
        ):
            save_checkpoint(
                checkpoint_output_dir=checkpoint_output_dir,
                epoch=epoch,
                classifier=classifier,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        save_checkpoint_time_taken = time.time() - start_save_checkpoint_time

        # Train
        start_train_time = time.time()
        train_log_dict = iterate_through_dataloader(
            phase=Phase.TRAIN,
            dataloader=train_loader,
            classifier=classifier,
            device=device,
            loss_fns=loss_fns,
            task_type=task_type,
            checkpoint_output_dir=checkpoint_output_dir,
            training_cfg=training_cfg,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        wandb_log_dict.update(train_log_dict)
        train_time_taken = time.time() - start_train_time

        # Val
        # Can do this before or after training (decided on after since before it was always at -ln(1/N_CLASSES) ~ 0.69)
        start_val_time = time.time()
        if epoch % training_cfg.val_freq == 0 and (
            epoch != 0 or training_cfg.val_on_epoch_0
        ):
            classifier.eval()
            val_log_dict = iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                classifier=classifier,
                device=device,
                loss_fns=loss_fns,
                task_type=task_type,
                checkpoint_output_dir=checkpoint_output_dir,
            )
            wandb_log_dict.update(val_log_dict)
        val_time_taken = time.time() - start_val_time

        classifier.train()

        if wandb.run is not None:
            wandb.log(wandb_log_dict)

        # Set description
        description = " | ".join(
            [
                training_loop_base_description + " (s)",
                f"Save: {save_checkpoint_time_taken:.0f}",
                f"Train: {train_time_taken:.0f}",
                f"Val: {val_time_taken:.0f}",
            ]
        )
        pbar.set_description(description)


# %%
wandb.watch(classifier, log="gradients", log_freq=100)


# %%
# Want to weight loss function because of imbalances
# Have success rates that are unique discrete classes [0, 0.1, 0.2, ...]
# Need to map these to classes
@localscope.mfc
def get_unique_classes(success_rates: np.ndarray) -> np.ndarray:
    # [0.5, 0.2, 0.1, 0.5, 0.0, 0.0, ...] => [0, 0.1, 0.2, ...]
    assert success_rates.ndim == 1
    unique_classes = np.unique(success_rates)
    return np.sort(unique_classes)


@localscope.mfc
def get_class_from_success_rate(
    success_rates: np.ndarray, unique_classes: np.ndarray
) -> int:
    # [0.5, 0.2, 0.1, 0.5, 0.0, 0.0, ...], [0, 0.1, 0.2, ...] => [5, 2, 1, 5, 0, 0, ...]
    assert unique_classes.ndim == 1
    errors = np.absolute(unique_classes[None, :] - success_rates[:, None])
    return errors.argmin(axis=-1)


@localscope.mfc
def _compute_class_weights_success_rates(
    success_rates: np.ndarray, unique_classes: np.ndarray
) -> np.ndarray:
    classes = np.arange(len(unique_classes))
    y_classes = get_class_from_success_rate(
        success_rates=success_rates, unique_classes=unique_classes
    )
    return compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_classes,
    )


@localscope.mfc
def compute_class_weight_np(
    train_dataset: Union[Subset, Any], input_dataset_full_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # TODO: May break if train_dataset is not a subset, but separate val/test files
    print("Loading grasp success data for class weighting...")
    t1 = time.time()
    with h5py.File(input_dataset_full_path, "r") as hdf5_file:
        y_picks_np = np.array(hdf5_file["/y_pick"][()])
        y_coll_np = np.array(hdf5_file["/y_coll"][()])
        y_PGS_np = np.array(hdf5_file["/y_PGS"][()])
    t2 = time.time()
    print(f"Loaded grasp success data in {t2 - t1:.2f} s")

    print("Extracting training indices...")
    t3 = time.time()
    if isinstance(train_dataset, Subset):
        y_picks_np = y_picks_np[train_dataset.indices]
        y_coll_np = y_coll_np[train_dataset.indices]
        y_PGS_np = y_PGS_np[train_dataset.indices]
    t4 = time.time()
    print(f"Extracted training indices in {t4 - t3:.2f} s")

    print("Computing class weight with this data...")
    t5 = time.time()

    unique_y_picks = get_unique_classes(y_picks_np)
    unique_y_coll = get_unique_classes(y_coll_np)
    unique_y_PGS = get_unique_classes(y_PGS_np)

    # argmax required to make binary classes
    # y_pick_class_weight_np = _compute_class_weights_success_rates(
    #     success_rates=y_picks_np, unique_classes=unique_y_picks
    # )
    # y_coll_class_weight_np = _compute_class_weights_success_rates(
    #     success_rates=y_coll_np,
    #     unique_classes=unique_y_coll,
    # )
    # y_PGS_class_weight_np = _compute_class_weights_success_rates(
    #     success_rates=y_PGS_np, unique_classes=unique_y_PGS
    # )

    # With soft labels, the counts of labels are a bit weird
    # Instead, let's say num of label 0 is sum(1 - label)
    # and num of label 1 is sum(label)
    # This would be same as count if we had hard labels
    # class_weight = n_samples / (n_classes * np.bincount(y)) https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    assert y_picks_np.ndim == y_coll_np.ndim == y_PGS_np.ndim == 1
    y_pick_class_weight_np = np.array(
        [
            y_picks_np.shape[0] / (2 * (1 - y_picks_np).sum()),
            y_picks_np.shape[0] / (2 * y_picks_np.sum()),
        ]
    )
    y_coll_class_weight_np = np.array(
        [
            y_coll_np.shape[0] / (2 * (1 - y_coll_np).sum()),
            y_coll_np.shape[0] / (2 * y_coll_np.sum()),
        ]
    )
    y_PGS_class_weight_np = np.array(
        [
            y_PGS_np.shape[0] / (2 * (1 - y_PGS_np).sum()),
            y_PGS_np.shape[0] / (2 * y_PGS_np.sum()),
        ]
    )
    t6 = time.time()
    print(f"Computed class weight in {t6 - t5:.2f} s")

    return (
        y_pick_class_weight_np,
        y_coll_class_weight_np,
        y_PGS_class_weight_np,
        unique_y_picks,
        unique_y_coll,
        unique_y_PGS,
    )


(
    y_pick_class_weight,
    y_coll_class_weight,
    y_PGS_class_weight,
    unique_y_picks,
    unique_y_coll,
    unique_y_PGS,
) = compute_class_weight_np(
    train_dataset=train_dataset, input_dataset_full_path=input_dataset_full_path
)
y_pick_class_weight = torch.from_numpy(y_pick_class_weight).float().to(device)
y_coll_class_weight = torch.from_numpy(y_coll_class_weight).float().to(device)
y_PGS_class_weight = torch.from_numpy(y_PGS_class_weight).float().to(device)
unique_y_picks = torch.from_numpy(unique_y_picks).float().to(device)
unique_y_coll = torch.from_numpy(unique_y_coll).float().to(device)
unique_y_PGS = torch.from_numpy(unique_y_PGS).float().to(device)
print(f"y_pick_class_weight = {y_pick_class_weight}")
print(f"y_coll_class_weight = {y_coll_class_weight}")
print(f"y_PGS_class_weight = {y_PGS_class_weight}")
print(f"unique_y_picks = {unique_y_picks}")
print(f"unique_y_coll = {unique_y_coll}")
print(f"unique_y_PGS = {unique_y_PGS}")

if cfg.training.extra_punish_false_positive_factor != 0.0:
    print(
        f"cfg.training.extra_punish_false_positive_factor = {cfg.training.extra_punish_false_positive_factor}"
    )
    y_pick_class_weight[1] *= 1 + cfg.training.extra_punish_false_positive_factor
    y_coll_class_weight[1] *= 1 + cfg.training.extra_punish_false_positive_factor
    y_PGS_class_weight[1] *= 1 + cfg.training.extra_punish_false_positive_factor
    print(f"After adjustment, y_pick_class_weight: {y_pick_class_weight}")
    print(f"After adjustment, y_pick_class_weight: {y_coll_class_weight}")
    print(f"After adjustment, y_pick_class_weight: {y_PGS_class_weight}")


# %%
class SoftmaxL1Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.softmax(input, dim=-1)
        return self.l1_loss(input, target).mean(dim=-1)


class SoftmaxL2Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.l2_loss = nn.MSELoss(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.softmax(input, dim=-1)
        return self.l2_loss(input, target).mean(dim=-1)


class WeightedSoftmaxL1(nn.Module):
    def __init__(
        self,
        unique_label_weights: torch.Tensor,
        unique_labels: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.unique_label_weights = unique_label_weights
        self.unique_labels = unique_labels

        (N,) = self.unique_labels.shape
        (N2,) = self.unique_label_weights.shape
        assert_equals(N, N2)

        self.l1_loss = nn.L1Loss(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        assert_equals(input.shape, (batch_size, 2))
        assert_equals(target.shape, (batch_size, 2))

        # Compute the raw L1 losses
        input = torch.softmax(input, dim=-1)
        l1_losses = self.l1_loss(input, target).mean(dim=-1)

        assert_equals(l1_losses.shape, (batch_size,))

        # Find the closest label for each target and its index
        target_indices = (
            (self.unique_labels.unsqueeze(0) - target[:, 1:2]).abs().argmin(dim=1)
        )

        # Gather the weights for each sample
        weights = self.unique_label_weights[target_indices]

        # Weight the L1 losses
        weighted_losses = weights * l1_losses

        assert len(weighted_losses.shape) == 1
        return weighted_losses


class WeightedSoftmaxL2(nn.Module):
    def __init__(
        self,
        unique_label_weights: torch.Tensor,
        unique_labels: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.unique_label_weights = unique_label_weights
        self.unique_labels = unique_labels

        (N,) = self.unique_labels.shape
        (N2,) = self.unique_label_weights.shape
        assert_equals(N, N2)

        self.l2_loss = nn.MSELoss(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        assert_equals(input.shape, (batch_size, 2))
        assert_equals(target.shape, (batch_size, 2))

        # Compute the raw L2 losses
        input = torch.softmax(input, dim=-1)
        l2_losses = self.l2_loss(input, target).mean(dim=-1)

        assert_equals(l2_losses.shape, (batch_size,))

        # Find the closest label for each target and its index
        target_indices = (
            (self.unique_labels.unsqueeze(0) - target[:, 1:2]).abs().argmin(dim=1)
        )

        # Gather the weights for each sample
        weights = self.unique_label_weights[target_indices]

        # Weight the L2 losses
        weighted_losses = weights * l2_losses

        assert len(weighted_losses.shape) == 1
        return weighted_losses


if cfg.training.loss_fn == "l1":
    print("=" * 80)
    print("Using L1 loss")
    print("=" * 80 + "\n")
    y_pick_loss_fn = SoftmaxL1Loss(reduction="none")
    y_coll_loss_fn = SoftmaxL1Loss(reduction="none")
    y_PGS_loss_fn = SoftmaxL1Loss(reduction="none")
elif cfg.training.loss_fn == "l2":
    print("=" * 80)
    print("Using L2 loss")
    print("=" * 80 + "\n")
    y_pick_loss_fn = SoftmaxL2Loss(reduction="none")
    y_coll_loss_fn = SoftmaxL2Loss(reduction="none")
    y_PGS_loss_fn = SoftmaxL2Loss(reduction="none")
elif cfg.training.loss_fn == "weighted_l1":
    print("=" * 80)
    print("Using Weighted L1 loss")
    print("=" * 80 + "\n")
    y_pick_loss_fn = WeightedSoftmaxL1(
        unique_label_weights=y_pick_class_weight,
        unique_labels=unique_y_picks,
        reduction="none",
    )
    y_coll_loss_fn = WeightedSoftmaxL1(
        unique_label_weights=y_coll_class_weight,
        unique_labels=unique_y_coll,
        reduction="none",
    )
    y_PGS_loss_fn = WeightedSoftmaxL1(
        unique_label_weights=y_PGS_class_weight,
        unique_labels=unique_y_PGS,
        reduction="none",
    )
elif cfg.training.loss_fn == "weighted_l2":
    print("=" * 80)
    print("Using Weighted L2 loss")
    print("=" * 80 + "\n")
    y_pick_loss_fn = WeightedSoftmaxL2(
        unique_label_weights=y_pick_class_weight,
        unique_labels=unique_y_picks,
        reduction="none",
    )
    y_coll_loss_fn = WeightedSoftmaxL2(
        unique_label_weights=y_coll_class_weight,
        unique_labels=unique_y_coll,
        reduction="none",
    )
    y_PGS_loss_fn = WeightedSoftmaxL2(
        unique_label_weights=y_PGS_class_weight,
        unique_labels=unique_y_PGS,
        reduction="none",
    )

elif cfg.training.loss_fn == "cross_entropy":
    print("=" * 80)
    print("Using CE loss")
    print("=" * 80 + "\n")
    y_pick_loss_fn = nn.CrossEntropyLoss(
        weight=y_pick_class_weight,
        label_smoothing=cfg.training.label_smoothing,
        reduction="none",
    )
    y_coll_loss_fn = nn.CrossEntropyLoss(
        weight=y_coll_class_weight,
        label_smoothing=cfg.training.label_smoothing,
        reduction="none",
    )
    y_PGS_loss_fn = nn.CrossEntropyLoss(
        weight=y_PGS_class_weight,
        label_smoothing=cfg.training.label_smoothing,
        reduction="none",
    )
else:
    raise ValueError("Unknown loss function")

if cfg.task_type == TaskType.Y_PICK:
    loss_fns = [y_pick_loss_fn]
elif cfg.task_type == TaskType.Y_COLL:
    loss_fns = [y_coll_loss_fn]
elif cfg.task_type == TaskType.Y_PGS:
    loss_fns = [y_PGS_loss_fn]
elif cfg.task_type == TaskType.Y_PICK_AND_Y_COLL:
    loss_fns = [
        y_pick_loss_fn,
        y_coll_loss_fn,
    ]
elif cfg.task_type == TaskType.Y_PICK_AND_Y_COLL_AND_Y_PGS:
    loss_fns = [
        y_pick_loss_fn,
        y_coll_loss_fn,
        y_PGS_loss_fn,
    ]
else:
    raise ValueError(f"Unknown task_type: {cfg.task_type}")

# %%

# Save out config to file if we haven't yet.
cfg_path = pathlib.Path(cfg.checkpoint_workspace.output_dir) / "config.yaml"
if not cfg_path.exists():
    cfg_yaml = tyro.extras.to_yaml(cfg)
    with open(cfg_path, "w") as f:
        f.write(cfg_yaml)

print(cfg)
if cfg.data.debug_shuffle_labels:
    print(
        "WARNING: Shuffle labels is turned on! Random labels are being passed. Press 'c' to continue"
    )

# # %% [markdown]
# # Debug/Analyze model
# DEBUG_phase = Phase.EVAL_TRAIN
# if DEBUG_phase == Phase.EVAL_TRAIN:
#     DEBUG_loader = train_loader
# elif DEBUG_phase == Phase.VAL:
#     DEBUG_loader = val_loader
# elif DEBUG_phase == Phase.TEST:
#     DEBUG_loader = test_loader
# else:
#     raise ValueError(f"Unknown phase: {DEBUG_phase}")
#
#
# loop_timer = LoopTimer()
# (
#     DEBUG_losses_dict,
#     DEBUG_predictions_dict,
#     DEBUG_ground_truths_dict,
# ) = _iterate_through_dataloader(
#     loop_timer=loop_timer,
#     phase=DEBUG_phase,
#     dataloader=DEBUG_loader,
#     classifier=classifier,
#     device=device,
#     loss_fns=loss_fns,
#     task_type=cfg.task_type,
#     max_num_batches=None,
# )
#
# # %%
# DEBUG_losses_dict.keys()
#
# # %%
# DEBUG_losses_dict["y_pick_loss"][:10]
#
# # %%
# DEBUG_predictions_dict["y_pick"][:10]
#
# # %%
# DEBUG_ground_truths_dict["y_pick"][:10]
#
# # %%
# DEBUG_predictions_dict
#
# # %%
# import matplotlib.pyplot as plt
#
# # Small circles
# gaussian_noise = np.random.normal(
#     0, 0.01, len(DEBUG_ground_truths_dict["y_pick"])
# )
# plt.scatter(
#     DEBUG_ground_truths_dict["y_pick"] + gaussian_noise,
#     DEBUG_predictions_dict["y_pick"],
#     s=0.1,
# )
# plt.xlabel("Ground Truth")
# plt.ylabel("Prediction")
# plt.title(f"y_pick Scatter Plot")
# plt.show()
#
# # %%
# np.unique(DEBUG_ground_truths_dict["y_pick"], return_counts=True)
#
# # %%
# unique_labels = np.unique(DEBUG_ground_truths_dict["y_pick"])
# fig, axes = plt.subplots(len(unique_labels), 1, figsize=(10, 10))
# axes = axes.flatten()
#
# unique_label_to_preds = {}
# for i, unique_val in enumerate(unique_labels):
#     preds = np.array(DEBUG_predictions_dict["y_pick"])
#     idxs = np.array(DEBUG_ground_truths_dict["y_pick"]) == unique_val
#     unique_label_to_preds[unique_val] = preds[idxs]
#
# for i, (unique_label, preds) in enumerate(sorted(unique_label_to_preds.items())):
#     # axes[i].hist(preds, bins=50, alpha=0.7, color="blue", log=True)
#     axes[i].hist(preds, bins=50, alpha=0.7, color="blue")
#     axes[i].set_title(f"Ground Truth: {unique_label}")
#     axes[i].set_xlim(0, 1)
#
# # Matching ylims
# max_y_val = max(ax.get_ylim()[1] for ax in axes)
# for i in range(len(axes)):
#     axes[i].set_ylim(0, max_y_val)
#
# fig.tight_layout()
#
#
# # %%
# DEBUG_log_dict = create_log_dict(
#     loop_timer=loop_timer,
#     phase=DEBUG_phase,
#     task_type=cfg.task_type,
#     losses_dict=DEBUG_losses_dict,
#     predictions_dict=DEBUG_predictions_dict,
#     ground_truths_dict=DEBUG_ground_truths_dict,
#     optimizer=optimizer,
# )
#
# # %%
# DEBUG_log_dict[f"{DEBUG_phase.name.lower()}_loss"]
#
# # %%
# DEBUG_log_dict_modified = {f"{k}_v2": v for k, v in DEBUG_log_dict.items()}
#
# # %%
# wandb.log(DEBUG_log_dict_modified)
#
#
# # %%
# loop_timer = LoopTimer()
# (
#     train_losses_dict,
#     train_predictions_dict,
#     train_ground_truths_dict,
# ) = _iterate_through_dataloader(
#     loop_timer=loop_timer,
#     phase=Phase.EVAL_TRAIN,
#     dataloader=train_loader,
#     classifier=classifier,
#     device=device,
#     loss_fns=loss_fns,
#     task_type=cfg.task_type,
#     max_num_batches=10,
# )
#
# # %%
# loss_names = [
#     "y_pick_loss",
#     "y_coll_loss",
# ]
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
#
# fig = make_subplots(rows=len(loss_names), cols=1, subplot_titles=loss_names)
# for i, loss_name in enumerate(loss_names):
#     fig.add_trace(
#         go.Scatter(y=val_losses_dict[loss_name], name=loss_name, mode="markers"),
#         row=i + 1,
#         col=1,
#     )
# fig.show()
#
#
# # %%
# def plot_distribution(data: np.ndarray, name: str) -> None:
#     # Calculating statistics
#     import scipy.stats as stats
#
#     data = np.array(data)
#     mean = np.mean(data)
#     max_value = np.max(data)
#     min_value = np.min(data)
#     data_range = np.ptp(data)  # Range as max - min
#     std_dev = np.std(data)
#     median = np.median(data)
#     mode = stats.mode(data).mode[0]
#     iqr = stats.iqr(data)  # Interquartile range
#     percentile_25 = np.percentile(data, 25)
#     percentile_75 = np.percentile(data, 75)
#
#     import matplotlib.pyplot as plt
#
#     # Create histogram
#     plt.hist(data, bins=50, alpha=0.7, color="blue", log=True)
#
#     # Printing results
#     print(
#         f"Mean: {mean}, Max: {max_value}, Min: {min_value}, Range: {data_range}, Standard Deviation: {std_dev}"
#     )
#     print(
#         f"Median: {median}, Mode: {mode}, IQR: {iqr}, 25th Percentile: {percentile_25}, 75th Percentile: {percentile_75}"
#     )
#
#     # Add lines for mean, median, and mode
#     plt.axvline(
#         mean, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.4f}"
#     )
#     plt.axvline(
#         median,
#         color="green",
#         linestyle="dashed",
#         linewidth=2,
#         label=f"Median: {median:.4f}",
#     )
#     plt.axvline(
#         mode, color="yellow", linestyle="dashed", linewidth=2, label=f"Mode: {mode:.4f}"
#     )
#
#     # Add lines for percentiles
#     plt.axvline(
#         percentile_25,
#         color="orange",
#         linestyle="dotted",
#         linewidth=2,
#         label=f"25th percentile: {percentile_25:.4f}",
#     )
#     plt.axvline(
#         percentile_75,
#         color="purple",
#         linestyle="dotted",
#         linewidth=2,
#         label=f"75th percentile: {percentile_75:.4f}",
#     )
#
#     # Add standard deviation
#     plt.axvline(
#         mean - std_dev,
#         color="cyan",
#         linestyle="dashdot",
#         linewidth=2,
#         label=f"Std Dev: {std_dev:.4f}",
#     )
#     plt.axvline(mean + std_dev, color="cyan", linestyle="dashdot", linewidth=2)
#
#     # Add legend
#     plt.legend()
#     plt.title(f"{name} histogram")
#
#     # Show plot
#     plt.show()
#
#
# plot_distribution(
#     data=val_losses_dict["y_coll_loss"],
#     name="y_coll_loss",
# )
#
# # %%
# plot_distribution(
#     data=val_losses_dict["y_pick_loss"], name="y_pick_loss"
# )

# %%


run_training_loop(
    training_cfg=cfg.training,
    train_loader=train_loader,
    val_loader=val_loader,
    classifier=classifier,
    device=device,
    loss_fns=loss_fns,
    optimizer=optimizer,
    checkpoint_output_dir=cfg.checkpoint_workspace.output_dir,
    task_type=cfg.task_type,
    lr_scheduler=lr_scheduler,
    start_epoch=start_epoch,
)

# %% [markdown]
# # Test

# %%
classifier.eval()
wandb_log_dict = {}
print(f"Running test metrics on epoch {cfg.training.n_epochs}")
wandb_log_dict["epoch"] = cfg.training.n_epochs
test_log_dict = iterate_through_dataloader(
    phase=Phase.TEST,
    dataloader=test_loader,
    classifier=classifier,
    device=device,
    loss_fns=loss_fns,
    task_type=cfg.task_type,
    checkpoint_output_dir=cfg.checkpoint_workspace.output_dir,
)
wandb_log_dict.update(test_log_dict)
wandb.log(wandb_log_dict)

# %% [markdown]
# # Save Model

# %%
save_checkpoint(
    checkpoint_output_dir=cfg.checkpoint_workspace.output_dir,
    epoch=cfg.training.n_epochs,
    classifier=classifier,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

# %%
wandb.finish()
