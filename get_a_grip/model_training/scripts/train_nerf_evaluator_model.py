from __future__ import annotations

import pathlib
import time
from collections import defaultdict
from dataclasses import asdict
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pypose as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import tyro
from clean_loop_timer import LoopTimer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import (
    DataLoader,
    Subset,
    random_split,
)
from tqdm import tqdm as std_tqdm

import wandb
from get_a_grip import get_data_folder
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    sample_random_rotate_transforms_only_around_y,
)
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
    PlotConfig,
    TaskType,
    TrainingConfig,
)
from get_a_grip.model_training.models.nerf_evaluator_model import NerfEvaluatorModel
from get_a_grip.model_training.utils.nerf_evaluator_model_batch_data import (
    BatchData,
    BatchDataInput,
    BatchDataOutput,
)
from get_a_grip.model_training.utils.nerf_grasp_evaluator_dataset import (
    NerfGraspEvalDataset,
)
from get_a_grip.model_training.utils.plot_utils import (
    plot_mesh_and_high_density_points,
    plot_mesh_and_query_points,
)
from get_a_grip.model_training.utils.scheduler import get_scheduler
from get_a_grip.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.utils.seed import set_seed
from wandb.util import generate_id
from wandb.viz import CustomChart

tqdm = partial(std_tqdm, dynamic_ncols=True)


# Make atol and rtol larger than default to avoid errors due to floating point precision.
# Otherwise we get errors about invalid rotation matrices
PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4
NUM_XYZ = 3


class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    EVAL_TRAIN = auto()


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def setup_checkpoint_workspace(
    cfg: NerfEvaluatorModelConfig,
) -> str:
    # If input_dir != output_dir, then we create a new output_dir and wandb_run_id
    if cfg.checkpoint_workspace.input_dir != cfg.checkpoint_workspace.output_dir:
        cfg.checkpoint_workspace.output_dir.mkdir(parents=True, exist_ok=False)
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
    return wandb_run_id


########## Dataset and Dataloader Start ##########
def create_grid_dataset(
    input_hdf5_filepath: pathlib.Path, cfg: NerfEvaluatorModelConfig
) -> NerfGraspEvalDataset:
    assert cfg.nerf_grasp_dataset_config.fingertip_config is not None
    return NerfGraspEvalDataset(
        input_hdf5_filepath=input_hdf5_filepath,
        fingertip_config=cfg.nerf_grasp_dataset_config.fingertip_config,
        max_num_data_points=cfg.data.max_num_data_points,
    )


def create_train_val_test_dataset(
    cfg: NerfEvaluatorModelConfig,
) -> Tuple[
    Union[Subset, NerfGraspEvalDataset],
    Union[Subset, NerfGraspEvalDataset],
    Union[Subset, NerfGraspEvalDataset],
]:
    if cfg.create_val_test_from_train:
        print(
            f"Creating val and test datasets from train dataset: {cfg.actual_train_dataset_path}"
        )
        full_dataset = create_grid_dataset(
            input_hdf5_filepath=cfg.actual_train_dataset_path, cfg=cfg
        )

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [cfg.data.frac_train, cfg.data.frac_val, cfg.data.frac_test],
            generator=torch.Generator().manual_seed(cfg.random_seed),
        )
        assert_equals(
            len(set.intersection(set(train_dataset.indices), set(val_dataset.indices))),
            0,
        )
        assert_equals(
            len(
                set.intersection(set(train_dataset.indices), set(test_dataset.indices))
            ),
            0,
        )
        assert_equals(
            len(set.intersection(set(val_dataset.indices), set(test_dataset.indices))),
            0,
        )
    else:
        print(
            f"Using actual val and test datasets: cfg.actual_train_dataset_filepath = {cfg.actual_train_dataset_path}, cfg.actual_val_dataset_filepath = {cfg.actual_val_dataset_path}, cfg.actual_test_dataset_filepath = {cfg.actual_test_dataset_path}"
        )
        train_dataset = create_grid_dataset(
            input_hdf5_filepath=cfg.actual_train_dataset_path, cfg=cfg
        )
        val_dataset = create_grid_dataset(
            input_hdf5_filepath=cfg.actual_val_dataset_path, cfg=cfg
        )
        test_dataset = create_grid_dataset(
            input_hdf5_filepath=cfg.actual_test_dataset_path, cfg=cfg
        )
    return train_dataset, val_dataset, test_dataset


def create_train_val_test_dataloader(
    cfg: NerfEvaluatorModelConfig,
    train_dataset: Union[Subset, NerfGraspEvalDataset],
    val_dataset: Union[Subset, NerfGraspEvalDataset],
    test_dataset: Union[Subset, NerfGraspEvalDataset],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_collate_fn = partial(
        custom_collate_fn,
        fingertip_config=cfg.nerf_grasp_dataset_config.fingertip_config,
        use_random_rotations=cfg.data.use_random_rotations,
        debug_shuffle_labels=cfg.data.debug_shuffle_labels,
    )
    val_test_collate_fn = partial(
        custom_collate_fn,
        use_random_rotations=False,
        fingertip_config=cfg.nerf_grasp_dataset_config.fingertip_config,
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
    return train_loader, val_loader, test_loader


def custom_collate_fn(
    batch,
    fingertip_config: EvenlySpacedFingertipConfig,
    use_random_rotations: bool = True,
    debug_shuffle_labels: bool = False,
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


########## Dataset and Dataloader End ##########


########## Plot Utils Start ##########
def nerf_densities_plot_example(
    batch_data: BatchData,
    fingertip_config: EvenlySpacedFingertipConfig,
    idx_to_visualize: int = 0,
    augmented: bool = False,
) -> go.Figure:
    NUM_FINGERS = fingertip_config.n_fingers
    NUM_PTS_X = fingertip_config.num_pts_x
    NUM_PTS_Y = fingertip_config.num_pts_y
    NUM_PTS_Z = fingertip_config.num_pts_z

    if augmented:
        query_points = batch_data.input.augmented_coords[idx_to_visualize]
        additional_mesh_transform = (
            batch_data.input.random_rotate_transform.matrix()
            .cpu()
            .numpy()[idx_to_visualize]
            if batch_data.input.random_rotate_transform is not None
            else None
        )
    else:
        query_points = batch_data.input.coords[idx_to_visualize]
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
    MESHDATA_ROOT = get_data_folder() / "meshdata"
    mesh_path = MESHDATA_ROOT / object_code / "coacd" / "decomposed.obj"

    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))
    if additional_mesh_transform is not None:
        mesh.apply_transform(additional_mesh_transform)

    # Get query points from grasp_transforms
    assert_equals(
        query_points.shape,
        (
            NUM_FINGERS,
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        ),
    )
    query_points = query_points.permute((0, 2, 3, 4, 1))
    assert_equals(
        query_points.shape,
        (
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
            NUM_XYZ,
        ),
    )
    query_points = torch.stack(
        [
            query_points[finger_idx].reshape(-1, NUM_XYZ)
            for finger_idx in range(NUM_FINGERS)
        ],
        dim=0,
    )
    query_point_colors = torch.stack(
        [colors[finger_idx].reshape(-1) for finger_idx in range(NUM_FINGERS)],
        dim=0,
    )
    fig = plot_mesh_and_query_points(
        mesh=mesh,
        all_query_points=query_points.detach().cpu().numpy(),
        all_query_points_colors=query_point_colors.detach().cpu().numpy(),
        num_fingers=NUM_FINGERS,
    )
    # Set title to label
    fig.update_layout(
        title_text=f"NERF DENSITIES: y_pick = {y_pick}, y_coll = {y_coll}, y_PGS = {y_PGS} ({object_code}) (augmented = {augmented})"
    )
    return fig


def nerf_densities_global_plot_example(
    batch_data: BatchData,
    idx_to_visualize: int = 0,
    augmented: bool = False,
) -> go.Figure:
    NUM_PTS_X = NERF_DENSITIES_GLOBAL_NUM_X
    NUM_PTS_Y = NERF_DENSITIES_GLOBAL_NUM_Y
    NUM_PTS_Z = NERF_DENSITIES_GLOBAL_NUM_Z

    if augmented:
        query_points = batch_data.input.augmented_coords_global[idx_to_visualize]
        additional_mesh_transform = (
            batch_data.input.random_rotate_transform.matrix()
            .cpu()
            .numpy()[idx_to_visualize]
            if batch_data.input.random_rotate_transform is not None
            else None
        )
    else:
        query_points = batch_data.input.coords_global[idx_to_visualize]
        additional_mesh_transform = None

    # Extract data
    colors = batch_data.input.nerf_alphas_global[idx_to_visualize]
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

    assert_equals(colors.shape, (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z))

    nerf_config = pathlib.Path(batch_data.nerf_config[idx_to_visualize])
    object_code_and_scale_str = nerf_config.parents[2].name
    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)

    # Path to meshes
    MESHDATA_ROOT = get_data_folder() / "meshdata"
    mesh_path = MESHDATA_ROOT / object_code / "coacd" / "decomposed.obj"

    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))
    if additional_mesh_transform is not None:
        mesh.apply_transform(additional_mesh_transform)

    # Get query points from grasp_transforms
    assert_equals(
        query_points.shape,
        (
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        ),
    )
    query_points = query_points.permute((1, 2, 3, 0))
    assert_equals(
        query_points.shape,
        (
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
            NUM_XYZ,
        ),
    )
    query_points = query_points.reshape(-1, NUM_XYZ)
    query_point_colors = colors.reshape(-1)
    fig = plot_mesh_and_high_density_points(
        mesh=mesh,
        all_query_points=query_points.detach().cpu().numpy(),
        all_query_points_colors=query_point_colors.detach().cpu().numpy(),
        density_threshold=0.01,
    )
    # Set title to label
    fig.update_layout(
        title_text=f"NERF DENSITIES GLOBAL: y_pick = {y_pick}, y_coll = {y_coll}, y_PGS = {y_PGS} ({object_code}) (augmented = {augmented})"
    )
    return fig


def debug_plot(
    batch_data: BatchData,
    fingertip_config: EvenlySpacedFingertipConfig,
    idx_to_visualize: int = 0,
) -> None:
    fig = nerf_densities_plot_example(
        batch_data=batch_data,
        fingertip_config=fingertip_config,
        idx_to_visualize=idx_to_visualize,
        augmented=False,
    )
    fig.show()
    fig2 = nerf_densities_global_plot_example(
        batch_data=batch_data,
        idx_to_visualize=idx_to_visualize,
        augmented=False,
    )
    fig2.show()

    fig = nerf_densities_plot_example(
        batch_data=batch_data,
        fingertip_config=fingertip_config,
        idx_to_visualize=idx_to_visualize,
        augmented=True,
    )
    fig.show()
    fig2 = nerf_densities_global_plot_example(
        batch_data=batch_data,
        idx_to_visualize=idx_to_visualize,
        augmented=True,
    )
    fig2.show()


########## Plot Utils End ##########


########## Train Setup Start ##########
def create_and_optionally_load(
    cfg: NerfEvaluatorModelConfig,
    device: torch.device,
    num_training_steps: int,
) -> Tuple[
    NerfEvaluatorModel,
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler._LRScheduler],
    int,
]:
    nerf_evaluator_model: NerfEvaluatorModel = cfg.model_config.create_model(
        fingertip_config=cfg.nerf_grasp_dataset_config.fingertip_config,
        n_tasks=cfg.task_type.n_tasks,
    ).to(device)
    start_epoch = 0
    optimizer = torch.optim.AdamW(
        params=nerf_evaluator_model.parameters(),
        lr=cfg.training.lr,
        betas=cfg.training.betas,
        weight_decay=cfg.training.weight_decay,
    )
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_scheduler_num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=start_epoch - 1,
    )

    # Load Checkpoint
    if cfg.checkpoint_workspace.input_dir is not None:
        assert cfg.checkpoint_workspace.input_dir.exists(), f"checkpoint_workspace.input_dir does not exist at {cfg.checkpoint_workspace.input_dir}"
        print(f"Loading checkpoint ({cfg.checkpoint_workspace.input_dir})...")
        latest_checkpoint_path = cfg.checkpoint_workspace.latest_input_checkpoint_path
        assert (
            latest_checkpoint_path is not None and latest_checkpoint_path.exists()
        ), f"latest_checkpoint_path does not exist at {latest_checkpoint_path}"

        checkpoint = torch.load(latest_checkpoint_path)
        nerf_evaluator_model.load_state_dict(checkpoint["nerf_evaluator_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("Done loading checkpoint")
    return (
        nerf_evaluator_model,
        optimizer,
        lr_scheduler,
        start_epoch,
    )


########## Train Setup End ##########


########## Loss Start ##########
def setup_loss_fns(
    cfg: NerfEvaluatorModelConfig,
) -> List[nn.Module]:
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
    elif cfg.training.loss_fn == "cross_entropy":
        print("=" * 80)
        print("Using CE loss")
        print("=" * 80 + "\n")
        y_pick_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=cfg.training.label_smoothing,
            reduction="none",
        )
        y_coll_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=cfg.training.label_smoothing,
            reduction="none",
        )
        y_PGS_loss_fn = nn.CrossEntropyLoss(
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
    return loss_fns


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


########## Loss End ##########


########## Train Loop Start ##########
def save_checkpoint(
    checkpoint_output_dir: pathlib.Path,
    epoch: int,
    nerf_evaluator_model: NerfEvaluatorModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    checkpoint_filepath = checkpoint_output_dir / f"ckpt_{epoch:04}.pth"
    print(f"Saving checkpoint to {checkpoint_filepath}")
    torch.save(
        {
            "epoch": epoch,
            "nerf_evaluator_model": nerf_evaluator_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


def save_checkpoint_batch(
    checkpoint_output_dir: pathlib.Path,
    epoch: int,
    batch_idx: int,
    nerf_evaluator_model: NerfEvaluatorModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    # Save more frequently to be safe (epochs may take many hours each)
    checkpoint_filepath = checkpoint_output_dir / f"ckpt_batch_{batch_idx:04}.pth"
    print(f"Saving checkpoint to {checkpoint_filepath}")
    torch.save(
        {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "nerf_evaluator_model": nerf_evaluator_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


# Global variables to keep track for saving checkpoints
GLOBAL_EPOCH = 0
GLOBAL_BATCH_IDX = 0


def _iterate_through_dataloader(
    loop_timer: LoopTimer,
    phase: Phase,
    dataloader: DataLoader,
    nerf_evaluator_model: NerfEvaluatorModel,
    device: torch.device,
    loss_fns: List[nn.Module],
    task_type: TaskType,
    checkpoint_output_dir: Optional[pathlib.Path] = None,
    training_cfg: Optional[TrainingConfig] = None,
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
        nerf_evaluator_model.train()
        assert (
            training_cfg is not None
            and optimizer is not None
            and checkpoint_output_dir is not None
        )
    else:
        nerf_evaluator_model.eval()
        assert training_cfg is None and optimizer is None

    assert_equals(len(loss_fns), nerf_evaluator_model.n_tasks)
    assert_equals(len(task_type.task_names), nerf_evaluator_model.n_tasks)

    with torch.set_grad_enabled(phase == Phase.TRAIN):
        dataload_section_timer = loop_timer.add_section_timer("Data").start()
        for batch_idx, batch_data in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            dataload_section_timer.stop()

            batch_idx = int(batch_idx)

            if batch_idx % 1000 == 0 and phase == Phase.TRAIN:
                assert optimizer is not None
                assert checkpoint_output_dir is not None
                global GLOBAL_BATCH_IDX
                save_checkpoint_batch(
                    checkpoint_output_dir=checkpoint_output_dir,
                    batch_idx=GLOBAL_BATCH_IDX,
                    epoch=GLOBAL_EPOCH,
                    nerf_evaluator_model=nerf_evaluator_model,
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
                all_logits = nerf_evaluator_model.get_all_logits(batch_data.input)
                assert_equals(
                    all_logits.shape,
                    (
                        batch_data.batch_size,
                        nerf_evaluator_model.n_tasks,
                        nerf_evaluator_model.n_classes,
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

                assert_equals(len(task_targets), nerf_evaluator_model.n_tasks)

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
                    task_losses.shape,
                    (nerf_evaluator_model.n_tasks, batch_data.batch_size),
                )
                total_losses = torch.mean(task_losses, dim=0)
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
                            nerf_evaluator_model.parameters(),
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


def _create_wandb_scatter_plot(
    ground_truths: List[float],
    predictions: List[float],
    title: str,
) -> CustomChart:
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


def create_log_dict(
    phase: Phase,
    loop_timer: LoopTimer,
    task_type: TaskType,
    losses_dict: Dict[str, List[float]],
    predictions_dict: Dict[str, List[float]],
    ground_truths_dict: Dict[str, List[float]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    plot_cfg: Optional[PlotConfig] = None,
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
        PLOT_SCATTER = plot_cfg is not None and plot_cfg.scatter_predicted_vs_actual
        if PLOT_SCATTER:
            # Make scatter plot of predicted vs ground truth
            for task_name in task_type.task_names:
                predictions = predictions_dict[task_name]
                ground_truths = ground_truths_dict[task_name]
                temp_log_dict[f"{task_name}_scatter"] = _create_wandb_scatter_plot(
                    ground_truths=ground_truths,
                    predictions=predictions,
                    title=f"{phase.name.title()} {task_name} Scatter Plot",
                )

    with loop_timer.add_section_timer("Histogram"):
        PLOT_HISTOGRAM = plot_cfg is not None and plot_cfg.histogram_predictions
        if PLOT_HISTOGRAM:
            # For each task, make multiple histograms of prediction (one per label)
            for task_name in task_type.task_names:
                temp_log_dict[f"{task_name}_histogram"] = _create_wandb_histogram_plot(
                    ground_truths=ground_truths_dict[task_name],
                    predictions=predictions_dict[task_name],
                    title=f"{phase.name.title()} {task_name} Histogram",
                )

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
        PLOT_CONFUSION_MATRIX = plot_cfg is not None and plot_cfg.confusion_matrices
        if PLOT_CONFUSION_MATRIX:
            for task_name in task_type.task_names:
                predictions = (
                    np.array(predictions_dict[task_name]).round().astype(int).tolist()
                )
                ground_truths = (
                    np.array(ground_truths_dict[task_name]).round().astype(int).tolist()
                )
                temp_log_dict[f"{task_name}_confusion_matrix"] = (
                    wandb.plot.confusion_matrix(
                        preds=predictions,
                        y_true=ground_truths,
                        class_names=["failure", "success"],
                        title=f"{phase.name.title()} {task_name} Confusion Matrix",
                    )
                )

    log_dict = {}
    for key, value in temp_log_dict.items():
        log_dict[f"{phase.name.lower()}_{key}"] = value
    return log_dict


def iterate_through_dataloader(
    phase: Phase,
    dataloader: DataLoader,
    nerf_evaluator_model: NerfEvaluatorModel,
    device: torch.device,
    loss_fns: List[nn.Module],
    task_type: TaskType,
    checkpoint_output_dir: Optional[pathlib.Path] = None,
    training_cfg: Optional[TrainingConfig] = None,
    plot_cfg: Optional[PlotConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    assert_equals(len(loss_fns), nerf_evaluator_model.n_tasks)
    assert_equals(len(task_type.task_names), nerf_evaluator_model.n_tasks)

    loop_timer = LoopTimer()

    # Iterate through dataloader and get logged results
    losses_dict, predictions_dict, ground_truths_dict = _iterate_through_dataloader(
        loop_timer=loop_timer,
        phase=phase,
        dataloader=dataloader,
        nerf_evaluator_model=nerf_evaluator_model,
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
        plot_cfg=plot_cfg,
    )

    loop_timer.pretty_print_section_times()
    print()
    print()

    return log_dict


def run_training_loop(
    training_cfg: TrainingConfig,
    plot_cfg: PlotConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nerf_evaluator_model: NerfEvaluatorModel,
    device: torch.device,
    loss_fns: List[nn.Module],
    optimizer: torch.optim.Optimizer,
    checkpoint_output_dir: pathlib.Path,
    task_type: TaskType,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    start_epoch: int = 0,
) -> None:
    global GLOBAL_EPOCH
    GLOBAL_EPOCH = start_epoch

    wandb.watch(nerf_evaluator_model, log="gradients", log_freq=100)

    training_loop_base_description = "Training Loop"
    for epoch in (
        pbar := tqdm(
            range(start_epoch, training_cfg.n_epochs),
            desc=training_loop_base_description,
        )
    ):
        epoch = int(epoch)
        GLOBAL_EPOCH = epoch

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
                nerf_evaluator_model=nerf_evaluator_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        save_checkpoint_time_taken = time.time() - start_save_checkpoint_time

        # Train
        start_train_time = time.time()
        train_log_dict = iterate_through_dataloader(
            phase=Phase.TRAIN,
            dataloader=train_loader,
            nerf_evaluator_model=nerf_evaluator_model,
            device=device,
            loss_fns=loss_fns,
            task_type=task_type,
            checkpoint_output_dir=checkpoint_output_dir,
            training_cfg=training_cfg,
            plot_cfg=plot_cfg,
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
            nerf_evaluator_model.eval()
            val_log_dict = iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                nerf_evaluator_model=nerf_evaluator_model,
                device=device,
                loss_fns=loss_fns,
                task_type=task_type,
                checkpoint_output_dir=checkpoint_output_dir,
                plot_cfg=plot_cfg,
            )
            wandb_log_dict.update(val_log_dict)
        val_time_taken = time.time() - start_val_time

        nerf_evaluator_model.train()

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


########## Train Loop End ##########


def main() -> None:
    # Load Config
    cfg = tyro.cli(tyro.conf.FlagConversionOff[NerfEvaluatorModelConfig])
    print("=" * 80)
    print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
    print("=" * 80 + "\n")

    set_seed(cfg.random_seed)

    # Setup Checkpoint Workspace and Maybe Resume Previous Run
    wandb_run_id = setup_checkpoint_workspace(
        cfg=cfg,
    )

    # Save out config to file
    cfg_path = pathlib.Path(cfg.checkpoint_workspace.output_dir) / "config.yaml"
    if not cfg_path.exists():
        cfg_yaml = tyro.extras.to_yaml(cfg)
        with open(cfg_path, "w") as f:
            f.write(cfg_yaml)

    # Setup Wandb Logging
    wandb.init(
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

    # Create Dataset and Dataloader
    train_dataset, val_dataset, test_dataset = create_train_val_test_dataset(cfg=cfg)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader, val_loader, test_loader = create_train_val_test_dataloader(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Debug prints shapes
    if cfg.data.use_random_rotations:
        print("Using random rotations for training")
    else:
        print("Not using random rotations for training")
    if cfg.data.debug_shuffle_labels:
        print(
            "WARNING: Shuffle labels is turned on! Random labels are being passed. Press 'c' to continue"
        )
    EXAMPLE_BATCH_DATA: BatchData = next(iter(train_loader))
    EXAMPLE_BATCH_DATA.print_shapes()
    print(f"y_pick = {EXAMPLE_BATCH_DATA.output.y_pick[:5]}")
    print(f"y_coll = {EXAMPLE_BATCH_DATA.output.y_coll[:5]}")
    print(f"y_PGS = {EXAMPLE_BATCH_DATA.output.y_PGS[:5]}")

    # Debug plot
    PLOT_EXAMPLE_BATCH_DATA = cfg.plot.batch_data
    if PLOT_EXAMPLE_BATCH_DATA:
        debug_plot(
            batch_data=EXAMPLE_BATCH_DATA,
            fingertip_config=cfg.nerf_grasp_dataset_config.fingertip_config,
            idx_to_visualize=2,
        )

    # Setup for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf_evaluator_model, optimizer, lr_scheduler, start_epoch = (
        create_and_optionally_load(
            cfg=cfg,
            device=device,
            num_training_steps=len(train_loader) * cfg.training.n_epochs,
        )
    )
    print(f"nerf_evaluator_model = {nerf_evaluator_model}")
    print(f"optimizer = {optimizer}")
    print(f"lr_scheduler = {lr_scheduler}")

    # Setup loss
    loss_fns = setup_loss_fns(
        cfg=cfg,
    )

    # Train
    run_training_loop(
        training_cfg=cfg.training,
        plot_cfg=cfg.plot,
        train_loader=train_loader,
        val_loader=val_loader,
        nerf_evaluator_model=nerf_evaluator_model,
        device=device,
        loss_fns=loss_fns,
        optimizer=optimizer,
        checkpoint_output_dir=cfg.checkpoint_workspace.output_dir,
        task_type=cfg.task_type,
        lr_scheduler=lr_scheduler,
        start_epoch=start_epoch,
    )

    # Test
    nerf_evaluator_model.eval()
    wandb_log_dict = {}
    print(f"Running test metrics on epoch {cfg.training.n_epochs}")
    wandb_log_dict["epoch"] = cfg.training.n_epochs
    test_log_dict = iterate_through_dataloader(
        phase=Phase.TEST,
        dataloader=test_loader,
        nerf_evaluator_model=nerf_evaluator_model,
        device=device,
        loss_fns=loss_fns,
        task_type=cfg.task_type,
        checkpoint_output_dir=cfg.checkpoint_workspace.output_dir,
        plot_cfg=cfg.plot,
    )
    wandb_log_dict.update(test_log_dict)
    wandb.log(wandb_log_dict)

    # Save Model
    save_checkpoint(
        checkpoint_output_dir=cfg.checkpoint_workspace.output_dir,
        epoch=cfg.training.n_epochs,
        nerf_evaluator_model=nerf_evaluator_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
