# %%
"""
Visualization script for nerf_evaluator_model
Useful to interactively understand the data and model performance
Percent script can be run like a Jupyter notebook or as a script
"""

# %%
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import tyro
from clean_loop_timer import LoopTimer
from plotly.subplots import make_subplots
from tqdm import tqdm as std_tqdm

import wandb
from get_a_grip import get_repo_folder
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)
from get_a_grip.model_training.scripts.train_nerf_evaluator_model import (
    Phase,
    _iterate_through_dataloader,
    create_and_optionally_load,
    create_log_dict,
    create_train_val_test_dataloader,
    create_train_val_test_dataset,
    debug_plot,
    setup_loss_fns,
)
from get_a_grip.model_training.utils.nerf_evaluator_model_batch_data import (
    BatchData,
)

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
    arguments = [
        "--train-dataset-path",
        "data/SMALL_DATASET/nerf_grasp_dataset/train_dataset.h5",
        "--task-type",
        "Y_PICK_AND_Y_COLL_AND_Y_PGS",
        "--dataloader.batch-size",
        "128",
        "--wandb.name",
        "probe",
        # Load checkpoint
        "--checkpoint-workspace.input_leaf_dir_name",
        "MY_NERF_EXPERIMENT_NAME_2024-07-05_01-13-06-844447",
        "--training.loss_fn",
        "l2",
        "--training.save_checkpoint_freq",
        "1",
        "model-config:cnn-xyz-global-cnn-model-config",
    ]
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")

cfg = tyro.cli(tyro.conf.FlagConversionOff[NerfEvaluatorModelConfig], args=arguments)
print("=" * 80)
print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
print("=" * 80 + "\n")

# %%
wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    name=cfg.wandb.name,
    group=cfg.wandb.group,
    job_type=cfg.wandb.job_type,
    config=asdict(cfg),
    resume=cfg.wandb.resume,
    reinit=True,
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# %%
EXAMPLE_BATCH_DATA: BatchData = next(iter(train_loader))
EXAMPLE_BATCH_DATA.print_shapes()
print(f"y_pick = {EXAMPLE_BATCH_DATA.output.y_pick[:5]}")
print(f"y_coll = {EXAMPLE_BATCH_DATA.output.y_coll[:5]}")
print(f"y_PGS = {EXAMPLE_BATCH_DATA.output.y_PGS[:5]}")

# %%
# Debug plot
debug_plot(
    batch_data=EXAMPLE_BATCH_DATA,
    fingertip_config=cfg.nerf_grasp_dataset_config.fingertip_config,
    idx_to_visualize=2,
)


# %%
# Create model, optimizer, lr_scheduler
nerf_evaluator_model, optimizer, lr_scheduler, start_epoch = create_and_optionally_load(
    cfg=cfg,
    device=device,
    num_training_steps=len(train_loader) * cfg.training.n_epochs,
)
print(f"nerf_evaluator_model = {nerf_evaluator_model}")
print(f"optimizer = {optimizer}")
print(f"lr_scheduler = {lr_scheduler}")

# %%
# Loss function
loss_fns = setup_loss_fns(
    cfg=cfg,
)


# %%
DEBUG_phase = Phase.EVAL_TRAIN
if DEBUG_phase == Phase.EVAL_TRAIN:
    DEBUG_loader = train_loader
elif DEBUG_phase == Phase.VAL:
    DEBUG_loader = val_loader
elif DEBUG_phase == Phase.TEST:
    DEBUG_loader = test_loader
else:
    raise ValueError(f"Unknown phase: {DEBUG_phase}")


loop_timer = LoopTimer()
(
    DEBUG_losses_dict,
    DEBUG_predictions_dict,
    DEBUG_ground_truths_dict,
) = _iterate_through_dataloader(
    loop_timer=loop_timer,
    phase=DEBUG_phase,
    dataloader=DEBUG_loader,
    nerf_evaluator_model=nerf_evaluator_model,
    device=device,
    loss_fns=loss_fns,
    task_type=cfg.task_type,
    max_num_batches=None,
)

# %%
DEBUG_losses_dict.keys()

# %%
DEBUG_losses_dict["y_pick_loss"][:10]

# %%
DEBUG_predictions_dict["y_pick"][:10]

# %%
DEBUG_ground_truths_dict["y_pick"][:10]

# %%
DEBUG_predictions_dict

# %%

# Small circles
gaussian_noise = np.random.normal(0, 0.01, len(DEBUG_ground_truths_dict["y_pick"]))
plt.scatter(
    DEBUG_ground_truths_dict["y_pick"] + gaussian_noise,
    DEBUG_predictions_dict["y_pick"],
    s=0.1,
)
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.title("y_pick Scatter Plot")
plt.show()

# %%
np.unique(DEBUG_ground_truths_dict["y_pick"], return_counts=True)

# %%
unique_labels = np.unique(DEBUG_ground_truths_dict["y_pick"])
fig, axes = plt.subplots(len(unique_labels), 1, figsize=(10, 10))
axes = axes.flatten()

unique_label_to_preds = {}
for i, unique_val in enumerate(unique_labels):
    preds = np.array(DEBUG_predictions_dict["y_pick"])
    idxs = np.array(DEBUG_ground_truths_dict["y_pick"]) == unique_val
    unique_label_to_preds[unique_val] = preds[idxs]

for i, (unique_label, preds) in enumerate(sorted(unique_label_to_preds.items())):
    # axes[i].hist(preds, bins=50, alpha=0.7, color="blue", log=True)
    axes[i].hist(preds, bins=50, alpha=0.7, color="blue")
    axes[i].set_title(f"Ground Truth: {unique_label}")
    axes[i].set_xlim(0, 1)

# Matching ylims
max_y_val = max(ax.get_ylim()[1] for ax in axes)
for i in range(len(axes)):
    axes[i].set_ylim(0, max_y_val)

fig.tight_layout()


# %%
DEBUG_log_dict = create_log_dict(
    loop_timer=loop_timer,
    phase=DEBUG_phase,
    task_type=cfg.task_type,
    losses_dict=DEBUG_losses_dict,
    predictions_dict=DEBUG_predictions_dict,
    ground_truths_dict=DEBUG_ground_truths_dict,
    optimizer=optimizer,
)

# %%
DEBUG_log_dict[f"{DEBUG_phase.name.lower()}_loss"]

# %%
DEBUG_log_dict_modified = {f"{k}_v2": v for k, v in DEBUG_log_dict.items()}

# %%
wandb.log(DEBUG_log_dict_modified)


# %%
loop_timer = LoopTimer()
(
    train_losses_dict,
    train_predictions_dict,
    train_ground_truths_dict,
) = _iterate_through_dataloader(
    loop_timer=loop_timer,
    phase=Phase.EVAL_TRAIN,  # Not TRAIN because we don't want to update the model
    dataloader=train_loader,
    nerf_evaluator_model=nerf_evaluator_model,
    device=device,
    loss_fns=loss_fns,
    task_type=cfg.task_type,
    max_num_batches=10,
)

# %%
loop_timer = LoopTimer()
(
    val_losses_dict,
    val_predictions_dict,
    val_ground_truths_dict,
) = _iterate_through_dataloader(
    loop_timer=loop_timer,
    phase=Phase.VAL,
    dataloader=val_loader,
    nerf_evaluator_model=nerf_evaluator_model,
    device=device,
    loss_fns=loss_fns,
    task_type=cfg.task_type,
    max_num_batches=10,
)


# %%
loss_names = [
    "y_pick_loss",
    "y_coll_loss",
]

fig = make_subplots(rows=len(loss_names), cols=1, subplot_titles=loss_names)
for i, loss_name in enumerate(loss_names):
    fig.add_trace(
        go.Scatter(y=val_losses_dict[loss_name], name=loss_name, mode="markers"),
        row=i + 1,
        col=1,
    )
fig.show()


# %%
def plot_distribution(data: np.ndarray, name: str) -> None:
    # Calculating statistics
    import scipy.stats as stats

    data = np.array(data)
    mean = np.mean(data)
    max_value = np.max(data)
    min_value = np.min(data)
    data_range = np.ptp(data)  # Range as max - min
    std_dev = np.std(data)
    median = np.median(data)
    mode = stats.mode(data).mode[0]
    iqr = stats.iqr(data)  # Interquartile range
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)

    import matplotlib.pyplot as plt

    # Create histogram
    plt.hist(data, bins=50, alpha=0.7, color="blue", log=True)

    # Printing results
    print(
        f"Mean: {mean}, Max: {max_value}, Min: {min_value}, Range: {data_range}, Standard Deviation: {std_dev}"
    )
    print(
        f"Median: {median}, Mode: {mode}, IQR: {iqr}, 25th Percentile: {percentile_25}, 75th Percentile: {percentile_75}"
    )

    # Add lines for mean, median, and mode
    plt.axvline(
        mean, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.4f}"
    )
    plt.axvline(
        median,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {median:.4f}",
    )
    plt.axvline(
        mode, color="yellow", linestyle="dashed", linewidth=2, label=f"Mode: {mode:.4f}"
    )

    # Add lines for percentiles
    plt.axvline(
        percentile_25,
        color="orange",
        linestyle="dotted",
        linewidth=2,
        label=f"25th percentile: {percentile_25:.4f}",
    )
    plt.axvline(
        percentile_75,
        color="purple",
        linestyle="dotted",
        linewidth=2,
        label=f"75th percentile: {percentile_75:.4f}",
    )

    # Add standard deviation
    plt.axvline(
        mean - std_dev,
        color="cyan",
        linestyle="dashdot",
        linewidth=2,
        label=f"Std Dev: {std_dev:.4f}",
    )
    plt.axvline(mean + std_dev, color="cyan", linestyle="dashdot", linewidth=2)

    # Add legend
    plt.legend()
    plt.title(f"{name} histogram")

    # Show plot
    plt.show()


plot_distribution(
    data=val_losses_dict["y_coll_loss"],
    name="y_coll_loss",
)

# %%
plot_distribution(data=val_losses_dict["y_pick_loss"], name="y_pick_loss")

# %%
