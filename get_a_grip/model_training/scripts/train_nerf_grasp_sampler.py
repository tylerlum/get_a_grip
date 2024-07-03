import pathlib
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import tyro

from get_a_grip.model_training.config.diffusion_config import (
    DataConfig,
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.models.dex_sampler import (
    NerfSampler,
)
from get_a_grip.model_training.utils.diffusion import (
    train,
    train_multigpu,
)
from get_a_grip.model_training.utils.nerf_grasp_dataset import (
    NerfGraspSampleDataset,
)


@dataclass
class TrainBpsGraspSamplerConfig:
    train_hdf5_path: pathlib.Path = pathlib.Path("TODO")
    val_hdf5_path: pathlib.Path = pathlib.Path("TODO")
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


def main() -> None:
    config = tyro.cli(
        TrainBpsGraspSamplerConfig,
        default=TrainBpsGraspSamplerConfig(
            diffusion=DiffusionConfig(
                data=DataConfig(
                    num_workers=4,
                ),
                training=TrainingConfig(
                    n_epochs=20000,
                    batch_size=256,
                ),
                use_nerf_sampler=True,
                multigpu=True,
                wandb_log=True,
            ),
        ),
    )

    train_dataset = NerfGraspSampleDataset(
        input_hdf5_filepath=config.train_hdf5_path,
        get_all_labels=False,
    )
    val_dataset = NerfGraspSampleDataset(
        input_hdf5_filepath=config.val_hdf5_path,
        get_all_labels=False,
    )

    model = NerfSampler(
        global_grid_shape=(
            4,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        ),
        grasp_dim=config.diffusion.data.grasp_dim,
        d_model=128,
        virtual_seq_len=4,
        conv_channels=(32, 64, 128),
    )

    if config.diffusion.multigpu:
        mp.spawn(
            train_multigpu,
            args=(
                config.diffusion,
                train_dataset,
                val_dataset,
                model,
            ),
            nprocs=torch.cuda.device_count(),
            join=True,
        )
    else:
        train(
            config=config.diffusion,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            rank=0,
        )
