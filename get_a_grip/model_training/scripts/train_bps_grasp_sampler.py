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
from get_a_grip.model_training.models.dex_sampler import (
    DexSampler,
)
from get_a_grip.model_training.utils.bps_grasp_dataset import (
    BpsGraspSampleDataset,
)
from get_a_grip.model_training.utils.diffusion import (
    train,
    train_multigpu,
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
                    batch_size=16384,
                ),
                use_nerf_sampler=False,
                multigpu=True,
                wandb_log=True,
            ),
        ),
    )

    train_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=config.train_hdf5_path,
        get_all_labels=False,
    )
    val_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=config.val_hdf5_path,
        get_all_labels=False,
    )
    model = DexSampler(
        n_pts=config.diffusion.data.n_pts,
        grasp_dim=config.diffusion.data.grasp_dim,
        d_model=128,
        virtual_seq_len=4,
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
