import datetime
import pathlib
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import tyro

from get_a_grip import get_data_folder
from get_a_grip.model_training.config.diffusion_config import (
    DataConfig,
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.models.bps_sampler import (
    BpsSampler,
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
    train_dataset_path: pathlib.Path = (
        get_data_folder() / "large/bps_grasp_dataset/train_dataset.h5"
    )
    val_dataset_path: pathlib.Path = (
        get_data_folder() / "large/bps_grasp_dataset/val_dataset.h5"
    )
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
                    log_path=(
                        get_data_folder()
                        / (
                            f"logs/bps_grasp_sampler/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                        )
                    ),
                ),
                multigpu=True,
                wandb_log=True,
            ),
        ),
    )

    train_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=config.train_dataset_path,
    )
    val_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=config.val_dataset_path,
    )
    model = BpsSampler(
        n_pts=config.diffusion.data.n_pts,
        grasp_dim=config.diffusion.data.grasp_dim,
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


if __name__ == "__main__":
    main()
