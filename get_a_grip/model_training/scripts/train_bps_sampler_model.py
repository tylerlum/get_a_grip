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
    WandbConfig,
)
from get_a_grip.model_training.models.bps_sampler_model import (
    BpsSamplerModel,
)
from get_a_grip.model_training.utils.bps_grasp_dataset import (
    BpsGraspSampleDataset,
)
from get_a_grip.model_training.utils.diffusion import (
    train,
    train_multigpu,
)


@dataclass
class TrainBpsSamplerModelConfig:
    train_dataset_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/bps_grasp_dataset/train_dataset.h5"
    )
    val_dataset_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/bps_grasp_dataset/val_dataset.h5"
    )
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


def main() -> None:
    config = tyro.cli(
        tyro.conf.FlagConversionOff[TrainBpsSamplerModelConfig],
        default=TrainBpsSamplerModelConfig(
            diffusion=DiffusionConfig(
                data=DataConfig(
                    num_workers=4,
                ),
                training=TrainingConfig(
                    n_epochs=20000,
                    batch_size=16384,
                    output_dir=(
                        get_data_folder()
                        / (
                            f"models/NEW/bps_sampler_model/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                        )
                    ),
                ),
                multigpu=True,
                wandb=WandbConfig(
                    log=True,
                    project="bps_sampler_model",
                ),
            ),
        ),
    )

    train_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=config.train_dataset_path,
    )
    val_dataset = BpsGraspSampleDataset(
        input_hdf5_filepath=config.val_dataset_path,
    )
    model = BpsSamplerModel(
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
