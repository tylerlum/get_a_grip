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
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
)
from get_a_grip.model_training.models.nerf_sampler import (
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
    train_dataset_path: pathlib.Path = (
        get_data_folder() / "large/nerf_grasp_dataset/train_dataset.h5"
    )
    val_dataset_path: pathlib.Path = (
        get_data_folder() / "large/nerf_grasp_dataset/val_dataset.h5"
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
                    batch_size=256,
                    log_path=(
                        get_data_folder()
                        / (
                            f"logs/nerf_grasp_sampler/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                        )
                    ),
                ),
                multigpu=True,
                wandb_log=True,
            ),
        ),
    )

    train_dataset = NerfGraspSampleDataset(
        input_hdf5_filepath=config.train_dataset_path,
    )
    val_dataset = NerfGraspSampleDataset(
        input_hdf5_filepath=config.val_dataset_path,
    )

    model = NerfSampler(
        global_grid_shape=(
            4,
            NERF_DENSITIES_GLOBAL_NUM_X,
            NERF_DENSITIES_GLOBAL_NUM_Y,
            NERF_DENSITIES_GLOBAL_NUM_Z,
        ),
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
