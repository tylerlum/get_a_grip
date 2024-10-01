import os
import pathlib
import subprocess
from dataclasses import dataclass

import tyro
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.scripts.train import _set_random_seed


@dataclass
class TrainNerfArgs:
    nerfdata_folder: pathlib.Path
    nerfcheckpoints_folder: pathlib.Path
    max_num_iterations: int = 400


def train_loop_return_trainer(
    local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0
) -> Trainer:
    """Main training function that sets up and runs the trainer per process
    THIS IS A COPY OF train_loop in https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/train.py
    BUT RETURNS THE TRAINER

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()
    return trainer


def get_nerfacto_default_config() -> TrainerConfig:
    # From nerfstudio/configs/method_configs.py nerfacto
    from nerfstudio.configs.method_configs import all_methods

    return all_methods["nerfacto"]


def train_nerf_return_trainer(
    args: TrainNerfArgs,
) -> Trainer:
    os.environ["WANDB_MODE"] = "disabled"

    # Should be equivalent to train_nerf
    config = get_nerfacto_default_config()

    # Check config types are what we expect
    assert isinstance(
        config.pipeline.model, NerfactoModelConfig
    ), f"Expected NerfactoModelConfig, got {config.pipeline.model}"
    assert isinstance(
        config.pipeline.datamanager, ParallelDataManagerConfig
    ), f"Expected ParallelDataManagerConfig, got {config.pipeline.datamanager}"
    assert isinstance(
        config.pipeline.datamanager.dataparser, NerfstudioDataParserConfig
    ), f"Expected NerfstudioDataParserConfig, got {config.pipeline.datamanager.dataparser}"

    # Modifications
    config.data = args.nerfdata_folder
    config.pipeline.datamanager.data = args.nerfdata_folder
    config.max_num_iterations = args.max_num_iterations
    config.output_dir = args.nerfcheckpoints_folder
    config.vis = "wandb"  # "viewer" is blocking, "wandb" has setup time so set "WANDB_MODE=disabled" to disable

    config.pipeline.model.disable_scene_contraction = True
    config.pipeline.datamanager.dataparser.auto_scale_poses = False
    config.pipeline.datamanager.dataparser.scale_factor = 1.0
    config.pipeline.datamanager.dataparser.center_method = "none"
    config.pipeline.datamanager.dataparser.orientation_method = "none"

    # Need to set timestamp
    config.set_timestamp()

    # print and save config
    config.print_to_terminal()
    config.save_config()

    trainer = train_loop_return_trainer(local_rank=0, world_size=1, config=config)
    return trainer


def print_and_run(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def train_nerf(
    args: TrainNerfArgs,
) -> None:
    # Should be equivalent to train_nerf_return_trainer
    os.environ["WANDB_MODE"] = "disabled"
    command = " ".join(
        [
            "ns-train nerfacto",
            f"--data {str(args.nerfdata_folder)}",
            f"--max-num-iterations {args.max_num_iterations}",
            f"--output-dir {str(args.nerfcheckpoints_folder)}",
            "--vis wandb",  # "viewer" is blocking, "wandb" has setup time so set "WANDB_MODE=disabled" to disable
            "--pipeline.model.disable-scene-contraction True",
            "nerfstudio-data",
            "--auto-scale-poses False",
            "--scale-factor 1.",
            "--center-method none",
            "--orientation-method none",
        ]
    )
    print_and_run(command)


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[TrainNerfArgs])
    train_nerf_return_trainer(args)


if __name__ == "__main__":
    main()
