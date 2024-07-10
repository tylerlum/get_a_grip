import pathlib
import subprocess
from dataclasses import dataclass

from nerfstudio.engine.trainer import Trainer, TrainerConfig
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


def get_nerfacto_default_config():
    # From nerfstudio/configs/method_configs.py nerfacto
    from nerfstudio.configs.method_configs import all_methods

    return all_methods["nerfacto"]


def train_nerf_return_trainer(
    args: TrainNerfArgs,
) -> Trainer:
    # Should be equivalent to train_nerf
    config = get_nerfacto_default_config()

    # Modifications
    config.data = args.nerfdata_folder
    config.pipeline.datamanager.data = args.nerfdata_folder
    config.max_num_iterations = args.max_num_iterations
    config.output_dir = args.nerfcheckpoints_folder
    config.vis = "none"

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
    command = " ".join(
        [
            "ns-train nerfacto",
            f"--data {str(args.nerfdata_folder)}",
            f"--max-num-iterations {args.max_num_iterations}",
            f"--output-dir {str(args.nerfcheckpoints_folder)}",
            "--vis wandb",
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
    args = TrainNerfArgs(
        nerfdata_folder=pathlib.Path(
            "experiments/2024-04-15_DEBUG/nerfdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
        ),
        nerfcheckpoints_folder=pathlib.Path(
            "experiments/2024-04-15_DEBUG/nerfcheckpoints/"
        ),
    )
    train_nerf_return_trainer(args)


if __name__ == "__main__":
    main()
