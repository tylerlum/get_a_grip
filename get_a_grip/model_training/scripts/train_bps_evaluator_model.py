import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import wandb
from get_a_grip import get_data_folder
from get_a_grip.model_training.models.bps_evaluator_model import BpsEvaluatorModel
from get_a_grip.model_training.utils.bps_grasp_dataset import BpsGraspEvalDataset
from wandb.util import generate_id


@dataclass
class TrainBpsEvaluatorModelConfig:
    # dataset paths
    train_dataset_path: Path = (
        get_data_folder() / "dataset/NEW/bps_grasp_dataset/train_dataset.h5"
    )
    val_dataset_path: Path = (
        get_data_folder() / "dataset/NEW/bps_grasp_dataset/val_dataset.h5"
    )

    # training parameters
    batch_size: int = 32768
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 42

    # validation, printing, and saving
    snapshot_freq: int = 5
    output_dir: Path = (
        get_data_folder()
        / f"models/NEW/bps_evaluator_model/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # wandb
    wandb_project: str = "bps_evaluator_model"
    wandb_log: bool = True

    train_frac_throw_away: float = 0.0  # Increase for dataset size ablation

    # whether to use multigpu training
    multigpu: bool = True
    num_gpus: int = torch.cuda.device_count()
    num_workers: int = 4


def setup(cfg: TrainBpsEvaluatorModelConfig, rank: int = 0):
    """Sets up the training loop."""
    # set random seed
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # local device
    if cfg.multigpu:
        device = torch.device("cuda", rank)
    else:
        device = torch.device(cfg.device)

    if cfg.multigpu:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=cfg.num_gpus)

    # get datasets
    train_dataset = BpsGraspEvalDataset(
        input_hdf5_filepath=cfg.train_dataset_path,
        frac_throw_away=cfg.train_frac_throw_away,
    )
    val_dataset = BpsGraspEvalDataset(
        input_hdf5_filepath=cfg.val_dataset_path,
        frac_throw_away=0.0,
    )

    # make dataloaders
    if cfg.multigpu:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=cfg.num_gpus,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=cfg.num_gpus,
            rank=rank,
            shuffle=False,
        )
        train_shuffle = None
        val_shuffle = None
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        pin_memory=True,
        num_workers=cfg.num_workers,
        sampler=train_sampler,
        multiprocessing_context="fork",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=val_shuffle,
        pin_memory=True,
        num_workers=cfg.num_workers,
        sampler=val_sampler,
        multiprocessing_context="fork",
    )

    # make other stuff we need
    if cfg.multigpu:
        model = DDP(BpsEvaluatorModel(in_grasp=37).to(device), device_ids=[rank])
    else:
        model = BpsEvaluatorModel(in_grasp=37).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.9)

    # logging
    wandb_id = generate_id()
    if cfg.wandb_log and rank == 0:
        wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    return (
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        train_sampler,
        wandb_id,
    )


def train(cfg: TrainBpsEvaluatorModelConfig, rank: int = 0) -> None:
    """Training function."""
    train_loader, val_loader, model, optimizer, scheduler, train_sampler, wandb_id = (
        setup(cfg, rank=rank)
    )
    if cfg.multigpu:
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make log path
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # update tqdm bar with train and val loss
    val_loss = 0.0
    with trange(cfg.num_epochs, desc="Epoch", leave=False, disable=(rank != 0)) as pbar:
        for epoch in range(cfg.num_epochs):
            if cfg.multigpu:
                dist.barrier()
                train_sampler.set_epoch(epoch)

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            model.train()
            train_loss = 0.0
            for i, (g_O, f_O, y) in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc="Iterations",
                leave=False,
                disable=(rank != 0),
            ):
                f_O, g_O, y = f_O.to(device), g_O.to(device), y.to(device)

                # [DEBUG]
                #######################
                # f_O = 0.0 * f_O
                # g_O[..., 1:] = 0.0
                # g_O[..., 0] = y
                # y = torch.nn.Sigmoid()(g_O[..., 0])
                #######################

                optimizer.zero_grad()
                y_pred = model(f_O, g_O)

                assert y_pred.shape == y.shape
                loss = torch.nn.functional.mse_loss(y_pred, y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                # [DEBUG]
                #######################
                # scheduler.step(loss)
                # print(f_O)
                # print(g_O)
                # print(y)
                # breakpoint()
                # if i == 0:
                #     break
                #######################

                if i % 100 == 0:
                    pbar.set_postfix(train_loss=train_loss / (i + 1), val_loss=val_loss)

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (g_O, f_O, y) in enumerate(val_loader):
                    f_O, g_O, y = f_O.to(device), g_O.to(device), y.to(device)
                    y_pred = model(f_O, g_O)
                    assert y_pred.shape == y.shape
                    loss = torch.nn.functional.mse_loss(y_pred, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            # scheduler.step(val_loss)
            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
            if cfg.wandb_log and rank == 0:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            if (
                epoch % cfg.snapshot_freq == 0 or epoch == cfg.num_epochs - 1
            ) and rank == 0:
                print(f"Saving model at epoch {epoch}!")
                torch.save(
                    getattr(model, "module", model).state_dict(),
                    cfg.output_dir / f"ckpt-{wandb_id}-step-{epoch}.pth",
                )

                if epoch == cfg.num_epochs - 1:
                    print(
                        f"Saving final model at path {cfg.output_dir / f'ckpt-{wandb_id}-step-final.pth'}!"
                    )
                    torch.save(
                        getattr(model, "module", model).state_dict(),
                        cfg.output_dir / f"ckpt-{wandb_id}-step-final.pth",
                    )

    if cfg.multigpu:
        dist.destroy_process_group()


def _train_multigpu(rank, cfg):
    train(cfg, rank)


def main() -> None:
    cfg = tyro.cli(tyro.conf.FlagConversionOff[TrainBpsEvaluatorModelConfig])
    if cfg.multigpu:
        mp.spawn(_train_multigpu, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        train(cfg, rank=0)


if __name__ == "__main__":
    main()
