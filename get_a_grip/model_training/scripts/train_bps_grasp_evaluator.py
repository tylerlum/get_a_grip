import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from wandb.util import generate_id

from nerf_grasping.dexdiffuser.dex_evaluator import DexEvaluator
from nerf_grasping.dexdiffuser.diffusion import get_bps_datasets, get_bps_datasets_small_train_set


@dataclass
class DexEvaluatorTrainingConfig:
    # training parameters
    batch_size: int = 32768
    learning_rate: float = 1e-4
    num_epochs: int = 10000
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 42

    # validation, printing, and saving
    snapshot_freq: int = 5000
    log_path: Path = Path("logs/dexdiffuser_evaluator")

    # wandb
    wandb_project: str = "dexdiffuser-evaluator"
    wandb_log: bool = True

    # whether to train gg-nerf or the dexdiffuser baseline
    train_ablation: bool = False

    # whether to use multigpu training
    multigpu: bool = True
    num_gpus: int = torch.cuda.device_count()
    num_workers: int = 4


def setup(cfg: DexEvaluatorTrainingConfig, rank: int = 0):
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
    USE_SMALL_TRAIN_SET = True
    if USE_SMALL_TRAIN_SET:
        print("Using small train set!")
        train_dataset, val_dataset, _ = get_bps_datasets_small_train_set(
            use_evaluator_dataset=True,
            get_all_labels=cfg.train_ablation,  # if we're training the ablation, get all the labels
            frac_throw_away=0.5,
        )
    else:
        print("Using full train set!")
        train_dataset, val_dataset, _ = get_bps_datasets(
            use_evaluator_dataset=True,
            get_all_labels=cfg.train_ablation,  # if we're training the ablation, get all the labels
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
        model = DDP(DexEvaluator(in_grasp=37).to(device), device_ids=[rank])
    else:
        model = DexEvaluator(in_grasp=37).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.9)

    # logging
    wandb_id = generate_id()
    if cfg.wandb_log and rank == 0:
        wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    return train_loader, val_loader, model, optimizer, scheduler, train_sampler, wandb_id

def train(cfg: DexEvaluatorTrainingConfig, rank: int = 0) -> None:
    """Training function."""
    train_loader, val_loader, model, optimizer, scheduler, train_sampler, wandb_id = setup(cfg, rank=rank)
    if cfg.multigpu:
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make log path
    cfg.log_path.mkdir(parents=True, exist_ok=True)

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
                if not cfg.train_ablation:
                    y_pred = model(f_O, g_O)[..., -1]
                else:
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
                    y_pred = model(f_O, g_O)[..., -1]
                    assert y_pred.shape == y.shape
                    loss = torch.nn.functional.mse_loss(y_pred, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            # scheduler.step(val_loss)
            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
            if cfg.wandb_log and rank == 0:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            if (epoch % cfg.snapshot_freq == 0 or epoch == cfg.num_epochs - 1) and rank == 0:
                print(f"Saving model at epoch {epoch}!")
                torch.save(
                    getattr(model, "module", model).state_dict(),
                    cfg.log_path / f"ckpt-{wandb_id}-step-{epoch}.pth",
                )

                if epoch == cfg.num_epochs - 1:
                    print(f"Saving final model at path {cfg.log_path / f'ckpt-{wandb_id}-step-final.pth'}!")
                    torch.save(
                        getattr(model, "module", model).state_dict(),
                        cfg.log_path / f"ckpt-{wandb_id}-step-final.pth",
                    )
    
    if cfg.multigpu:
        dist.destroy_process_group()


def _train_multigpu(rank, cfg):
    train(cfg, rank)


if __name__ == "__main__":
    cfg = DexEvaluatorTrainingConfig(
        num_epochs=1000,
        batch_size=4096 * 8,
        learning_rate=1e-4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        random_seed=42,
        snapshot_freq=5,
        log_path=Path(f"logs/dexdiffuser_evaluator/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        wandb_project="dexdiffuser-evaluator",
        wandb_log=True,
        multigpu=True,
        num_gpus=torch.cuda.device_count(),
        num_workers=4,
        train_ablation=False,
    )
    if cfg.multigpu:
        mp.spawn(_train_multigpu, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        train(cfg, rank=0)
    # testing loading
    # device = torch.device("cuda")
    # dex_evaluator = DexEvaluator(3 + 6 + 16 + 12, 4096).to(device)
    # ckpt_path = "/home/albert/research/nerf_grasping/nerf_grasping/dexdiffuser/logs/dexdiffuser_evaluator/20240602_165946/ckpt-p9u7vl8l-step-0.pth"
    # dex_evaluator.load_state_dict(torch.load(f"{ckpt_path}"))
    # batch_size = 2
    # f_O = torch.rand(batch_size, 4096).to(device)
    # g_O = torch.rand(batch_size, 3 + 6 + 16 + 12).to(device)
    # labels = dex_evaluator(f_O=f_O, g_O=g_O)
    breakpoint()
