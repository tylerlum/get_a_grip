"""The goal of this file to implement the diffusion process for grasp sampling.
Implementation based on: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
"""

import os
import time
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import wandb
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
)
from wandb.util import generate_id


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_optimizer(config: DiffusionConfig, parameters):
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )
    elif config.optim.optimizer == "RMSProp":
        return optim.RMSprop(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == "SGD":
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            "Optimizer {} not understood.".format(config.optim.optimizer)
        )


def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    cond: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(f_O=cond, g_t=x, t=t.float().view(-1, 1))
    if keepdim:
        return (e - output).square().sum(dim=1)
    else:
        return (e - output).square().sum(dim=1).mean(dim=0)


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
    return a


def generalized_steps(x, cond, seq, model, b, eta):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to("cuda")
            et = model(g_t=xt, f_O=cond, t=t.float().view(-1, 1))
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to("cpu"))

    return xs, x0_preds


def ddpm_steps(x, cond, seq, model, b):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")

            output = model(g_t=x, f_O=cond, t=t.float().view(-1, 1))
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to("cpu"))
    return xs, x0_preds


class Diffusion(object):
    def __init__(
        self,
        config: DiffusionConfig,
        model: nn.Module,
        device: Optional[torch.device] = None,
        rank: int = 0,
        load_multigpu_ckpt: bool = False,
    ):
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.schedule.beta_schedule,
            beta_start=config.schedule.beta_start,
            beta_end=config.schedule.beta_end,
            num_diffusion_timesteps=config.schedule.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        model = model.to(self.device)
        if config.multigpu and not load_multigpu_ckpt:
            self.model = DDP(model, device_ids=[rank])
        else:
            self.model = model

    def load_checkpoint(self, config: DiffusionConfig, filename: str) -> None:
        # Use given filename
        assert filename.endswith(".pth"), f"Invalid filename: {filename}"
        checkpoint_path = config.training.output_dir / filename
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

        states = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        model_state_dict = states[0]
        self.model.load_state_dict(model_state_dict)

    def load_latest_checkpoint(self, config: DiffusionConfig) -> None:
        pth_filepaths = list(config.training.output_dir.glob("*.pth"))

        if len(pth_filepaths) == 0:
            raise FileNotFoundError(
                f"No checkpoints found in {config.training.output_dir}"
            )

        # Sort by created time
        latest_pth_filepath = sorted(
            pth_filepaths, key=lambda path: path.stat().st_ctime
        )[-1]

        self.load_checkpoint(config, latest_pth_filepath.name)

    def sample(self, xT: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            x = self._sample(xT, cond, self.model, last=True)
            assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
            x = x.to(xT.device)
            return x

    def sample_and_return_all_steps(
        self, xT: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        with torch.no_grad():
            output = self._sample(xT, cond, self.model, last=False)
            assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"

            xs, x0_preds = output

            # Keep on cpu because will be large
            xs = torch.stack([x.cpu() for x in xs], dim=0)
            x0_preds = torch.stack([x.cpu() for x in x0_preds], dim=0)
            n_steps = xs.shape[0]
            assert xs.shape == (
                n_steps,
                *xT.shape,
            ), f"Expected shape {(n_steps, *xT.shape)}, got {xs.shape}"
            assert x0_preds.shape == (
                n_steps - 1,
                *xT.shape,
            ), f"Expected shape {(n_steps - 1, *xT.shape)}, got {x0_preds.shape}"

            return xs, x0_preds

    def _sample(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        model: nn.Module,
        last: bool = True,
        sample_type: Literal["generalized", "ddpm_noisy"] = "generalized",
        skip_type: Literal["uniform", "quad"] = "quad",
        skip: int = 1,
        timesteps: int = 1000,
        eta: float = 0.0,
    ) -> Union[torch.Tensor, tuple]:
        if skip_type == "uniform":
            skip = self.num_timesteps // timesteps
            seq = range(0, self.num_timesteps, skip)
        elif skip_type == "quad":
            seq = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), timesteps) ** 2
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        if sample_type == "generalized":
            xs, x0_preds = generalized_steps(x, cond, seq, model, self.betas, eta=eta)
        elif sample_type == "ddpm_noisy":
            xs, x0_preds = ddpm_steps(x, cond, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            return xs[-1]
        return xs, x0_preds


def train(
    config: DiffusionConfig,
    train_dataset: data.Dataset,
    val_dataset: data.Dataset,
    model: nn.Module,
    rank: int = 0,
) -> None:
    num_gpus = torch.cuda.device_count()
    if config.multigpu:
        device = torch.device("cuda", rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=num_gpus)
    else:
        device = torch.device("cuda")

    config = config
    wandb_id = generate_id()
    if config.wandb.log and rank == 0:
        wandb.init(project=config.wandb.project, id=wandb_id, resume="allow")

    # datasets and dataloader
    if config.multigpu:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_gpus,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=num_gpus,
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

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=train_shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        multiprocessing_context="fork",  # SUPER IMPORTANT THIS IS FORK AND NOT SPAWN FOR SPEED!
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=val_shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        multiprocessing_context="fork",  # SUPER IMPORTANT THIS IS FORK AND NOT SPAWN FOR SPEED!
    )

    # making the model
    runner = Diffusion(config=config, model=model, device=device, rank=rank)
    optimizer = get_optimizer(config, runner.model.parameters())

    if runner.config.model.ema:
        ema_helper = EMAHelper(mu=runner.config.model.ema_rate)
        ema_helper.register(runner.model)
    else:
        ema_helper = None

    start_epoch, step = 0, 0
    with trange(
        start_epoch,
        runner.config.training.n_epochs,
        initial=start_epoch,
        total=runner.config.training.n_epochs,
        desc="Epoch",
        leave=False,
        disable=(rank != 0),
    ) as pbar:
        for epoch in range(start_epoch, runner.config.training.n_epochs):
            if config.multigpu:
                dist.barrier()
                train_sampler.set_epoch(epoch)

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch + 1}/{runner.config.training.n_epochs}")

            data_start = time.time()
            data_time = 0
            train_time = 0
            train_loss = 0

            # training loop
            runner.model.train()
            for i, (grasps, bpss, _) in tqdm(
                enumerate(train_loader),
                desc="Iterations",
                total=len(train_loader),
                leave=False,
                disable=(rank != 0),
            ):
                n = grasps.size(0)
                data_time += time.time() - data_start
                train_start = time.time()
                runner.model.train()
                step += 1

                grasps = grasps.to(device)
                bpss = bpss.to(device)
                e = torch.randn_like(grasps)
                b = runner.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=runner.num_timesteps, size=(n // 2 + 1,)
                ).to(device)
                t = torch.cat([t, runner.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(
                    model=runner.model, x0=grasps, cond=bpss, t=t, e=e, b=b
                )
                train_loss += loss.item()

                # if step % config.training.print_freq == 0 or step == 1:
                #     print(
                #         f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                #     )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        runner.model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()
                train_time += time.time() - train_start
                data_start = time.time()

            train_loss /= len(train_loader)

            # val step
            runner.model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, (grasps, bpss, _) in enumerate(val_loader):
                    n = grasps.size(0)
                    grasps = grasps.to(device)
                    bpss = bpss.to(device)
                    e = torch.randn_like(grasps)
                    b = runner.betas

                    t = torch.randint(
                        low=0, high=runner.num_timesteps, size=(n // 2 + 1,)
                    ).to(device)
                    t = torch.cat([t, runner.num_timesteps - t - 1], dim=0)[:n]
                    _val_loss = noise_estimation_loss(
                        model=runner.model, x0=grasps, cond=bpss, t=t, e=e, b=b
                    )
                    val_loss += _val_loss.item()
                val_loss /= len(val_loader)

            # logging
            pbar.set_postfix(
                step=step,
                train_loss=train_loss,
                val_loss=val_loss,
                data_time=data_time,
                train_time=train_time,
            )
            if runner.config.wandb.log and rank == 0:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            if runner.config.model.ema:
                ema_helper.update(runner.model)

            is_last_epoch = epoch == runner.config.training.n_epochs - 1
            if (
                step % runner.config.training.snapshot_freq == 0
                or step == 1
                or is_last_epoch
            ) and rank == 0:
                print(f"Saving model at step {step}!")
                states = [
                    getattr(runner.model, "module", runner.model).state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if runner.config.model.ema:
                    states.append(ema_helper.state_dict())

                output_dir = config.training.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                if is_last_epoch:
                    torch.save(states, output_dir / "ckpt_final.pth")
                else:
                    torch.save(states, output_dir / f"ckpt_{step}.pth")

    if config.multigpu:
        dist.destroy_process_group()


def train_multigpu(
    rank: int,
    config: DiffusionConfig,
    train_dataset: data.Dataset,
    val_dataset: data.Dataset,
    model: nn.Module,
) -> None:
    train(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        rank=rank,
    )
