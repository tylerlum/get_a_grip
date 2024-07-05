from __future__ import annotations

import math
import pathlib
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pypose as pp
import torch
import tyro
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from tqdm import tqdm as std_tqdm

import wandb
from get_a_grip.grasp_planning.config.optimization_config import OptimizationConfig
from get_a_grip.grasp_planning.config.optimizer_config import (
    RandomSamplingConfig,
    SGDOptimizerConfig,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import (
    AllegroGraspConfig,
)
from get_a_grip.grasp_planning.utils.joint_limit_utils import (
    get_joint_limits,
)
from get_a_grip.grasp_planning.utils.nerf_evaluator_wrapper import (
    NerfEvaluatorWrapper,
)
from get_a_grip.grasp_planning.utils.optimizer_utils import (
    sample_random_rotate_transforms_only_around_y,
)

tqdm = partial(std_tqdm, dynamic_ncols=True)


class Optimizer:
    """
    A base class for grasp optimizers.
    """

    def __init__(
        self,
        init_grasp_config: AllegroGraspConfig,
        nerf_evaluator_wrapper: NerfEvaluatorWrapper,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nerf_evaluator_wrapper = nerf_evaluator_wrapper.to(device=self.device)
        init_grasp_config = init_grasp_config.to(device=self.device)

        self.grasp_config = init_grasp_config
        self.nerf_evaluator_wrapper = nerf_evaluator_wrapper

    def compute_grasp_losses(self) -> torch.Tensor:
        return self.nerf_evaluator_wrapper.get_failure_probability(self.grasp_config)

    def compute_grasp_losses_no_grad(self) -> torch.Tensor:
        with torch.no_grad():
            return self.nerf_evaluator_wrapper.get_failure_probability(
                self.grasp_config
            )

    def step(self):
        raise NotImplementedError()


class SGDOptimizer(Optimizer):
    def __init__(
        self,
        init_grasp_config: AllegroGraspConfig,
        nerf_evaluator_wrapper: NerfEvaluatorWrapper,
        optimizer_config: SGDOptimizerConfig,
    ):
        """
        Constructor for SGDOptimizer.

        Args:
            init_grasp_config: Initial grasp configuration.
            nerf_evaluator_wrapper: NerfEvaluatorWrapper object defining the metric to optimize.
            optimizer_config: SGDOptimizerConfig object defining the optimizer configuration.
        """
        super().__init__(init_grasp_config, nerf_evaluator_wrapper)
        self.optimizer_config = optimizer_config

        # Add requires_grad to grasp config.
        init_grasp_config.joint_angles.requires_grad = optimizer_config.opt_fingers
        init_grasp_config.wrist_pose.requires_grad = optimizer_config.opt_wrist_pose
        init_grasp_config.grasp_orientations.requires_grad = (
            optimizer_config.opt_grasp_dirs
        )

        USE_ADAMW = True
        if optimizer_config.opt_fingers:
            self.joint_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.joint_angles],
                    lr=optimizer_config.finger_lr,
                    momentum=optimizer_config.momentum,
                )
                if not USE_ADAMW
                else torch.optim.AdamW(
                    [self.grasp_config.joint_angles],
                    lr=optimizer_config.finger_lr,
                )
            )

        if optimizer_config.opt_wrist_pose:
            self.wrist_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.wrist_pose],
                    lr=optimizer_config.wrist_lr,
                    momentum=optimizer_config.momentum,
                )
                if not USE_ADAMW
                else torch.optim.AdamW(
                    [self.grasp_config.wrist_pose],
                    lr=optimizer_config.wrist_lr,
                )
            )

        if optimizer_config.opt_grasp_dirs:
            self.grasp_dir_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.grasp_orientations],
                    lr=optimizer_config.grasp_dir_lr,
                    momentum=optimizer_config.momentum,
                )
                if not USE_ADAMW
                else torch.optim.AdamW(
                    [self.grasp_config.grasp_orientations],
                    lr=optimizer_config.grasp_dir_lr,
                )
            )

        self.joint_lower_limits, self.joint_upper_limits = get_joint_limits(
            device=self.grasp_config.device
        )

        assert self.joint_lower_limits.shape == (16,)
        assert self.joint_upper_limits.shape == (16,)

        self.grasp_config.joint_angles.data = torch.clamp(
            self.grasp_config.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )

    def step(self):
        if self.optimizer_config.opt_fingers:
            self.joint_optimizer.zero_grad()
        if self.optimizer_config.opt_wrist_pose:
            self.wrist_optimizer.zero_grad()
        if self.optimizer_config.opt_grasp_dirs:
            self.grasp_dir_optimizer.zero_grad()

        losses = self.compute_grasp_losses()
        assert losses.shape == (len(self.grasp_config),)

        losses.sum().backward()  # Should be sum so gradient magnitude per parameter is invariant to batch size.

        if self.optimizer_config.opt_fingers:
            self.joint_optimizer.step()
        if self.optimizer_config.opt_wrist_pose:
            self.wrist_optimizer.step()
        if self.optimizer_config.opt_grasp_dirs:
            self.grasp_dir_optimizer.step()

        # Clip joint angles to feasible range.
        self.grasp_config.joint_angles.data = torch.clamp(
            self.grasp_config.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )


class RandomSamplingOptimizer(Optimizer):
    def __init__(
        self,
        init_grasp_config: AllegroGraspConfig,
        nerf_evaluator_wrapper: NerfEvaluatorWrapper,
        optimizer_config: RandomSamplingConfig,
    ):
        """
        Constructor for RandomSamplingOptimizer.

        Args:
            init_grasp_config: Initial grasp configuration.
            nerf_evaluator_wrapper: NerfEvaluatorWrapper object defining the metric to optimize.
            optimizer_config: RandomSamplingConfig object defining the optimizer configuration.
        """
        super().__init__(init_grasp_config, nerf_evaluator_wrapper)
        self.optimizer_config = optimizer_config

        self.joint_lower_limits, self.joint_upper_limits = get_joint_limits(
            device=self.grasp_config.device
        )

        assert self.joint_lower_limits.shape == (16,)
        assert self.joint_upper_limits.shape == (16,)

        self.grasp_config.joint_angles.data = torch.clamp(
            self.grasp_config.joint_angles,
            min=self.joint_lower_limits,
            max=self.joint_upper_limits,
        )

    def step(self):
        with torch.no_grad():
            # Eval old
            old_losses = (
                self.nerf_evaluator_wrapper.get_failure_probability(self.grasp_config)
                .detach()
                .cpu()
                .numpy()
            )

            # Sample new
            new_grasp_config = AllegroGraspConfig.from_grasp_config_dict(
                self.grasp_config.as_dict(),
                check=False,  # Check causing issues, probably just numerical issues
            ).to(device=self.device)

            wrist_trans_perturbations = (
                torch.randn((*new_grasp_config.wrist_pose.lshape, 3))
                * self.optimizer_config.wrist_trans_noise
            ).to(device=self.device)
            wrist_rot_perturbations = (
                (
                    pp.randn_so3(new_grasp_config.wrist_pose.lshape)
                    * self.optimizer_config.wrist_rot_noise
                )
                .Exp()
                .to(device=self.device)
            )
            wrist_pose_perturbations = pp.SE3(
                torch.cat([wrist_trans_perturbations, wrist_rot_perturbations], dim=-1)
            ).to(device=self.device)

            joint_angle_perturbations = (
                torch.randn_like(new_grasp_config.joint_angles)
                * self.optimizer_config.joint_angle_noise
            ).to(device=self.device)
            grasp_orientation_perturbations = (
                (
                    pp.randn_so3(new_grasp_config.grasp_orientations.lshape)
                    * self.optimizer_config.grasp_orientation_noise
                )
                .Exp()
                .to(device=self.device)
            )

            new_grasp_config.hand_config.set_wrist_pose(
                wrist_pose_perturbations @ new_grasp_config.hand_config.wrist_pose
            )
            new_grasp_config.hand_config.set_joint_angles(
                new_grasp_config.hand_config.joint_angles + joint_angle_perturbations
            )
            new_grasp_config.set_grasp_orientations(
                grasp_orientation_perturbations @ new_grasp_config.grasp_orientations
            )

            # Clip joint angles to feasible range.
            new_grasp_config.joint_angles.data = torch.clamp(
                new_grasp_config.joint_angles,
                min=self.joint_lower_limits,
                max=self.joint_upper_limits,
            )

            # Eval new
            new_losses = (
                self.nerf_evaluator_wrapper.get_failure_probability(new_grasp_config)
                .detach()
                .cpu()
                .numpy()
            )

            # Update grasp config
            old_dict = self.grasp_config.as_dict()
            new_dict = new_grasp_config.as_dict()
            improved_idxs = new_losses < old_losses

            improved_dict = {}
            for key in old_dict.keys():
                assert old_dict[key].shape == new_dict[key].shape
                improved_dict[key] = old_dict[key].copy()
                improved_dict[key][improved_idxs] = new_dict[key][improved_idxs]

            self.grasp_config = AllegroGraspConfig.from_grasp_config_dict(
                improved_dict,
                check=False,  # Check causing issues, probably just numerical issues
            ).to(device=self.device)


def run_optimizer_loop(
    optimizer: Optimizer,
    optimizer_config: Union[SGDOptimizerConfig, RandomSamplingConfig],
    print_freq: int,
    save_grasps_freq: int,
    output_path: pathlib.Path,
    use_rich: bool = False,
    console=Console(),
) -> Tuple[torch.Tensor, AllegroGraspConfig]:
    """
    Convenience function for running the optimizer loop.
    """

    with (
        Progress(
            TextColumn("[bold green]{task.description}[/bold green]"),
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TextColumn("=>"),
            TimeElapsedColumn(),
            console=console,
        )
        if use_rich
        else nullcontext()
    ) as progress:
        task_id = (
            progress.add_task(
                "Optimizing grasps...",
                total=optimizer_config.num_steps,
            )
            if progress is not None
            else None
        )

        for iter in tqdm(range(optimizer_config.num_steps), desc="Optimizing grasps"):
            losses_np = optimizer.compute_grasp_losses_no_grad().detach().cpu().numpy()

            if iter % print_freq == 0:
                print(
                    f"Iter: {iter} | Losses: {np.round(losses_np.tolist(), decimals=3)} | Min loss: {losses_np.min():.3f} | Max loss: {losses_np.max():.3f} | Mean loss: {losses_np.mean():.3f} | Std dev: {losses_np.std():.3f}"
                )

            optimizer.step()

            # Update progress bar.
            if progress is not None and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                )

            # Log to wandb.
            if wandb.run is not None:
                wandb_log_dict = {}
                wandb_log_dict["optimization_step"] = iter
                for i, loss in enumerate(losses_np.tolist()):
                    wandb_log_dict[f"loss_{i}"] = loss
                wandb_log_dict["min_loss"] = losses_np.min().item()
                wandb_log_dict["max_loss"] = losses_np.max().item()
                wandb_log_dict["mean_loss"] = losses_np.mean().item()
                wandb_log_dict["std_loss"] = losses_np.std().item()

                wandb.log(wandb_log_dict)

            if iter % save_grasps_freq == 0:
                # Save mid optimization grasps to file
                grasp_config_dict = optimizer.grasp_config.as_dict()
                grasp_config_dict["loss"] = losses_np

                # To interface with mid optimization visualizer, need to create new folder (mid_optimization_folder_path)
                # that has folders with iteration number
                # TODO: Decide if this should just store in output_path.parent (does this cause issues?) or store in new folder
                # <mid_optimization_folder_path>
                #    - 0
                #        - <object_code_and_scale_str>.py
                #    - x
                #        - <object_code_and_scale_str>.py
                #    - 2x
                #        - <object_code_and_scale_str>.py
                #    - 3x
                #        - <object_code_and_scale_str>.py
                main_output_folder_path, filename = (
                    output_path.parent,
                    output_path.name,
                )
                mid_optimization_folder_path = (
                    main_output_folder_path / "mid_optimization"
                )
                this_iter_folder_path = mid_optimization_folder_path / f"{iter}"
                this_iter_folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving mid opt grasp config dict to {this_iter_folder_path}")
                np.save(
                    this_iter_folder_path / filename,
                    grasp_config_dict,
                    allow_pickle=True,
                )

    optimizer.nerf_evaluator_wrapper.eval()

    return (
        optimizer.compute_grasp_losses_no_grad(),
        optimizer.grasp_config,
    )


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_evaluator_wrapper: Optional[NerfEvaluatorWrapper] = None,
) -> Dict[str, np.ndarray]:
    # print("=" * 80)
    # print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
    # print("=" * 80 + "\n")

    # Create rich.Console object.
    if cfg.random_seed is not None:
        torch.random.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

    console = Console(width=120)

    if cfg.wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=asdict(cfg),
        )

    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        )
        if cfg.use_rich
        else nullcontext()
    ) as progress:
        task = (
            progress.add_task("Loading grasp data", total=1)
            if progress is not None
            else None
        )

        # TODO: Find a way to load a particular split of the grasp_data.
        init_grasp_config_dict = np.load(
            cfg.init_grasp_config_dict_path, allow_pickle=True
        ).item()
        num_grasps_in_dict = init_grasp_config_dict["trans"].shape[0]
        print(f"Found {num_grasps_in_dict} grasps in grasp config dict dataset")

        if (
            cfg.max_num_grasps_to_eval is not None
            and num_grasps_in_dict > cfg.max_num_grasps_to_eval
        ):
            print(f"Limiting to {cfg.max_num_grasps_to_eval} grasps from dataset.")
            # randomize the order, keep at most max_num_grasps_to_eval
            init_grasp_config_dict = {
                k: v[
                    np.random.choice(
                        a=v.shape[0], size=cfg.max_num_grasps_to_eval, replace=False
                    )
                ]
                for k, v in init_grasp_config_dict.items()
            }

        init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            init_grasp_config_dict
        )
        print(f"Loaded {len(init_grasp_configs)} initial grasp configs.")

        # HACK: For now, just take the first num_grasps.
        # init_grasp_configs = init_grasp_configs[: cfg.optimizer.num_grasps]

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    # Create grasp metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nerf_evaluator_wrapper is None:
        print(
            f"Loading nerf_evaluator config from {cfg.nerf_evaluator_wrapper.nerf_evaluator_config_path}"
        )
        nerf_evaluator_wrapper = NerfEvaluatorWrapper.from_config(
            cfg.nerf_evaluator_wrapper,
            console=console,
        )
    else:
        print("Using provided grasp metric.")
    nerf_evaluator_wrapper = nerf_evaluator_wrapper.to(device=device)

    # Put this here to ensure that the random seed is set before sampling random rotations.
    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    BATCH_SIZE = cfg.eval_batch_size
    all_success_preds = []
    with torch.no_grad():
        # Sample random rotations
        N_SAMPLES = 1 + cfg.n_random_rotations_per_grasp
        new_grasp_configs_list = []
        for i in range(N_SAMPLES):
            new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                init_grasp_config_dict
            )
            if i != 0:
                random_rotate_transforms = (
                    sample_random_rotate_transforms_only_around_y(
                        len(new_grasp_configs)
                    )
                )
                new_grasp_configs.hand_config.set_wrist_pose(
                    random_rotate_transforms @ new_grasp_configs.hand_config.wrist_pose
                )
            new_grasp_configs_list.append(new_grasp_configs)

        new_grasp_config_dicts = defaultdict(list)
        for i in range(N_SAMPLES):
            config_dict = new_grasp_configs_list[i].as_dict()
            for k, v in config_dict.items():
                new_grasp_config_dicts[k].append(v)
        for k, v in new_grasp_config_dicts.items():
            new_grasp_config_dicts[k] = np.concatenate(v, axis=0)
        new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            new_grasp_config_dicts
        )
        assert len(new_grasp_configs) == len(init_grasp_configs) * N_SAMPLES

        # Filter grasps that are less IK feasible
        if cfg.filter_less_feasible_grasps:
            new_grasp_configs = new_grasp_configs.filter_less_feasible(
                fingers_forward_theta_deg=cfg.fingers_forward_theta_deg,
                palm_upwards_theta_deg=cfg.palm_upwards_theta_deg,
            )

        # Evaluate grasp metric and collisions
        n_batches = math.ceil(len(new_grasp_configs) / BATCH_SIZE)
        for batch_i in tqdm(
            range(n_batches), desc=f"Evaling grasp metric with batch_size={BATCH_SIZE}"
        ):
            start_idx = batch_i * BATCH_SIZE
            end_idx = np.clip(
                (batch_i + 1) * BATCH_SIZE,
                a_min=None,
                a_max=len(new_grasp_configs),
            )

            temp_grasp_configs = new_grasp_configs[start_idx:end_idx].to(device=device)

            # Metric
            success_preds = (
                (1 - nerf_evaluator_wrapper.get_failure_probability(temp_grasp_configs))
                .detach()
                .cpu()
                .numpy()
            )
            all_success_preds.append(success_preds)

        # Aggregate
        all_success_preds = np.concatenate(all_success_preds)
        assert all_success_preds.shape == (len(new_grasp_configs),)

        ordered_idxs_best_first = np.argsort(all_success_preds)[::-1].copy()
        print(
            f"ordered_idxs_best_first = {ordered_idxs_best_first[:cfg.optimizer.num_grasps]}"
        )

        new_grasp_configs = new_grasp_configs[ordered_idxs_best_first]

    init_grasp_configs = new_grasp_configs[: cfg.optimizer.num_grasps]

    # Create Optimizer.
    if isinstance(cfg.optimizer, SGDOptimizerConfig):
        optimizer = SGDOptimizer(
            init_grasp_configs,
            nerf_evaluator_wrapper,
            cfg.optimizer,
        )
    elif isinstance(cfg.optimizer, RandomSamplingConfig):
        optimizer = RandomSamplingOptimizer(
            init_grasp_configs,
            nerf_evaluator_wrapper,
            cfg.optimizer,
        )
    else:
        raise ValueError(f"Invalid optimizer config: {cfg.optimizer}")

    init_losses = optimizer.compute_grasp_losses_no_grad()

    table = Table(title="Grasp loss")
    table.add_column("Iteration", justify="right")
    table.add_column("Min loss")
    table.add_column("Mean loss")
    table.add_column("Max loss")
    table.add_column("Std dev.")

    table.add_row(
        "0",
        f"{init_losses.min():.5f}",
        f"{init_losses.mean():.5f}",
        f"{init_losses.max():.5f}",
        f"{init_losses.std():.5f}",
    )

    final_losses, final_grasp_configs = run_optimizer_loop(
        optimizer,
        optimizer_config=cfg.optimizer,
        print_freq=cfg.print_freq,
        save_grasps_freq=cfg.save_grasps_freq,
        output_path=cfg.output_path,
        use_rich=cfg.use_rich,
        console=console,
    )

    assert final_losses.shape[0] == len(
        final_grasp_configs
    ), f"{final_losses.shape[0]} != {len(final_grasp_configs)}"

    print(f"Initial grasp loss: {np.round(init_losses.tolist(), decimals=3)}")
    print(f"Final grasp loss: {np.round(final_losses.tolist(), decimals=3)}")

    table.add_row(
        f"{cfg.optimizer.num_steps}",
        f"{final_losses.min():.5f}",
        f"{final_losses.mean():.5f}",
        f"{final_losses.max():.5f}",
        f"{final_losses.std():.5f}",
    )
    if cfg.use_rich:
        console.print(table)
    else:
        table_str = f"""
        Iteration: {0:4d} | Min loss: {init_losses.min():.5f} | Mean loss: {init_losses.mean():.5f} | Max loss: {init_losses.max():.5f} | Std dev: {init_losses.std():.5f}
        Iteration: {cfg.optimizer.num_steps:4d} | Min loss: {final_losses.min():.5f} | Mean loss: {final_losses.mean():.5f} | Max loss: {final_losses.max():.5f} | Std dev: {final_losses.std():.5f}
        """
        print(table_str)

    # HACK
    # grasp_config_dict = COPY.as_dict()
    grasp_config_dict = final_grasp_configs.as_dict()
    grasp_config_dict["loss"] = final_losses.detach().cpu().numpy()

    print(f"Saving final grasp config dict to {cfg.output_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cfg.output_path), grasp_config_dict, allow_pickle=True)

    if wandb.run is not None:
        wandb.finish()
    return grasp_config_dict


def main() -> None:
    cfg = tyro.cli(OptimizationConfig)
    get_optimized_grasps(cfg)


if __name__ == "__main__":
    main()
