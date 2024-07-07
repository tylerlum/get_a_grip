from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pypose as pp
import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from get_a_grip.grasp_planning.config.nerf_evaluator_wrapper_config import (
    NerfEvaluatorWrapperConfig,
)
from get_a_grip.grasp_planning.utils.allegro_grasp_config import AllegroGraspConfig
from get_a_grip.model_training.config.diffusion_config import (
    DiffusionConfig,
    TrainingConfig,
)
from get_a_grip.model_training.config.fingertip_config import (
    EvenlySpacedFingertipConfig,
)
from get_a_grip.model_training.config.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from get_a_grip.model_training.config.nerf_evaluator_model_config import (
    NerfEvaluatorModelConfig,
)
from get_a_grip.model_training.models.bps_evaluator_model import BpsEvaluatorModel
from get_a_grip.model_training.models.bps_sampler_model import BpsSamplerModel
from get_a_grip.model_training.models.nerf_evaluator_model import (
    NerfEvaluatorModel,
)
from get_a_grip.model_training.models.nerf_sampler_model import NerfSamplerModel
from get_a_grip.model_training.utils.diffusion import Diffusion
from get_a_grip.model_training.utils.nerf_grasp_evaluator_batch_data import (
    BatchDataInput,
)
from get_a_grip.model_training.utils.nerf_load_utils import (
    load_nerf_field,
)
from get_a_grip.model_training.utils.nerf_ray_utils import (
    get_ray_origins_finger_frame,
    get_ray_samples,
)
from get_a_grip.model_training.utils.nerf_utils import (
    get_densities_in_grid,
)
from get_a_grip.model_training.utils.point_utils import (
    transform_point,
)
import torch.nn as nn


class Sampler(ABC):
    @abstractmethod
    def sample(self) -> AllegroGraspConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_sample(self) -> int:
        raise NotImplementedError


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        """Returns losses where lower is better"""
        raise NotImplementedError

    @property
    @abstractmethod
    def top_K(self) -> int:
        raise NotImplementedError

    def get_top_K(self, grasp_config: AllegroGraspConfig) -> AllegroGraspConfig:
        B = len(grasp_config)
        losses = self.evaluate(grasp_config)

        assert losses.shape == (
            B,
        ), f"Expected losses.shape == ({B},), got {losses.shape}"

        sorted_idxs = torch.argsort(losses)
        _sorted_losses = losses[sorted_idxs]
        sorted_grasp_config = grasp_config[sorted_idxs]
        top_K_grasp_configs = sorted_grasp_config[: self.top_K]
        return top_K_grasp_configs


class EvaluatorOptimizer(ABC):
    @abstractmethod
    def optimize(self, grasp_config: AllegroGraspConfig) -> AllegroGraspConfig:
        raise NotImplementedError


class GraspPlanner:
    def __init__(
        self,
        sampler: Sampler,
        evaluator: Evaluator,
        evaluator_optimizer: EvaluatorOptimizer,
    ) -> None:
        self.sampler = sampler
        self.evaluator = evaluator
        self.evaluator_optimizer = evaluator_optimizer

    def plan(self) -> AllegroGraspConfig:
        # Sample
        sampled_grasp_configs = self.sampler.sample()

        # Sample random rotations around y
        new_grasp_configs = AllegroGraspConfig.from_multiple_grasp_configs(
            [sampled_grasp_configs]
            + [
                sampled_grasp_configs.sample_random_rotations_only_around_y()
                for _ in range(cfg.n_random_rotations_per_grasp)
            ]
        )
        assert len(new_grasp_configs) == len(sampled_grasp_configs) * (
            1 + cfg.n_random_rotations_per_grasp
        )

        # Filter
        if True:
            filtered_grasp_configs = sampled_grasp_configs.filter_less_feasible(
                fingers_forward_theta_deg=TODO,
                palm_upwards_theta_deg=TODO,
            )
        else:
            filtered_grasp_configs = sampled_grasp_configs

        # Evaluate and rank
        top_K_grasp_configs = self.evaluator.get_top_K(filtered_grasp_configs)

        # Optimize
        optimized_grasp_configs = self.evaluator_optimizer.optimize(top_K_grasp_configs)

        return optimized_grasp_configs


class BpsSampler(Sampler):
    def __init__(self, bps_values: torch.Tensor, ckpt_path: Path) -> None:
        # Save BPS values
        assert bps_values.shape == (
            4096,
        ), f"Expected shape (4096,), got {bps_values.shape}"
        self.bps_values = bps_values

        # Load model
        config = DiffusionConfig(
            training=TrainingConfig(
                log_path=ckpt_path.parent,
            )
        )
        self.model = BpsSamplerModel(
            n_pts=config.data.n_pts,
            grasp_dim=config.data.grasp_dim,
        )
        self.runner = Diffusion(
            config=config, model=self.model, load_multigpu_ckpt=True
        )
        self.runner.load_checkpoint(config, filename=ckpt_path.name)

    @property
    def n_sample(self) -> int:
        return 10

    def sample(self) -> AllegroGraspConfig:
        # Repeat BPS values
        bps_values_repeated = self.bps_values.unsqueeze(dim=0).repeat_interleave(
            self.n_sample, dim=0
        )
        assert bps_values_repeated.shape == (
            self.n_sample,
            4096,
        ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

        # Sample
        xT = torch.randn(self.n_sample, self.model.grasp_dim, device=self.runner.device)
        x = self.runner.sample(xT=xT, cond=bps_values_repeated)
        grasp_configs = AllegroGraspConfig.from_grasp(grasp=x)
        return grasp_configs


class NerfSampler(Sampler):
    def __init__(
        self, nerf_densities_global_with_coords: torch.Tensor, ckpt_path: Path
    ) -> None:
        assert (
            nerf_densities_global_with_coords.shape
            == (
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape (4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords.shape}"
        self.nerf_densities_global_with_coords = nerf_densities_global_with_coords

        # Load model
        config = DiffusionConfig(
            training=TrainingConfig(
                log_path=ckpt_path.parent,
            ),
        )
        self.model = NerfSamplerModel(
            global_grid_shape=(
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            ),
            grasp_dim=config.data.grasp_dim,
        )
        self.model._HACK_MODE_FOR_PERFORMANCE = True  # Big hack to speed up from sampling wasting dumb compute since it has the same cnn each time

        self.runner = Diffusion(
            config=config, model=self.model, load_multigpu_ckpt=True
        )
        self.runner.load_checkpoint(config, filename=ckpt_path.name)

    @property
    def n_sample(self) -> int:
        return 10

    def sample(self) -> AllegroGraspConfig:
        # Repeat NeRF values
        nerf_densities_global_with_coords_repeated = (
            self.nerf_densities_global_with_coords.unsqueeze(0).repeat(
                self.n_sample, 1, 1, 1, 1
            )
        )

        assert (
            nerf_densities_global_with_coords_repeated.shape
            == (
                self.n_sample,
                4,
                NERF_DENSITIES_GLOBAL_NUM_X,
                NERF_DENSITIES_GLOBAL_NUM_Y,
                NERF_DENSITIES_GLOBAL_NUM_Z,
            )
        ), f"Expected shape ({self.n_sample}, 4, {NERF_DENSITIES_GLOBAL_NUM_X}, {NERF_DENSITIES_GLOBAL_NUM_Y}, {NERF_DENSITIES_GLOBAL_NUM_Z}), got {nerf_densities_global_with_coords_repeated.shape}"

        # Sample
        xT = torch.randn(self.n_sample, self.model.grasp_dim, device=self.runner.device)
        x = self.runner.sample(xT=xT, cond=nerf_densities_global_with_coords_repeated)
        grasp_configs = AllegroGraspConfig.from_grasp(grasp=x)
        return grasp_configs


class NoEvaluator(Evaluator):
    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        return torch.linspace(0, 0.001, len(grasp_config))

    @property
    def top_K(self) -> int:
        return 5


class BpsEvaluator(Evaluator):
    def __init__(self, bps_values: torch.Tensor, ckpt_path: Path) -> None:
        # Save BPS values
        assert bps_values.shape == (
            4096,
        ), f"Expected shape (4096,), got {bps_values.shape}"
        self.bps_values = bps_values

        self.bps_evaluator = BpsEvaluatorModel(
            in_grasp=3 + 6 + 16 + 4 * 3, in_bps=4096
        ).to(bps_values.device)
        self.bps_evaluator.eval()
        self.bps_evaluator.load_state_dict(
            torch.load(ckpt_path, map_location=bps_values.device)
        )

    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        B = len(grasp_config)

        # Repeat BPS values
        bps_values_repeated = self.bps_values.unsqueeze(dim=0).repeat_interleave(
            B, dim=0
        )
        assert bps_values_repeated.shape == (
            B,
            4096,
        ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

        # Evaluate
        g_O = grasp_config.as_grasp()
        assert g_O.shape == (B, 3 + 6 + 16 + 4 * 3)

        f_O = bps_values_repeated[:B]
        assert f_O.shape == (B, 4096)

        success_preds = self.bps_evaluator(f_O=f_O, g_O=g_O)[:, -1]
        assert success_preds.shape == (
            B,
        ), f"success_preds.shape = {success_preds.shape}, expected ({B},)"

        losses = 1 - success_preds
        return losses

    @property
    def top_K(self) -> int:
        return 5


class NerfEvaluator(nn.Module, Evaluator):
    def __init__(
        self,
        nerf_field: Field,
        nerf_evaluator_model: NerfEvaluatorModel,
        fingertip_config: EvenlySpacedFingertipConfig,
        X_N_Oy: np.ndarray,
    ) -> None:
        super().__init__()
        self.nerf_field = nerf_field
        self.nerf_evaluator_model = nerf_evaluator_model
        self.fingertip_config = fingertip_config
        self.X_N_Oy = X_N_Oy
        self.ray_origins_finger_frame = get_ray_origins_finger_frame(fingertip_config)

    def forward(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        ray_samples = self.compute_ray_samples(grasp_config)

        # Query NeRF at RaySamples.
        densities = self.compute_nerf_densities(
            ray_samples,
        )
        assert densities.shape == (
            len(grasp_config),
            4,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        # Query NeRF in grid
        # BRITTLE: Require that the nerf_evaluator_model has the word "Global" in it if needed
        need_to_query_global = (
            "global" in self.nerf_evaluator_model.__class__.__name__.lower()
        )
        if need_to_query_global:
            lb_N = transform_point(T=self.X_N_Oy, point=lb_Oy)
            ub_N = transform_point(T=self.X_N_Oy, point=ub_Oy)
            nerf_densities_global, _query_points_global_N = get_densities_in_grid(
                field=self.nerf_field,
                lb=lb_N,
                ub=ub_N,
                num_pts_x=NERF_DENSITIES_GLOBAL_NUM_X,
                num_pts_y=NERF_DENSITIES_GLOBAL_NUM_Y,
                num_pts_z=NERF_DENSITIES_GLOBAL_NUM_Z,
            )
            nerf_densities_global = (
                torch.from_numpy(nerf_densities_global)
                .float()[None, ...]
                .repeat_interleave(len(grasp_config), dim=0)
            )
        else:
            nerf_densities_global, _query_points_global_N = None, None

        batch_data_input = BatchDataInput(
            nerf_densities=densities,
            grasp_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=self.fingertip_config,
            grasp_configs=grasp_config.as_tensor(),
            nerf_densities_global=(
                nerf_densities_global if nerf_densities_global is not None else None
            ),
        ).to(grasp_config.device)

        # Pass grasp transforms, densities into nerf_evaluator.
        return self.nerf_evaluator_model.get_failure_probability(batch_data_input)

    def evaluate(self, grasp_config: AllegroGraspConfig) -> torch.Tensor:
        return self(grasp_config)

    def compute_ray_samples(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> RaySamples:
        # Let Oy be object yup frame (centroid of object)
        # Let N be nerf frame (where the nerf is defined)
        # For NeRFs trained from sim data, Oy and N are the same.
        # But for real-world data, Oy and N are different (N is a point on the table, used as NeRF origin)
        # When sampling from the NeRF, we must give ray samples in N frame
        # But nerf_evaluator is trained on Oy frame
        # Thus, we must transform grasp_frame_transforms from Oy frame to N frame
        # Let Fi be the finger frame (origin at each fingertip i)
        # Let p_Fi be points in Fi frame
        # self.ray_origins_finger_frame = p_Fi
        # grasp_frame_transforms = T_{Oy <- Fi}
        # X_N_Oy = T_{N <- Oy}
        # TODO: Batch this to avoid OOM (refer to create_nerf_grasp_dataset.py)

        # Prepare transforms
        T_Oy_Fi = grasp_config.grasp_frame_transforms
        assert T_Oy_Fi.lshape == (len(grasp_config), grasp_config.num_fingers)

        assert self.X_N_Oy.shape == (
            4,
            4,
        )
        X_N_Oy_repeated = (
            torch.from_numpy(self.X_N_Oy)
            .float()
            .unsqueeze(dim=0)
            .repeat_interleave(len(grasp_config) * grasp_config.num_fingers, dim=0)
            .reshape(len(grasp_config), grasp_config.num_fingers, 4, 4)
        )

        T_N_Oy = pp.from_matrix(
            X_N_Oy_repeated.to(T_Oy_Fi.device),
            pp.SE3_type,
        )

        # Transform grasp_frame_transforms to nerf frame
        T_N_Fi = T_N_Oy @ T_Oy_Fi

        # Generate RaySamples.
        ray_samples = get_ray_samples(
            self.ray_origins_finger_frame,
            T_N_Fi,
            self.fingertip_config,
        )
        return ray_samples

    def compute_nerf_densities(
        self,
        ray_samples,
    ) -> torch.Tensor:
        # Query NeRF at RaySamples.
        densities = self.nerf_field.get_density(ray_samples.to("cuda"))[0][
            ..., 0
        ]  # Shape [B, 4, n_x, n_y, n_z]
        return densities

    @classmethod
    def from_config(
        cls,
        nerf_evaluator_wrapper_config: NerfEvaluatorWrapperConfig,
        console: Optional[Console] = None,
    ) -> NerfEvaluator:
        assert nerf_evaluator_wrapper_config.X_N_Oy is not None
        return cls.from_configs(
            nerf_config=nerf_evaluator_wrapper_config.nerf_config,
            nerf_evaluator_config=nerf_evaluator_wrapper_config.nerf_evaluator_config,
            X_N_Oy=nerf_evaluator_wrapper_config.X_N_Oy,
            nerf_evaluator_checkpoint=nerf_evaluator_wrapper_config.nerf_evaluator_checkpoint,
            console=console,
        )

    @classmethod
    def from_configs(
        cls,
        nerf_config: pathlib.Path,
        nerf_evaluator_config: NerfEvaluatorModelConfig,
        X_N_Oy: np.ndarray,
        nerf_evaluator_checkpoint: int = -1,
        console: Optional[Console] = None,
    ) -> NerfEvaluator:
        # Load nerf
        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description} "),
                TimeElapsedColumn(),
                console=console,
            )
            if console is not None
            else nullcontext()
        ) as progress:
            task = (
                progress.add_task("Loading NeRF", total=1)
                if progress is not None
                else None
            )

            nerf_field = load_nerf_field(nerf_config)

            if progress is not None and task is not None:
                progress.update(task, advance=1)

        # Load nerf_evaluator
        nerf_evaluator = load_nerf_evaluator(
            nerf_evaluator_config=nerf_evaluator_config,
            nerf_evaluator_checkpoint=nerf_evaluator_checkpoint,
            console=console,
        )

        return cls(
            nerf_field,
            nerf_evaluator,
            nerf_evaluator_config.nerfdata_config.fingertip_config,
            X_N_Oy,
        )

    @property
    def top_K(self) -> int:
        return 5


def load_nerf_evaluator(
    nerf_evaluator_config: NerfEvaluatorModelConfig,
    nerf_evaluator_checkpoint: int = -1,
    console: Optional[Console] = None,
) -> NerfEvaluatorModel:
    # Load nerf_evaluator
    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        )
        if console is not None
        else nullcontext()
    ) as progress:
        task = (
            progress.add_task("Loading nerf_evaluator", total=1)
            if progress is not None
            else None
        )

        # (should device thing be here? probably since saved on gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nerf_evaluator = (
            nerf_evaluator_config.model_config.get_nerf_evaluator_from_fingertip_config(
                fingertip_config=nerf_evaluator_config.nerfdata_config.fingertip_config,
                n_tasks=nerf_evaluator_config.task_type.n_tasks,
            )
        ).to(device)

        # Load nerf_evaluator weights
        assert nerf_evaluator_config.checkpoint_workspace.output_dir.exists(), f"checkpoint_workspace.output_dir does not exist at {nerf_evaluator_config.checkpoint_workspace.output_dir}"
        print(
            f"Loading checkpoint ({nerf_evaluator_config.checkpoint_workspace.output_dir})..."
        )

        output_checkpoint_paths = (
            nerf_evaluator_config.checkpoint_workspace.output_checkpoint_paths
        )
        assert (
            len(output_checkpoint_paths) > 0
        ), f"No checkpoints found in {nerf_evaluator_config.checkpoint_workspace.output_checkpoint_paths}"
        assert (
            nerf_evaluator_checkpoint < len(output_checkpoint_paths)
        ), f"Requested checkpoint {nerf_evaluator_checkpoint} does not exist in {nerf_evaluator_config.checkpoint_workspace.output_checkpoint_paths}"
        checkpoint_path = output_checkpoint_paths[nerf_evaluator_checkpoint]

        checkpoint = torch.load(checkpoint_path)
        nerf_evaluator.load_state_dict(checkpoint["nerf_evaluator"])
        nerf_evaluator.load_state_dict(torch.load(checkpoint_path)["nerf_evaluator"])

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    return nerf_evaluator
