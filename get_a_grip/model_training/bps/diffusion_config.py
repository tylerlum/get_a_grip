from dataclasses import dataclass, field
import datetime
import pathlib


@dataclass
class DataConfig:
    num_workers: int = 0
    n_pts: int = 4096  # Number of points in bps (from DexDiffuser)
    grasp_dim: int = 3 + 6 + 16 + 4 * 3 # Grasp xyz + rot6d + joint angles + grasp directions


@dataclass
class ModelConfig:
    var_type: str = "fixedlarge"
    ema_rate: float = 0.9999
    ema: bool = True


@dataclass
class DiffusionConfig:
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000


@dataclass
class TrainingConfig:
    batch_size: int = 16384 * 2
    n_epochs: int = 20000
    print_freq: int = 10
    snapshot_freq: int = 100
    log_path: pathlib.Path = pathlib.Path(f"logs/dexdiffuser_sampler/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    # log_path: pathlib.Path = pathlib.Path(f"logs/dexdiffuser_sampler/stable_jun2")  # [vulcan] first one trained, tylers arch, not converged, no_noisy

@dataclass
class OptimConfig:
    weight_decay: float = 0.000
    optimizer: str = "Adam"
    lr: float = 0.0002
    beta1: float = 0.9
    amsgrad: bool = False
    eps: float = 0.00000001
    grad_clip: float = 1.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wandb_log: bool = True
    multigpu: bool = True
    use_nerf_sampler: bool = False  # DexSampler or NeRFSampler
