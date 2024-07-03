from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Literal

# A general date-time string for naming runs -- shared across all config modules.
CONFIG_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


@dataclass(frozen=True)
class WandbConfig:
    """Parameters for logging to wandb."""

    project: str
    """Name of the wandb project."""

    entity: Optional[str] = None
    """Account associated with the wandb project."""

    name: str = CONFIG_DATETIME_STR
    """Name of the run."""

    group: Optional[str] = None
    """Name of the run group."""

    job_type: Optional[str] = None
    """Name of the job type."""

    resume: Literal["allow", "never"] = "never"
    """Whether to allow wandb to resume a previous run."""