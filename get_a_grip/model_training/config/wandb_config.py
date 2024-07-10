from dataclasses import dataclass
from typing import Literal, Optional

from get_a_grip.model_training.config.datetime_str import get_datetime_str


@dataclass(frozen=True)
class WandbConfig:
    """Parameters for logging to wandb."""

    project: str
    """Name of the wandb project."""

    entity: Optional[str] = None
    """Account associated with the wandb project."""

    name: str = get_datetime_str()
    """Name of the run."""

    group: Optional[str] = None
    """Name of the run group."""

    job_type: Optional[str] = None
    """Name of the job type."""

    resume: Literal["allow", "never"] = "never"
    """Whether to allow wandb to resume a previous run."""
