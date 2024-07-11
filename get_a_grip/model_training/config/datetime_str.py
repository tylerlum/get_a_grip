from datetime import datetime
from functools import lru_cache


@lru_cache
def get_datetime_str() -> str:
    # Get datetime str that is fixed for a given run
    # A general date-time string for naming runs -- shared across all config modules.
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
