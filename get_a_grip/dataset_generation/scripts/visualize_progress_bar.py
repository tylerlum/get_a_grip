import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm


@dataclass
class Args:
    path: Path
    """Path to directory to monitor"""
    total: int
    """Total number of folders to wait for"""
    time_interval_sec: float = 5
    """Time interval in seconds to check for new folders"""

    def __post_init__(self):
        assert self.path.is_dir(), f"{self.path} is not a directory"


def visualize_progress_bar(args: Args) -> None:
    prev_num_folders = len(list(args.path.iterdir()))
    pbar = tqdm(total=args.total, desc=f"{args.path}", dynamic_ncols=True)
    pbar.update(prev_num_folders)

    # Wait a bit for the progress bar to show up
    for i in range(10):
        pbar.update(0)
        time.sleep(0.1)

    while prev_num_folders < args.total:
        num_folders = len(list(args.path.iterdir()))
        pbar.update(num_folders - prev_num_folders)
        prev_num_folders = num_folders
        time.sleep(args.time_interval_sec)
    print("DONE!")


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[Args])
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    visualize_progress_bar(args)


if __name__ == "__main__":
    main()
