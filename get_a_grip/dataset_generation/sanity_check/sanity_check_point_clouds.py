from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.utils.parse_object_code_and_scale import is_object_code_and_scale_str


@dataclass
class Args:
    point_clouds_path: Path = get_data_folder() / "dataset/large/point_clouds"

    def __post_init__(self) -> None:
        assert (
            self.point_clouds_path.exists()
        ), f"{self.point_clouds_path} does not exist"


def sanity_check_point_clouds(args: Args) -> None:
    """
    point_clouds_path
    └── <object_code_and_scale>
        └── point_cloud.ply
    └── ...
    """
    object_code_and_scale_paths = sorted(
        [
            x
            for x in args.point_clouds_path.iterdir()
            if is_object_code_and_scale_str(x.stem)
        ]
    )

    path_and_issue_list = []
    for object_code_and_scale_path in tqdm(object_code_and_scale_paths):
        if not (object_code_and_scale_path / "point_cloud.ply").exists():
            path_and_issue_list.append(
                (object_code_and_scale_path, "point_cloud.ply does not exist")
            )
            print(path_and_issue_list[-1])

    if len(path_and_issue_list) == 0:
        print("All good!")
        return

    print("\n" + "=" * 80)
    print(f"{len(path_and_issue_list)} issues found")
    print("=" * 80 + "\n")
    for path, issue in path_and_issue_list:
        print(f"{path}: {issue}")


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")
    sanity_check_point_clouds(args)


if __name__ == "__main__":
    main()
