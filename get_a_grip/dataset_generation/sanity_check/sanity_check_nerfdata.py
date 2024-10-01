from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.utils.parse_object_code_and_scale import is_object_code_and_scale_str


@dataclass
class Args:
    nerfdata_path: Path = get_data_folder() / "dataset/large/nerfdata"

    def __post_init__(self) -> None:
        assert self.nerfdata_path.exists(), f"{self.nerfdata_path} does not exist"


def sanity_check_nerfdata(args: Args) -> None:
    """
    nerfdata_path
    └── <object_code_and_scale>
        └── images
            └── <filename>.png
            └── <filename>.png
            └── ...
        └── transforms.json

            └── <datetime>
                └── nerfstudio_models
                    └── <filename>.ckpt
    └── ...
    """
    object_code_and_scale_paths = sorted(
        [
            x
            for x in args.nerfdata_path.iterdir()
            if is_object_code_and_scale_str(x.stem)
        ]
    )

    path_and_issue_list = []
    for object_code_and_scale_path in tqdm(object_code_and_scale_paths):
        if not (object_code_and_scale_path / "images").exists():
            path_and_issue_list.append(
                (object_code_and_scale_path, "images does not exist")
            )
            print(path_and_issue_list[-1])

        if not (object_code_and_scale_path / "transforms.json").exists():
            path_and_issue_list.append(
                (object_code_and_scale_path, "transforms.json does not exist")
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
    sanity_check_nerfdata(args)


if __name__ == "__main__":
    main()
