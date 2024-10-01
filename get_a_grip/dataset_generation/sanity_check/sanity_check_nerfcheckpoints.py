from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.utils.parse_object_code_and_scale import is_object_code_and_scale_str


@dataclass
class Args:
    nerfcheckpoints_path: Path = get_data_folder() / "dataset/large/nerfcheckpoints"

    def __post_init__(self) -> None:
        assert (
            self.nerfcheckpoints_path.exists()
        ), f"{self.nerfcheckpoints_path} does not exist"


def sanity_check_nerfcheckpoints(args: Args) -> None:
    """
    nerfcheckpoints_path
    └── <object_code_and_scale>
        └── nerfacto
            └── <datetime>
                └── nerfstudio_models
                    └── <filename>.ckpt
    └── ...
    """
    object_code_and_scale_paths = sorted(
        [
            x
            for x in args.nerfcheckpoints_path.iterdir()
            if is_object_code_and_scale_str(x.stem)
        ]
    )

    path_and_issue_list = []
    for object_code_and_scale_path in tqdm(object_code_and_scale_paths):
        nerfacto_path = object_code_and_scale_path / "nerfacto"

        if not nerfacto_path.exists():
            path_and_issue_list.append((nerfacto_path, "nerfacto does not exist"))
            print(path_and_issue_list[-1])
            continue

        datetime_paths = list(nerfacto_path.iterdir())
        if len(datetime_paths) == 0:
            path_and_issue_list.append((nerfacto_path, "No datetime folders"))
            print(path_and_issue_list[-1])
            continue

        if len(datetime_paths) > 1:
            path_and_issue_list.append(
                (nerfacto_path, f"Multiple datetime folders: {datetime_paths}")
            )
            print(path_and_issue_list[-1])
            continue

        for datetime_path in datetime_paths:
            nerfstudio_models_path = datetime_path / "nerfstudio_models"
            if not nerfstudio_models_path.exists():
                path_and_issue_list.append(
                    (datetime_path, "nerfstudio_models does not exist")
                )
                print(path_and_issue_list[-1])
                continue

            ckpt_files = [
                x for x in nerfstudio_models_path.iterdir() if x.suffix == ".ckpt"
            ]
            if len(ckpt_files) == 0:
                path_and_issue_list.append((nerfstudio_models_path, "No ckpt files"))
                print(path_and_issue_list[-1])
                continue

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
    sanity_check_nerfcheckpoints(args)


if __name__ == "__main__":
    main()
