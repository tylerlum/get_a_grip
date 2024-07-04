import pathlib
import subprocess
from dataclasses import dataclass
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip.model_training.utils.nerf_load_utils import get_nerf_configs


@dataclass
class Args:
    nerfcheckpoints_path: pathlib.Path
    nerf_is_z_up: bool
    density_of_0_level_set: float = 15.0
    n_pts_each_dim_marching_cubes: int = 31
    rescale: bool = True
    min_num_edges: Optional[int] = 200
    output_dir_path: pathlib.Path = pathlib.Path(__file__).parent / "nerf_meshdata"
    add_1cm_vertical_offset: bool = False
    only_largest_component: bool = False


def print_and_run(cmd: str) -> None:
    print(cmd)
    subprocess.run(cmd, shell=True)


def nerf_to_urdf_all(args: Args) -> None:
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert (
        args.nerfcheckpoints_path.exists()
    ), f"{args.nerfcheckpoints_path} does not exist"
    object_nerfcheckpoint_folders = sorted(
        x for x in args.nerfcheckpoints_path.iterdir()
    )
    for object_nerfcheckpoint_folder in tqdm(
        object_nerfcheckpoint_folders, desc="object_nerfcheckpoint_folders"
    ):
        nerf_config = get_nerf_configs(object_nerfcheckpoint_folder)[-1]
        command = " ".join(
            [
                "python nerf_grasping/baselines/nerf_to_urdf.py",
                f"--nerfcheckpoint-filepath {nerf_config}",
                f"--nerf-is-z-up {args.nerf_is_z_up}",
                f"--density-of-0-level-set {args.density_of_0_level_set}",
                f"--n-pts-each-dim-marching-cubes {args.n_pts_each_dim_marching_cubes}",
                "--rescale" if args.rescale else "--no-rescale",
                f"--min-num-edges {args.min_num_edges}",
                f"--output-dir-path {str(args.output_dir_path)}",
                "--add-1cm-vertical-offset" if args.add_1cm_vertical_offset else "",
                "--only-largest-component" if args.only_largest_component else "",
            ]
        )
        print_and_run(command)


def main() -> None:
    args = tyro.cli(Args)
    nerf_to_urdf_all(args)


if __name__ == "__main__":
    main()
