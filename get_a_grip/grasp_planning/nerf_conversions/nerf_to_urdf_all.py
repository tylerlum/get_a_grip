import pathlib
from dataclasses import dataclass
from typing import Optional

import tyro
from tqdm import tqdm

from get_a_grip.grasp_planning.nerf_conversions.nerf_to_urdf import (
    NerfToUrdfArgs,
    nerf_to_urdf,
)
from get_a_grip.utils.nerf_load_utils import get_latest_nerf_config


@dataclass
class NerfToUrdfAllArgs:
    nerfcheckpoints_path: pathlib.Path
    nerf_is_z_up: bool
    density_of_0_level_set: float = 15.0
    n_pts_each_dim_marching_cubes: int = 31
    rescale: bool = True
    min_num_edges: Optional[int] = 200
    output_dir_path: pathlib.Path = pathlib.Path(__file__).parent / "nerf_meshdata"
    add_1cm_vertical_offset: bool = False
    only_largest_component: bool = False


def nerf_to_urdf_all(args: NerfToUrdfAllArgs) -> None:
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
        nerf_config = get_latest_nerf_config(object_nerfcheckpoint_folder)

        nerf_to_urdf(
            NerfToUrdfArgs(
                nerf_config=nerf_config,
                nerf_is_z_up=args.nerf_is_z_up,
                density_of_0_level_set=args.density_of_0_level_set,
                n_pts_each_dim_marching_cubes=args.n_pts_each_dim_marching_cubes,
                rescale=args.rescale,
                min_num_edges=args.min_num_edges,
                output_dir_path=args.output_dir_path,
                add_1cm_vertical_offset=args.add_1cm_vertical_offset,
                only_largest_component=args.only_largest_component,
            )
        )


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[NerfToUrdfAllArgs])
    nerf_to_urdf_all(args)


if __name__ == "__main__":
    main()
