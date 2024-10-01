import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from get_a_grip.grasp_planning.nerf_conversions.nerf_to_mesh import nerf_to_mesh
from get_a_grip.utils.nerf_load_utils import load_nerf_field
from get_a_grip.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)

# Hardcoded intentionally for consistency
NERF_TO_MESH_MIN_HEIGHT = 0
NERF_TO_MESH_MAX_HEIGHT = 0.3
NERF_TO_MESH_MIN_WIDTH = -0.2
NERF_TO_MESH_MAX_WIDTH = 0.2


def nerf_to_mesh_bounding_box_min(nerf_is_z_up: bool) -> np.ndarray:
    min_height = NERF_TO_MESH_MIN_HEIGHT
    min_width = NERF_TO_MESH_MIN_WIDTH

    if nerf_is_z_up:
        min_z = min_height

        min_x = min_width
        min_y = min_width
    else:
        min_y = min_height

        min_x = min_width
        min_z = min_width
    return np.array([min_x, min_y, min_z])


def nerf_to_mesh_bounding_box_max(nerf_is_z_up: bool) -> np.ndarray:
    max_height = NERF_TO_MESH_MAX_HEIGHT
    max_width = NERF_TO_MESH_MAX_WIDTH

    if nerf_is_z_up:
        max_z = max_height

        max_x = max_width
        max_y = max_width
    else:
        max_y = max_height

        max_x = max_width
        max_z = max_width
    return np.array([max_x, max_y, max_z])


@dataclass
class NerfToUrdfArgs:
    nerf_config: pathlib.Path
    nerf_is_z_up: bool
    density_of_0_level_set: float = 15.0
    n_pts_each_dim_marching_cubes: int = 31
    rescale: bool = True
    min_num_edges: Optional[int] = 200
    output_dir_path: pathlib.Path = pathlib.Path(__file__).parent / "nerf_meshdata"
    add_1cm_vertical_offset: bool = False
    only_largest_component: bool = False

    @property
    def lb(self) -> np.ndarray:
        min_height = 0.01 if self.add_1cm_vertical_offset else 0.0
        if self.nerf_is_z_up:
            return np.array([-0.2, -0.2, min_height])
        else:
            return np.array([-0.2, min_height, -0.2])

    @property
    def ub(self) -> np.ndarray:
        if self.nerf_is_z_up:
            return np.array([0.2, 0.2, 0.3])
        else:
            return np.array([0.2, 0.3, 0.2])


def create_urdf(
    obj_path: pathlib.Path,
    output_urdf_filename: str,
    ixx: float = 0.1,
    iyy: float = 0.1,
    izz: float = 0.1,
) -> pathlib.Path:
    assert obj_path.exists(), f"{obj_path} does not exist"
    assert output_urdf_filename.endswith(
        ".urdf"
    ), f"{output_urdf_filename} does not end with .urdf"
    output_folder = obj_path.parent
    obj_filename = obj_path.name

    urdf_content = f"""<robot name="root">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <material name="">
        <color rgba="0.75 0.75 0.75 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""

    output_urdf_path = output_folder / output_urdf_filename
    assert not output_urdf_path.exists(), f"{output_urdf_path} already exists"
    with open(output_urdf_path, "w") as urdf_file:
        urdf_file.write(urdf_content)
    return output_urdf_path


def nerf_to_urdf(args: NerfToUrdfArgs) -> Tuple[pathlib.Path, pathlib.Path]:
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert args.nerf_config.exists(), f"{args.nerf_config} does not exist"
    assert (
        args.nerf_config.name == "config.yml"
    ), f"{args.nerf_config} is not a config.yml file"
    assert (
        args.nerf_config.parent.parent.name == "nerfacto"
    ), f"{args.nerf_config.parent.parent.name} should be nerfacto"
    # Eg. path=data/2023-10-13_13-12-28/nerfcheckpoints/sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98_0_1076/nerfacto/2023-10-13_131849/config.yml
    # object_code_and_scale=sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98_0_1076
    object_code_and_scale = args.nerf_config.parent.parent.parent.name
    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale)

    nerf_field = load_nerf_field(args.nerf_config)
    lb = args.lb
    ub = args.ub

    # Should match existing meshdata folder structure
    # <output_dir_path>
    # └── <object_code>
    #     └── coacd
    #         ├── coacd.urdf
    #         └── decomposed.obj
    output_folder = args.output_dir_path / object_code / "coacd"
    if output_folder.exists():
        print(f"WARNING: {output_folder} already exists, skipping")
        return output_folder / "decomposed.obj", output_folder / "coacd.urdf"
    output_folder.mkdir(exist_ok=False, parents=True)

    obj_path = output_folder / "decomposed.obj"
    scale = 1.0 / object_scale if args.rescale else 1.0
    nerf_to_mesh(
        nerf_field,
        level=args.density_of_0_level_set,
        npts=args.n_pts_each_dim_marching_cubes,
        lb=lb,
        ub=ub,
        scale=scale,
        min_len=args.min_num_edges,
        save_path=obj_path,
        only_largest_component=args.only_largest_component,
    )

    urdf_path = create_urdf(obj_path=obj_path, output_urdf_filename="coacd.urdf")

    assert urdf_path.exists(), f"{urdf_path} does not exist"
    assert obj_path.exists(), f"{obj_path} does not exist"
    print(f"Created {urdf_path} and {obj_path}")
    return obj_path, urdf_path


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[NerfToUrdfArgs])
    nerf_to_urdf(args)


if __name__ == "__main__":
    main()
