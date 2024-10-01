import pathlib
from typing import Optional, Tuple

from curobo.geom.types import WorldConfig


def get_table_collision_dict() -> dict:
    return {
        "cuboid": {
            "table": {
                "dims": [1.3208, 1.8288, 0.02],
                "pose": [0.50165, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }


def get_object_collision_dict(
    file_path: pathlib.Path,
    xyz: Tuple[float, float, float],
    quat_wxyz: Tuple[float, float, float, float],
    obj_name: str = "object",
) -> dict:
    return {
        "mesh": {
            obj_name: {
                "pose": [*xyz, *quat_wxyz],
                "file_path": str(file_path),
            }
        }
    }


def get_dummy_collision_dict() -> dict:
    FAR_AWAY_POS = 10.0
    return {
        "cuboid": {
            "dummy": {
                "dims": [0.1, 0.1, 0.1],
                "pose": [FAR_AWAY_POS, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }


def get_world_cfg(
    collision_check_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = None,
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision_check_table: bool = True,
    obj_name: str = "object",
) -> WorldConfig:
    world_dict = {}
    if collision_check_table:
        world_dict.update(get_table_collision_dict())
    if collision_check_object and obj_filepath is not None:
        world_dict.update(
            get_object_collision_dict(
                file_path=obj_filepath,
                xyz=obj_xyz,
                quat_wxyz=obj_quat_wxyz,
                obj_name=obj_name,
            )
        )
    if len(world_dict) == 0:
        # Error if there are no objects, so add a dummy object
        world_dict.update(get_dummy_collision_dict())
    world_cfg = WorldConfig.from_dict(world_dict)
    return world_cfg
