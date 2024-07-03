from dataclasses import dataclass
from typing import Optional

import tyro


@dataclass(unsafe_hash=True)
class BaseFingertipConfig:
    # Dec 15, 2023:
    # Allegro fingertip width/height ~28mm (http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Dimensions_and_Directions)
    # Moving in fingers by ~30mm
    # Add some buffer so we can have gradients for improvement
    num_pts_x: Optional[int] = None
    num_pts_y: Optional[int] = None
    num_pts_z: Optional[int] = None
    finger_width_mm: float = 60.0
    finger_height_mm: float = 60.0
    grasp_depth_mm: float = 80.0
    n_fingers: int = 4


@dataclass(unsafe_hash=True)
class VanillaFingertipConfig(BaseFingertipConfig):
    pass


# Not frozen, since we need to modify the num_pts_x, num_pts_y, num_pts_z in the custom constructor.
@dataclass(unsafe_hash=True)
class EvenlySpacedFingertipConfig(BaseFingertipConfig):
    distance_between_pts_mm: float = 2.0

    def __init__(
        self,
        distance_between_pts_mm: float = 2,
        finger_width_mm: float = 60.0,
        finger_height_mm: float = 60.0,
        grasp_depth_mm: float = 80.0,
        num_pts_x: Optional[int] = None,
        num_pts_y: Optional[int] = None,
        num_pts_z: Optional[int] = None,
        n_fingers=4,
    ):
        self.distance_between_pts_mm = distance_between_pts_mm
        num_pts_x = int(finger_width_mm / distance_between_pts_mm) + 1
        num_pts_y = int(finger_height_mm / distance_between_pts_mm) + 1
        num_pts_z = int(grasp_depth_mm / distance_between_pts_mm) + 1

        super().__init__(
            num_pts_x,
            num_pts_y,
            num_pts_z,
            finger_width_mm=finger_width_mm,
            finger_height_mm=finger_height_mm,
            grasp_depth_mm=grasp_depth_mm,
            n_fingers=n_fingers,
        )


UnionFingertipConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "vanilla": VanillaFingertipConfig(20, 30, 40),
        "even": EvenlySpacedFingertipConfig(),
        "big_even": EvenlySpacedFingertipConfig(
            finger_width_mm=35,
            finger_height_mm=35,
            grasp_depth_mm=50,
            distance_between_pts_mm=2.5,
        ),
    }
)


if __name__ == "__main__":
    cfg = tyro.cli(UnionFingertipConfig)
    print(cfg)
