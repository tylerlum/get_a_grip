from dataclasses import dataclass

import tyro


@dataclass
class EvenlySpacedFingertipConfig:
    # Allegro fingertip width/height ~28mm (http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Dimensions_and_Directions)
    # Moving in fingers by ~30mm
    # Add some buffer so we can have gradients for improvement
    distance_between_pts_mm: float = 2.0
    finger_width_mm: float = 60.0
    finger_height_mm: float = 60.0
    grasp_depth_mm: float = 80.0
    n_fingers: int = 4

    @property
    def num_pts_x(self) -> int:
        return int(self.finger_width_mm / self.distance_between_pts_mm) + 1

    @property
    def num_pts_y(self) -> int:
        return int(self.finger_height_mm / self.distance_between_pts_mm) + 1

    @property
    def num_pts_z(self) -> int:
        return int(self.grasp_depth_mm / self.distance_between_pts_mm) + 1


if __name__ == "__main__":
    cfg = tyro.cli(tyro.conf.FlagConversionOff[EvenlySpacedFingertipConfig])
    print(cfg)
