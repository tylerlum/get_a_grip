from dataclasses import dataclass
from typing import Optional

import numpy as np

from get_a_grip.model_training.config.fingertip_config import UnionFingertipConfig


@dataclass
class CameraConfig:
    """
    Container for all camera intrinsics.
    """

    H: int
    """Image height in pixels."""

    W: int
    """Image width in pixels."""

    cx: Optional[float] = None
    """Image center in x, in pixels."""

    cy: Optional[float] = None
    """Image center in y, in pixels."""

    fx: Optional[float] = None
    """Focal length in x, in pixels."""

    fy: Optional[float] = None
    """Focal length in y, in pixels."""

    def set_intrisics_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ):
        """
        Convenience method creating a CameraIntrinsics object
        from a desired frustum defined by a fingertip_config.
        """
        # First, check if cx, cy are set. If not, set them to the center of the image.
        if self.cx is None:
            self.cx = self.W / 2
        if self.cy is None:
            self.cy = self.H / 2

        w_frustum_m, h_frustum_m = (
            fingertip_config.finger_width_mm / 1000,
            fingertip_config.finger_height_mm / 1000,
        )

        assert np.isclose(
            self.H / self.W, h_frustum_m / w_frustum_m
        ), "Aspect ratios don't match!"

        z_frustum_m = fingertip_config.grasp_depth_mm / 1000

        # Compute focal length in pixels.
        if self.fx is None:
            self.fx = z_frustum_m * self.W / w_frustum_m
        if self.fy is None:
            self.fy = z_frustum_m * self.H / h_frustum_m
