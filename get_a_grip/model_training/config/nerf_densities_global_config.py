import numpy as np

# HACK: Just some hardcoded values to reference, make into nice config later
(
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
) = (30, 30, 30)

lb_Oy = np.array([-0.15, -0.15, -0.15])
ub_Oy = np.array([0.15, 0.15, 0.15])
