# leap_hand_simplified

`leap_hand_right.urdf`:

- Copied from: https://github.com/dexsuite/dex-urdf/tree/main/robots/hands/leap_hand

- Could not use https://github.com/leap-hand/LEAP_Hand_Sim/tree/master/assets/leap_hand because we want primitive collision shapes as much as possible (otherwise penetration checking would be very slow or OOM GPU)

- Modified with the following changes:
  - Removed `base_link` and `base_joint`
  - Change `palm_lower` to `palm_link`
  - Current codebase expects each link to have at most 1 collision geometry (possible to refactor, but would be error-prone, so we're keeping it simple for now). Thus, for all links `my_link_name` that have multiple collision geometries, we've added additional links `my_link_name_collision_link_0`, `my_link_name_collision_link_1`, ... each with a single collision geometry.

`leap_hand_right_with_virtual_joints.urdf`:

- Copied `leap_hand_right.urdf`, then added 6 joints to control translation xyz and rotation RPY

`penetration_points.json`:

- Self-collision avoidance

`contact_points_precision_grasp.json`:

- Desired contact points with the object for precision grasps

- Created using `generate_allegro_contact_points_precision_grasp.py`
