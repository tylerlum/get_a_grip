# allegro_hand_description

`allegro_hand_description_right.urdf`:

- Copied from: https://github.com/simlabrobotics/allegro_hand_ros_v4/tree/master/src/allegro_hand_description

`allegro_hand_description_right_with_virtual_joints.urdf`:

- Copied `allegro_hand_description_right.urdf`, then added 6 joints to control translation xyz and rotation RPY

`penetration_points.json`:

- Self-collision avoidance

`contact_points_precision_grasp.json`:

- Desired contact points with the object for precision grasps

- Created using `generate_allegro_contact_points_precision_grasp.py`
