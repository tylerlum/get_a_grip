# Additional Details

## Frames and Conventions

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/e12abaef-cbbe-4cb4-ae0a-0a25a7935708" alt="Ny" style="width:30%;">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/2274a609-7348-484e-9751-32f3bd04a60a" alt="Oy" style="width:30%;">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/2861558c-436f-4669-8a77-d187d6e57705" alt="H" style="width:30%;">
</p>

### Isaac Gym World Y-up Frame (Wy Frame)

Isaac Gym is set up so that Y-axis is up and gravity points along -Y. An object with 0 translation and 0 rotation is shown below.

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/507903bd-1e6d-4ac2-b4c5-fe68588c95b1" alt="Wy" style="width:30%;">
</p>

### NeRF Y-up Frame (Ny Frame)

### Object Y-up Frame (Oy Frame)

### Isaacgym => NeRF Image Frames

Following [NeRF Studio data conventions](https://docs.nerf.studio/quickstart/data_conventions.html), we define camera orientation such that X = right, Y = up, Z = optical axis into camera. The translation xyz are the same as Isaac Gym coordinates.

Note that in some intermediate computation, we first compute the camera transform with the following orientation convention (before adding a rotation to modify it).

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/5486b518-2abf-4009-bd7b-0ff58a7736aa" alt="nerfstudio_frames" style="width:30%;">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/b2bf5405-a5b4-4c44-824e-ffe591e6c1d4" alt="intermediate_frames" style="width:27%;">
</p>

## Robot

This codebase currently only supports the Allegro hand, which has four fingers.

Allegro wrist pose frame:

- The Allegro Hand's origin is roughly at the base of its middle finger, with Y-axis (Green) along the direction from middle finger to index finger and Z-axis (Blue) along the direction from middle finger base to tip. An Allegro hand with 0 translation, 0 rotation, and 0 joint angles is shown below.

<p align="center">
  <img src="https://github.com/tylerlum/get_a_grip/assets/26510814/bc383cc4-3710-4187-ac5d-d30e0a361453" alt="allegro_frame" style="width:30%;">
</p>

Allegro urdf links:

- link 0 - 3 is fore finger (link 3 tip is fore fingertip)

- link 4 - 7 is middle finger (link 7 tip is middle fingertip)

- link 8 - 11 is ring finger (link 11 tip is ring fingertip)

- link 12 - 15 is thumb (link 15 tip is thumbtip)

Allegro urdf details:

- The urdf comes from [simlabrobotics/allegro_hand_ros_v4](https://github.com/simlabrobotics/allegro_hand_ros_v4/blob/master/src/allegro_hand_description/allegro_hand_description_right.urdf)

- We modify the urdf to have 6 "virtual joints" (one each for translation xyz an rotation RPY), which allows us to move the gripper in 6 DOF (used to move the gripper during grasp evaluation)

## DexGraspNet

Many major components of the dataset generation code originated from [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet). For additional information about mesh processing, other hand models, and more, please refer to their repository.