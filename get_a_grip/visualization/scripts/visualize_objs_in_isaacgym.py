"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


1080 balls of solitude
-------------------------
Demonstrates the use of collision filtering to limit collisions to actors within an environment,
simulate all collisions including between actors in different environments, or simulate no collisions between
actors - they will still collide with the ground plane.

Modes can be set via command line arguments:
    --no_collisions to have no actors colide with other actors
    --all_collisions to have all actors, even those from different environments, collide

Press 'R' to reset the  simulation
"""

import pathlib
import random
from math import sqrt
from typing import List, Literal, Tuple

import numpy as np
from isaacgym import gymapi, gymutil
from tqdm import tqdm

from get_a_grip import get_assets_folder, get_data_folder
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_FILE,
)
from get_a_grip.dataset_generation.utils.leap_hand_info import (
    LEAP_HAND_FILE,
)


def sample_object_codes_and_scales(
    max_n_objects: int,
    meshdata_root_path: pathlib.Path,
) -> Tuple[List[str], List[float]]:
    # urdf_paths
    assert (
        meshdata_root_path.exists()
    ), f"Meshdata root path {meshdata_root_path} does not exist"

    all_object_codes = [x.name for x in meshdata_root_path.iterdir() if x.is_dir()]
    filtered_object_codes = [
        object_code
        for object_code in all_object_codes
        if "pistol" not in object_code and "gun" not in object_code
    ]
    n_objects = (
        max_n_objects
        if len(filtered_object_codes) >= max_n_objects
        else len(filtered_object_codes)
    )

    selected_object_codes = random.sample(filtered_object_codes, k=n_objects)

    scales = [random.uniform(0.05, 0.1) for _ in range(n_objects)]
    return selected_object_codes, scales


def get_urdf_paths(object_codes: List[str], meshdata_root_path: pathlib.Path) -> list:
    # urdf_paths
    assert (
        meshdata_root_path.exists()
    ), f"Meshdata root path {meshdata_root_path} does not exist"

    selected_object_paths = [
        meshdata_root_path / object_code
        for object_code in object_codes
        if (meshdata_root_path / object_code).exists()
    ]
    print(f"Expected {len(object_codes)} object_paths")
    print(f"Selected {len(selected_object_paths)} object_paths")

    selected_urdf_paths = []
    for x in tqdm(selected_object_paths, desc="Finding URDFs"):
        urdf_path = x / "coacd" / "coacd.urdf"
        if not urdf_path.exists():
            print(f"WARNING: {urdf_path} does not exist")
            continue

        selected_urdf_paths.append(urdf_path)
    print(f"Found {len(selected_urdf_paths)} urdf_paths")
    return selected_urdf_paths


def load_assets(gym, sim, selected_urdf_paths: List[pathlib.Path]) -> list:
    # Asset options
    obj_asset_options = gymapi.AssetOptions()
    obj_asset_options.override_com = True
    obj_asset_options.override_inertia = True
    obj_asset_options.density = 500
    obj_asset_options.vhacd_enabled = (
        False  # Don't need convex decomposition for visualization only
    )
    obj_asset_options.fix_base_link = True

    assets = [
        gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, obj_asset_options)
        for urdf_path in tqdm(selected_urdf_paths, desc="Loading assets")
    ]
    return assets


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {
            "name": "--all_collisions",
            "action": "store_true",
            "help": "Simulate all collisions",
        },
        {
            "name": "--no_collisions",
            "action": "store_true",
            "help": "Ignore all collisions",
        },
        {
            "name": "--meshdata_root_path",
            "type": pathlib.Path,
            "help": "Path to meshdata root",
            "default": get_data_folder() / "meshdata",
        },
    ],
)

# Get urdf paths
object_codes, scales = sample_object_codes_and_scales(
    max_n_objects=100,
    meshdata_root_path=args.meshdata_root_path,
)
urdf_paths = get_urdf_paths(
    object_codes=object_codes, meshdata_root_path=args.meshdata_root_path
)

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
assets = load_assets(gym=gym, sim=sim, selected_urdf_paths=urdf_paths)

num_envs = len(assets)

# set lighting
light_index = 0
intensity = gymapi.Vec3(0.75, 0.75, 0.75)
ambient = gymapi.Vec3(0.75, 0.75, 0.75)
direction = gymapi.Vec3(0.0, 0.0, -1.0)
gym.set_light_parameters(sim, light_index, intensity, ambient, direction)

# set up the env grid
num_per_row = int(sqrt(num_envs))
env_spacing = 0.15
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
np.random.seed(17)

colors = [
    (242 / 255, 139 / 255, 130 / 255),  # Light Coral
    (251 / 255, 188 / 255, 4 / 255),  # Light Orange
    (255 / 255, 244 / 255, 117 / 255),  # Light Yellow
    (204 / 255, 255 / 255, 144 / 255),  # Light Green
    (167 / 255, 255 / 255, 235 / 255),  # Light Cyan
    (203 / 255, 240 / 255, 248 / 255),  # Light Sky Blue
    (174 / 255, 203 / 255, 250 / 255),  # Light Blue
    (215 / 255, 174 / 255, 251 / 255),  # Light Purple
    (253 / 255, 207 / 255, 232 / 255),  # Light Pink
    (230 / 255, 201 / 255, 168 / 255),  # Light Brown
]
# colors = [
#     (1.0, 0.0, 0.0),  # Red
#     (1.0, 0.5, 0.0),  # Orange
#     (1.0, 1.0, 0.0),  # Yellow
#     (0.5, 1.0, 0.0),  # Lime
#     (0.0, 1.0, 0.0),  # Green
#     (0.0, 1.0, 0.5),  # Spring Green
#     (0.0, 1.0, 1.0),  # Cyan
#     (0.0, 0.5, 1.0),  # Ocean
#     (0.0, 0.0, 1.0),  # Blue
#     (0.5, 0.0, 1.0),  # Violet
# ]

for i in range(num_envs):
    asset = assets[i]

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # create ball pyramid
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    pose.p = gymapi.Vec3(0, scales[i] * 1.02, 0)

    # Set up collision filtering.
    if args.all_collisions:
        # Everything should collide.
        # Put all actors in the same group, with filtering mask set to 0 (no filtering).
        collision_group = 0
        collision_filter = 0

    elif args.no_collisions:
        # Nothing should collide.
        # Use identical filtering masks for all actors to filter collisions between them.
        # Group assignment doesn't matter in this case.
        # Alternative would be to put each actor in a different group.
        collision_group = 0
        collision_filter = 1

    else:
        # Balls in the same env should collide, but not with balls from different envs.
        # Use one group per env, and filtering masks set to 0.
        collision_group = i
        collision_filter = 0

    ahandle = gym.create_actor(
        env, asset, pose, None, collision_group, collision_filter
    )
    gym.set_actor_scale(env, ahandle, scales[i])
    num_rbs = gym.get_actor_rigid_body_count(env, ahandle)

    COLOR_OR_TEXTURE = "COLOR"

    if COLOR_OR_TEXTURE == "COLOR":
        row = i // num_per_row
        color = gymapi.Vec3(colors[row][0], colors[row][1], colors[row][2])

        for i in range(num_rbs):
            gym.set_rigid_body_color(
                env, ahandle, i, gymapi.MESH_VISUAL_AND_COLLISION, color
            )
    elif COLOR_OR_TEXTURE == "TEXTURE":
        # Set table texture
        table_texture = gym.create_texture_from_file(sim, "table/wood.png")
        for i in range(num_rbs):
            gym.set_rigid_body_texture(
                env, ahandle, i, gymapi.MESH_VISUAL_AND_COLLISION, table_texture
            )
    else:
        raise ValueError(f"Invalid COLOR_OR_TEXTURE: {COLOR_OR_TEXTURE}")


# Add hand
hand_asset_options = gymapi.AssetOptions()
hand_asset_options.disable_gravity = True
hand_asset_options.collapse_fixed_joints = True
hand_asset_options.fix_base_link = True

# No virtual joints so it won't move
hand_root = str(get_assets_folder())

hand_to_show: Literal["ALLEGRO", "LEAP"] = "LEAP"
if hand_to_show == "ALLEGRO":
    hand_file = ALLEGRO_HAND_FILE
elif hand_to_show == "LEAP":
    hand_file = LEAP_HAND_FILE
else:
    raise ValueError(f"Invalid hand_to_show: {hand_to_show}")
hand_asset = gym.load_asset(sim, hand_root, hand_file, hand_asset_options)

hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(-0.25, 0.1, 0)
IDX = num_envs // 2
hand_handle = gym.create_actor(
    envs[IDX], hand_asset, hand_pose, None, IDX, collision_filter
)

cam_pos = gymapi.Vec3(-2, 2, 0)
cam_target = gymapi.Vec3(1.5, 0, 0)
gym.viewer_camera_look_at(viewer, envs[IDX], cam_pos, cam_target)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


step_idx = 0
while not gym.query_viewer_has_closed(viewer):
    step_idx += 1

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    if step_idx % 10000 == 0:
        print(f"Step: {step_idx}")
        gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
