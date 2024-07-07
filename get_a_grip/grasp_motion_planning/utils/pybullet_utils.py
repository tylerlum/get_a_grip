from collections import defaultdict

import pybullet as pb


def get_link_com_xyz_orn(pb, body_id, link_id):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = pb.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = pb.getLinkState(
            body_id, link_id, computeForwardKinematics=1
        )
    return list(link_com), list(link_quat)


def create_primitive_shape(
    pb,
    mass,
    shape,
    dim,
    color=(0.6, 0, 0, 1),
    collidable=True,
    init_xyz=(0, 0, 0),
    init_quat=(0, 0, 0, 1),
):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == pb.GEOM_BOX:
        visual_shape_id = pb.createVisualShape(
            shapeType=shape, halfExtents=dim, rgbaColor=color
        )
        if collidable:
            collision_shape_id = pb.createCollisionShape(
                shapeType=shape, halfExtents=dim
            )
    elif shape == pb.GEOM_CYLINDER:
        visual_shape_id = pb.createVisualShape(
            shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color
        )
        if collidable:
            collision_shape_id = pb.createCollisionShape(
                shape, dim[0], [1, 1, 1], dim[1]
            )
    elif shape == pb.GEOM_SPHERE:
        visual_shape_id = pb.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, radius=dim[0])

    sid = pb.createMultiBody(
        baseMass=mass,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=init_xyz,
        baseOrientation=init_quat,
    )
    return sid


def draw_collision_spheres(robot, config: dict) -> None:
    link_names = {"world": -1}
    for i in range(pb.getNumJoints(robot)):
        link_names[pb.getJointInfo(robot, i)[12].decode("utf-8")] = i

    color_codes = [[1, 0, 0, 0.7], [0, 1, 0, 0.7]]

    if not hasattr(draw_collision_spheres, "cached_spheres"):
        draw_collision_spheres.cached_spheres = defaultdict(list)
        for i, link in enumerate(config["collision_spheres"].keys()):
            if link not in link_names:
                continue

            link_id = link_names[link]
            link_pos, link_ori = get_link_com_xyz_orn(pb, robot, link_id)
            for sphere in config["collision_spheres"][link]:
                s = create_primitive_shape(
                    pb,
                    0.0,
                    shape=pb.GEOM_SPHERE,
                    dim=(sphere["radius"],),
                    collidable=False,
                    color=color_codes[i % 2],
                )
                # Place the sphere relative to the link
                world_coord = list(
                    pb.multiplyTransforms(
                        link_pos, link_ori, sphere["center"], [0, 0, 0, 1]
                    )[0]
                )
                world_coord[1] += 0.0
                pb.resetBasePositionAndOrientation(s, world_coord, [0, 0, 0, 1])
                draw_collision_spheres.cached_spheres[link].append(s)
    else:
        cached_spheres = draw_collision_spheres.cached_spheres
        for i, link in enumerate(config["collision_spheres"].keys()):
            if link not in link_names:
                continue

            link_id = link_names[link]
            link_pos, link_ori = get_link_com_xyz_orn(pb, robot, link_id)
            for j, sphere in enumerate(config["collision_spheres"][link]):
                s = cached_spheres[link][j]

                # Place the sphere relative to the link
                world_coord = list(
                    pb.multiplyTransforms(
                        link_pos, link_ori, sphere["center"], [0, 0, 0, 1]
                    )[0]
                )
                world_coord[1] += 0.0
                pb.resetBasePositionAndOrientation(s, world_coord, [0, 0, 0, 1])


def remove_collision_spheres() -> None:
    if hasattr(draw_collision_spheres, "cached_spheres"):
        for link, spheres in draw_collision_spheres.cached_spheres.items():
            for s in spheres:
                pb.resetBasePositionAndOrientation(s, [100, 0, 0], [0, 0, 0, 1])
