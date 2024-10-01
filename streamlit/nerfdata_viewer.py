import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh
import tyro
from PIL import Image
from tqdm import tqdm

from get_a_grip.model_training.utils.plot_utils import (
    get_scene_dict,
    get_yup_camera,
    get_zup_camera,
)


@dataclass
class VisualizeNerfDataArgs:
    nerfdata_path: pathlib.Path
    meshdata_path: Optional[pathlib.Path] = None
    mesh_scale: float = 0.1
    max_num_cameras: int = 10
    show_background: bool = True
    show_grid: bool = True
    show_ticklabels: bool = True
    is_z_up: bool = False


def load_image(fpath: str, sz: int = 128) -> np.ndarray:
    img = Image.open(fpath)
    img = img.resize((sz, sz))
    return np.asarray(img)[:, :, :3]


def compute_fov_deg(
    fl_x: float, fl_y: float, img_w: int, img_h: int
) -> Tuple[float, float]:
    fov_x = 2 * np.arctan(img_w / (2 * fl_x))
    fov_y = 2 * np.arctan(img_h / (2 * fl_y))
    return np.rad2deg(fov_x), np.rad2deg(fov_y)


def load_nerf(
    root_path: pathlib.Path,
    max_num_cameras: Optional[int] = 10,
    randomize_camera_order_seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[str], List[str], List[Optional[str]], float, float]:
    poses = []
    legends = []
    colors = []
    image_paths = []

    pose_path = root_path / "transforms.json"
    print(f"Load poses from {pose_path}")

    with open(pose_path, "r") as fin:
        jdata = json.load(fin)

    frames = jdata["frames"]

    if randomize_camera_order_seed is not None:
        np.random.seed(randomize_camera_order_seed)
        np.random.shuffle(frames)

    for i, frm in tqdm(enumerate(frames), total=len(frames)):
        if max_num_cameras is not None and i >= max_num_cameras:
            break

        c2w = np.array(frm["transform_matrix"])
        poses.append(c2w)
        colors.append("blue")

        if "file_path" in frm:
            fpath = frm["file_path"]
            fname = os.path.basename(fpath)

            legends.append(fname)
            image_paths.append(root_path / fpath)
        else:
            legends.append(str(i))
            image_paths.append(None)

    fl_x, fl_y, img_w, img_h = jdata["fl_x"], jdata["fl_y"], jdata["w"], jdata["h"]

    fov_deg_x, fov_deg_y = compute_fov_deg(
        fl_x=fl_x, fl_y=fl_y, img_w=img_w, img_h=img_h
    )
    return (
        poses,
        legends,
        colors,
        image_paths,
        fov_deg_x,
        fov_deg_y,
    )


def encode_image(raw_image: np.ndarray) -> Tuple[Image.Image, List[List[Any]]]:
    dum_img = Image.fromarray(np.ones((3, 3, 3), dtype="uint8")).convert(
        "P", palette="WEB"
    )
    idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

    bit_image = Image.fromarray(raw_image).convert("P", palette="WEB", dither=None)

    colorscale = []
    for i in range(256):
        rgb = idx_to_color[i] if i < len(idx_to_color) else (0, 0, 0)
        colorscale.append([i / 255.0, f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"])

    return bit_image, colorscale


def calc_cam_cone_pts_3d(
    c2w: np.ndarray, fov_deg_x: float, fov_deg_y: float, cam_cone_scale: float = 0.1
) -> np.ndarray:
    fov_rad_x = np.deg2rad(fov_deg_x)
    fov_rad_y = np.deg2rad(fov_deg_y)

    cam_xyz = c2w[:3, -1]

    corners = [
        [np.tan(fov_rad_x / 2.0), np.tan(fov_rad_y / 2.0), -1.0],
        [-np.tan(fov_rad_x / 2.0), np.tan(fov_rad_y / 2.0), -1.0],
        [-np.tan(fov_rad_x / 2.0), -np.tan(fov_rad_y / 2.0), -1.0],
        [np.tan(fov_rad_x / 2.0), -np.tan(fov_rad_y / 2.0), -1.0],
        [0, np.tan(fov_rad_y / 2.0), -1.0],
    ]

    corners = [np.dot(c2w[:3, :3], corner) for corner in corners]

    # Now attach as offset to actual 3D camera position:
    corners = [
        (np.array(corner) / np.linalg.norm(corner, ord=2)) * cam_cone_scale
        for corner in corners
    ]
    corners = [cam_xyz + corner for corner in corners]

    xs = [cam_xyz[0]] + [corner[0] for corner in corners]
    ys = [cam_xyz[1]] + [corner[1] for corner in corners]
    zs = [cam_xyz[2]] + [corner[2] for corner in corners]

    return np.array([xs, ys, zs]).T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    N = points.shape[0]
    assert T.shape == (4, 4), f"T.shape: {T.shape}"
    assert points.shape == (N, 3), f"points.shape: {points.shape}"

    return np.matmul(np.concatenate([points, np.ones((N, 1))], axis=1), T.T)[:, :3]


def visualize_nerfdata(
    nerfdata_path: pathlib.Path,
    mesh: Optional[trimesh.Trimesh],
    max_num_cameras: Optional[int] = 10,
    randomize_camera_order_seed: Optional[int] = None,
    show_background: bool = True,
    show_grid: bool = True,
    show_ticklabels: bool = True,
    show_legend: bool = True,
    is_z_up: bool = False,
    title: str = "NeRF Data",
) -> go.Figure:
    poses, legends, colors, image_paths, fov_deg_x, fov_deg_y = load_nerf(
        nerfdata_path,
        max_num_cameras=max_num_cameras,
        randomize_camera_order_seed=randomize_camera_order_seed,
    )

    images = [load_image(str(fpath)) if fpath else None for fpath in image_paths]

    bit_images, image_colorscale = zip(*(encode_image(img) for img in images))

    fig = go.Figure()

    if mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color="lightpink",
                opacity=0.8,
                lighting={"ambient": 1},
            )
        )

    for pose, color, legend, bit_image, colorscale in zip(
        poses, colors, legends, bit_images, image_colorscale
    ):
        cone = calc_cam_cone_pts_3d(c2w=pose, fov_deg_x=fov_deg_x, fov_deg_y=fov_deg_y)

        W, H = bit_image.size

        CAMERA_IMAGE_DISTANCE = 0.0
        z = np.zeros((W, H)) + CAMERA_IMAGE_DISTANCE
        (x, y) = np.meshgrid(
            np.linspace(-0.1, 0.1, W),
            np.linspace(0.1, -0.1, H) * H / W,
        )

        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        assert xyz.shape == (W, H, 3), f"xyz.shape: {xyz.shape}"
        rot_xyz = transform_points(T=pose, points=xyz.reshape(W * H, 3)).reshape(
            W, H, 3
        )
        x, y, z = rot_xyz[:, :, 0], rot_xyz[:, :, 1], rot_xyz[:, :, 2]

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=bit_image,
                cmin=0,
                cmax=255,
                colorscale=colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3,
            )
        )

        # Have redundant edges to make the lines continuous so it can all be one trace
        edges = [0, 1, 2, 0, 3, 2, 0, 4, 3, 4, 1, 0, 5]
        fig.add_trace(
            go.Scatter3d(
                x=[cone[edge, 0] for edge in edges],
                y=[cone[edge, 1] for edge in edges],
                z=[cone[edge, 2] for edge in edges],
                mode="lines",
                line=dict(color=color, width=3),
                name=legend,
                showlegend=show_legend,
            )
        )

        # Add label.
        TEXT_CAMERA_DISTANCE = 0.05
        text_pos_C = np.array([0, 0, TEXT_CAMERA_DISTANCE])
        text_pos_W = transform_points(T=pose, points=text_pos_C[None, :]).squeeze(
            axis=0
        )

        fig.add_trace(
            go.Scatter3d(
                x=[text_pos_W[0]],
                y=[text_pos_W[1]],
                z=[text_pos_W[2]],
                showlegend=False,
                mode="text",
                text=legend,
                textposition="middle center",
            )
        )

    # look at the center of scene
    fig.update_layout(
        autosize=True,
        hovermode=False,
        # margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(
                showticklabels=show_ticklabels,
                showgrid=show_grid,
                zeroline=False,
                showbackground=show_background,
                showspikes=False,
                showline=False,
                ticks="",
            ),
            yaxis=dict(
                showticklabels=show_ticklabels,
                showgrid=show_grid,
                zeroline=False,
                showbackground=show_background,
                showspikes=False,
                showline=False,
                ticks="",
            ),
            zaxis=dict(
                showticklabels=show_ticklabels,
                showgrid=show_grid,
                zeroline=False,
                showbackground=show_background,
                showspikes=False,
                showline=False,
                ticks="",
            ),
        ),
        title=title,
    )
    fig.update_layout(
        scene=get_scene_dict(),
        scene_camera=get_zup_camera() if is_z_up else get_yup_camera(),
    )
    return fig


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeNerfDataArgs])
    print("=" * 80)
    print(f"Arguments: {args}")
    print("=" * 80 + "\n")

    assert args.nerfdata_path.exists(), f"{args.nerfdata_path} does not exist"

    if args.meshdata_path is not None:
        assert args.meshdata_path.exists(), f"{args.meshdata_path} does not exist"
        mesh = trimesh.load_mesh(str(args.meshdata_path))
        mesh.apply_scale(args.mesh_scale)
    else:
        mesh = None

    fig = visualize_nerfdata(
        nerfdata_path=args.nerfdata_path,
        mesh=mesh,
        max_num_cameras=args.max_num_cameras,
        show_background=args.show_background,
        show_grid=args.show_grid,
        show_ticklabels=args.show_ticklabels,
        is_z_up=args.is_z_up,
    )
    fig.show()


if __name__ == "__main__":
    main()
