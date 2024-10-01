import pathlib
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import trimesh
import tyro
from plotly.subplots import make_subplots

from get_a_grip import get_data_folder


@dataclass
class VisualizeMeshCompareToOriginalArgs:
    obj_filepath: pathlib.Path
    opacity: float = 1.0
    original_meshdata_dir_path: pathlib.Path = get_data_folder() / "meshdata"


def create_mesh_3d(
    vertices: np.ndarray, faces: np.ndarray, opacity: float = 1.0
) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
    )


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeMeshCompareToOriginalArgs])
    assert args.obj_filepath.exists(), f"{args.obj_filepath} does not exist"
    assert (
        args.original_meshdata_dir_path.exists()
    ), f"{args.original_meshdata_dir_path} does not exist"

    assert args.obj_filepath.name == "decomposed.obj"
    assert args.obj_filepath.parent.name == "coacd"
    object_code = args.obj_filepath.parent.parent.name

    original_mesh_filepath = (
        args.original_meshdata_dir_path / object_code / "coacd" / "decomposed.obj"
    ).resolve()
    assert original_mesh_filepath.exists(), f"{original_mesh_filepath} does not exist"

    # Load your .obj file
    mesh = trimesh.load(str(args.obj_filepath))
    original_mesh = trimesh.load(str(original_mesh_filepath))

    # Extract vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    original_vertices = np.array(original_mesh.vertices)
    original_faces = np.array(original_mesh.faces)

    # Create a 3D Plotly figure
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
        subplot_titles=(
            f"{object_code}",
            f"{object_code} (original)",
        ),
    )
    fig.add_trace(
        create_mesh_3d(vertices, faces, opacity=args.opacity),
        row=1,
        col=1,
    )
    fig.add_trace(
        create_mesh_3d(original_vertices, original_faces, opacity=args.opacity),
        row=1,
        col=2,
    )
    fig.show()


if __name__ == "__main__":
    main()
