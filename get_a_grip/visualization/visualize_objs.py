import math
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh
import tyro
from plotly.subplots import make_subplots

from get_a_grip import get_data_folder


@dataclass
class VisualizeObjsArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "large/meshes"
    max_num_objects_to_visualize: int = 10


def load_obj_mesh(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return vertices, faces


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
    args = tyro.cli(VisualizeObjsArgs)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert (
        args.meshdata_root_path.exists()
    ), f"args.meshdata_root_path {args.meshdata_root_path} does not exist"

    obj_files = sorted(
        [
            path / "coacd" / "decomposed.obj"
            for path in args.meshdata_root_path.iterdir()
        ]
    )
    obj_files = obj_files[: args.max_num_objects_to_visualize]

    # Create subplots
    N = len(obj_files)
    nrows = math.ceil(math.sqrt(N))
    ncols = math.ceil(N / nrows)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "mesh3d"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=[obj_file.parent.parent.name for obj_file in obj_files],
    )

    # Add each obj file to the subplot
    for index, obj_file in enumerate(obj_files):
        row, col = divmod(index, ncols)
        vertices, faces = load_obj_mesh(obj_file)
        fig.add_trace(
            create_mesh_3d(vertices, faces),
            row=row + 1,
            col=col + 1,
        )

    # Update layout and show figure
    fig.update_layout(
        title=f"3D Models in {args.meshdata_root_path.name}", showlegend=False
    )
    fig.show()


if __name__ == "__main__":
    main()
