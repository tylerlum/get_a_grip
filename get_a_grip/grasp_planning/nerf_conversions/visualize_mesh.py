import pathlib
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import trimesh
import tyro


@dataclass
class VisualizeMeshArgs:
    obj_filepath: pathlib.Path
    opacity: float = 1.0


def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[VisualizeMeshArgs])

    # Load your .obj file
    mesh = trimesh.load(str(args.obj_filepath))

    # Extract vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Create a 3D Plotly figure
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=args.opacity,
            )
        ]
    )

    # Often in format <object_code>/coacd/decomposed.obj, but not always
    fig.update_layout(
        title=f"{'/'.join(args.obj_filepath.parts[-3:])}",
    )
    fig.show()


if __name__ == "__main__":
    main()
