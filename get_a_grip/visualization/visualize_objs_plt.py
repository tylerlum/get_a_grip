import trimesh
import numpy as np
import math
import pathlib
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from dataclasses import dataclass
import tyro


@dataclass
class VisualizeObjsPltArgs:
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_v2")
    max_num_objects_to_visualize: int = 10


def load_obj_mesh(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return vertices, faces


def main() -> None:
    args = tyro.cli(VisualizeObjsPltArgs)
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
    fig = plt.figure(figsize=(15, 15))
    for i, obj_file in enumerate(obj_files):
        vertices, faces = load_obj_mesh(obj_file)
        triang = Triangulation(vertices[:, 0], vertices[:, 1], faces)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(
            triang, vertices[:, 2], edgecolor="k", linewidth=0.5, antialiased=True
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        ax.set_title(obj_file.parent.parent.name)

    fig.suptitle(f"3D Models in {args.meshdata_root_path.name}")
    plt.show()


if __name__ == "__main__":
    main()
