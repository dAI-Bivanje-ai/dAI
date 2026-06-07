import numpy as np
import matplotlib.pyplot as plt
from voxel_grid import VoxelGrid


def visualize_voxels(grid: VoxelGrid, title: str = "Vokselski model") -> None:
    """
    Prikaže snovne voksle (oznaka 1) v 3D z matplotlib.
    Barvna shema skladna z visualizer.py (svetlo modra / temno siva).
    Args:
        grid:  VoxelGrid — mreža po voxelize_interior (labels vsebuje {0, 1})
        title: str       — naslov okna
    Returns:
        None
    """
    filled = grid.labels == 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    facecolors = np.where(
        filled[..., np.newaxis], [0.71, 0.83, 0.96, 0.4], [0, 0, 0, 0]
    )
    edgecolors = np.where(
        filled[..., np.newaxis], [0.27, 0.27, 0.25, 0.3], [0, 0, 0, 0]
    )

    ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)

    # osi v mm (pretvori iz indeksov v svetovne koordinate)
    nx, ny, nz = grid.shape
    x0, y0, z0 = grid.origin
    vs = grid.voxel_size

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([nx, ny, nz])

    ax.set_xlabel("X (voksli)")
    ax.set_ylabel("Y (voksli)")
    ax.set_zlabel("Z (voksli)")
    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    from stl_reader import read_stl
    from voxel_grid import build_grid
    from voxelizer import voxelize_surface, voxelize_interior

    # triangles = read_stl("models/spodnji.stl")
    triangles = read_stl("models/pokrov.stl")

    grid = build_grid(triangles, voxel_size=2.0)
    voxelize_surface(grid, triangles)
    voxelize_interior(grid)
    visualize_voxels(grid, title="Voksli — pokrov.stl")
    # visualize_voxels(grid, title="Voksli — spodnji.stl")
