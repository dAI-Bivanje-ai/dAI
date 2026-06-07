import sys

from stl_reader import read_stl
from visualizer import visualize
from watertight import check_watertight
from voxel_grid import build_grid
from voxelizer import voxelize_surface, voxelize_interior
from voxel_visualizer import visualize_voxels
from volume import compute_volume

VOXEL_SIZE = 2.0  # mm — mora biti < debelina stene T=4mm


def process(filepath: str) -> float:
    print(f"\n=== {filepath} ===")
    triangles = read_stl(filepath)

    ok = check_watertight(triangles)
    print(f"Vodotesen: {'DA' if ok else 'NE'}")

    visualize(triangles, title=filepath)

    grid = build_grid(triangles, voxel_size=VOXEL_SIZE)
    voxelize_surface(grid, triangles)
    voxelize_interior(grid)
    visualize_voxels(grid, title=f"Voksli — {filepath}")

    return compute_volume(grid)


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["models/spodnji.stl", "models/pokrov.stl"]
    total = 0.0
    for p in paths:
        total += process(p)
    print(f"\nSkupni volumen ohišja: {total:.2f} mm³")
