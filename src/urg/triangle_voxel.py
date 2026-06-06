import numpy as np
from voxel_grid import VoxelGrid, voxel_corners


def plane_from_triangle(v0, v1, v2):
    """Vrne (normala n, D) za ravnino Ax+By+Cz+D=0 skozi trikotnik."""
    n = np.cross(v1 - v0, v2 - v0)
    n = n / np.linalg.norm(n)
    d = -np.dot(n, v0)
    return n, d


if __name__ == "__main__":
    # Test 1: trikotnik v ravnini XY → normala mora biti (0,0,1), d=0
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    n, d = plane_from_triangle(v0, v1, v2)
    print(f"Test 1 — trikotnik v XY ravnini:")
    print(f"  normala = {n}  (pričakovano: [0, 0, 1])")
    print(f"  d       = {d}  (pričakovano: 0.0)")

    # Test 2: trikotnik dvignjen na z=5 → normala še vedno (0,0,1), d=-5
    v0 = np.array([0.0, 0.0, 5.0])
    v1 = np.array([1.0, 0.0, 5.0])
    v2 = np.array([0.0, 1.0, 5.0])
    n, d = plane_from_triangle(v0, v1, v2)
    print(f"\nTest 2 — trikotnik v ravnini z=5:")
    print(f"  normala = {n}  (pričakovano: [0, 0, 1])")
    print(f"  d       = {d}  (pričakovano: -5.0)")

    # Test 3: trikotnik v ravnini XZ → normala mora biti (0,1,0) ali (0,-1,0)
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 1.0])
    n, d = plane_from_triangle(v0, v1, v2)
    print(f"\nTest 3 — trikotnik v XZ ravnini:")
    print(f"  normala = {n}  (pričakovano: [0, ±1, 0])")
    print(f"  d       = {d}  (pričakovano: 0.0)")
