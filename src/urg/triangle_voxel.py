import numpy as np
from voxel_grid import VoxelGrid, voxel_corners


def plane_from_triangle(v0, v1, v2):
    """Vrne (normala n, D) za ravnino Ax+By+Cz+D=0 skozi trikotnik."""
    n = np.cross(v1 - v0, v2 - v0)
    n = n / np.linalg.norm(n)
    d = -np.dot(n, v0)
    return n, d


EPS = 1e-9


def edge_plane_intersection(p1, p2, n, d):
    """
    Presek roba (daljice p1-p2) z ravnino (n, d).
    Vrne:
      - None         če rob ne seka ravnine (vzporeden ali na isti strani)
      - np.ndarray   eno presečišče (točka na robu)
      - [p1, p2]     če rob leži v ravnini (obe krajišči)
    """
    denom = np.dot(n, p2 - p1)
    num = -(np.dot(n, p1) + d)

    if abs(denom) < EPS:
        # Rob je vzporeden z ravnino
        if abs(num) < EPS:
            # Rob leži v ravnini → obe krajišči sta presečišči
            return [p1, p2]
        else:
            # Rob je vzporeden in ne seka ravnine
            return None

    t = num / denom
    if 0.0 <= t <= 1.0:
        return p1 + t * (p2 - p1)
    return None


if __name__ == "__main__":
    # Test 1: trikotnik v ravnini XY -> normala mora biti (0,0,1), d=0
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    n, d = plane_from_triangle(v0, v1, v2)
    print(f"Test 1 — trikotnik v XY ravnini:")
    print(f"  normala = {n}  (pričakovano: [0, 0, 1])")
    print(f"  d       = {d}  (pričakovano: 0.0)")

    # Test 2: trikotnik dvignjen na z=5 -> normala še vedno (0,0,1), d=-5
    v0 = np.array([0.0, 0.0, 5.0])
    v1 = np.array([1.0, 0.0, 5.0])
    v2 = np.array([0.0, 1.0, 5.0])
    n, d = plane_from_triangle(v0, v1, v2)
    print(f"\nTest 2 — trikotnik v ravnini z=5:")
    print(f"  normala = {n}  (pričakovano: [0, 0, 1])")
    print(f"  d       = {d}  (pričakovano: -5.0)")

    # Test 3: trikotnik v ravnini XZ -> normala mora biti (0,1,0) ali (0,-1,0)
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 1.0])
    n, d = plane_from_triangle(v0, v1, v2)
    print(f"\nTest 3 — trikotnik v XZ ravnini:")
    print(f"  normala = {n}  (pričakovano: [0, ±1, 0])")
    print(f"  d       = {d}  (pričakovano: 0.0)")

    # --- edge_plane_intersection() testi ---
    print("\n--- edge_plane_intersection ---")
    n_xy = np.array([0.0, 0.0, 1.0])  # ravnina XY (z=0)
    d_xy = 0.0

    # Test 4: rob seka ravnino z=0 na sredini -> presečišče (0,0,0)
    res = edge_plane_intersection(
        np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]), n_xy, d_xy
    )
    print(f"Test 4 — rob seka z=0:     {res}  (pričakovano: [0, 0, 0])")

    # Test 5: rob je v celoti nad ravnino -> None
    res = edge_plane_intersection(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0]), n_xy, d_xy
    )
    print(f"Test 5 — rob nad ravnino:  {res}  (pričakovano: None)")

    # Test 6: rob leži točno v ravnini z=0 -> obe krajišči
    res = edge_plane_intersection(
        np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), n_xy, d_xy
    )
    print(f"Test 6 — rob v ravnini:    {res}  (pričakovano: obe krajišči)")
