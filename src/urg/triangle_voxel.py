import numpy as np
from voxel_grid import VoxelGrid, voxel_corners
from convex_hull import (
    point_side,
    triangle_area,
    graham_scan,
)  # pomožne funkcije, ki so že implementirane od prej


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


# 12 robov voksla kot pari indeksov oglišč iz voxel_corners()
# oglišča: index = dx*4 + dy*2 + dz (po vrstnem redu zanke v voxel_corners)
VOXEL_EDGES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),  # robovi vzdolž z osi (navpicni)
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),  # robovi vzdolž y osi
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # robovi vzdolž x osi
]


def voxel_plane_intersection(corners, n, d):
    """
    Poišče presečišča ravnine (n,d) z vseh 12 robov voksla.
    Vrne array oblike (m, 3), m v {0, 1, ..., 6}.
    """
    I = []

    for i, j in VOXEL_EDGES:
        res = edge_plane_intersection(corners[i], corners[j], n, d)
        if res is None:
            continue

        pts = res if isinstance(res, list) else [res]
        for pt in pts:
            # Dodamo samo, če točka še ni v seznamu
            if not any(np.linalg.norm(pt - q) < EPS for q in I):
                I.append(pt)
            if len(I) == 6:
                break
        if len(I) == 6:
            break

    return np.array(I) if I else np.empty((0, 3))


def project_to_2d(points_3d, n, origin):
    """
    Projicira 3D tocke (ki lezijo v ravnini z normalo n) v 2D koordinate.
    Vrne seznam tuplev (u, v).
    """
    # normala ravnine = os, ki kaze ven iz ravnine
    normal = n / np.linalg.norm(n)

    # pomozni vektor: vzamemo (0,0,1), razen ce je normala skoraj enaka njemu, v tem primeru vzamemo (1,0,0)
    if abs(np.dot(normal, np.array([0.0, 0.0, 1.0]))) < 1.0 - EPS:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])

    # prva os v ravnini: pravokotna na normalo in helper
    axis_x = np.cross(helper, normal)
    axis_x = axis_x / np.linalg.norm(axis_x)

    # druga os v ravnini: pravokotna na normalo in axis_x
    axis_y = np.cross(normal, axis_x)

    # za vsako tocko: odstejes origin, projiciras na axis_x in axis_y
    pts_2d = []
    for p in points_3d:
        offset = p - origin  # da dobis
        u = np.dot(offset, axis_x)
        v = np.dot(offset, axis_y)
        pts_2d.append((u, v))

    return pts_2d


def polygon_area(poly):
    """Ploščina konveksnega mnogokotnika — fan triangulacija iz prvega oglišča."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(1, len(poly) - 1):
        area += triangle_area(poly[0], poly[i], poly[i + 1])
    return area


def segments_intersect(a, b, c, d):
    """Ali se daljici AB in CD pravilno sekata (dotik ne šteje)."""
    return (
        point_side(a, b, c) * point_side(a, b, d) < 0
        and point_side(c, d, a) * point_side(c, d, b) < 0
    )


def polygons_intersect(poly_a, poly_b):
    """
    Test sekanja dveh konveksnih 2D mnogokotnikov (lista tuplev (u,v)).
    Dotik ne šteje. Vrne True ali False.
    """
    n_a, n_b = len(poly_a), len(poly_b)

    # 1. preverimo ali se kateri par robov seka
    for i in range(n_a):
        a1, a2 = poly_a[i], poly_a[(i + 1) % n_a]
        for j in range(n_b):
            b1, b2 = poly_b[j], poly_b[(j + 1) % n_b]
            if segments_intersect(a1, a2, b1, b2):
                return True

    # 2. noben rob se ne seka — preverimo vsebovanost
    # če je eden znotraj drugega, bo konveksna lupina unije enaka večjemu liku
    hull = graham_scan(list(poly_a) + list(poly_b))
    if len(hull) < 3:
        return False
    hull_area = polygon_area(hull)
    area_a = polygon_area(list(poly_a))
    area_b = polygon_area(list(poly_b))
    return hull_area <= max(area_a, area_b) + EPS


def on_same_voxel_face(points, corners):
    """Preveri ali vse točke ležijo na isti ploskvi voksla."""
    x0, y0, z0 = corners[0]  # 0,0,0
    x1, y1, z1 = corners[7]  # 1,1,1
    for axis, val in [(0, x0), (0, x1), (1, y0), (1, y1), (2, z0), (2, z1)]:
        if all(abs(pt[axis] - val) < EPS for pt in points):
            return True
    return False


def triangle_intersects_voxel(triangle: dict, grid, i: int, j: int, k: int) -> bool:
    """Ali se trikotnik seka z vokslom [i,j,k]."""
    v0, v1, v2 = triangle["v0"], triangle["v1"], triangle["v2"]

    n, d = plane_from_triangle(v0, v1, v2)  # izračunamo ravnino trikotnika
    corners = voxel_corners(grid, i, j, k)  # dobimo vseh 8 voglišč voksla

    I = voxel_plane_intersection(
        corners, n, d
    )  # najde presečišča ravnine z vsemi 12 robovi voksla

    if len(I) < 3:
        return False

    if len(I) == 4 and on_same_voxel_face(I, corners):  # dotik
        return False

    origin = corners[0]
    I_2d = project_to_2d(I, n, origin)  # projeciramo v 2D
    tri_2d = project_to_2d([v0, v1, v2], n, origin)  # projeciramo v 2D

    hull_I = graham_scan(I_2d)  # vzame konveksno lupino
    hull_tri = graham_scan(tri_2d)  # vzame konveksno lupino

    if len(hull_I) < 3 or len(hull_tri) < 3:
        return False

    return polygons_intersect(hull_I, hull_tri)  # dobimo končno odločitev


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

    # --- voxel_plane_intersection testi ---
    print("\n--- voxel_plane_intersection ---")
    from voxel_grid import VoxelGrid, voxel_corners, build_grid
    import math

    # Enotski voksel [0,1]^3
    grid = VoxelGrid(np.array([0.0, 0.0, 0.0]), 1.0, (1, 1, 1))
    corners = voxel_corners(grid, 0, 0, 0)

    # Test 7: ravnina z=0.5 (sredina voksla) -> 4 točke (kvadrat)
    n7 = np.array([0.0, 0.0, 1.0])
    d7 = -0.5
    I7 = voxel_plane_intersection(corners, n7, d7)
    print(f"Test 7 — ravnina z=0.5:    {len(I7)} točk  (pričakovano: 4)")

    # Test 8: ravnina z=0 (spodnja stranica voksla) -> 4 točke, ki so ravno na stranici
    n8 = np.array([0.0, 0.0, 1.0])
    d8 = 0.0
    I8 = voxel_plane_intersection(corners, n8, d8)
    print(
        f"Test 8 — ravnina z=0:      {len(I8)} točk  (pričakovano: 4, dotik stranice)"
    )

    # Test 9: ravnina z=2 (nad vokslom) -> 0 točk
    n9 = np.array([0.0, 0.0, 1.0])
    d9 = -2.0
    I9 = voxel_plane_intersection(corners, n9, d9)
    print(f"Test 9 — ravnina z=2:      {len(I9)} točk  (pričakovano: 0)")

    # Test 10: diagonalna ravnina x+y+z=1.5 -> 3-6 točk
    n10 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    d10 = -1.5 / math.sqrt(3)
    I10 = voxel_plane_intersection(corners, n10, d10)
    print(f"Test 10 — diag. ravnina:   {len(I10)} točk  (pričakovano: 3-6)")

    # --- project_to_2d testi ---
    print("\n--- project_to_2d ---")

    # Test 11: točke v XY ravnini (n=(0,0,1)), origin=(0,0,0)
    # (1,0,0) mora dati (1,0) ali (0,1), (0,1,0) mora dati (0,1) ali (1,0)
    pts3d = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    n11 = np.array([0.0, 0.0, 1.0])
    origin11 = np.array([0.0, 0.0, 0.0])
    pts2d = project_to_2d(pts3d, n11, origin11)
    print(f"Test 11 — točke v XY ravnini:")
    for p3, p2 in zip(pts3d, pts2d):
        print(f"  {p3} → {p2}")

    # --- polygons_intersect testi ---
    print("\n--- polygons_intersect ---")

    # Test 12: dva kvadrata, ki se prekrivata -> True
    sq1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
    sq2 = [(1, 1), (3, 1), (3, 3), (1, 3)]
    print(
        f"Test 12 — prekrivata se:      {polygons_intersect(sq1, sq2)}  (pričakovano: True)"
    )

    # Test 13: kvadrat znotraj trikotnika (vsebovanost) -> True
    big_tri = [(-5, 0), (5, 0), (0, 5)]
    small_sq = [(-0.5, 0.5), (0.5, 0.5), (0.5, 1.5), (-0.5, 1.5)]
    print(
        f"Test 13 — kvadrat v trikotniku: {polygons_intersect(big_tri, small_sq)}  (pričakovano: True)"
    )

    # Test 14: dva kvadrata, ki se le dotikata -> False
    sq3 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    sq4 = [(1, 0), (2, 0), (2, 1), (1, 1)]
    print(
        f"Test 14 — samo dotik:          {polygons_intersect(sq3, sq4)}  (pričakovano: False)"
    )

    # Test 15: dva kvadrata, ki sta povsem ločena -> False
    sq5 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    sq6 = [(3, 3), (4, 3), (4, 4), (3, 4)]
    print(
        f"Test 15 — povsem ločena:       {polygons_intersect(sq5, sq6)}  (pričakovano: False)"
    )
