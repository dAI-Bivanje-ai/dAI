"""
Vokselizacija modela: označevanje površja in polnjenje notranjosti.

Modul najprej označi voksle, ki jih seka trikotniška mreža (površje), nato
pa z večkratnim flood fillom loči zunanjo okolico, lupino, notranjost in
vrzeli ter na koncu mrežo preštevilči v oznaki {0, 1}.
"""

from collections import deque
import numpy as np
from voxel_grid import VoxelGrid
from triangle_voxel import triangle_intersects_voxel


def flood_fill(labels, start, match, new_label):
    """
    BFS flood fill — označi vse voksle z oznako match, ki so povezani s start, z new_label.
    Iterativno z deque, 6-povezanost.
    Args:
        labels:    np.ndarray (nx,ny,nz) uint8 — oznake vokslov
        start:     tuple(int,int,int)          — začetni voksel (i,j,k)
        match:     int                         — oznaka ki jo iščemo
        new_label: int                         — oznaka ki jo dodelimo
    Returns:
        int — število označenih vokslov
    """
    nx, ny, nz = labels.shape

    if labels[start] != match:
        return 0

    queue = deque()
    queue.append(start)
    labels[start] = new_label
    count = 1

    # 6 sosedov: 1 po vsaki osi (ker ima voksel 6 ploskev)
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while queue:
        i, j, k = queue.popleft()
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                if labels[ni, nj, nk] == match:
                    labels[ni, nj, nk] = new_label
                    count += 1
                    queue.append((ni, nj, nk))

    return count


def voxelize_surface(grid: VoxelGrid, triangles: list) -> None:
    """
    Označi voksle, ki jih seka trikotniška mreža, kot snovne (oznaka 1).
    Za vsak trikotnik izračuna AABB in preveri samo voksle znotraj tega razpona.
    Args:
        grid:      VoxelGrid  — prazna mreža (labels vse 0)
        triangles: list[dict] — trikotniki iz read_stl()
    Returns:
        None — modificira grid.labels direktno
    """
    nx, ny, nz = grid.shape
    marked = 0
    total = nx * ny * nz

    for tri in triangles:
        # AABB trikotnika
        verts = np.array([tri["v0"], tri["v1"], tri["v2"]])
        tri_min = verts.min(axis=0)
        tri_max = verts.max(axis=0)

        # pretvori v razpon vokselnih indeksov
        i_min = max(0, int(np.floor((tri_min[0] - grid.origin[0]) / grid.voxel_size)))
        i_max = min(
            nx - 1, int(np.ceil((tri_max[0] - grid.origin[0]) / grid.voxel_size))
        )
        j_min = max(0, int(np.floor((tri_min[1] - grid.origin[1]) / grid.voxel_size)))
        j_max = min(
            ny - 1, int(np.ceil((tri_max[1] - grid.origin[1]) / grid.voxel_size))
        )
        k_min = max(0, int(np.floor((tri_min[2] - grid.origin[2]) / grid.voxel_size)))
        k_max = min(
            nz - 1, int(np.ceil((tri_max[2] - grid.origin[2]) / grid.voxel_size))
        )

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    if grid.labels[i, j, k] == 0:
                        if triangle_intersects_voxel(tri, grid, i, j, k):
                            grid.labels[i, j, k] = 1
                            marked += 1

    print(f"Vokselizacija površja: označenih {marked} / {total} vokslov kot snovni")


def voxelize_interior(grid: VoxelGrid) -> None:
    """
    Loči notranjost modela od zunanje okolice in vrzeli z večkratnim flood fillom.
    Vmesne oznake: 2=zunanja okolica, 3=zunanja lupina, 4=notranjost, 5=vrzel.
    Na koncu preštevilči v {0, 1}.
    Args:
        grid: VoxelGrid — mreža po voxelize_surface (labels vsebuje {0, 1})
    Returns:
        None — modificira grid.labels direktno
    """
    labels = grid.labels
    nx, ny, nz = grid.shape

    # korak 1: zunanja okolica 0 → 2
    n2 = flood_fill(labels, (0, 0, 0), match=0, new_label=2)
    print(f"  Zunanja okolica (2): {n2} vokslov")

    # po koraku 1: stene ostanejo 1, zrak okoli sten postane 2, zrak znotraj sten ostane še vedno 0

    # korak 2: najdi voksel z oznako 1 ki meji na 2 -> zunanja lupina 1 -> 3
    # po koraku 2: 2 -> zunanjost, 3 -> stene modela, 0 -> zrak znotraj modela
    start_shell = None
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if labels[i, j, k] == 1:
                    for di, dj, dk in [
                        (1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            if labels[ni, nj, nk] == 2:
                                start_shell = (i, j, k)
                                break
                if start_shell:
                    break
            if start_shell:
                break
        if start_shell:
            break

    n3 = flood_fill(labels, start_shell, match=1, new_label=3)
    print(f"  Zunanja lupina (3): {n3} vokslov")

    # korak 3: preostale 0-regije -> notranjost (4) ali vrzel (5)
    # po koraku 3: 2 -> zunanjost, 3 -> stene, 4 -> notranjost, 5 -> vrzeli
    n4 = n5 = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if labels[i, j, k] == 0:
                    filled = flood_fill(labels, (i, j, k), match=0, new_label=4)
                    borders_shell = False
                    for ii in range(nx):
                        for jj in range(ny):
                            for kk in range(nz):
                                if labels[ii, jj, kk] == 4:
                                    for di, dj, dk in [
                                        (1, 0, 0),
                                        (-1, 0, 0),
                                        (0, 1, 0),
                                        (0, -1, 0),
                                        (0, 0, 1),
                                        (0, 0, -1),
                                    ]:
                                        ni, nj, nk = ii + di, jj + dj, kk + dk
                                        if (
                                            0 <= ni < nx
                                            and 0 <= nj < ny
                                            and 0 <= nk < nz
                                        ):
                                            if labels[ni, nj, nk] == 3:
                                                borders_shell = True
                    if borders_shell:
                        n4 += filled
                    else:
                        flood_fill(labels, (i, j, k), match=4, new_label=5)
                        n5 += filled

    print(f"  Notranjost (4): {n4} vokslov")
    print(f"  Vrzeli (5): {n5} vokslov")

    # korak 4: preštevilčenje {1->1, 2->0, 3->1, 4->1, 5->0}
    labels[labels == 2] = 0
    labels[labels == 3] = 1
    # tu more po algoritmu bit 1, sam mi je bolj logično 0, ker gledamo koliko materiala bi porabili?
    labels[labels == 4] = 0
    labels[labels == 5] = 0

    print(f"  Po preštevilčenju: {int((labels == 1).sum())} snovnih vokslov")


if __name__ == "__main__":
    # Test: 3x3x3 mreža, sredinski voksel je ovira (1), ostalo zrak (0)
    # flood fill iz (0,0,0) mora zapolniti vse razen sredinskega
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[2, 2, 2] = 1  # ovira

    n = flood_fill(labels, (0, 0, 0), match=0, new_label=2)
    print(f"Označenih vokslov: {n}  (pričakovano: {5*5*5 - 1} = 124)")
    print(f"Vrednost ovire:    {labels[2,2,2]}  (pričakovano: 1, nespremenjena)")
    print(f"Vrednost (0,0,0):  {labels[0,0,0]}  (pričakovano: 2)")
    print(f"Vrednost (4,4,4):  {labels[4,4,4]}  (pričakovano: 2)")

    # --- voxelize_surface test ---
    print("\n--- voxelize_surface ---")
    from stl_reader import read_stl
    from voxel_grid import build_grid

    triangles = read_stl("models/spodnji.stl")
    grid = build_grid(triangles, voxel_size=2.0)  # voxel 2 mm
    voxelize_surface(grid, triangles)

    unique, counts = np.unique(grid.labels, return_counts=True)
    print(f"Oznake po vokselizaciji površja: {dict(zip(unique, counts))}")

    print("\n--- voxelize_interior ---")
    voxelize_interior(grid)
    unique, counts = np.unique(grid.labels, return_counts=True)
    print(f"Oznake po koncu: {dict(zip(unique, counts))}")
    assert set(unique.tolist()) <= {0, 1}, "labels mora vsebovati samo {0, 1}!"
    print("OK — labels vsebuje samo {0, 1}")
