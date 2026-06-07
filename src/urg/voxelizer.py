from collections import deque
import numpy as np
from voxel_grid import VoxelGrid
from triangle_voxel import triangle_intersects_voxel


def flood_fill(labels, start, match, new_label):
    """
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
