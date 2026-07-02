"""
3D prikaz trikotniškega STL modela z matplotlib.

Modul izriše trikotnike kot poltransparentne ploskve z obrobami in
samodejno nastavi razmerja osi glede na obseg modela.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize(triangles: list[dict], title: str = "STL model") -> None:
    """
    Prikaže trikotniški model v 3D.
    Robovi: temno siva, linewidth=0.3
    Ploskve: svetlo modra, alpha=0.3
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    verts = [[t["v0"], t["v1"], t["v2"]] for t in triangles]
    poly = Poly3DCollection(
        verts, facecolor="#B5D4F4", edgecolor="#444441", linewidth=0.3, alpha=0.4
    )
    ax.add_collection3d(poly)

    all_verts = np.array([v for t in triangles for v in (t["v0"], t["v1"], t["v2"])])
    mins = all_verts.min(axis=0)
    maxs = all_verts.max(axis=0)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ranges = maxs - mins
    ax.set_box_aspect(ranges / ranges.max())

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    import sys
    from stl_reader import read_stl

    paths = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else ["models/spodnji.stl", "models/pokrov.stl"]
    )
    for path in paths:
        triangles = read_stl(path)
        visualize(triangles, title=path)
