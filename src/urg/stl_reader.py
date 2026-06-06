import struct
import numpy as np
import sys


def read_ascii(f):

    tokens = f.read().decode("ascii").split()

    if tokens[0].lower() != "solid":
        raise ValueError("ni ASCII STL")

    triangles = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() == "facet":
            normal = np.array(
                [float(tokens[i + 2]), float(tokens[i + 3]), float(tokens[i + 4])]
            )
            verts = []
            j = i + 7  # preskoči 'facet normal nx ny nz outer loop'
            for _ in range(3):
                verts.append(
                    np.array(
                        [
                            float(tokens[j + 1]),
                            float(tokens[j + 2]),
                            float(tokens[j + 3]),
                        ]
                    )
                )
                j += 4  # 'vertex x y z'
            triangles.append(
                {"normal": normal, "v0": verts[0], "v1": verts[1], "v2": verts[2]}
            )
            i = j
        else:
            i += 1
    return triangles
