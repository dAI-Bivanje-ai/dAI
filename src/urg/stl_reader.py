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


def read_binary(f) -> list[dict]:
    header = f.read(80)
    count = struct.unpack("<I", f.read(4))[0]  # tu mamo število trikotnikov

    triangles = []
    for _ in range(count):
        data = f.read(50)  # vsak trikotnik tocno 50 bajtov
        values = struct.unpack("<12fH", data)
        normal = np.array(values[0:3])
        v0 = np.array(values[3:6])
        v1 = np.array(values[6:9])
        v2 = np.array(values[9:12])
        triangles.append({"normal": normal, "v0": v0, "v1": v1, "v2": v2})
    return triangles


def read_stl(filepath: str) -> list[dict]:
    """
    Prebere STL datoteko in vrne seznam trikotnikov.

    Returns:
        Seznam slovarjev, vsak z:
        {
            'normal': np.ndarray shape (3,),   # normala trikotnika
            'v0':     np.ndarray shape (3,),   # prvo oglišče
            'v1':     np.ndarray shape (3,),   # drugo oglišče
            'v2':     np.ndarray shape (3,),   # tretje oglišče
        }
    """
    with open(filepath, "rb") as f:
        try:
            f.seek(0)
            triangles = read_ascii(f)
        except (ValueError, UnicodeDecodeError, IndexError):
            f.seek(0)
            triangles = read_binary(f)

    assert len(triangles) > 0, "STL je prazen"

    all_verts = np.array([v for t in triangles for v in (t["v0"], t["v1"], t["v2"])])
    mins = all_verts.min(axis=0)
    maxs = all_verts.max(axis=0)

    print(f"Uvoženo {len(triangles)} trikotnikov iz {filepath}")
    print(f"AABB: min={mins} max={maxs}")

    return triangles


if __name__ == "__main__":

    paths = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else ["models/spodnji.stl", "models/pokrov.stl"]
    )
    for path in paths:
        read_stl(path)
        print()
