from collections import defaultdict
import numpy as np


def check_watertight(triangles: list[dict]) -> bool:

    edge_count = defaultdict(int)

    for t in triangles:
        verts = [
            tuple(np.round(t["v0"], 6)),
            tuple(np.round(t["v1"], 6)),
            tuple(np.round(t["v2"], 6)),
        ]

        edges = [
            (min(verts[0], verts[1]), max(verts[0], verts[1])),
            (min(verts[1], verts[2]), max(verts[1], verts[2])),
            (min(verts[0], verts[2]), max(verts[0], verts[2])),
        ]

        for e in edges:
            edge_count[e] += 1

    bad = {e: c for e, c in edge_count.items() if c != 2}
    print(f"Skupaj robov: {len(edge_count)}")
    print(f"Problematičnih robov (ne == 2): {len(bad)}")
    return len(bad) == 0
