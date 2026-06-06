import numpy as np


class VoxelGrid:
    """
    origin:     np.ndarray (3,)  — koordinate vogala voksel[0,0,0] v prostoru modela (pove kje se mreža dejansko začne)
    voxel_size: float            — dolžina stranice voksla [mm]
    shape:      tuple[int,int,int] — (nx, ny, nz) število vokslov po posamezni osi
    labels:     np.ndarray shape (nx,ny,nz), dtype=uint8 — oznake vokslov
    """

    def __init__(self, origin, voxel_size, shape):
        self.origin = origin
        self.voxel_size = voxel_size
        self.shape = shape
        self.labels = np.zeros(shape, dtype=np.uint8)


def build_grid(
    triangles: list[dict], voxel_size: float, offset_voxels: int = 1
) -> VoxelGrid:
    """
    Izračuna AABB modela, ga razširi za offset_voxels vokslov na vsako stran,
    zaokroži dimenzije navzgor na cela števila vokslov in vrne prazno mrežo
    (vsi voksli označeni kot zrak, oznaka 0).
    """
    all_verts = np.array([v for t in triangles for v in (t["v0"], t["v1"], t["v2"])])
    aabb_min = all_verts.min(axis=0)
    aabb_max = all_verts.max(axis=0)

    span = aabb_max - aabb_min
    counts = (
        np.ceil(span / voxel_size).astype(int) + 2 * offset_voxels
    )  # tu dodamo + 2 * offset_voxels, da je obdano z zrakom
    shape = (int(counts[0]), int(counts[1]), int(counts[2]))

    origin = aabb_min - offset_voxels * voxel_size

    grid = VoxelGrid(origin, voxel_size, shape)

    total = shape[0] * shape[1] * shape[2]
    print(
        f"Mreža: {shape[0]}x{shape[1]}x{shape[2]} vokslov "
        f"(voxel_size={voxel_size} mm, skupaj {total} vokslov)"
    )

    return grid
