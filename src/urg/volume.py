"""
Izračun volumna materiala iz vokselske mreže.

Modul prešteje snovne voksle (oznaka 1) in jih pomnoži s kubom stranice
voksla, da dobi približni volumen modela.
"""

from voxel_grid import VoxelGrid


def compute_volume(grid: VoxelGrid) -> float:
    """
    Izračuna volumen materiala iz vokselske mreže.
    Vsak voksel z oznako 1 predstavlja material — volumen je število takih vokslov
    pomnoženo s kubom stranice voksla.
    Args:
        grid: VoxelGrid — mreža po voxelize_interior (labels vsebuje {0, 1})
    Returns:
        float — volumen v kubičnih mm
    """
    solid = int((grid.labels == 1).sum())
    volume = solid * grid.voxel_size**3
    print(
        f"Volumen: {volume:.2f} mm(3) "
        f"({solid} snovnih vokslov, velikost voksla {grid.voxel_size} mm)"
    )
    return volume
