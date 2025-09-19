import numpy as np


def load_bath_grid(path: str) -> np.ndarray:
    """Load a MOST-format bathymetry grid.

    The file starts with two integers (nx, ny), followed by grid axes and the flattened grid values."""
    with open(path, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    nx, ny = map(int, tokens[:2])
    idx = 2 + nx + ny
    data = np.array(tokens[idx:idx + nx * ny], dtype=float)
    return data.reshape((ny, nx))