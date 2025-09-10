from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_basis_dir(path: str | Path) -> np.ndarray:
    """Load basis_*.wave as array (k, H, W), with zeros mapped to NaN.

    Files are sorted by their numeric index in the filename.
    """
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Basis directory not found: {p}")

    arrays: list[np.ndarray] = []
    indices: list[int] = []
    for f in p.iterdir():
        name = f.name
        if name.startswith("basis_") and name.endswith(".wave") and f.is_file():
            try:
                idx = int(name.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            data = np.loadtxt(f).astype(float, copy=False)
            data[data == 0] = np.nan
            arrays.append(data)
            indices.append(idx)

    if not arrays:
        raise RuntimeError(f"No basis_*.wave in {p}")

    arrays_sorted = [arr for _, arr in sorted(zip(indices, arrays))]
    B = np.stack(arrays_sorted, axis=0)
    return B


def save_basis_masks(masks: Iterable[np.ndarray], out_dir: str | Path) -> None:
    """Save a sequence of boolean/float masks to out_dir/basis_{i}.wave.

    True/non-NaN -> 1.0, False/NaN -> 0.0
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(masks):
        arr = np.asarray(m, dtype=float)
        if arr.dtype == bool:
            arr = arr.astype(float)
        arr = np.where(np.isfinite(arr) & (arr != 0), 1.0, 0.0)
        np.savetxt(out / f"basis_{i}.wave", arr, fmt="%.1f")


def load_functions_wave(path: str | Path) -> np.ndarray:
    return np.loadtxt(path)


def save_functions_wave(arr: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(arr, dtype=float), fmt="%.6f")

