from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Size:
    x: int  # width (columns)
    y: int  # height (rows)


def xy_to_rc(x: int, y: int) -> tuple[int, int]:
    """Convert Cartesian (x, y) to numpy array indices (row, col) = (y, x)."""
    return int(y), int(x)


def rc_to_xy(r: int, c: int) -> tuple[int, int]:
    """Convert numpy array indices (row, col) to Cartesian (x, y)."""
    return int(c), int(r)

