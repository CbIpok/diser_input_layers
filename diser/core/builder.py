from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def build_rect_grid_masks(nx: int,
                          ny: int,
                          polygon_points: np.ndarray,
                          n: int,
                          theta_deg: float) -> list[np.ndarray]:
    """Build n x n rectangular masks clipped by polygon, rotated by theta_deg.

    Returns list of boolean masks, each shape (ny, nx).
    """
    verts = np.asarray(polygon_points, dtype=float)

    xs = np.arange(0.5, nx, 1.0)
    ys = np.arange(0.5, ny, 1.0)
    X, Y = np.meshgrid(xs, ys)

    mask_poly = Path(verts).contains_points(np.vstack((X.ravel(), Y.ravel())).T)
    mask_poly = mask_poly.reshape(ny, nx)

    x0, y0 = verts[:, 0].min(), verts[:, 1].min()
    rad = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    Xc = X - x0
    Yc = Y - y0
    Xr = cos_t * Xc + sin_t * Yc
    Yr = -sin_t * Xc + cos_t * Yc

    Xr_in = Xr[mask_poly]
    Yr_in = Yr[mask_poly]
    xr_min, xr_max = float(Xr_in.min()), float(Xr_in.max())
    yr_min, yr_max = float(Yr_in.min()), float(Yr_in.max())
    cell = max((xr_max - xr_min) / n, (yr_max - yr_min) / n)
    xr0 = xr_min + ((xr_max - xr_min) - cell * n) / 2
    yr0 = yr_min + ((yr_max - yr_min) - cell * n) / 2
    i_edges = xr0 + cell * np.arange(n + 1)
    j_edges = yr0 + cell * np.arange(n + 1)

    masks: list[np.ndarray] = []
    for i0 in i_edges[:-1]:
        for j0 in j_edges[:-1]:
            cell_mask = (
                (Xr >= i0) & (Xr < i0 + cell) & (Yr >= j0) & (Yr < j0 + cell)
            )
            m = mask_poly & cell_mask
            if np.any(m):
                masks.append(m)
    return masks

