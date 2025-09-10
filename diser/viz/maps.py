from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def plot_grid_scalar(arr: np.ndarray, title: str = "", cmap: str = "viridis"):
    fig, ax = plt.subplots()
    im = ax.imshow(arr.T, origin='lower', cmap=cmap, interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig, ax


def plot_triangulated(xs: np.ndarray, ys: np.ndarray, values: np.ndarray,
                      title: str = "", levels: int = 100):
    mask = np.isfinite(values)
    triang = tri.Triangulation(xs[mask], ys[mask])
    fig, ax = plt.subplots()
    tpc = ax.tricontourf(triang, values[mask], levels=levels)
    fig.colorbar(tpc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    return fig, ax

