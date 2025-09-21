import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # Allow plotting without display
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from plot_basis_maps import load_basis


def load_basis_arrays(directory: str, count: int) -> List[np.ndarray]:
    """Load ``count`` basis arrays from ``directory``.

    Each file is expected to be named ``basis_{j}.wave`` for ``0 <= j < count``
    and contain a 2-D grid readable by :func:`numpy.loadtxt`.
    """
    arrays: List[np.ndarray] = []
    for j in range(count):
        path = os.path.join(directory, f"basis_{j}.wave")
        arrays.append(np.loadtxt(path))
    return arrays


def compute_pointwise_mse(
    xs: np.ndarray,
    ys: np.ndarray,
    coefs: List[np.ndarray],
    basis_arrays: List[np.ndarray],
    true_surface: np.ndarray,
) -> np.ndarray:
    """Return MSE values for each ``(x, y)`` coordinate.

    For every point ``(x, y)`` the approximation is computed as
    ``sum(c_j * basis_j[y, x])`` and compared with ``true_surface``.
    Coordinates are expected to correspond to grid cell centres; indices are
    derived using ``floor`` which maps 0.5 -> 0, 1.5 -> 1, etc.
    """
    num_points = xs.shape[0]
    mse_vals = np.full(num_points, np.nan)
    for idx in range(num_points):
        xi = int(np.floor(xs[idx]))
        yi = int(np.floor(ys[idx]))
        if (
            yi < 0
            or xi < 0
            or yi >= true_surface.shape[0]
            or xi >= true_surface.shape[1]
        ):
            continue
        approx = 0.0
        for j, basis in enumerate(basis_arrays):
            approx += coefs[j][idx] * basis[yi, xi]
        real = true_surface[yi, xi]
        mse_vals[idx] = (real - approx) ** 2
    return mse_vals


def plot_mse(
    xs: np.ndarray,
    ys: np.ndarray,
    mse_vals: np.ndarray,
    out_path: str | None,
    title: str,
) -> str | None:
    """Plot a triangulated MSE map and optionally save it."""
    mask = np.isfinite(mse_vals)
    if mask.sum() < 3:
        return None
    triang = tri.Triangulation(xs[mask], ys[mask])
    fig, ax = plt.subplots()
    tpc = ax.tricontourf(triang, mse_vals[mask], levels=100)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    else:
        plt.show()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MSE between approximation and true surface"
    )
    parser.add_argument(
        "--coefs-folder",
        default="coefs_process",
        help="Directory containing basis_{i}.json files",
    )
    parser.add_argument(
        "--i",
        type=int,
        default=4,
        help="Number of basis functions (selects basis_{i}.json)",
    )
    parser.add_argument(
        "--basis-folder",
        default="data",
        help="Directory containing basis_{i}/basis_{j}.wave arrays",
    )
    parser.add_argument(
        "--func-path",
        default="data/functions.wave",
        help="Path to the reference surface",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save the resulting PNG. If omitted, plot is shown",
    )
    args = parser.parse_args()

    # Load coefficients for all points
    json_path = os.path.join(args.coefs_folder, f"basis_{args.i}.json")
    xs, ys, _approx_err, coef_arrays = load_basis(json_path)

    # Determine where basis functions are stored
    basis_dir = os.path.join(args.basis_folder, f"basis_{args.i}")
    if not os.path.isdir(basis_dir):
        basis_dir = os.path.join(args.basis_folder, "basis")
    basis_arrays = load_basis_arrays(basis_dir, args.i)

    # Load reference surface
    true_surface = np.loadtxt(args.func_path)

    # Compute MSE values
    mse_vals = compute_pointwise_mse(xs, ys, coef_arrays, basis_arrays, true_surface)
    mean_mse = float(np.nanmean(mse_vals))
    print(f"Mean MSE over domain: {mean_mse:.6g}")

    # Plot
    out_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, f"mse_{args.i}.png")
    plot_mse(xs, ys, mse_vals, out_path, title=f"MSE for i={args.i}")


if __name__ == "__main__":
    main()
