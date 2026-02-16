"""Compute and plot mean squared deviation between reconstructed form and the original wave.

This script uses coefficients produced by ``plot_basis_maps.py`` and basis
functions stored in ``data/basis_i`` to reconstruct an approximation of the
``data/functions.wave`` field. For each grid point it computes the squared
error between the approximation and the real field and displays or saves the
resulting map.
"""
import argparse
import os
from typing import List

import numpy as np

from plot_basis_maps import load_basis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def _load_basis_arrays(basis_dir: str, count: int) -> List[np.ndarray]:
    """Load ``count`` basis arrays from ``basis_dir``."""
    basis_arrays = []
    for j in range(count):
        path = os.path.join(basis_dir, f"basis_{j}.wave")
        basis_arrays.append(np.loadtxt(path))
    return basis_arrays


def _compute_sq_errors(xs: np.ndarray, ys: np.ndarray, coefs: List[np.ndarray],
                       basis_arrays: List[np.ndarray], func: np.ndarray) -> np.ndarray:
    """Return squared errors of approximation at given coordinates."""
    sq_errors: List[float] = []
    for idx, (x, y) in enumerate(zip(xs, ys)):
        xi, yi = int(round(x)), int(round(y))
        approx = 0.0
        for j, basis in enumerate(basis_arrays):
            approx += coefs[j][idx] * basis[yi, xi]
        real_val = func[yi, xi]
        sq_errors.append((approx - real_val) ** 2)
    return np.asarray(sq_errors)


def _plot_mse(xs: np.ndarray, ys: np.ndarray, mse: np.ndarray, out_file: str | None) -> None:
    """Plot mean squared error map and optionally save to ``out_file``."""
    mask = np.isfinite(mse)
    if mask.sum() < 3:
        return
    triang = tri.Triangulation(xs[mask], ys[mask])
    fig, ax = plt.subplots()
    tpc = ax.tricontourf(triang, mse[mask], levels=100)
    ax.set_title("mse")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax)
    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean squared deviation between approximation and original wave",
    )
    parser.add_argument("--coef-folder", default="coefs_process", help="Folder with basis_{i}.json")
    parser.add_argument("--data-dir", default="data", help="Directory with basis_i folders and functions.wave")
    parser.add_argument("--i", type=int, default=4, help="Number of basis functions")
    parser.add_argument("--save-dir", default=None, help="Directory to save the MSE plot")
    args = parser.parse_args()

    basis_json = os.path.join(args.coef_folder, f"basis_{args.i}.json")
    xs, ys, _approx_err, coefs = load_basis(basis_json)
    func = np.loadtxt(os.path.join(args.data_dir, "functions.wave"))
    basis_dir = os.path.join(args.data_dir, f"basis_{args.i}")
    basis_arrays = _load_basis_arrays(basis_dir, args.i)
    sq_errors = _compute_sq_errors(xs, ys, coefs, basis_arrays, func)

    output = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        output = os.path.join(args.save_dir, f"mse_{args.i}.png")
    _plot_mse(xs, ys, sq_errors, output)

    print(f"MSE for i={args.i}: {sq_errors.mean():.6f}")


if __name__ == "__main__":
    main()
