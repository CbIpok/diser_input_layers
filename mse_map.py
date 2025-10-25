import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from plot_basis_maps import load_basis


def load_basis_arrays(basis_dir: str, count: int) -> List[np.ndarray]:
    """Load basis_j.wave arrays from ``basis_dir``.

    Parameters
    ----------
    basis_dir: str
        Directory containing files ``basis_{j}.wave``.
    count: int
        Number of basis functions ``i``. Files ``0 <= j < i``
        are expected to be present.
    """
    arrays: List[np.ndarray] = []
    for j in range(count):
        path = os.path.join(basis_dir, f"basis_{j}.wave")
        arrays.append(np.loadtxt(path))
    return arrays


def compute_mse(xs: np.ndarray, ys: np.ndarray, coefs: List[np.ndarray],
                basis_arrays: List[np.ndarray], func: np.ndarray) -> np.ndarray:
    """Compute mean squared error between reconstructed and real surface.

    ``xs`` and ``ys`` define coordinates where coefficients ``coefs`` are
    specified. ``basis_arrays`` contains basis functions ``basis_j.wave`` and
    ``func`` is the real surface ``functions.wave``. Only points where
    coefficients are provided are evaluated.
    """
    mse = np.full(xs.shape, np.nan, dtype=float)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        xi = int(round(x))
        yi = int(round(y))
        approx = 0.0
        for b_arr, c_arr in zip(basis_arrays, coefs):
            approx += c_arr[idx] * b_arr[yi, xi]
        real_val = func[yi, xi]
        mse[idx] = (real_val - approx) ** 2
    return mse


def plot_map(xs: np.ndarray, ys: np.ndarray, mse: np.ndarray,
             out_path: str | None = None) -> None:
    """Plot triangulated map of ``mse`` values at ``xs``, ``ys`` coordinates."""
    mask = np.isfinite(mse)
    if mask.sum() < 3:
        return
    triang = tri.Triangulation(xs[mask], ys[mask])
    fig, ax = plt.subplots()
    tpc = ax.tricontourf(triang, mse[mask], levels=100)
    ax.set_title("mean squared error")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    fig.colorbar(tpc, ax=ax)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSE map for basis approximation",
    )
    parser.add_argument("--i", type=int, default=4,
                        help="number of basis coefficients")
    parser.add_argument("--coefs-folder", default="coefs_process",
                        help="folder with basis_{i}.json files")
    parser.add_argument("--data-dir", default="data",
                        help="directory with basis_{i} folders and functions.wave")
    parser.add_argument("--save", default=None,
                        help="path to save resulting PNG; shows plot if omitted")
    args = parser.parse_args()

    json_path = os.path.join(args.coefs_folder, f"basis_{args.i}.json")
    xs, ys, _, coefs = load_basis(json_path)

    basis_dir = os.path.join(args.data_dir, f"basis_{args.i}")
    basis_arrays = load_basis_arrays(basis_dir, args.i)
    func_path = os.path.join(args.data_dir, "functions.wave")
    func = np.loadtxt(func_path)

    mse = compute_mse(xs, ys, coefs, basis_arrays, func)
    plot_map(xs, ys, mse, args.save)


if __name__ == "__main__":
    main()
