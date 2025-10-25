import argparse
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from plot_basis_maps import load_basis


def _load_basis_waves(basis_root: str, i: int) -> List[np.ndarray]:
    """Load basis_j.wave arrays for j < i from ``basis_root/basis_i``."""
    basis_dir = os.path.join(basis_root, f"basis_{i}")
    waves: List[np.ndarray] = []
    for j in range(i):
        path = os.path.join(basis_dir, f"basis_{j}.wave")
        waves.append(np.loadtxt(path))
    return waves


def _mean_coeffs(coefs: List[np.ndarray]) -> List[float]:
    """Return a list of average coefficients ignoring NaN values."""
    return [float(np.nanmean(arr)) for arr in coefs]


def compute_mse(function: np.ndarray, basis_waves: List[np.ndarray],
                coeffs: List[float]) -> List[float]:
    """Compute cumulative MSE for approximations using first ``j`` basis
    functions. Returns a list of length ``len(coeffs)`` where the ``j``-th
    element corresponds to approximation with ``j+1`` basis functions."""
    approx = np.zeros_like(function)
    mses: List[float] = []
    for wave, coef in zip(basis_waves, coeffs):
        approx += coef * wave
        mses.append(float(np.mean((approx - function) ** 2)))
    return mses


def plot_mse(mses: List[float], i: int, out_path: str | None = None) -> None:
    """Plot MSE values.

    Parameters
    ----------
    mses:
        Sequence of mean squared error values for progressively more basis
        functions.
    i:
        Index of the current basis set. Used for titling and default file name
        when saving.
    out_path:
        Optional path to save the plot. If omitted, the plot is shown when the
        backend is interactive. In non-interactive environments, the plot is
        automatically saved to ``mse_basis_{i}.png`` in the current directory.
    """
    fig, ax = plt.subplots()
    xs = np.arange(1, len(mses) + 1)
    ax.plot(xs, mses, marker="o")
    ax.set_xlabel("Number of basis functions")
    ax.set_ylabel("Mean squared error")
    ax.set_title(f"MSE for basis_{i}")
    ax.grid(True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return

    backend = plt.get_backend().lower()
    if "agg" in backend:
        auto_path = f"mse_basis_{i}.png"
        fig.savefig(auto_path)
        warnings.warn(
            "Current Matplotlib backend is non-interactive; "
            f"plot saved to {auto_path}.",
            RuntimeWarning,
        )
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSE between sum(c_j*basis_j) and data/functions.wave"
    )
    parser.add_argument("--i", type=int, required=True,
                        help="Number of basis functions to use")
    parser.add_argument("--coef-folder", default="coefs_process",
                        help="Directory with basis_{i}.json files")
    parser.add_argument("--basis-root", default="data",
                        help="Root directory containing basis_i folders")
    parser.add_argument("--function-file", default="data/functions.wave",
                        help="Path to the real function array")
    parser.add_argument(
        "--save",
        default=None,
        help=(
            "Path to save the MSE plot (PNG). If omitted, the plot is shown "
            "when possible; otherwise it is written to mse_basis_<i>.png."
        ),
    )
    args = parser.parse_args()

    # Load real function
    function = np.loadtxt(args.function_file)

    # Load coefficients using plot_basis_maps module
    coef_path = os.path.join(args.coef_folder, f"basis_{args.i}.json")
    _xs, _ys, _approx, coef_arrays = load_basis(coef_path)
    coeffs = _mean_coeffs(coef_arrays)

    # Load basis wave arrays
    basis_waves = _load_basis_waves(args.basis_root, args.i)

    # Compute MSE for incremental number of basis functions
    mses = compute_mse(function, basis_waves, coeffs)

    # Plot results
    plot_mse(mses, args.i, args.save)


if __name__ == "__main__":
    main()
