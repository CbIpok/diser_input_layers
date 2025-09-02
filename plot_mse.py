import argparse
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_basis_maps import load_basis


def reconstruct(basis_dir: str, xs: np.ndarray, ys: np.ndarray,
                coefs: list[np.ndarray], shape: Tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct approximation using basis functions and coefficients."""
    approx = np.zeros(shape, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    ix = xs.astype(int)
    iy = ys.astype(int)
    for j, coef in enumerate(coefs):
        basis_path = os.path.join(basis_dir, f"basis_{j}.wave")
        basis = np.loadtxt(basis_path)
        mask |= basis != 0
        coef_grid = np.zeros(shape, dtype=float)
        coef_grid[iy, ix] = coef
        approx += coef_grid * basis
    return approx, mask


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot MSE of approximation for given i.")
    parser.add_argument("--i", type=int, required=True,
                        help="Number of basis functions (folder basis_i and file basis_i.json)")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing basis_i folders and functions.wave")
    parser.add_argument("--coefs-dir", default="coefs_process",
                        help="Directory containing basis_{i}.json")
    parser.add_argument("--out", default="mse_diff.png",
                        help="Path to save the difference plot")
    args = parser.parse_args()

    func_path = os.path.join(args.data_dir, "functions.wave")
    real = np.loadtxt(func_path)

    coef_file = os.path.join(args.coefs_dir, f"basis_{args.i}.json")
    xs, ys, _, coefs = load_basis(coef_file)

    basis_dir = os.path.join(args.data_dir, f"basis_{args.i}")
    approx, mask = reconstruct(basis_dir, xs, ys, coefs, real.shape)

    diff = approx - real
    mse = np.mean((diff[mask]) ** 2)
    print(f"MSE for i={args.i}: {mse}")

    fig, ax = plt.subplots()
    im = ax.imshow(np.where(mask, diff, np.nan), origin="lower")
    ax.set_title(f"Approximation error for i={args.i}\nMSE={mse:.4f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, label="Approx - Real")
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
