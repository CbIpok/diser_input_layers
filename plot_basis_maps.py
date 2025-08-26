"""Plot 2D height maps for approximation error and coefficients.

This script reads files of the form ``coefs_process/basis_{i}.json``
where each entry has a key ``"[x,y]"`` describing coordinates and a
value with ``"aprox_error"`` and a list of ``i`` coefficients.
The coordinates may form a non-rectangular domain, so plots are built
using triangulation.

Example usage for ``i=4``:

    python plot_basis_maps.py --i 4 --save-dir output

This will create separate PNG files for the approximation error and
each coefficient in the ``output`` directory.
"""
import argparse
import ast
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # Ensure plots can be created without a display
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def load_basis(path: str):
    """Load coordinate and coefficient data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    xs: List[float] = []
    ys: List[float] = []
    approx_errors: List[float] = []
    coefs: List[List[float]] = []

    for key, val in data.items():
        x, y = ast.literal_eval(key)
        xs.append(x)
        ys.append(y)
        approx_errors.append(val["aprox_error"])
        coef_vals = val["coefs"]
        if not coefs:
            coefs = [[] for _ in coef_vals]
        for idx, c in enumerate(coef_vals):
            coefs[idx].append(c)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    approx_arr = np.array(approx_errors)
    coef_arrays = [np.array(c) for c in coefs]
    return xs_arr, ys_arr, approx_arr, coef_arrays


def plot_maps(xs: np.ndarray, ys: np.ndarray, approx_err: np.ndarray,
              coefs: List[np.ndarray], out_dir: str | None = None):
    """Generate height maps for approximation error and coefficients."""
    triang = tri.Triangulation(xs, ys)
    variables = [("approx_error", approx_err)]
    variables += [(f"coef_{i+1}", arr) for i, arr in enumerate(coefs)]

    output_files = []
    for title, arr in variables:
        fig, ax = plt.subplots()
        tpc = ax.tricontourf(triang, arr, levels=100)
        ax.set_title(title.replace("_", " "))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        fig.colorbar(tpc, ax=ax)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{title}.png"
            path = os.path.join(out_dir, filename)
            fig.savefig(path)
            output_files.append(path)
            plt.close(fig)
        else:
            plt.show()
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D height maps from basis_i.json files"
    )
    parser.add_argument(
        "--folder", default="coefs_process",
        help="Directory containing basis_{i}.json"
    )
    parser.add_argument(
        "--i", type=int, default=4,
        help="Number of coefficients (file basis_{i}.json)"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save PNG plots. If omitted, plots are shown"
    )
    args = parser.parse_args()

    file_path = os.path.join(args.folder, f"basis_{args.i}.json")
    xs, ys, approx_err, coefs = load_basis(file_path)
    plot_maps(xs, ys, approx_err, coefs, args.save_dir)


if __name__ == "__main__":
    main()
