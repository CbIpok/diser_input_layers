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
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def _to_float(value) -> float:
    """Return float(value) or NaN if conversion fails or value is None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_basis_coofs(path: str):
    """Load coordinate and coefficient data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    xs: List[float] = []
    ys: List[float] = []
    approx_errors: List[float] = []
    coefs: List[List[float]] = []

    for key, val in data.items():
        x, y = ast.literal_eval(key)
        xs.append(float(x))
        ys.append(float(y))
        approx_errors.append(_to_float(val.get("aprox_error")))
        coef_vals = [_to_float(c) for c in val.get("coefs", [])]
        if not coefs:
            coefs = [[] for _ in coef_vals]
        for idx, c in enumerate(coef_vals):
            coefs[idx].append(c)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    approx_arr = np.asarray(approx_errors, dtype=float)
    coef_arrays = [np.asarray(c, dtype=float) for c in coefs]
    return xs_arr, ys_arr, approx_arr, coef_arrays


def load_basis(path: str):
    arrays = []
    indices = []

    # Проверяем, что путь существует
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist")

    # Перебираем файлы в директории
    for filename in os.listdir(path):
        if filename.startswith("basis_") and filename.endswith(".wave"):
            # Извлекаем числовую часть между 'basis_' и '.wave'
            try:
                index = int(filename.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                continue  # Пропускаем файлы с неправильным форматом

            full_path = os.path.join(path, filename)
            # Загружаем данные из файла
            data = np.loadtxt(full_path)
            # Заменяем 0 на NaN
            data[data == 0] = np.nan
            arrays.append(data)
            indices.append(index)

    # Сортируем массивы по индексам
    sorted_arrays = [array for _, array in sorted(zip(indices, arrays))]
    return sorted_arrays


def plot_maps(xs: np.ndarray, ys: np.ndarray, approx_err: np.ndarray,
              coefs: List[np.ndarray], out_dir: str | None = None):
    """Generate height maps for approximation error and coefficients."""
    variables = [("approx_error", approx_err)]
    variables += [(f"coef_{i+1}", arr) for i, arr in enumerate(coefs)]

    output_files = []
    for title, arr in variables:
        mask = np.isfinite(arr)
        if mask.sum() < 3:
            # Not enough points to build triangulation
            continue
        triang = tri.Triangulation(xs[mask], ys[mask])
        fig, ax = plt.subplots()
        tpc = ax.tricontourf(triang, arr[mask], levels=100)
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
    xs, ys, approx_err, coefs = load_basis_coofs(file_path)
    plot_maps(xs, ys, approx_err, coefs, args.save_dir)


if __name__ == "__main__":
    main()
