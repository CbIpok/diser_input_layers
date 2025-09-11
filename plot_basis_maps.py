"""Plot 2D height maps for approximation error and coefficients.

Reads coefs_process/basis_{i}.json and can:
- compute RMSE grid (heavy computation) and optionally save to .npy
- plot precomputed RMSE grid
- plot triangulated approx_error and coefficient maps

Backwards-compatible functions load_basis_coofs/load_basis/calc_mse are kept for tests,
but calc_mse now delegates to a compute helper.
"""
import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import numpy as np

from loader import load_bath_grid
from diser.io.coeffs import read_coef_json
from diser.io.basis import load_basis_dir
from diser.core.restore import pointwise_rmse_from_coefs, gaussian_smooth_nan
from diser.viz.figio import save_figure_bundle


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_basis_coofs(path: str):
    samples = read_coef_json(path)
    xs_arr = np.asarray(samples.xs, dtype=float)
    ys_arr = np.asarray(samples.ys, dtype=float)
    approx_arr = (np.asarray(samples.approx_error, dtype=float)
                  if samples.approx_error is not None else np.array([]))
    coef_arrays = [np.asarray(c, dtype=float) for c in samples.coefs]
    return xs_arr, ys_arr, approx_arr, coef_arrays


def load_basis(path: str):
    B = load_basis_dir(path)
    return [B[i] for i in range(B.shape[0])]


def build_rmse_grid(to_restore, bases, xs_arr, ys_arr, coef_arrays, dtype=np.float64):
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])

    k, Hb, Wb = bases.shape
    Hg, Wg = grid.shape
    assert (Hb, Wb) == (Hg, Wg), "bases и grid должны совпадать по размеру"
    assert to_restore.shape == (Hg, Wg), "to_restore должен совпадать по размеру с grid"

    stride = int(cfg.get('save_interval', 1))
    # 1) Считаем RMSE в точках как есть (без stride) и размещаем во временной решётке
    xs_un = np.asarray(xs_arr, dtype=int)
    ys_un = np.asarray(ys_arr, dtype=int)
    rmse_un = pointwise_rmse_from_coefs(to_restore, bases, xs_un, ys_un, coef_arrays, dtype=dtype)

    # 2) Перекладываем значения в итоговую решётку по координатам с учётом stride
    out = np.full_like(rmse_un, np.nan)
    xi = (xs_un * stride).astype(int)
    yi = (ys_un * stride).astype(int)
    # берём значения из не-страйдовой решётки в исходных координатах
    vals = rmse_un[ys_un, xs_un]
    # и кладём их в страйдовые координаты
    valid = np.isfinite(vals)
    out[yi[valid], xi[valid]] = vals[valid]
    return out


def calc_mse(to_restore, bases, xs_arr, ys_arr, coef_arrays, block=512, dtype=np.float32):
    mse = build_rmse_grid(to_restore, bases, xs_arr, ys_arr, coef_arrays, dtype=np.float64)

    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(mse.T, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='MSE')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title('MSE Distribution')
    plt.show()

    return mse


def plot_aprox_error_raw(xs: np.ndarray,
                         ys: np.ndarray,
                         approx_err: np.ndarray,
                         out_dir: str | None = None):
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])
    H, W = grid.shape

    stride = int(cfg.get('save_interval', 1))
    xi = np.asarray(xs, dtype=int) * stride  # x -> col
    yi = np.asarray(ys, dtype=int) * stride  # y -> row

    raw = np.full((H, W), np.nan, dtype=float)
    mask = (
        (yi >= 0) & (yi < H) &
        (xi >= 0) & (xi < W) &
        np.isfinite(approx_err)
    )
    raw[yi[mask], xi[mask]] = approx_err[mask]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(raw.T, origin='lower', interpolation='nearest', cmap='viridis')
    plt.colorbar(im, label='aprox_error (raw)')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title('aprox_error (raw)')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'aprox_error_raw.png')
        plt.savefig(path)
        plt.close()
        return [path]
    else:
        plt.show()
        return []


def plot_maps(xs: np.ndarray, ys: np.ndarray, approx_err: np.ndarray,
              coefs: List[np.ndarray], out_dir: str | None = None):
    variables = [("approx_error", approx_err)]
    variables += [(f"coef_{i+1}", arr) for i, arr in enumerate(coefs)]

    try:
        with open('data/config.json', 'r', encoding='utf-8') as f:
            stride = int(json.load(f).get('save_interval', 1))
    except FileNotFoundError:
        stride = 1
    xs_tri = np.asarray(xs, dtype=float) * stride
    ys_tri = np.asarray(ys, dtype=float) * stride

    output_files = []
    for title, arr in variables:
        mask = np.isfinite(arr)
        if mask.sum() < 3:
            continue
        triang = tri.Triangulation(xs_tri[mask], ys_tri[mask])
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
        description="Compute/plot maps from basis_i.json"
    )
    parser.add_argument("--folder", default="coefs_process",
                        help="Directory containing basis_{i}.json")
    parser.add_argument("--i", type=int, default=4,
                        help="Number of coefficients (file basis_{i}.json)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save PNG plots. If omitted, plots are shown")
    parser.add_argument("--mode", choices=["compute", "plot", "both"], default="both",
                        help="compute: only compute RMSE and optionally save; plot: only plot (optionally from --rmse-in); both: compute and plot")
    parser.add_argument("--rmse-out", default=None,
                        help="Path to save computed RMSE grid (.npy) in compute/both mode")
    parser.add_argument("--rmse-in", default=None,
                        help="Path to load RMSE grid (.npy) in plot mode")
    args = parser.parse_args()

    file_path = os.path.join(args.folder, f"basis_{args.i}.json")
    xs, ys, approx_err, coefs = load_basis_coofs(file_path)

    basis = np.array(load_basis(f"data/basis_{args.i}"))
    to_restore = np.loadtxt("data/functions.wave")

    if args.mode in ("compute", "both"):
        rmse = build_rmse_grid(to_restore, basis, xs, ys, coefs)
        if args.rmse_out:
            os.makedirs(os.path.dirname(args.rmse_out) or '.', exist_ok=True)
            np.save(args.rmse_out, rmse)

    if args.mode in ("plot", "both"):
        if args.rmse_in and args.mode == "plot":
            rmse = np.load(args.rmse_in)

        # Interpolate RMSE from sparse points ONLY (no background/areas)
        H, W = to_restore.shape
        yy, xx = np.nonzero(np.isfinite(rmse))
        vals = rmse[yy, xx]
        if xx.size >= 3:
            triang = tri.Triangulation(xx.astype(float), yy.astype(float))
            interpolator = tri.LinearTriInterpolator(triang, vals)
            Xi, Yi = np.meshgrid(np.arange(W), np.arange(H))
            rmse_full = interpolator(Xi, Yi)
            rmse_full = np.array(rmse_full.filled(np.nan))
        else:
            rmse_full = rmse

        fig = plt.figure(figsize=(10, 8))
        im = plt.imshow(rmse_full, origin='upper', cmap='inferno')
        plt.colorbar(im, label='RMSE (interpolated)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RMSE (interpolated)')
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_figure_bundle(fig, os.path.join(args.save_dir, 'rmse_interpolated'), formats=("png","svg"), with_pickle=True)
            plt.close(fig)
        else:
            plt.show()


# --- Programmatic API for averaging RMSE over multiple i ---
def compute_rmse_for_i(i: int,
                       folder: str = 'coefs_process',
                       basis_root: str = 'data',
                       functions_path: str = 'data/functions.wave') -> np.ndarray:
    file_path = os.path.join(folder, f"basis_{i}.json")
    xs, ys, _, coefs = load_basis_coofs(file_path)
    basis = np.array(load_basis(os.path.join(basis_root, f"basis_{i}")))
    to_restore = np.loadtxt(functions_path)
    return build_rmse_grid(to_restore, basis, xs, ys, coefs)


def average_rmse_over_i(i_list,
                        folder: str = 'coefs_process',
                        basis_root: str = 'data',
                        functions_path: str = 'data/functions.wave',
                        smooth_sigma: float | None = None):
    grids = [compute_rmse_for_i(i, folder, basis_root, functions_path) for i in i_list]
    mean_grid = np.nanmean(np.stack(grids, axis=0), axis=0)
    smoothed = gaussian_smooth_nan(mean_grid, sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else None
    return mean_grid, smoothed


if __name__ == "__main__":
    main()
