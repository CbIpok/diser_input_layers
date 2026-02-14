#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np

# Ensure project root is on sys.path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
from diser.io.coeffs import read_coef_json, resolve_coeffs_dir
from diser.viz.figio import save_figure_bundle


def _resolve_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    p = os.path.expanduser(str(path_value))
    if os.path.isabs(p) or os.path.exists(p):
        return p
    candidate = os.path.join(REPO_ROOT, p)
    return candidate


def _box_sum(arr: np.ndarray, radius: int) -> np.ndarray:
    r = int(radius)
    if r <= 0:
        return np.asarray(arr, dtype=np.float64)
    H, W = arr.shape
    padded = np.pad(arr, ((r, r), (r, r)), mode="constant", constant_values=0)
    integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integ = integ.cumsum(axis=0).cumsum(axis=1)
    y1 = np.arange(0, H)
    y2 = y1 + 2 * r
    x1 = np.arange(0, W)
    x2 = x1 + 2 * r
    A = integ[np.ix_(y2 + 1, x2 + 1)]
    B = integ[np.ix_(y1, x2 + 1)]
    C = integ[np.ix_(y2 + 1, x1)]
    D = integ[np.ix_(y1, x1)]
    return A - B - C + D


def _window_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    finite = np.isfinite(arr)
    vals = np.where(finite, arr, 0.0)
    counts = _box_sum(finite.astype(np.int32), radius)
    sums = _box_sum(vals, radius)
    mean = np.full_like(arr, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return mean


def interpolate_nan_window(arr: np.ndarray, max_radius: int = 5) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    max_r = int(max_radius)
    for radius in range(1, max_r + 1):
        missing = ~np.isfinite(out)
        if not missing.any():
            break
        mean = _window_mean(out, radius)
        fillable = missing & np.isfinite(mean)
        out[fillable] = mean[fillable]
    return out


def build_aprox_error_grid(i_value: int,
                           folder: str,
                           functions_path: str,
                           stride: int = 1,
                           shape: tuple[int, int] | None = None) -> np.ndarray:
    folder = _resolve_path(folder)
    functions_path = _resolve_path(functions_path)
    coefs_dir = resolve_coeffs_dir(folder, functions_path)
    file_path = coefs_dir / f"basis_{i_value}.json"
    samples = read_coef_json(file_path)
    xs = np.asarray(samples.xs, dtype=int)
    ys = np.asarray(samples.ys, dtype=int)
    approx = np.asarray(samples.approx_error, dtype=float) if samples.approx_error is not None else np.array([])

    stride = int(stride)
    xi = xs * stride
    yi = ys * stride
    if shape is None:
        if xi.size and yi.size:
            H = int(np.nanmax(yi)) + 1
            W = int(np.nanmax(xi)) + 1
        else:
            H, W = 0, 0
        shape = (H, W)
    out = np.full(shape, np.nan, dtype=float)
    if approx.size:
        mask = (
            (yi >= 0) & (yi < out.shape[0]) &
            (xi >= 0) & (xi < out.shape[1]) &
            np.isfinite(approx)
        )
        out[yi[mask], xi[mask]] = approx[mask]
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Build aprox_error grid from basis_{i}.json with NaN interpolation window."
    )
    p.add_argument("--i", type=int, required=True, help="Coefficient count i (basis_{i}.json)")
    p.add_argument("--folder", default="coefs_process", help="Root folder with coefficients")
    p.add_argument("--functions", default="data/functions.wave", help="Functions file to resolve coeffs folder")
    p.add_argument("--out", required=True, help="Output .npy path for aprox_error grid")
    p.add_argument("--stride", type=int, default=1, help="Stride to map decimated coords to full grid")
    p.add_argument("--shape", default=None, help="Optional output shape H,W (e.g. 3000,2000)")
    p.add_argument("--interp-radius", type=int, default=5, help="Max interpolation window radius (<=5 recommended)")
    p.add_argument("--save-dir", default=None, help="Directory to save PNG/SVG visualization (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    args.folder = _resolve_path(args.folder)
    args.functions = _resolve_path(args.functions)
    shape = None
    if args.shape:
        parts = [p.strip() for p in str(args.shape).split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--shape must be 'H,W'")
        shape = (int(parts[0]), int(parts[1]))
    grid = build_aprox_error_grid(
        args.i, args.folder, args.functions, stride=args.stride, shape=shape
    )
    if args.interp_radius and args.interp_radius > 0:
        grid = interpolate_nan_window(grid, max_radius=args.interp_radius)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, grid)

    if args.save_dir is not None:
        import matplotlib.pyplot as plt

        os.makedirs(args.save_dir, exist_ok=True)
        fig = plt.figure(figsize=(10, 8))
        im = plt.imshow(grid, origin="upper", cmap="viridis")
        plt.colorbar(im, label="aprox_error (interpolated)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"aprox_error (interpolated), i={args.i}, radius={args.interp_radius}")
        save_figure_bundle(
            fig,
            os.path.join(args.save_dir, "aprox_error_interpolated"),
            formats=("png", "svg"),
            with_pickle=True,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
