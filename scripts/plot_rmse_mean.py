#!/usr/bin/env python
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from diser.viz.figio import save_figure_bundle


def parse_args():
    p = argparse.ArgumentParser(description='Plot averaged RMSE grids from .npy (interpolated)')
    p.add_argument('--rmse', required=True, help='Path to mean RMSE grid .npy')
    p.add_argument('--rmse-smooth', default=None, help='Path to smoothed mean RMSE grid .npy (optional)')
    p.add_argument('--save-dir', default=None, help='Directory to save PNGs (otherwise show)')
    return p.parse_args()


def interpolate_and_plot(grid: np.ndarray, title: str, path: str | None = None):
    H, W = grid.shape
    yy, xx = np.nonzero(np.isfinite(grid))
    vals = grid[yy, xx]
    if xx.size >= 3:
        triang = tri.Triangulation(xx.astype(float), yy.astype(float))
        interpolator = tri.LinearTriInterpolator(triang, vals)
        Xi, Yi = np.meshgrid(np.arange(W), np.arange(H))
        full = np.array(interpolator(Xi, Yi).filled(np.nan))
    else:
        full = grid

    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(full, origin='upper', cmap='inferno')
    plt.colorbar(im, label=title)
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(title)
    if path:
        save_figure_bundle(fig, path, formats=("png", "svg"), with_pickle=True)
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()
    rmse = np.load(args.rmse)
    out_dir = args.save_dir
    if out_dir:
        interpolate_and_plot(rmse, 'RMSE mean (interpolated)', os.path.join(out_dir, 'rmse_mean_interpolated.png'))
    else:
        interpolate_and_plot(rmse, 'RMSE mean (interpolated)', None)

    if args.rmse_smooth and os.path.exists(args.rmse_smooth):
        sm = np.load(args.rmse_smooth)
        if out_dir:
            interpolate_and_plot(sm, 'RMSE mean (smoothed, interpolated)', os.path.join(out_dir, 'rmse_mean_smoothed_interpolated.png'))
        else:
            interpolate_and_plot(sm, 'RMSE mean (smoothed, interpolated)', None)


if __name__ == '__main__':
    main()
