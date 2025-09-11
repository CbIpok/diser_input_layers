#!/usr/bin/env python
import argparse
import os
import numpy as np
from plot_basis_maps import average_rmse_over_i
from diser.viz.figio import save_figure_bundle


def parse_args():
    p = argparse.ArgumentParser(description='Mean RMSE grid over multiple i (optionally smoothed)')
    p.add_argument('--i-list', required=True, help='Comma-separated list of i values, e.g. 4,16,25')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--smooth-sigma', type=float, default=None, help='Gaussian sigma for smoothing (optional)')
    p.add_argument('--rmse-out', required=True, help='Output .npy for mean RMSE grid')
    p.add_argument('--rmse-smooth-out', default=None, help='Output .npy for smoothed mean RMSE grid (optional)')
    p.add_argument('--save-dir', default=None, help='Directory to save PNG visualizations (interpolated)')
    return p.parse_args()


def main():
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]
    mean_grid, smoothed = average_rmse_over_i(
        i_list,
        folder=args.folder,
        basis_root=args.basis_root,
        functions_path=args.functions,
        smooth_sigma=args.smooth_sigma,
    )
    os.makedirs(os.path.dirname(args.rmse_out) or '.', exist_ok=True)
    np.save(args.rmse_out, mean_grid)
    if args.rmse_smooth_out and smoothed is not None:
        os.makedirs(os.path.dirname(args.rmse_smooth_out) or '.', exist_ok=True)
        np.save(args.rmse_smooth_out, smoothed)

    out_dir = args.save_dir or (os.path.dirname(args.rmse_out) or '.')
    os.makedirs(out_dir, exist_ok=True)
    # Render interpolated PNG/SVG/PKL of mean grid (no background)
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    H, W = mean_grid.shape
    yy, xx = np.nonzero(np.isfinite(mean_grid))
    vals = mean_grid[yy, xx]
    if xx.size >= 3:
        triang = tri.Triangulation(xx.astype(float), yy.astype(float))
        interpolator = tri.LinearTriInterpolator(triang, vals)
        Xi, Yi = np.meshgrid(np.arange(W), np.arange(H))
        mean_full = np.array(interpolator(Xi, Yi).filled(np.nan))
    else:
        mean_full = mean_grid
    title_suffix = f"i_list={i_list}"
    if args.smooth_sigma:
        title_suffix += f", sigma={args.smooth_sigma}"
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(mean_full, origin='upper', cmap='inferno')
    plt.colorbar(im, label='RMSE mean (interpolated)')
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'RMSE mean (interpolated) [{title_suffix}]')
    save_figure_bundle(fig, os.path.join(out_dir, 'rmse_mean_interpolated'), formats=("png","svg"), with_pickle=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
