#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plot_basis_maps import average_mean_error_over_i
from diser.viz.figio import save_figure_bundle


def parse_args():
    p = argparse.ArgumentParser(description='Mean (signed) error grid over multiple i (optionally smoothed)')
    p.add_argument('--i-list', required=True, help='Comma-separated list of i values, e.g. 4,16,25')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--smooth-sigma', type=float, default=None, help='Gaussian sigma for smoothing (optional)')
    p.add_argument('--mean-out', required=True, help='Output .npy for mean error grid')
    p.add_argument('--mean-smooth-out', default=None, help='Output .npy for smoothed mean error grid (optional)')
    p.add_argument('--save-dir', default=None, help='Directory to save PNG visualizations (interpolated)')
    return p.parse_args()


def main():
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]
    mean_grid, smoothed = average_mean_error_over_i(
        i_list,
        folder=args.folder,
        basis_root=args.basis_root,
        functions_path=args.functions,
        smooth_sigma=args.smooth_sigma,
    )
    os.makedirs(os.path.dirname(args.mean_out) or '.', exist_ok=True)
    np.save(args.mean_out, mean_grid)
    if args.mean_smooth_out and smoothed is not None:
        os.makedirs(os.path.dirname(args.mean_smooth_out) or '.', exist_ok=True)
        np.save(args.mean_smooth_out, smoothed)

    out_dir = args.save_dir or (os.path.dirname(args.mean_out) or '.')
    os.makedirs(out_dir, exist_ok=True)
    # Interpolated visualization (RMSE-like style) but for mean error
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
    im = plt.imshow(mean_full, origin='upper', cmap='plasma')
    plt.colorbar(im, label='Mean error (interpolated)')
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'Mean error (interpolated) [{title_suffix}]')
    save_figure_bundle(fig, os.path.join(out_dir, 'mean_interpolated'), formats=("png","svg"), with_pickle=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
