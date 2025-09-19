#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
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
    base_name = os.path.splitext(os.path.basename(args.rmse_out))[0]
    fig_base = os.path.join(out_dir, base_name)
    # Render interpolated PNG/SVG/PKL of mean grid (no background)
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    def render_interpolated(grid, fig_base_path, title_suffix):
        finite_mask = np.isfinite(grid)
        H, W = grid.shape
        if np.count_nonzero(finite_mask) >= 3:
            yy, xx = np.nonzero(finite_mask)
            vals = grid[yy, xx]
            triang = tri.Triangulation(xx.astype(float), yy.astype(float))
            interpolator = tri.LinearTriInterpolator(triang, vals)
            Xi, Yi = np.meshgrid(np.arange(W), np.arange(H))
            grid_full = np.array(interpolator(Xi, Yi).filled(np.nan))
        else:
            grid_full = grid
        fig = plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_full, origin='upper', cmap='inferno')
        plt.colorbar(im, label='RMSE mean (interpolated)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'RMSE mean (interpolated) [{title_suffix}]')
        save_figure_bundle(fig, fig_base_path, formats=("png","svg"), with_pickle=True)
        plt.close(fig)

    title_suffix = f"i_list={i_list}"
    render_interpolated(mean_grid, fig_base, title_suffix)
    if args.rmse_smooth_out and smoothed is not None:
        smooth_title = title_suffix
        if args.smooth_sigma is not None:
            smooth_title = f"{title_suffix}, sigma={args.smooth_sigma}"
        smooth_base_name = os.path.splitext(os.path.basename(args.rmse_smooth_out))[0]
        smooth_fig_base = os.path.join(out_dir, smooth_base_name)
        render_interpolated(smoothed, smooth_fig_base, smooth_title)


if __name__ == '__main__':
    main()
