#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from plot_basis_maps import average_rmse_over_i
from diser.core.restore import gaussian_smooth_nan
from diser.viz.figio import save_figure_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Mean RMSE grid over multiple i (optionally pre-smoothed reconstructions)')
    p.add_argument('--i-list', required=True, help='Comma-separated list of i values, e.g. 4,16,25')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--smooth-sigma', type=float, default=None,
                   help='Gaussian sigma to pre-smooth reconstructed forms before RMSE (also used for the smoothed preview)')
    p.add_argument('--rmse-out', required=True, help='Base output .npy path for raw mean RMSE grid')
    p.add_argument('--save-dir', default=None, help='Directory to save PNG/SVG/PKL visualizations (defaults to rmse-out directory)')
    return p.parse_args()


def _suffix_path(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _format_sigma(sigma: float) -> str:
    if sigma is None:
        return 'none'
    text = f"{sigma:g}"
    return text.replace('.', '_')


def render_interpolated(grid: np.ndarray,
                        fig_base_path: str,
                        title_suffix: str,
                        display_sigma: float | None) -> None:
    import matplotlib.pyplot as plt

    finite_mask = np.isfinite(grid)
    H, W = grid.shape
    finite_points = int(np.count_nonzero(finite_mask))
    total_points = grid.size
    threshold = min(50000, 0.5 * total_points)
    fill_ratio = finite_points / total_points if total_points else 0.0

    if finite_points >= 3 and (finite_points <= threshold or fill_ratio <= 0.5):
        sigma = float(display_sigma) if display_sigma and display_sigma > 0 else 3.0
        grid_full = gaussian_smooth_nan(grid, sigma=sigma)
    else:
        grid_full = grid

    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(grid_full, origin='upper', cmap='inferno')
    plt.colorbar(im, label='RMSE mean' if finite_points == total_points else 'RMSE mean (interpolated)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'RMSE mean [{title_suffix}]')
    save_figure_bundle(fig, fig_base_path, formats=("png", "svg"), with_pickle=True)
    plt.close(fig)


def save_grid_bundle(grid: np.ndarray,
                     out_path: str,
                     title_suffix: str,
                     display_sigma: float | None,
                     save_dir: str | None) -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, grid)

    fig_base = os.path.splitext(out_path)[0]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(fig_base)
        fig_base_path = os.path.join(save_dir, base_name)
    else:
        fig_base_path = fig_base
    render_interpolated(grid, fig_base_path, title_suffix, display_sigma)


def main() -> None:
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]

    raw_grid, _ = average_rmse_over_i(
        i_list,
        folder=args.folder,
        basis_root=args.basis_root,
        functions_path=args.functions,
        smooth_sigma=None,
    )

    base_title = f"i_list={i_list}"
    save_grid_bundle(raw_grid, args.rmse_out, f"{base_title}, recon_sigma=none", None, args.save_dir)

    if args.smooth_sigma and args.smooth_sigma > 0:
        smooth_grid, _ = average_rmse_over_i(
            i_list,
            folder=args.folder,
            basis_root=args.basis_root,
            functions_path=args.functions,
            smooth_sigma=args.smooth_sigma,
        )
        sigma_tag = _format_sigma(args.smooth_sigma)
        smooth_path = _suffix_path(args.rmse_out, f"__smooth_sigma{sigma_tag}")
        save_grid_bundle(
            smooth_grid,
            smooth_path,
            f"{base_title}, recon_sigma={args.smooth_sigma:g}",
            float(args.smooth_sigma),
            args.save_dir,
        )


if __name__ == '__main__':
    main()
