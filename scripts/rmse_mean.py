#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from plot_basis_maps import average_rmse_over_i
from diser.viz.figio import save_figure_bundle


def _sanitize_tag(value: str) -> str:
    return value.replace('.', '_')


def _build_base_name(i_list: List[int], functions_path: str) -> str:
    i_tag = '_'.join(str(i) for i in i_list)
    func_tag = _sanitize_tag(Path(functions_path).stem)
    return f"rmse_mean__i_{i_tag}__func_{func_tag}"


def _format_sigma_tag(sigma: float | None) -> str:
    if sigma is None or sigma == 0:
        return 'none'
    return _sanitize_tag(str(sigma))


def _save_grid(grid: np.ndarray,
               out_path: Path,
               title_suffix: str,
               fig_dir: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, grid)

    import matplotlib.pyplot as plt
    from matplotlib import cm

    finite_mask = np.isfinite(grid)
    finite_points = int(np.count_nonzero(finite_mask))
    total_points = grid.size

    cmap = cm.get_cmap('inferno').copy()
    cmap.set_bad('white', alpha=0.0)

    masked = np.ma.masked_invalid(grid)
    if finite_points:
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
    else:
        vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(masked, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    label = 'RMSE mean'
    if finite_points != total_points:
        label += ' (sparse)'
    plt.colorbar(im, label=label)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'RMSE mean [{title_suffix}]')

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_base = fig_dir / out_path.stem
    save_figure_bundle(fig, fig_base, formats=("png", "svg"), with_pickle=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Mean RMSE grid over multiple i (optionally pre-smoothed reconstructions)')
    p.add_argument('--i-list', required=True, help='Comma-separated list of i values, e.g. 4,16,25')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--smooth-sigma', type=float, default=None,
                   help='Gaussian sigma to pre-smooth reconstructed forms before RMSE')
    p.add_argument('--out-dir', default='output/rmse_mean', help='Directory where output grids/figures will be stored')
    p.add_argument('--visualize-only', action='store_true',
                   help='Skip RMSE recompute; regenerate figures from existing npy files')
    return p.parse_args()


def _find_smoothed_path(base_name: str, sigma: float, out_dir: Path) -> Path | None:
    candidates = [
        out_dir / f"{base_name}__recon_sigma_{_format_sigma_tag(sigma)}.npy",
        out_dir / f"{base_name}__recon_sigma_{_sanitize_tag(f'{sigma:g}')}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _visualize_only(i_list: List[int], functions_path: str, smooth_sigma: float | None, out_dir: Path) -> None:
    fig_dir = out_dir
    base_name = _build_base_name(i_list, functions_path)

    raw_path = out_dir / f"{base_name}__recon_sigma_none.npy"
    if raw_path.exists():
        raw_grid = np.load(raw_path)
        _save_grid(raw_grid, raw_path, f"i_list={i_list}, recon_sigma=none", fig_dir)
    else:
        print(f"[warn] raw grid not found: {raw_path}")

    if smooth_sigma and smooth_sigma > 0:
        smooth_path = _find_smoothed_path(base_name, smooth_sigma, out_dir)
        if smooth_path:
            smooth_grid = np.load(smooth_path)
            _save_grid(
                smooth_grid,
                smooth_path,
                f"i_list={i_list}, recon_sigma={smooth_sigma:g}",
                fig_dir,
            )
        else:
            print(f"[warn] smoothed grid not found for sigma={smooth_sigma}: {base_name}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir

    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]

    if args.visualize_only:
        _visualize_only(i_list, args.functions, args.smooth_sigma, out_dir)
        return

    base_name = _build_base_name(i_list, args.functions)

    raw_grid, _ = average_rmse_over_i(
        i_list,
        folder=args.folder,
        basis_root=args.basis_root,
        functions_path=args.functions,
        smooth_sigma=None,
    )
    raw_path = out_dir / f"{base_name}__recon_sigma_none.npy"
    _save_grid(raw_grid, raw_path, f"i_list={i_list}, recon_sigma=none", fig_dir)

    if args.smooth_sigma and args.smooth_sigma > 0:
        smooth_grid, _ = average_rmse_over_i(
            i_list,
            folder=args.folder,
            basis_root=args.basis_root,
            functions_path=args.functions,
            smooth_sigma=args.smooth_sigma,
        )
        sigma_tag = _format_sigma_tag(args.smooth_sigma)
        smooth_path = out_dir / f"{base_name}__recon_sigma_{sigma_tag}.npy"
        _save_grid(
            smooth_grid,
            smooth_path,
            f"i_list={i_list}, recon_sigma={args.smooth_sigma:g}",
            fig_dir,
        )


if __name__ == '__main__':
    main()
