#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import numpy as np

from diser.io.basis import load_basis_dir, load_functions_wave
from diser.io.coeffs import read_coef_json, resolve_coeffs_dir
from diser.core.restore import pointwise_rmse_from_coefs
from diser.viz.maps import plot_triangulated, plot_grid_scalar


def parse_args():
    p = argparse.ArgumentParser(description='Plot RMSE map and coefficient maps')
    p.add_argument('--i', type=int, default=16, help='Number of coefficients (basis_{i}.json)')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--save-dir', default=None, help='Directory to save figures (otherwise show)')
    return p.parse_args()


def main():
    args = parse_args()
    coefs_dir = resolve_coeffs_dir(args.folder, args.functions)
    coef_path = coefs_dir / f"basis_{args.i}.json"
    basis_path = os.path.join(args.basis_root, f"basis_{args.i}")

    samples = read_coef_json(coef_path)
    bases = load_basis_dir(basis_path)
    target = load_functions_wave(args.functions)

    rmse = pointwise_rmse_from_coefs(target, bases, samples.xs, samples.ys, samples.coefs)

    figs = []
    fig, ax = plot_triangulated(samples.xs, samples.ys, samples.approx_error, title='approx_error')
    figs.append(('approx_error.png', fig))
    fig2, ax2 = plot_grid_scalar(rmse, title='RMSE map (points only)')
    figs.append(('rmse_points.png', fig2))

    for j, c in enumerate(samples.coefs, start=1):
        figj, axj = plot_triangulated(samples.xs, samples.ys, c, title=f'coef_{j}')
        figs.append((f'coef_{j}.png', figj))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for name, fig in figs:
            fig.savefig(os.path.join(args.save_dir, name))
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()

