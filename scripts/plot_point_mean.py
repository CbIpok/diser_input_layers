#!/usr/bin/env python
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from diser.viz.figio import save_figure_bundle


def parse_args():
    p = argparse.ArgumentParser(description='Plot averaged reconstruction surfaces from .npy')
    p.add_argument('--mean', required=True, help='Path to mean reconstruction .npy')
    p.add_argument('--smooth', default=None, help='Path to smoothed mean reconstruction .npy (optional)')
    p.add_argument('--save-dir', default=None, help='Directory to save PNGs (otherwise show)')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave to subtract')
    p.add_argument('--subtract-functions', action='store_true',
                   help='Subtract functions.wave from the loaded arrays before plotting')
    return p.parse_args()


def plot_grid(arr: np.ndarray, title: str, path: str | None = None):
    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, origin='upper', cmap='viridis')
    plt.colorbar(im, label=title)
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(title)
    if path:
        # Accept both base path and a path with extension; normalize to base
        from pathlib import Path
        p = Path(path)
        base = p.with_suffix('') if p.suffix else p
        save_figure_bundle(fig, base, formats=("png", "svg"), with_pickle=True)
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()
    mean = np.load(args.mean)
    if args.subtract_functions:
        func = np.loadtxt(args.functions)
        if func.shape != mean.shape:
            raise AssertionError(f"functions.wave shape {func.shape} differs from mean shape {mean.shape}")
        mean = mean - func
    out_dir = args.save_dir
    if out_dir:
        title = 'Mean reconstruction - functions.wave' if args.subtract_functions else 'Mean reconstruction'
        fname = 'mean_reconstruction_minus_functions' if args.subtract_functions else 'mean_reconstruction'
        plot_grid(mean, title, os.path.join(out_dir, fname))
    else:
        plot_grid(mean, 'Mean reconstruction - functions.wave' if args.subtract_functions else 'Mean reconstruction', None)

    if args.smooth and os.path.exists(args.smooth):
        sm = np.load(args.smooth)
        if args.subtract_functions:
            func = np.loadtxt(args.functions)
            if func.shape != sm.shape:
                raise AssertionError(f"functions.wave shape {func.shape} differs from smoothed mean shape {sm.shape}")
            sm = sm - func
        if out_dir:
            title2 = 'Mean reconstruction (smoothed) - functions.wave' if args.subtract_functions else 'Mean reconstruction (smoothed)'
            fname2 = 'mean_reconstruction_smoothed_minus_functions' if args.subtract_functions else 'mean_reconstruction_smoothed'
            plot_grid(sm, title2, os.path.join(out_dir, fname2))
        else:
            plot_grid(sm, 'Mean reconstruction (smoothed) - functions.wave' if args.subtract_functions else 'Mean reconstruction (smoothed)', None)


if __name__ == '__main__':
    main()
