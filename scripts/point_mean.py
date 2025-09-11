#!/usr/bin/env python
import argparse
import os
import numpy as np
from point import reconstruct_mean_over_i
from diser.viz.figio import save_figure_bundle
# Edge-aware post-filtering removed per request; keep pure mean output


def parse_args():
    p = argparse.ArgumentParser(description='Mean reconstruction over multiple i (optionally smoothed)')
    p.add_argument('--i-list', required=True,
                   help='Comma-separated list of i values, e.g. 4,16,25')
    p.add_argument('--point', nargs=2, type=float, default=[100, 547],
                   help='Point (x y) to pick coefficients for')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--smooth-sigma', type=float, default=None, help='Gaussian sigma for smoothing (optional)')
    p.add_argument('--save-mean', required=True, help='Output .npy for mean reconstruction')
    p.add_argument('--save-smooth', default=None, help='Output .npy for smoothed mean (optional)')
    p.add_argument('--save-dir', default=None, help='Directory to save PNG visualizations')
    p.add_argument('--subtract-functions', action='store_true',
                   help='Subtract functions.wave from the resulting mean (and smoothed) before saving and plotting')
    return p.parse_args()


def main():
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]
    mean_Z, smoothed = reconstruct_mean_over_i(
        i_list,
        folder=args.folder,
        basis_root=args.basis_root,
        functions_path=args.functions,
        point_xy=tuple(args.point),
        smooth_sigma=args.smooth_sigma,
    )
    # Note: previously applied an edge-aware filter to mean_Z here; removed.
    # Optionally subtract functions.wave from the resulting forms
    if args.subtract_functions:
        func = np.loadtxt(args.functions)
        if func.shape != mean_Z.shape:
            raise AssertionError(f"functions.wave shape {func.shape} differs from mean shape {mean_Z.shape}")
        mean_Z = mean_Z - func
        if smoothed is not None:
            if smoothed.shape != func.shape:
                raise AssertionError(f"functions.wave shape {func.shape} differs from smoothed mean shape {smoothed.shape}")
            smoothed = smoothed - func
    os.makedirs(os.path.dirname(args.save_mean) or '.', exist_ok=True)
    np.save(args.save_mean, mean_Z)
    if args.save_smooth and smoothed is not None:
        os.makedirs(os.path.dirname(args.save_smooth) or '.', exist_ok=True)
        np.save(args.save_smooth, smoothed)
    # Decide output directory for figures
    out_dir = args.save_dir or (os.path.dirname(args.save_mean) or '.')
    os.makedirs(out_dir, exist_ok=True)

    # Metadata for titles
    title_suffix = f"i_list={i_list}, point=({int(args.point[0])},{int(args.point[1])})"
    if args.smooth_sigma:
        title_suffix += f", sigma={args.smooth_sigma}"

    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(8, 6))
    plt.imshow(mean_Z, origin='upper', cmap='viridis')
    cbar_label = 'Mean reconstruction'
    title_base = 'Mean reconstruction'
    base_name = 'mean_reconstruction'
    if args.subtract_functions:
        cbar_label = 'Mean reconstruction - functions.wave'
        title_base = 'Mean reconstruction (minus functions.wave)'
        base_name = 'mean_reconstruction_minus_functions'
    plt.colorbar(label=cbar_label)
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'{title_base} [{title_suffix}]')
    save_figure_bundle(fig1, os.path.join(out_dir, base_name), formats=("png","svg"), with_pickle=True)
    plt.close(fig1)

    if smoothed is not None:
        fig2 = plt.figure(figsize=(8, 6))
        plt.imshow(smoothed, origin='upper', cmap='viridis')
        cbar_label2 = 'Mean reconstruction (smoothed)'
        title2 = 'Mean reconstruction (smoothed)'
        base2 = 'mean_reconstruction_smoothed'
        if args.subtract_functions:
            cbar_label2 = 'Mean reconstruction - functions.wave (smoothed input)'
            title2 = 'Mean reconstruction (smoothed) minus functions.wave'
            base2 = 'mean_reconstruction_smoothed_minus_functions'
        plt.colorbar(label=cbar_label2)
        plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'{title2} [{title_suffix}]')
        save_figure_bundle(fig2, os.path.join(out_dir, base2), formats=("png","svg"), with_pickle=True)
        plt.close(fig2)


if __name__ == '__main__':
    main()
