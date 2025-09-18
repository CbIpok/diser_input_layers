#!/usr/bin/env python
import argparse
import os
import numpy as np
from pathlib import Path

from point import load_basis_coefs
from diser.io.basis import load_basis_dir
from diser.core.restore import reconstruct_from_bases
from diser.viz.figio import save_figure_bundle


def parse_args():
    p = argparse.ArgumentParser(description='Reconstruct surface for a specific i and save to file')
    p.add_argument('--i', type=int, required=True, help='Number of basis functions (uses basis_{i}.json, basis_{i}/)')
    p.add_argument('--point', nargs=2, type=float, required=True, help='Point (x y) to pick coefficients for')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--save', required=True, help='Output .npy path for reconstruction')
    p.add_argument('--save-dir', default=None, help='Directory to save 2D image (optional)')
    p.add_argument('--img-formats', default='png,svg', help='Comma-separated formats for figure')
    p.add_argument('--subtract-functions', action='store_true', help='Subtract functions.wave (or .npy if provided) before saving')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave (for subtraction only)')
    return p.parse_args()


def main():
    args = parse_args()
    coefs_json = os.path.join(args.folder, f'basis_{args.i}.json')
    basis_dir = os.path.join(args.basis_root, f'basis_{args.i}')

    xs, ys, coefs = load_basis_coefs(coefs_json)
    bases = load_basis_dir(basis_dir)

    x_sel, y_sel = map(float, args.point)
    # Select coefficients nearest to the point
    xs_f = np.asarray(xs, dtype=float)
    ys_f = np.asarray(ys, dtype=float)
    d2 = (xs_f - x_sel) ** 2 + (ys_f - y_sel) ** 2
    idx = int(np.argmin(d2))
    c = np.array([coef[idx] for coef in coefs], dtype=float)

    Z = reconstruct_from_bases(c, bases)

    # Optional subtraction of functions
    if args.subtract_functions:
        if str(args.functions).lower().endswith('.npy'):
            func = np.load(args.functions)
        else:
            func = np.loadtxt(args.functions)
        if func.shape != Z.shape:
            raise AssertionError(f"functions shape {func.shape} differs from reconstruction {Z.shape}")
        Z = Z - func

    # Suffix
    x_i, y_i = int(round(x_sel)), int(round(y_sel))
    suffix = f"i-{args.i}_point-{x_i}_{y_i}"
    base, ext = os.path.splitext(args.save)
    out_npy = f"{base}__{suffix}{ext}"
    out_txt = f"{base}__{suffix}.txt"
    os.makedirs(os.path.dirname(out_npy) or '.', exist_ok=True)
    np.save(out_npy, Z)
    np.savetxt(out_txt, Z)

    # Optional 2D image
    out_dir = args.save_dir or (os.path.dirname(out_npy) or '.')
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(Z, origin='upper', cmap='viridis')
        cb = 'Reconstruction'
        if args.subtract_functions:
            cb = 'Reconstruction - functions'
        plt.colorbar(label=cb)
        plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'Reconstruction [{suffix}]')
        fmts = tuple([s.strip() for s in str(args.img_formats).split(',') if s.strip()]) or ('png',)
        save_figure_bundle(fig, os.path.join(out_dir, f"reconstruction__{suffix}"), formats=fmts, with_pickle=True)
        plt.close(fig)
    except Exception:
        pass

    print(out_npy)


if __name__ == '__main__':
    main()

