#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np

# Ensure project root on sys.path when executed as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    p.add_argument('--img-formats', default='png,svg',
                   help='Comma-separated image formats to save (e.g., png or png,svg)')
    p.add_argument('--subtract-functions', action='store_true',
                   help='Subtract functions.wave from the resulting mean (and smoothed) before saving and plotting')
    # Profiles (1D sections) of reconstruction vs original functions
    p.add_argument('--profile-rows', default=None,
                   help='Comma-separated Y indices to export row profiles (original vs mean)')
    p.add_argument('--profile-cols', default=None,
                   help='Comma-separated X indices to export column profiles (original vs mean)')
    p.add_argument('--profiles-dir', default=None,
                   help='Directory to save profile plots/CSV (defaults to save-dir)')
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
    mean_Z_raw = mean_Z.copy()
    # Note: previously applied an edge-aware filter to mean_Z here; removed.
    # Optionally subtract functions.wave from the resulting forms
    if args.subtract_functions:
        # Load reference functions array; support .npy for faster I/O
        if str(args.functions).lower().endswith('.npy'):
            func = np.load(args.functions)
        else:
            func = np.loadtxt(args.functions)
        if func.shape != mean_Z.shape:
            raise AssertionError(f"functions.wave shape {func.shape} differs from mean shape {mean_Z.shape}")
        mean_Z = mean_Z - func
        if smoothed is not None:
            if smoothed.shape != func.shape:
                raise AssertionError(f"functions.wave shape {func.shape} differs from smoothed mean shape {smoothed.shape}")
            smoothed = smoothed - func
    # Build a suffix for filenames that encodes i-list and point
    def _suffix_from(i_vals, pt_xy):
        i_part = '_'.join(str(i) for i in i_vals)
        x, y = int(pt_xy[0]), int(pt_xy[1])
        return f"i-list-{i_part}_point-{x}_{y}"

    suffix_tag = _suffix_from(i_list, args.point)

    def _with_suffix(path_in: str, suffix: str) -> str:
        base, ext = os.path.splitext(path_in)
        return f"{base}__{suffix}{ext}"

    # Save mean as .npy and additionally as .txt, both with suffix
    save_mean_path = _with_suffix(args.save_mean, suffix_tag)
    os.makedirs(os.path.dirname(save_mean_path) or '.', exist_ok=True)
    np.save(save_mean_path, mean_Z)
    # Also save as text (space-separated)
    mean_txt_path = os.path.splitext(save_mean_path)[0] + '.txt'
    np.savetxt(mean_txt_path, mean_Z)

    # If requested, save smoothed as well (npy + txt), with suffix
    if args.save_smooth and smoothed is not None:
        save_smooth_path = _with_suffix(args.save_smooth, suffix_tag)
        os.makedirs(os.path.dirname(save_smooth_path) or '.', exist_ok=True)
        np.save(save_smooth_path, smoothed)
        smooth_txt_path = os.path.splitext(save_smooth_path)[0] + '.txt'
        np.savetxt(smooth_txt_path, smoothed)
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
    formats = tuple([s.strip() for s in str(args.img_formats).split(',') if s.strip()]) or ("png",)
    base_with_suffix = f"{base_name}__{suffix_tag}"
    save_figure_bundle(fig1, os.path.join(out_dir, base_with_suffix), formats=formats, with_pickle=True)
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
        base2_with_suffix = f"{base2}__{suffix_tag}"
        save_figure_bundle(fig2, os.path.join(out_dir, base2_with_suffix), formats=formats, with_pickle=True)
        plt.close(fig2)

    # Export 1D profiles if requested (compare ORIGINAL functions vs MEAN RAW, not subtracted)
    rows_spec = [s for s in (args.profile_rows or '').split(',') if s.strip()]
    cols_spec = [s for s in (args.profile_cols or '').split(',') if s.strip()]
    if rows_spec or cols_spec:
        if str(args.functions).lower().endswith('.npy'):
            functions_arr = np.load(args.functions)
        else:
            functions_arr = np.loadtxt(args.functions)
        if functions_arr.shape != mean_Z_raw.shape:
            raise AssertionError(f"functions.wave shape {functions_arr.shape} differs from mean shape {mean_Z_raw.shape}")
        prof_dir = args.profiles_dir or out_dir
        os.makedirs(prof_dir, exist_ok=True)

        import csv
        import matplotlib.pyplot as plt
        H, W = mean_Z_raw.shape
        # Rows
        for rs in rows_spec:
            try:
                y = int(rs)
            except ValueError:
                continue
            if not (0 <= y < H):
                continue
            x_axis = np.arange(W)
            orig = functions_arr[y, :]
            meanv = mean_Z_raw[y, :]
            # CSV
            csv_path = os.path.join(prof_dir, f"profile_row_{y}__{suffix_tag}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['x', 'original', 'mean'])
                for x, o, m in zip(x_axis, orig, meanv):
                    w.writerow([int(x), float(o), float(m)])
            # Plot
            fig = plt.figure(figsize=(10, 4))
            plt.plot(x_axis, orig, label='original', linewidth=1.0)
            plt.plot(x_axis, meanv, label='mean', linewidth=1.0)
            plt.title(f'Row {y} profiles [{suffix_tag}]')
            plt.xlabel('x'); plt.ylabel('value'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(prof_dir, f"profile_row_{y}__{suffix_tag}.png"))
            plt.close(fig)
        # Cols
        for cs in cols_spec:
            try:
                x = int(cs)
            except ValueError:
                continue
            if not (0 <= x < W):
                continue
            y_axis = np.arange(H)
            orig = functions_arr[:, x]
            meanv = mean_Z_raw[:, x]
            csv_path = os.path.join(prof_dir, f"profile_col_{x}__{suffix_tag}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['y', 'original', 'mean'])
                for y, o, m in zip(y_axis, orig, meanv):
                    w.writerow([int(y), float(o), float(m)])
            fig = plt.figure(figsize=(6, 8))
            plt.plot(orig, y_axis, label='original', linewidth=1.0)
            plt.plot(meanv, y_axis, label='mean', linewidth=1.0)
            plt.gca().invert_yaxis()
            plt.title(f'Column {x} profiles [{suffix_tag}]')
            plt.xlabel('value'); plt.ylabel('y'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(prof_dir, f"profile_col_{x}__{suffix_tag}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
