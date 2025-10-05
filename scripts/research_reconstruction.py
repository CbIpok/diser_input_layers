#!/usr/bin/env python
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diser.io.basis import load_basis_dir
from diser.io.coeffs import read_coef_json, resolve_coeffs_dir
from diser.core.restore import reconstruct_from_bases
from plot_basis_maps import average_rmse_over_i, average_mean_error_over_i


@dataclass
class CoefSet:
    xs: np.ndarray
    ys: np.ndarray
    coefs: List[np.ndarray]  # list of k arrays (M,)


def _detect_i_from_coefs(folder: str | Path) -> List[int]:
    p = Path(folder)
    vals = []
    for f in p.glob('basis_*.json'):
        try:
            i = int(f.stem.split('_')[1])
            vals.append(i)
        except Exception:
            continue
    return sorted(set(vals))


def _load_coefs(folder: str | Path, i: int) -> CoefSet:
    path = Path(folder) / f"basis_{i}.json"
    samples = read_coef_json(path)
    return CoefSet(xs=np.asarray(samples.xs, dtype=float),
                   ys=np.asarray(samples.ys, dtype=float),
                   coefs=[np.asarray(c, dtype=float) for c in samples.coefs])


def _nearest_coef(cset: CoefSet, x: float, y: float) -> np.ndarray:
    xs = cset.xs; ys = cset.ys
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    idx = int(np.argmin(d2))
    return np.array([coef[idx] for coef in cset.coefs], dtype=float)


def _nonmax_suppress(points: np.ndarray, values: np.ndarray, k: int, min_dist: float) -> List[Tuple[int, int, float]]:
    # points: Nx2 (y, x), values: (N,)
    order = np.argsort(-values)  # descending
    selected = []
    used = np.zeros(len(order), dtype=bool)
    for oi in order:
        if used[oi]:
            continue
        y, x = points[oi]
        val = values[oi]
        selected.append((int(x), int(y), float(val)))
        if len(selected) >= k:
            break
        # mark neighbors within radius
        dy = points[:, 0] - y
        dx = points[:, 1] - x
        mask = (dx * dx + dy * dy) <= (min_dist * min_dist)
        used[mask] = True
    return selected


def _ensure_dir(d: str | os.PathLike) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def _reconstruct_profiles(bases: np.ndarray, c: np.ndarray, y: int, x: int) -> Tuple[np.ndarray, np.ndarray]:
    B = np.nan_to_num(bases, nan=0.0, copy=False)
    row = c @ B[:, y, :]
    col = c @ B[:, :, x]
    return row, col


def _reconstruct_patch(bases: np.ndarray, c: np.ndarray, y: int, x: int, half: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    H, W = bases.shape[1:]
    y0 = max(0, y - half)
    y1 = min(H, y + half + 1)
    x0 = max(0, x - half)
    x1 = min(W, x + half + 1)
    sub = np.nan_to_num(bases[:, y0:y1, x0:x1], nan=0.0, copy=False)
    patch = np.tensordot(c, sub, axes=(0, 0))
    return patch, (y0, y1, x0, x1)


def parse_args():
    p = argparse.ArgumentParser(description='Research reconstruction quality across i and selected points from an error metric')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Original field path (.wave or .npy)')

    p.add_argument('--i-all', default=None, help='Comma-separated list of i to analyze; default -> autodetect from coefs_process')
    p.add_argument('--groups', default='16,25,36,49', help='Groups of i for mean profiles; semicolon-separated groups, e.g. "16,25,36,49;4,9,16"')

    p.add_argument('--select-metric', choices=['rmse_mean', 'mean_signed'], default='rmse_mean', help='Metric for selecting points')
    p.add_argument('--select-i-list', default='4,16,25,36,49', help='i-list for selection metric (rmse_mean or mean_signed)')
    p.add_argument('--rmse-in', default=None, help='Optional precomputed RMSE mean .npy to select from')
    p.add_argument('--mean-in', default=None, help='Optional precomputed mean signed error .npy to select from')
    p.add_argument('--num-points', type=int, default=12, help='How many points to select')
    p.add_argument('--min-dist', type=float, default=64.0, help='Minimal spacing (pixels) between selected points')
    p.add_argument('--select-within-support', action='store_true', help='Restrict selection to |functions| > 1e-12 support')
    p.add_argument('--select-within-basis', action='store_true', help='Restrict selection to union of basis supports over --i-all')

    p.add_argument('--window', type=int, default=64, help='Half-size of patch window around (x,y) for local stats')
    p.add_argument('--out-dir', default='output/research', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_dir(args.out_dir)

    # Load original
    func = np.load(args.functions) if str(args.functions).lower().endswith('.npy') else np.loadtxt(args.functions)
    H, W = func.shape

    coefs_dir = resolve_coeffs_dir(args.folder, args.functions)

    # Resolve i lists
    if args.i_all:
        i_all = [int(s) for s in args.i_all.split(',') if s.strip()]
    else:
        i_all = _detect_i_from_coefs(coefs_dir)
    if not i_all:
        raise SystemExit('No i found in coefs_process')
    group_sets = []
    if args.groups:
        for grp in args.groups.split(';'):
            gs = [int(s) for s in grp.split(',') if s.strip()]
            if gs:
                group_sets.append(gs)

    select_i_list = [int(s) for s in str(args.select_i_list).split(',') if s.strip()]

    # Selection metric grid
    if args.select_metric == 'rmse_mean':
        if args.rmse_in and Path(args.rmse_in).exists():
            metric = np.load(args.rmse_in)
        else:
            metric, _ = average_rmse_over_i(select_i_list, folder=coefs_dir, basis_root=args.basis_root, functions_path=args.functions, smooth_sigma=None)
    else:
        if args.mean_in and Path(args.mean_in).exists():
            metric = np.load(args.mean_in)
        else:
            metric, _ = average_mean_error_over_i(select_i_list, folder=coefs_dir, basis_root=args.basis_root, functions_path=args.functions, smooth_sigma=None)

    finite = np.isfinite(metric)
    if args.select_within_support:
        support = np.abs(func) > 1e-12
        finite = finite & support
    if args.select_within_basis:
        union = np.zeros_like(func, dtype=bool)
        for i in i_all:
            B = load_basis_dir(os.path.join(args.basis_root, f'basis_{i}'))
            union |= np.any(np.isfinite(B), axis=0)
            # free
            del B
        finite = finite & union
    ys, xs = np.nonzero(finite)
    vals = metric[ys, xs]
    if ys.size == 0:
        raise SystemExit('Selection metric has no finite points')
    points = np.stack([ys, xs], axis=1)
    selected = _nonmax_suppress(points, vals, k=args.num_points, min_dist=args.min_dist)

    # Load save_interval for mapping grid -> sample coords
    cfg_path = Path('data') / 'config.json'
    save_interval = 1
    if cfg_path.exists():
        try:
            import json as _json
            save_interval = int(_json.loads(cfg_path.read_text(encoding='utf-8')).get('save_interval', 1))
        except Exception:
            save_interval = 1

    # Load coefficients and bases per i on demand; compute profiles/patches per point
    coefs_cache: dict[int, CoefSet] = {}
    bases_cache: dict[int, np.ndarray] = {}

    # Summary
    summary = {
        'i_all': i_all,
        'groups': group_sets,
        'select_metric': args.select_metric,
        'select_i_list': select_i_list,
        'selected_points': [{'x': x, 'y': y, 'metric': v} for x, y, v in selected],
        'per_point': []
    }

    # Prepare output dirs
    prof_dir = Path(args.out_dir) / 'profiles'
    _ensure_dir(prof_dir)
    patches_dir = Path(args.out_dir) / 'patches'
    _ensure_dir(patches_dir)

    import csv
    import matplotlib.pyplot as plt

    for pt_idx, (x, y, mv) in enumerate(selected, start=1):
        # Map grid coords (x,y) to sample coords (xs,ys) used in JSON
        xsamp = int(round(x / max(save_interval, 1)))
        ysamp = int(round(y / max(save_interval, 1)))
        rec_rows = {}
        rec_cols = {}
        patch_stats = {}
        # For each i compute profiles and patch stats
        for i in i_all:
            # load coefs/bases if needed
            if i not in coefs_cache:
                coefs_cache[i] = _load_coefs(coefs_dir, i)
            if i not in bases_cache:
                bases_cache[i] = load_basis_dir(os.path.join(args.basis_root, f'basis_{i}'))

            c = _nearest_coef(coefs_cache[i], xsamp, ysamp)
            row, col = _reconstruct_profiles(bases_cache[i], c, y, x)
            rec_rows[i] = row
            rec_cols[i] = col

            # Patch stats
            patch, (y0, y1, x0, x1) = _reconstruct_patch(bases_cache[i], c, y, x, args.window)
            fpatch = func[y0:y1, x0:x1]
            diff = patch - fpatch
            dv = diff[np.isfinite(diff)]
            if dv.size:
                patch_stats[i] = {
                    'mean': float(np.mean(dv)),
                    'median': float(np.median(dv)),
                    'std': float(np.std(dv)),
                    'p95': float(np.percentile(dv, 95)),
                    'p99': float(np.percentile(dv, 99)),
                    'mae': float(np.mean(np.abs(dv))),
                    'count': int(dv.size),
                }
            else:
                patch_stats[i] = {'count': 0}

        # Build group means (profiles/patch stats)
        group_profiles_row = {}
        group_profiles_col = {}
        group_patch_stats = {}
        for gidx, grp in enumerate(group_sets, start=1):
            # Only include i present in i_all
            use = [i for i in grp if i in rec_rows]
            if not use:
                continue
            row_mean = np.mean(np.stack([rec_rows[i] for i in use], axis=0), axis=0)
            col_mean = np.mean(np.stack([rec_cols[i] for i in use], axis=0), axis=0)
            group_profiles_row[f'group{gidx}'] = row_mean
            group_profiles_col[f'group{gidx}'] = col_mean

            # Patch stats: average patches via re-running (avoid huge mem)
            # Simple proxy: average stats across i in group
            ps = [patch_stats[i] for i in use if patch_stats.get(i, {}).get('count', 0) > 0]
            if ps:
                group_patch_stats[f'group{gidx}'] = {
                    'mean': float(np.mean([p['mean'] for p in ps])),
                    'median': float(np.mean([p['median'] for p in ps])),
                    'std': float(np.mean([p['std'] for p in ps])),
                    'p95': float(np.mean([p['p95'] for p in ps])),
                    'p99': float(np.mean([p['p99'] for p in ps])),
                    'mae': float(np.mean([p['mae'] for p in ps])),
                    'members': use,
                }

        # Save CSVs with original + per-i + group means
        # Row CSV
        x_axis = np.arange(W)
        row_csv = prof_dir / f"row_y{y}_x{x}_pt{pt_idx}.csv"
        with open(row_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ['x', 'original'] + [f'i{i}' for i in i_all] + list(group_profiles_row.keys())
            w.writerow(header)
            for xi in range(W):
                row_vals = [xi, float(func[y, xi])] + [float(rec_rows[i][xi]) for i in i_all]
                for gk in group_profiles_row:
                    row_vals.append(float(group_profiles_row[gk][xi]))
                w.writerow(row_vals)

        # Column CSV
        y_axis = np.arange(H)
        col_csv = prof_dir / f"col_x{x}_y{y}_pt{pt_idx}.csv"
        with open(col_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ['y', 'original'] + [f'i{i}' for i in i_all] + list(group_profiles_col.keys())
            w.writerow(header)
            for yi in range(H):
                col_vals = [yi, float(func[yi, x])] + [float(rec_cols[i][yi]) for i in i_all]
                for gk in group_profiles_col:
                    col_vals.append(float(group_profiles_col[gk][yi]))
                w.writerow(col_vals)

        # Plots (original + groups only)
        if group_profiles_row:
            fig = plt.figure(figsize=(10, 4))
            plt.plot(x_axis, func[y, :], label='original', linewidth=1.0)
            for gk, arr in group_profiles_row.items():
                plt.plot(x_axis, arr, label=gk, linewidth=1.0)
            plt.title(f'Row y={y}, x={x}'); plt.xlabel('x'); plt.ylabel('value'); plt.legend(); plt.tight_layout()
            plt.savefig(prof_dir / f"row_y{y}_x{x}_pt{pt_idx}.png")
            plt.close(fig)
        if group_profiles_col:
            fig = plt.figure(figsize=(6, 8))
            plt.plot(func[:, x], y_axis, label='original', linewidth=1.0)
            for gk, arr in group_profiles_col.items():
                plt.plot(arr, y_axis, label=gk, linewidth=1.0)
            plt.gca().invert_yaxis()
            plt.title(f'Col x={x}, y={y}'); plt.xlabel('value'); plt.ylabel('y'); plt.legend(); plt.tight_layout()
            plt.savefig(prof_dir / f"col_x{x}_y{y}_pt{pt_idx}.png")
            plt.close(fig)

        summary['per_point'].append({
            'point': {'x': x, 'y': y, 'metric': mv},
            'sample_coords': {'xs': xsamp, 'ys': ysamp, 'save_interval': save_interval},
            'patch_stats': patch_stats,
            'group_patch_stats': group_patch_stats,
            'row_csv': str(row_csv),
            'col_csv': str(col_csv),
        })

    # Free bases
    bases_cache.clear(); coefs_cache.clear()

    # Save summary
    with open(Path(args.out_dir) / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(Path(args.out_dir) / 'summary.json')


if __name__ == '__main__':
    main()
