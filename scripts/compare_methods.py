#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diser.io.basis import load_basis_dir
from diser.core.restore import gaussian_smooth_nan
from matplotlib.path import Path as MplPath


def _ensure_dir(d: str | os.PathLike) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def _load_functions(path: str) -> np.ndarray:
    return np.load(path) if str(path).lower().endswith('.npy') else np.loadtxt(path)


def _load_max_height(nc_path: str) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    if 'max_height' in ds:
        arr = ds['max_height'].values
    else:
        arr = np.nanmax(ds['height'].values, axis=0)
    return np.asarray(arr, dtype=np.float64)


def _fit_coeffs_from_nc(nc_root: str, i: int) -> np.ndarray:
    nc_functions = os.path.join(nc_root, 'functions.nc')
    nc_basis_dir = os.path.join(nc_root, f'basis_{i}')
    Mh_f = _load_max_height(nc_functions)
    basis_nc = []
    for j in range(i):
        path = os.path.join(nc_basis_dir, f'basis_{j}.nc')
        basis_nc.append(_load_max_height(path))
    Mh_B = np.stack(basis_nc, axis=0)
    region = np.isfinite(Mh_f)
    region &= (Mh_f > 0)
    region |= np.any(np.nan_to_num(Mh_B, nan=0.0) > 0, axis=0)
    yx = region.reshape(-1)
    t = Mh_f.reshape(-1)[yx]
    B = np.nan_to_num(Mh_B.reshape(i, -1), nan=0.0)[:, yx].T
    c, *_ = np.linalg.lstsq(B, t, rcond=None)
    return np.asarray(c, dtype=np.float64)


def _reconstruct_form_from_basis(i: int, basis_root: str, coeffs: np.ndarray) -> np.ndarray:
    B = load_basis_dir(os.path.join(basis_root, f'basis_{i}'))
    Z = np.tensordot(coeffs, np.nan_to_num(B, nan=0.0), axes=(0, 0))
    return np.asarray(Z, dtype=np.float64)


def _load_source_mask(nx: int, ny: int, area_path: str = 'data/areas/source.json') -> np.ndarray:
    verts = np.array(json.loads(Path(area_path).read_text(encoding='utf-8'))['points'], dtype=float)
    xs = np.arange(0.5, nx, 1.0)
    ys = np.arange(0.5, ny, 1.0)
    X, Y = np.meshgrid(xs, ys)
    mask_poly = MplPath(verts).contains_points(np.vstack((X.ravel(), Y.ravel())).T)
    return mask_poly.reshape(ny, nx)


def _sample_points(mask: np.ndarray, n: int, seed: int, forced: Tuple[int, int] | None) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    order = np.arange(xs.size)
    rng.shuffle(order)
    pts = []
    if forced is not None:
        fx, fy = forced
        if 0 <= fy < mask.shape[0] and 0 <= fx < mask.shape[1]:
            pts.append((fx, fy))
    for k in order:
        x = int(xs[k]); y = int(ys[k])
        if forced is not None and len(pts) > 0 and (x, y) == forced:
            continue
        pts.append((x, y))
        if len(pts) >= n:
            break
    return pts


def parse_args():
    p = argparse.ArgumentParser(description='Compare adaptive fusion vs mean-of-i (smoothed) on the same K points')
    p.add_argument('--nc-root', required=True, help='Folder with functions.nc and basis_{i}/basis_*.nc')
    p.add_argument('--basis-root', default='data', help='Root with data/basis_{i}')
    p.add_argument('--i-list', default='16,25,36,49', help='i values for baseline mean-of-i')
    p.add_argument('--smooth-sigma', type=float, default=1.5, help='Smoothing sigma for baseline')
    p.add_argument('--adaptive-form', default='output/adaptive/reconstruction.npy', help='Path to adaptive fused form .npy')
    p.add_argument('--functions', default='data/functions.wave', help='Original field .wave/.npy')
    p.add_argument('--n-points', type=int, default=5000, help='Number of points')
    p.add_argument('--seed', type=int, default=0, help='Seed for reproducible sampling')
    p.add_argument('--force-first', default='2000,1400', help='Force include this point (x,y)')
    p.add_argument('--out-dir', default='output/compare_adaptive', help='Output directory')
    return p.parse_args()


def main():
    import matplotlib.pyplot as plt

    args = parse_args()
    _ensure_dir(args.out_dir)
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]

    # Load original and mask
    F = _load_functions(args.functions).astype(np.float64)
    H, W = F.shape
    mask_source = _load_source_mask(W, H, area_path='data/areas/source.json')

    # Sample points
    fx, fy = [int(v) for v in args.force_first.split(',')] if args.force_first else (None, None)
    pts = _sample_points(mask_source, args.n_points, args.seed, (fx, fy) if args.force_first else None)

    # Load adaptive form
    ZA = np.load(args.adaptive_form).astype(np.float64)
    if ZA.shape != F.shape:
        raise AssertionError(f'adaptive form shape {ZA.shape} != original {F.shape}')

    # Build baseline mean-of-i from NC fit
    Zis = []
    for i in i_list:
        c = _fit_coeffs_from_nc(args.nc_root, i)
        Zi = _reconstruct_form_from_basis(i, args.basis_root, c)
        Zis.append(Zi)
    ZB = np.mean(np.stack(Zis, axis=0), axis=0)
    if args.smooth_sigma and args.smooth_sigma > 0:
        ZB = gaussian_smooth_nan(ZB, sigma=float(args.smooth_sigma))

    # Per-point metrics
    half = 12
    side = 2 * half + 1
    maeA = np.zeros(len(pts), dtype=np.float64)
    maeB = np.zeros(len(pts), dtype=np.float64)
    biasA = np.zeros(len(pts), dtype=np.float64)
    biasB = np.zeros(len(pts), dtype=np.float64)

    for pi, (x, y) in enumerate(pts):
        y0 = max(0, y - half); y1 = min(H, y + half + 1)
        x0 = max(0, x - half); x1 = min(W, x + half + 1)
        Fo = F[y0:y1, x0:x1]
        Pa = ZA[y0:y1, x0:x1]
        Pb = ZB[y0:y1, x0:x1]
        da = Pa - Fo
        db = Pb - Fo
        maeA[pi] = float(np.mean(np.abs(da)))
        maeB[pi] = float(np.mean(np.abs(db)))
        biasA[pi] = float(np.mean(da))
        biasB[pi] = float(np.mean(db))

    improved = (maeA < maeB).mean()
    summary = {
        'points': len(pts),
        'i_list_baseline': i_list,
        'smooth_sigma_baseline': float(args.smooth_sigma),
        'MAE_adaptive': {'mean': float(maeA.mean()), 'median': float(np.median(maeA)), 'p95': float(np.percentile(maeA,95))},
        'MAE_baseline': {'mean': float(maeB.mean()), 'median': float(np.median(maeB)), 'p95': float(np.percentile(maeB,95))},
        'bias_adaptive': float(biasA.mean()),
        'bias_baseline': float(biasB.mean()),
        'frac_improved_adaptive_vs_baseline': float(improved)
    }
    Path(os.path.join(args.out_dir, 'summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Save a few example previews
    order = np.argsort(maeB - maeA)  # biggest improvement first
    k = min(6, len(order))
    for idx in list(order[:k]) + list(order[-k:]):
        x, y = pts[int(idx)]
        y0 = max(0, y - half); y1 = min(H, y + half + 1)
        x0 = max(0, x - half); x1 = min(W, x + half + 1)
        Fo = F[y0:y1, x0:x1]
        Pa = ZA[y0:y1, x0:x1]
        Pb = ZB[y0:y1, x0:x1]
        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1); plt.imshow(Fo, origin='upper', cmap='viridis'); plt.title('orig'); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(Pb, origin='upper', cmap='viridis'); plt.title(f'baseline σ={args.smooth_sigma}'); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(Pa, origin='upper', cmap='viridis'); plt.title('adaptive'); plt.axis('off')
        plt.tight_layout()
        fig_path = Path(args.out_dir) / f'pt_{idx}_x{x}_y{y}.png'
        fig.savefig(fig_path)
        plt.close(fig)

    print(os.path.join(args.out_dir, 'summary.json'))


if __name__ == '__main__':
    main()

