#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diser.io.basis import load_basis_dir
from diser.core.restore import gaussian_smooth_nan
from diser.io.coeffs import read_coef_json


def _ensure_parent(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
    if not os.path.isfile(nc_functions):
        raise FileNotFoundError(nc_functions)
    if not os.path.isdir(nc_basis_dir):
        raise FileNotFoundError(nc_basis_dir)
    Mh_f = _load_max_height(nc_functions)
    basis_nc = []
    for j in range(i):
        path = os.path.join(nc_basis_dir, f'basis_{j}.nc')
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        basis_nc.append(_load_max_height(path))
    Mh_B = np.stack(basis_nc, axis=0)  # (i, Hs, Ws)
    region = np.isfinite(Mh_f)
    region &= (Mh_f > 0)
    region |= np.any(np.nan_to_num(Mh_B, nan=0.0) > 0, axis=0)
    yx = region.reshape(-1)
    t = Mh_f.reshape(-1)[yx]
    B = np.nan_to_num(Mh_B.reshape(i, -1), nan=0.0)[:, yx].T  # (N, i)
    # least squares
    c, *_ = np.linalg.lstsq(B, t, rcond=None)
    return np.asarray(c, dtype=np.float64)


def _reconstruct_form_from_basis(i: int, basis_root: str, coeffs: np.ndarray) -> np.ndarray:
    B = load_basis_dir(os.path.join(basis_root, f'basis_{i}'))  # (i,H,W)
    Z = np.tensordot(coeffs, np.nan_to_num(B, nan=0.0), axes=(0, 0))
    return np.asarray(Z, dtype=np.float64)


def _save_interval() -> int:
    cfg = json.loads(Path('data/config.json').read_text(encoding='utf-8'))
    return int(cfg.get('save_interval', 4))


def _build_errmap_from_json(i: int, H: int, W: int, folder: str = 'coefs_process') -> np.ndarray:
    path = Path(folder) / f'basis_{i}.json'
    samples = read_coef_json(path)
    xs = np.asarray(samples.xs, dtype=int)
    ys = np.asarray(samples.ys, dtype=int)
    if samples.approx_error is None:
        return np.zeros((H, W), dtype=np.float32)
    ae = np.asarray(samples.approx_error, dtype=np.float32)
    si = _save_interval()
    # Build decimated grid size by max sample index
    h_dec = int(np.ceil(H / si))
    w_dec = int(np.ceil(W / si))
    img = np.full((h_dec, w_dec), np.nan, dtype=np.float32)
    # clamp indices
    xs = np.clip(xs, 0, w_dec - 1)
    ys = np.clip(ys, 0, h_dec - 1)
    img[ys, xs] = ae[: img.size].reshape(-1)[: len(xs)]
    # simple nearest fill: replace NaN with nearest valid along rows then cols
    # forward fill rows
    for y in range(h_dec):
        row = img[y]
        last = np.nan
        for x in range(w_dec):
            if np.isnan(row[x]):
                row[x] = last
            else:
                last = row[x]
        last = np.nan
        for x in range(w_dec - 1, -1, -1):
            if np.isnan(row[x]):
                row[x] = last
            else:
                last = row[x]
    # forward fill cols
    for x in range(w_dec):
        col = img[:, x]
        last = np.nan
        for y in range(h_dec):
            if np.isnan(col[y]):
                col[y] = last
            else:
                last = col[y]
        last = np.nan
        for y in range(h_dec - 1, -1, -1):
            if np.isnan(col[y]):
                col[y] = last
            else:
                last = col[y]
    img = np.nan_to_num(img, nan=float(np.nanmean(img)) if np.isnan(img).any() else 0.0)
    # Upsample by Kronecker
    up = np.kron(img, np.ones((si, si), dtype=np.float32))
    return up[:H, :W]


def parse_args():
    p = argparse.ArgumentParser(description='Adaptive multi-i reconstruction from NC + approx_error-driven fusion')
    p.add_argument('--nc-root', required=True, help='Folder with functions.nc and basis_{i}/basis_*.nc')
    p.add_argument('--basis-root', default='data', help='Root with data/basis_{i}')
    p.add_argument('--i-list', default=None, help='Comma-separated i values; if omitted, use all with both NC and data/basis_i available')
    p.add_argument('--alpha', type=float, default=0.5, help='Weight exponent for i (scale preference)')
    p.add_argument('--beta', type=float, default=1.0, help='Weight exponent for approx_error penalty')
    p.add_argument('--smooth-sigma', type=float, default=0.0, help='Optional final Gaussian smoothing (global sigma)')
    p.add_argument('--save', default='output/adaptive/reconstruction.npy', help='Output fused form .npy')
    p.add_argument('--save-png', default='output/adaptive/reconstruction.png', help='PNG preview')
    p.add_argument('--save-json', default='output/adaptive/summary.json', help='Summary JSON')
    return p.parse_args()


def main():
    import matplotlib.pyplot as plt

    args = parse_args()
    # Determine i-list
    if args.i_list:
        i_list = [int(s) for s in args.i_list.split(',') if s.strip()]
    else:
        # intersect NC basis dirs and data/basis_{i}
        nc_dirs = set()
        for p in Path(args.nc_root).glob('basis_*'):
            try:
                nc_dirs.add(int(p.name.split('_')[1]))
            except Exception:
                pass
        data_dirs = set()
        for p in Path(args.basis_root).glob('basis_*'):
            try:
                data_dirs.add(int(p.name.split('_')[1]))
            except Exception:
                pass
        i_list = sorted(nc_dirs & data_dirs)
    if not i_list:
        raise SystemExit('No matching i found between NC and data basis dirs')

    # Fit coeffs from NC and reconstruct per i
    forms: Dict[int, np.ndarray] = {}
    errmaps: Dict[int, np.ndarray] = {}
    coeffs: Dict[int, np.ndarray] = {}
    for i in i_list:
        c = _fit_coeffs_from_nc(args.nc_root, i)
        coeffs[i] = c
        Z = _reconstruct_form_from_basis(i, args.basis_root, c)
        forms[i] = Z
        errmaps[i] = _build_errmap_from_json(i, Z.shape[0], Z.shape[1])

    # Compute gradient magnitude of coarse mean to identify edges
    Zmean = np.mean(np.stack([forms[i] for i in i_list], axis=0), axis=0)
    gx = np.zeros_like(Zmean)
    gy = np.zeros_like(Zmean)
    gx[:, 1:-1] = 0.5 * (Zmean[:, 2:] - Zmean[:, :-2])
    gy[1:-1, :] = 0.5 * (Zmean[2:, :] - Zmean[:-2, :])
    gmag = np.hypot(gx, gy)
    gmag = gmag / (np.nanmax(gmag) + 1e-12)

    # Adaptive fusion weights
    eps = 1e-6
    H, W = Zmean.shape
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    max_i = float(max(i_list))
    for i in i_list:
        Zi = forms[i]
        Ei = errmaps[i]
        # normalize error map
        E = Ei.astype(np.float64)
        # weights: i^alpha * (E+eps)^(-beta) * (1 + 0.5*g)
        wi = (i / max_i) ** float(args.alpha) * np.power(E + eps, -float(args.beta)) * (1.0 + 0.5 * gmag)
        num += wi * Zi
        den += wi
    fused = num / np.maximum(den, eps)

    if args.smooth_sigma and args.smooth_sigma > 0:
        fused = gaussian_smooth_nan(fused, sigma=float(args.smooth_sigma))

    # Save outputs
    _ensure_parent(args.save)
    np.save(args.save, fused)
    _ensure_parent(args.save_png)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1); plt.imshow(fused, origin='upper', cmap='viridis'); plt.title('Adaptive fused form'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(gmag, origin='upper', cmap='magma'); plt.title('Gradient (norm)'); plt.axis('off')
    plt.tight_layout(); plt.savefig(args.save_png); plt.close()

    _ensure_parent(args.save_json)
    summary = {
        'i_list': i_list,
        'alpha': float(args.alpha),
        'beta': float(args.beta),
        'smooth_sigma': float(args.smooth_sigma or 0.0),
        'coeffs': {str(i): [float(x) for x in coeffs[i].tolist()] for i in i_list},
        'shapes': {'H': int(H), 'W': int(W)}
    }
    Path(args.save_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.save)
    print(args.save_png)
    print(args.save_json)


if __name__ == '__main__':
    main()

