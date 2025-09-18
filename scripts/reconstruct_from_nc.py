#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diser.io.basis import load_basis_dir


def _ensure_dir(p: str | os.PathLike) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _downsample_max(mask: np.ndarray, fx: int, fy: int) -> np.ndarray:
    """Downsample binary/float mask by (fy, fx) using block max."""
    H, W = mask.shape
    assert H % fy == 0 and W % fx == 0, f"shape {mask.shape} not divisible by {(fy, fx)}"
    m = mask.reshape(H // fy, fy, W // fx, fx)
    return np.nanmax(m, axis=(1, 3))


def _load_max_height(nc_path: str) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    # prefer 'max_height' if exists, else compute from 'height'
    if 'max_height' in ds:
        arr = ds['max_height'].values
    else:
        # height dims: (time, lat, lon)
        h = ds['height'].values
        arr = np.nanmax(h, axis=0)
    return np.asarray(arr, dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description='Reconstruct initial form from NC max_height by fitting basis max_height (global L2)')
    p.add_argument('--nc-root', required=True, help='Folder with functions.nc and basis_{i}/basis_*.nc')
    p.add_argument('--i', type=int, default=16, help='i (number of basis functions)')
    p.add_argument('--basis-root', default='data', help='Root with data/basis_{i}/basis_*.wave for final form synthesis')
    p.add_argument('--functions-wave', default='data/functions.wave', help='Original form (for metrics)')
    p.add_argument('--save-coeffs', default='output/nc_fit/coeffs_i{I}.json', help='Where to save coeffs JSON')
    p.add_argument('--save-form', default='output/nc_fit/reconstructed_form_i{I}.npy', help='Where to save reconstructed initial form')
    p.add_argument('--save-diff', default='output/nc_fit/diff_form_i{I}.npy', help='Where to save (recon - original)')
    p.add_argument('--save-preview', default='output/nc_fit/preview_i{I}.png', help='Optional PNG diff preview')
    return p.parse_args()


def main():
    import json
    import matplotlib.pyplot as plt

    args = parse_args()
    I = int(args.i)
    save_coeffs = args.save_coeffs.format(I=I)
    save_form = args.save_form.format(I=I)
    save_diff = args.save_diff.format(I=I)
    save_preview = args.save_preview.format(I=I)

    nc_functions = os.path.join(args.nc_root, 'functions.nc')
    nc_basis_dir = os.path.join(args.nc_root, f'basis_{I}')

    if not os.path.isfile(nc_functions):
        raise SystemExit(f'functions.nc not found: {nc_functions}')
    if not os.path.isdir(nc_basis_dir):
        raise SystemExit(f'basis_{I} dir with NC files not found: {nc_basis_dir}')

    # Load 2D max_height fields (sample grid: lat x lon)
    Mh_f = _load_max_height(nc_functions)  # (lat, lon)
    # Load all basis max_height
    basis_nc = []
    for j in range(I):
        path = os.path.join(nc_basis_dir, f'basis_{j}.nc')
        if not os.path.isfile(path):
            raise SystemExit(f'Missing: {path}')
        basis_nc.append(_load_max_height(path))
    Mh_B = np.stack(basis_nc, axis=0)  # (I, lat, lon)

    # Build fit region mask: where Mh_f or any Mh_B are finite and > 0
    region = np.isfinite(Mh_f)
    region &= (Mh_f > 0)
    region |= np.any(np.nan_to_num(Mh_B, nan=0.0) > 0, axis=0)

    # Flatten design
    I_k, Hs, Ws = Mh_B.shape
    yx = region.reshape(-1)
    t = Mh_f.reshape(-1)[yx]
    B = np.nan_to_num(Mh_B.reshape(I_k, -1), nan=0.0)[:, yx].T  # (N, I)

    # Solve least squares for coefficients
    c, *_ = np.linalg.lstsq(B, t, rcond=None)

    # Reconstruct initial form with basis masks (full resolution)
    # Load basis masks at full resolution and synthesize Z(x,y) = sum c_j * mask_j
    Bm = load_basis_dir(os.path.join(args.basis_root, f'basis_{I}'))  # (I, H, W)
    # Map sample grid (lat,lon) to full grid: known save_interval=4 (from data/config.json)
    cfg = json.loads(Path('data/config.json').read_text(encoding='utf-8'))
    save_interval = int(cfg.get('save_interval', 4))
    # sanity shapes
    Hf, Wf = Bm.shape[1:]
    # Synthesize initial form
    Z = np.tensordot(c, np.nan_to_num(Bm, nan=0.0), axes=(0, 0))

    # Save coeffs
    _ensure_dir(save_coeffs)
    with open(save_coeffs, 'w', encoding='utf-8') as f:
        json.dump({'i': I, 'coeffs': [float(x) for x in c.tolist()]}, f, ensure_ascii=False, indent=2)

    # Save form
    _ensure_dir(save_form)
    np.save(save_form, Z)

    # If original form provided, compute diff & preview
    if args.functions_wave:
        if str(args.functions_wave).lower().endswith('.npy'):
            F = np.load(args.functions_wave)
        else:
            F = np.loadtxt(args.functions_wave)
        if F.shape == Z.shape:
            D = Z - F
            _ensure_dir(save_diff)
            np.save(save_diff, D)
            # quick PNG preview
            _ensure_dir(save_preview)
            fig = plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1); plt.imshow(F, origin='upper', cmap='viridis'); plt.title('original'); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(Z, origin='upper', cmap='viridis'); plt.title('reconstructed'); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(D, origin='upper', cmap='coolwarm'); plt.title('diff'); plt.axis('off')
            plt.tight_layout(); plt.savefig(save_preview); plt.close(fig)
        else:
            print(f'[warn] functions_wave shape {F.shape} != reconstructed {Z.shape}; skip diff')

    print(save_coeffs)
    print(save_form)
    if os.path.exists(save_diff):
        print(save_diff)
        print(save_preview)


if __name__ == '__main__':
    main()

