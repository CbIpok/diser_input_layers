#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diser.io.basis import load_basis_dir
from diser.core.restore import reconstruct_from_bases, valid_mask_from_bases, mse_on_valid_region


def _ensure_dir(d: str | os.PathLike) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def _load_functions(path: str) -> np.ndarray:
    return np.load(path) if str(path).lower().endswith('.npy') else np.loadtxt(path)


def compute_coeffs_for_i(i: int, basis_root: str, functions_path: str) -> dict:
    B = load_basis_dir(os.path.join(basis_root, f"basis_{i}"))  # (k,H,W)
    T = _load_functions(functions_path)  # (H,W)
    if T.shape != B.shape[1:]:
        raise AssertionError(f"functions shape {T.shape} differs from basis shape {B.shape[1:]}")

    mask = valid_mask_from_bases(B)  # (H,W)
    if not np.any(mask):
        raise RuntimeError(f"No valid region for basis_{i}")

    # Build design matrix over valid region
    k = B.shape[0]
    H, W = T.shape
    idx = np.flatnonzero(mask.reshape(-1))
    # Flatten bases over valid idx
    Bf = np.nan_to_num(B.reshape(k, -1), nan=0.0, copy=False)[:, idx].T  # (N,k)
    t = T.reshape(-1)[idx]  # (N,)

    # Least squares
    # c: (k,), residuals sumsq (not used), rank, s
    c, *_ = np.linalg.lstsq(Bf, t, rcond=None)

    # Reconstruction and stats (on valid region)
    Z = reconstruct_from_bases(c, B)
    mse = mse_on_valid_region(T, Z, mask)
    mae = float(np.mean(np.abs((T - Z)[mask])))

    return {
        'i': i,
        'k': int(k),
        'coeffs': [float(x) for x in c.tolist()],
        'stats': {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': mae,
            'valid_pixels': int(idx.size),
        },
        'reconstruction': Z,  # caller may choose to save
        'mask': mask,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Compute global L2 approximation coefficients for basis_{i} to approximate functions.wave')
    p.add_argument('--i', type=int, default=None, help='Single i to compute')
    p.add_argument('--i-list', default=None, help='Comma-separated i values to compute')
    p.add_argument('--basis-root', default='data', help='Root with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave (or .npy)')
    p.add_argument('--out-dir', default='output/global_coeffs', help='Where to save coeffs and reconstructions')
    p.add_argument('--save-recon', action='store_true', help='Save reconstruction arrays')
    return p.parse_args()


def main():
    args = parse_args()
    todo = []
    if args.i is not None:
        todo.append(int(args.i))
    if args.i_list:
        todo.extend([int(s) for s in args.i_list.split(',') if s.strip()])
    if not todo:
        raise SystemExit('Specify --i or --i-list')

    _ensure_dir(args.out_dir)
    summary = {'functions': args.functions, 'basis_root': args.basis_root, 'items': []}
    for i in sorted(set(todo)):
        res = compute_coeffs_for_i(i, args.basis_root, args.functions)
        coeffs = res['coeffs']
        stats = res['stats']
        # Save coeffs JSON
        out_json = os.path.join(args.out_dir, f'coeffs_global_{i}.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'i': i, 'coeffs': coeffs, 'stats': stats}, f, ensure_ascii=False, indent=2)
        # Optionally reconstruction
        if args.save_recon:
            np.save(os.path.join(args.out_dir, f'reconstruction_global_{i}.npy'), res['reconstruction'])
        summary['items'].append({'i': i, 'coeffs_json': out_json, 'stats': stats})

    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(os.path.join(args.out_dir, 'summary.json'))


if __name__ == '__main__':
    main()
