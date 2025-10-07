#!/usr/bin/env python
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from plot_basis_maps import average_rmse_over_i
from scripts.rmse_mean import _build_base_name, _format_sigma_tag, _save_grid

FUNCTIONS: List[str] = [
    'data/functions_pow0.5.wave',
    'data/functions_pow0.75.wave',
    'data/functions_pow1.wave',
    'data/functions_pow2.wave',
]
I_VALUES: List[int] = [4, 9, 16, 25, 36, 49, 64, 81, 100]
SMOOTH_SIGMAS: List[float] = [5.0, 10.0, 15.0]
FOLDER = 'coefs_process'
BASIS_ROOT = 'data'
OUT_DIR = Path('output/rmse_mean')


def run_batches() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for func_path, i_value in itertools.product(FUNCTIONS, I_VALUES):
        i_list = [i_value]
        base_name = _build_base_name(i_list, func_path)
        print(f"Processing functions={func_path}, i={i_value}")

        raw_grid, _ = average_rmse_over_i(
            i_list,
            folder=FOLDER,
            basis_root=BASIS_ROOT,
            functions_path=func_path,
            smooth_sigma=None,
        )
        raw_path = OUT_DIR / f"{base_name}__recon_sigma_none.npy"
        _save_grid(raw_grid, raw_path, f"i_list={i_list}, recon_sigma=none", OUT_DIR)

        for sigma in SMOOTH_SIGMAS:
            print(f"  smoothing sigma={sigma}")
            smooth_grid, _ = average_rmse_over_i(
                i_list,
                folder=FOLDER,
                basis_root=BASIS_ROOT,
                functions_path=func_path,
                smooth_sigma=sigma,
            )
            sigma_tag = _format_sigma_tag(sigma)
            smooth_path = OUT_DIR / f"{base_name}__recon_sigma_{sigma_tag}.npy"
            _save_grid(
                smooth_grid,
                smooth_path,
                f"i_list={i_list}, recon_sigma={sigma:g}",
                OUT_DIR,
            )


if __name__ == '__main__':
    run_batches()
