#!/usr/bin/env python3
# save_functions.py

import os
import json
import argparse
import numpy as np

from loader import load_bath_grid
from area import load_areas


def save_functions(config_path: str, out_path: str, power_override: float | None = None):
    # 1) Load config for scene size
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    size = cfg.get('size')
    if not size or 'x' not in size or 'y' not in size:
        raise ValueError("Config must contain 'size' with 'x' and 'y' values")
    nx = int(size['x'])
    ny = int(size['y'])

    # 2) Optional: verify bath grid matches scene size
    bath_path = os.path.normpath(cfg.get('bath_path', ''))
    if bath_path:
        grid = load_bath_grid(bath_path)
        gy, gx = grid.shape
        if gx != nx or gy != ny:
            print(f"Warning: bath grid shape ({gx}, {gy}) differs from config size ({nx}, {ny})")

    # 3) Create zero array for function overlay
    functions = np.zeros((ny, nx), dtype=float)

    # 4) Load all area definitions and overlay cosine waves
    areas_dir = os.path.normpath(cfg.get('areas_dir', 'data/areas'))
    areas = load_areas(areas_dir)
    if power_override is not None:
        for area_obj in areas:
            if hasattr(area_obj, 'power'):
                area_obj.power = power_override
    for area in areas:
        if hasattr(area, 'compute_wave'):
            mask = area.compute_mask(nx, ny)
            wave = area.compute_wave(mask)
            idx = ~np.isnan(wave)
            functions[idx] = wave[idx]

    # 5) Save the function array as text using numpy.savetxt
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savetxt(out_path, functions, fmt='%.6f')
    print(f"Saved function array ({ny}x{nx}) to {out_path} using numpy.savetxt")


def main():
    parser = argparse.ArgumentParser(
        description="Generate functions.wave by compositing cosine areas onto the simulation grid."
    )
    parser.add_argument(
        '-c', '--config', default='data/config.json',
        '-c', '--config', default='data/config.json',
        help="Path to config.json"
    parser.add_argument(
        '-o', '--output', default='data/functions.wave',
        '-o', '--output', default='data/functions.wave',
        help="Output path for the generated text file"
    parser.add_argument(
        '--power', type=float, default=None,
        help='Override cosine power exponent for CosineWaveArea (e.g., 2.0 sharper, 0.5 smoother)'
    )


    args = parser.parse_args()
    save_functions(args.config, args.output, power_override=args.power)


if __name__ == '__main__':
    main()


