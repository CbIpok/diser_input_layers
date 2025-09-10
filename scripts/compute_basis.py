#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diser.core.builder import build_rect_grid_masks
from diser.io.basis import save_basis_masks


def parse_args():
    p = argparse.ArgumentParser(description="Compute rectangular basis masks inside polygon")
    p.add_argument('-c', '--config', default='data/config.json', help='Path to config.json')
    p.add_argument('-a', '--area', default='data/areas/source.json', help='Path to polygon area JSON')
    p.add_argument('-i', '--count', type=int, default=16, help='Number of masks (perfect square)')
    p.add_argument('--theta', type=float, default=-30.0, help='Rotation angle in degrees')
    p.add_argument('-o', '--out', default=None, help='Output directory (default data/basis_{i})')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    nx = int(cfg['size']['x'])
    ny = int(cfg['size']['y'])
    verts = np.array(json.loads(Path(args.area).read_text(encoding='utf-8'))['points'], dtype=float)

    n = int(round(np.sqrt(args.count)))
    if n * n != args.count:
        raise SystemExit('--count must be a perfect square (e.g., 16, 25, 36)')

    masks = build_rect_grid_masks(nx, ny, verts, n=n, theta_deg=args.theta)
    out_dir = args.out or f"data/basis_{args.count}"
    save_basis_masks(masks, out_dir)
    print(f"Saved {len(masks)} masks to {out_dir}")


if __name__ == '__main__':
    main()

