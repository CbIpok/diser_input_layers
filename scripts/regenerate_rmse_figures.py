#!/usr/bin/env python
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPT = PROJECT_ROOT / 'scripts' / 'rmse_mean.py'
OUT_DIR = PROJECT_ROOT / 'output' / 'rmse_mean'

FUNCTIONS: List[str] = [
    'data/functions_pow0.5.wave',
    'data/functions_pow0.75.wave',
    'data/functions_pow1.wave',
    'data/functions_pow2.wave',
]
I_VALUES: List[int] = [4, 9, 16, 25, 36, 49, 64, 81, 100]
SMOOTH_SIGMAS: List[float] = [5.0, 10.0, 15.0]


def regenerate_figures() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    for func_path, i_val, sigma in itertools.product(FUNCTIONS, I_VALUES, SMOOTH_SIGMAS):
        cmd = [
            PYTHON,
            str(SCRIPT),
            '--i-list', str(i_val),
            '--functions', func_path,
            '--smooth-sigma', str(sigma),
            '--out-dir', str(OUT_DIR),
            '--visualize-only',
        ]
        print('Regenerating', func_path, i_val, sigma)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


if __name__ == '__main__':
    regenerate_figures()
