#!/usr/bin/env python
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPT = PROJECT_ROOT / 'scripts' / 'rmse_mean.py'
OUT_DIR = PROJECT_ROOT / 'output' / 'rmse_mean'
MAX_CONCURRENT = 6

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

COMBOS: List[Tuple[str, int, float]] = [
    (func, i_val, sigma)
    for func, i_val, sigma in itertools.product(FUNCTIONS, I_VALUES, SMOOTH_SIGMAS)
]


def launch_batches() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active: List[Tuple[Tuple[str, int, float], subprocess.Popen]] = []
    for func, i_val, sigma in COMBOS:
        cmd = [
            PYTHON,
            str(SCRIPT),
            '--i-list', str(i_val),
            '--functions', func,
            '--smooth-sigma', str(sigma),
            '--out-dir', str(OUT_DIR),
            '--folder', FOLDER,
            '--basis-root', BASIS_ROOT,
        ]
        print(f"Launching: i={i_val}, functions={func}, sigma={sigma}")
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        active.append(((func, i_val, sigma), proc))

        while len(active) >= MAX_CONCURRENT:
            time.sleep(1.0)
            active = [entry for entry in active if entry[1].poll() is None]

    while active:
        time.sleep(1.0)
        active = [entry for entry in active if entry[1].poll() is None]


if __name__ == '__main__':
    launch_batches()
