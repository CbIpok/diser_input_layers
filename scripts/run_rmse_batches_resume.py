#!/usr/bin/env python
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

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


def _base_name(i_val: int, functions_path: str) -> str:
    func_tag = Path(functions_path).stem.replace('.', '_')
    return f"rmse_mean__i_{i_val}__func_{func_tag}"


def _smooth_path(i_val: int, functions_path: str, sigma: float) -> Path:
    sigma_tag = str(sigma).replace('.', '_')
    return OUT_DIR / f"{_base_name(i_val, functions_path)}__recon_sigma_{sigma_tag}.npy"


def run_missing() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')

    combos = [
        (func, i_val, sigma)
        for func, i_val, sigma in itertools.product(FUNCTIONS, I_VALUES, SMOOTH_SIGMAS)
        if not _smooth_path(i_val, func, sigma).exists()
    ]

    print(f"Pending combinations: {len(combos)}")
    active: list[tuple[tuple[str, int, float], subprocess.Popen]] = []

    for func, i_val, sigma in combos:
        cmd = [
            PYTHON,
            str(SCRIPT),
            '--i-list', str(i_val),
            '--functions', func,
            '--smooth-sigma', str(sigma),
            '--out-dir', str(OUT_DIR),
        ]
        print(f"Launching: i={i_val}, functions={func}, sigma={sigma}")
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
        active.append(((func, i_val, sigma), proc))

        while len(active) >= MAX_CONCURRENT:
            time.sleep(5)
            active = [entry for entry in active if entry[1].poll() is None]

    # wait remaining
    while active:
        time.sleep(5)
        active = [entry for entry in active if entry[1].poll() is None]


if __name__ == '__main__':
    run_missing()
