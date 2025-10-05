from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class CoefSamples:
    xs: np.ndarray        # (M,) float64
    ys: np.ndarray        # (M,) float64
    coefs: List[np.ndarray]  # list of k arrays (M,)
    approx_error: Optional[np.ndarray] = None  # (M,) or None


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_coef_json(path: str | Path) -> CoefSamples:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    xs: list[float] = []
    ys: list[float] = []
    approx: list[float] = []
    coefs: List[List[float]] = []

    for key, val in data.items():
        if not isinstance(val, dict):
            # Some historical dumps store plain strings like 'nan'; skip them.
            continue
        x, y = ast.literal_eval(key)
        xs.append(float(x))
        ys.append(float(y))
        approx.append(_to_float(val.get("aprox_error")))
        coef_vals = [_to_float(c) for c in val.get("coefs", [])]
        if not coefs:
            coefs = [[] for _ in coef_vals]
        for j, c in enumerate(coef_vals):
            coefs[j].append(c)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    approx_arr = np.asarray(approx, dtype=float)
    coef_arrays = [np.asarray(c, dtype=float) for c in coefs]
    return CoefSamples(xs=xs_arr, ys=ys_arr, coefs=coef_arrays, approx_error=approx_arr)



def _has_basis_json(directory: Path) -> bool:
    '''Return True if the directory contains basis_*.json.'''
    if not directory.is_dir():
        return False
    return any(directory.glob('basis_*.json'))


def _is_default_coefs_root(folder_path: Path) -> bool:
    '''Check whether folder_path refers to the default coefs_process root.'''
    default_root = Path('coefs_process')
    try:
        return folder_path.resolve() == default_root.resolve()
    except FileNotFoundError:
        normalized_a = os.path.normcase(os.path.normpath(str(folder_path)))
        normalized_b = os.path.normcase(os.path.normpath(str(default_root)))
        return normalized_a == normalized_b


def _candidate_subdirs(functions_path: str | Path | None) -> list[str]:
    '''Generate candidate subdirectory names for a given functions file.'''
    if not functions_path:
        return []
    p = Path(functions_path)
    candidates: list[str] = []

    def _add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    stem = p.stem or p.name
    _add(stem)
    _add(p.name)
    if '.' in stem:
        _add(stem.replace('.', '_'))
    _add(stem.replace('-', '_'))
    if '.' in p.name:
        _add(p.name.split('.')[0])
    sanitized = re.sub(r'[^0-9A-Za-z_.-]+', '_', stem)
    _add(sanitized)
    if '.' in sanitized:
        _add(sanitized.replace('.', '_'))
    return candidates


def resolve_coeffs_dir(folder: str | Path | None = None,
                       functions_path: str | Path | None = None,
                       *,
                       ensure_exists: bool = True) -> Path:
    '''Resolve the directory containing basis_*.json for a functions dataset.'''
    base_path = Path(folder) if folder is not None else Path('coefs_process')
    folder_default = folder is None or _is_default_coefs_root(base_path)

    if not folder_default:
        if _has_basis_json(base_path):
            return base_path
        if ensure_exists:
            raise FileNotFoundError(f'Coefficient directory not found: {base_path}')
        return base_path

    checked: list[Path] = []
    for name in _candidate_subdirs(functions_path):
        candidate = base_path / name
        checked.append(candidate)
        if _has_basis_json(candidate):
            return candidate

    if _has_basis_json(base_path):
        return base_path

    if ensure_exists:
        attempted = ', '.join(str(p) for p in (checked + [base_path])) or str(base_path)
        raise FileNotFoundError(
            f'Unable to resolve coefficients directory from folder={folder!r} and functions_path={functions_path!r}. '
            f'Checked: {attempted}'
        )
    return base_path


def coeffs_json_path(i: int,
                     folder: str | Path | None = None,
                     functions_path: str | Path | None = None,
                     *,
                     ensure_exists: bool = True) -> Path:
    '''Return the path to basis_{i}.json matching folder/functions settings.'''
    directory = resolve_coeffs_dir(folder, functions_path, ensure_exists=ensure_exists)
    path = directory / f'basis_{i}.json'
    if ensure_exists and not path.exists():
        raise FileNotFoundError(f'Coefficient file not found: {path}')
    return path
