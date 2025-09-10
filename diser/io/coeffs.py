from __future__ import annotations

import ast
import json
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

