from __future__ import annotations

import numpy as np

from .restore import gaussian_smooth_nan


def guided_unsharp(form: np.ndarray, alpha: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    """Simple NaN-aware unsharp masking.

    - Computes a blurred version using Gaussian smoothing (NaN-aware)
    - Enhances details: out = form + alpha * (form - blur)
    - Preserves NaNs in the input
    """
    arr = np.asarray(form, dtype=np.float64)
    blur = gaussian_smooth_nan(arr, sigma=float(sigma))
    out = arr + float(alpha) * (arr - blur)
    # Preserve NaNs exactly where input was NaN
    out[~np.isfinite(arr)] = np.nan
    return out


def filter(form: np.ndarray, method=guided_unsharp, params: dict | None = None) -> np.ndarray:  # noqa: A003
    """Apply a post-filter to the form (2D array).

    - method: callable or string 'guided_unsharp'
    - params: optional dict of keyword args for the method
    """
    if isinstance(method, str):
        m = method.lower()
        if m == 'guided_unsharp':
            method_fn = guided_unsharp
        else:
            raise ValueError(f"Unknown filter method: {method}")
    else:
        method_fn = method

    kwargs = dict(params or {})
    return method_fn(form, **kwargs)

