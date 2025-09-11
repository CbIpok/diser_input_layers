from __future__ import annotations

import numpy as np


def reconstruct_from_bases(c: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Compute Z_hat = sum_j c_j * basis_j, treating NaN in bases as zeros."""
    c = np.asarray(c, dtype=float)
    B = np.asarray(bases, dtype=float)
    B = np.nan_to_num(B, nan=0.0, copy=False)
    return np.tensordot(c, B, axes=(0, 0))


def valid_mask_from_bases(bases: np.ndarray) -> np.ndarray:
    """Return boolean mask where any basis has finite value."""
    return np.any(np.isfinite(bases), axis=0)


def mse_on_valid_region(Z_true: np.ndarray, Z_hat: np.ndarray, valid_mask: np.ndarray) -> float:
    """Mean squared error only over True entries of valid_mask."""
    Zt = np.asarray(Z_true, dtype=float)
    Zh = np.asarray(Z_hat, dtype=float)
    m = np.asarray(valid_mask, dtype=bool)
    diff = (Zt - Zh)[m]
    return float(np.mean(diff * diff)) if diff.size else float('nan')


def pointwise_rmse_from_coefs(to_restore: np.ndarray,
                              bases: np.ndarray,
                              xs: np.ndarray,
                              ys: np.ndarray,
                              coefs: list[np.ndarray],
                              dtype=np.float64) -> np.ndarray:
    """Efficient RMSE evaluation at given (xs, ys) from coefficients.

    Implements the same math as plot_basis_maps.calc_mse but without I/O or plotting.
    Returns grid with NaN everywhere except at provided indices.
    """
    grid = np.asarray(to_restore, dtype=dtype)
    k, H, W = bases.shape
    xs_i = np.asarray(xs, dtype=int)
    ys_i = np.asarray(ys, dtype=int)
    assert xs_i.size == ys_i.size

    valid = np.any(np.isfinite(bases), axis=0).reshape(H * W)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.full_like(grid, np.nan)

    B = bases.reshape(k, -1)[:, idx]
    B = np.nan_to_num(B, nan=0.0, copy=False)
    t = grid.reshape(-1)[idx]

    G = B @ B.T
    h = B @ t
    tt = float(t @ t)
    N = t.size

    C = np.asarray(coefs, dtype=dtype)
    if C.ndim == 1:
        C = C[None, :]
    if C.shape[0] == k:
        C = C.T  # (M, k)

    term1 = np.einsum('ik,kl,il->i', C, G, C, optimize=True)
    term2 = 2.0 * (C @ h)
    mse_vals = (term1 - term2 + tt) / N

    out = np.full_like(grid, np.nan)
    # Place values at (row=y, col=x) indices (no stride here)
    out[ys_i, xs_i] = np.sqrt(mse_vals)
    return out


# --- Simple NaN-aware Gaussian smoothing (separable) ---
def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return k


def _convolve1d_nan(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(arr).astype(np.float64)
    arr_filled = np.where(np.isfinite(arr), arr, 0.0)

    def _conv1d(v):
        return np.convolve(v, kernel, mode='same')

    out = np.apply_along_axis(_conv1d, axis, arr_filled)
    w = np.apply_along_axis(_conv1d, axis, mask)
    with np.errstate(invalid='ignore'):
        out = out / w
    out[w == 0] = np.nan
    return out


def gaussian_smooth_nan(arr: np.ndarray, sigma: float = 1.0, radius: int | None = None) -> np.ndarray:
    """Gaussian smoothing for 2D array with NaN handling (separable)."""
    k = _gaussian_kernel_1d(sigma, radius)
    tmp = _convolve1d_nan(arr, k, axis=1)
    out = _convolve1d_nan(tmp, k, axis=0)
    return out
