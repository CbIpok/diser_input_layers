#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diser.io.basis import load_basis_dir
from diser.io.coeffs import read_coef_json
from diser.core.restore import gaussian_smooth_nan
from matplotlib.path import Path as MplPath


def _ensure_dir(d: str | os.PathLike) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def _load_functions(path: str) -> np.ndarray:
    return np.load(path) if str(path).lower().endswith('.npy') else np.loadtxt(path)


def _save_interval() -> int:
    cfg_path = Path('data') / 'config.json'
    if cfg_path.exists():
        try:
            return int(json.loads(cfg_path.read_text(encoding='utf-8')).get('save_interval', 1))
        except Exception:
            return 1
    return 1


def _sample_points_from_union(union: np.ndarray, n: int, seed: int | None = None,
                              force_first: Tuple[int, int] | None = None) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(union)
    idx = np.arange(xs.size)
    rng.shuffle(idx)
    pts = []
    if force_first is not None:
        fx, fy = force_first
        # Always include forced point if inside bounds (even if outside union)
        if 0 <= fy < union.shape[0] and 0 <= fx < union.shape[1]:
            pts.append((fx, fy))
    for k in idx:
        x = int(xs[k]); y = int(ys[k])
        if force_first is not None and len(pts) > 0 and (x, y) == force_first:
            continue
        pts.append((x, y))
        if len(pts) >= n:
            break
    return pts


def _load_source_mask(nx: int, ny: int, area_path: str = 'data/areas/source.json') -> np.ndarray:
    verts = np.array(json.loads(Path(area_path).read_text(encoding='utf-8'))['points'], dtype=float)
    xs = np.arange(0.5, nx, 1.0)
    ys = np.arange(0.5, ny, 1.0)
    X, Y = np.meshgrid(xs, ys)
    mask_poly = MplPath(verts).contains_points(np.vstack((X.ravel(), Y.ravel())).T)
    return mask_poly.reshape(ny, nx)


def _build_index(xs: np.ndarray, ys: np.ndarray):
    # Map (xs,ys) -> index for fast lookup
    d = {}
    for i, (x, y) in enumerate(zip(xs.astype(int), ys.astype(int))):
        d[(int(x), int(y))] = i
    return d


def _nearest_idx(xs: np.ndarray, ys: np.ndarray, x: int, y: int) -> int:
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    return int(np.argmin(d2))


def parse_args():
    p = argparse.ArgumentParser(description='Research 5000 points: local patch errors for mean-over-i reconstruction')
    p.add_argument('--i-list', default='16,25,36,49', help='i list for mean reconstruction (comma-separated)')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Original field')
    p.add_argument('--n-points', type=int, default=5000, help='Number of points to analyze (min 5000)')
    p.add_argument('--window', type=int, default=12, help='Half-size of patch window (12 -> 25x25)')
    p.add_argument('--seed', type=int, default=0, help='Random seed for sampling points')
    p.add_argument('--out-dir', default='output/research_5000', help='Output dir')
    p.add_argument('--force-first', default='2000,1400', help='Force first global point x,y (default: 2000,1400)')
    return p.parse_args()


def main():
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]
    _ensure_dir(args.out_dir)

    T = _load_functions(args.functions).astype(np.float32, copy=False)
    H, W = T.shape
    save_int = _save_interval()

    # Source polygon mask (approximate source zone mask)
    union = _load_source_mask(W, H, area_path='data/areas/source.json')

    if args.force_first:
        fx, fy = [int(v) for v in args.force_first.split(',')]
        force = (fx, fy)
    else:
        force = None
    pts = _sample_points_from_union(union, max(1, args.n_points), seed=args.seed, force_first=force)

    # Prepare per-point storage for original patches and mean reconstruction sum
    half = int(args.window)
    side = 2 * half + 1
    P = side * side
    N = len(pts)

    orig_patches = np.empty((N, P), dtype=np.float32)
    mean_sum = np.zeros((N, P), dtype=np.float32)

    # Also collect per-i MAE / approx_error per point (summary only) and store the patches themselves
    per_i_mae = {i: np.zeros(N, dtype=np.float32) for i in i_list}
    per_i_ae = {i: np.zeros(N, dtype=np.float32) for i in i_list}
    per_i_patch: dict[int, np.ndarray] = {i: np.zeros((N, P), dtype=np.float32) for i in i_list}
    # Adaptive vs baseline (mean+smooth) per point
    mae_adapt = np.zeros(N, dtype=np.float32)
    mae_baseS = np.zeros(N, dtype=np.float32)

    # Pre-fill original patches
    for pi, (xg, yg) in enumerate(pts):
        y0 = max(0, yg - half); y1 = min(H, yg + half + 1)
        x0 = max(0, xg - half); x1 = min(W, xg + half + 1)
        patch = np.zeros((side, side), dtype=np.float32)
        sub = T[y0:y1, x0:x1]
        patch[(y0-yg+half):(y1-yg+half), (x0-xg+half):(x1-xg+half)] = sub
        orig_patches[pi, :] = patch.reshape(-1)

    # For each i, load bases and coefs, then accumulate mean_sum and per-i MAE
    for i in i_list:
        # Load bases (float32) and coefs
        B = load_basis_dir(os.path.join(args.basis_root, f'basis_{i}')).astype(np.float32, copy=False)
        samples = read_coef_json(Path(args.folder) / f'basis_{i}.json')
        xs = np.asarray(samples.xs, dtype=float)
        ys = np.asarray(samples.ys, dtype=float)
        idx_map = _build_index(xs, ys)

        for pi, (xg, yg) in enumerate(pts):
            # Sample coordinates (xsamp, ysamp)
            xsamp = int(round(xg / max(save_int, 1)))
            ysamp = int(round(yg / max(save_int, 1)))
            key = (xsamp, ysamp)
            if key in idx_map:
                idx = idx_map[key]
            else:
                idx = _nearest_idx(xs, ys, xsamp, ysamp)
            c = np.array([np.asarray(ca, dtype=np.float32)[idx] for ca in samples.coefs], dtype=np.float32)
            # approx_error at sample
            ae_val = None
            try:
                if getattr(samples, 'approx_error', None) is not None and len(samples.approx_error) > idx:
                    ae_val = float(np.asarray(samples.approx_error, dtype=np.float32)[idx])
            except Exception:
                ae_val = None

            # Patch reconstruction via tensordot on subvolume
            y0 = max(0, yg - half); y1 = min(H, yg + half + 1)
            x0 = max(0, xg - half); x1 = min(W, xg + half + 1)
            sub = np.nan_to_num(B[:, y0:y1, x0:x1], nan=0.0, copy=False)
            patch = np.tensordot(c, sub, axes=(0, 0)).astype(np.float32, copy=False)

            # Place into fixed-size buffer
            buf = np.zeros((side, side), dtype=np.float32)
            buf[(y0-yg+half):(y1-yg+half), (x0-xg+half):(x1-xg+half)] = patch
            vec = buf.reshape(-1)
            mean_sum[pi, :] += vec
            per_i_patch[i][pi, :] = vec

            # per-i MAE
            dv = np.abs(vec - orig_patches[pi, :])
            per_i_mae[i][pi] = float(np.mean(dv))
            if ae_val is not None:
                per_i_ae[i][pi] = ae_val

        # free
        del B

    # Final mean patch and metrics
    mean_patch = (mean_sum / float(len(i_list))).astype(np.float32)
    # Baseline smoothing on mean patch (global sigma ~ 1.5, but patch-scale small; use sigma=1.0)
    def _smooth_patch(p: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        side = int(np.sqrt(p.size))
        M = p.reshape(side, side)
        S = gaussian_smooth_nan(M.astype(np.float64), sigma=float(sigma)).astype(np.float32)
        return S.reshape(-1)
    mean_patch_sm = np.vstack([_smooth_patch(mean_patch[pi, :], sigma=1.5) for pi in range(N)]).astype(np.float32)
    diff = mean_patch - orig_patches
    mae = np.mean(np.abs(diff), axis=1)
    bias = np.mean(diff, axis=1)
    std = np.std(diff, axis=1)

    # Baseline (mean+smooth) MAE per point
    diffB = mean_patch_sm - orig_patches
    mae_baseS = np.mean(np.abs(diffB), axis=1)

    # Adaptive fusion on patches using per-point approx_error and the local gradient of mean_patch
    # Precompute per-point gradient magnitude maps for mean_patch
    gmaps = np.zeros((N, side, side), dtype=np.float32)
    for pi in range(N):
        M = mean_patch[pi, :].reshape(side, side)
        gx = np.zeros_like(M); gy = np.zeros_like(M)
        gx[:, 1:-1] = 0.5 * (M[:, 2:] - M[:, :-2])
        gy[1:-1, :] = 0.5 * (M[2:, :] - M[:-2, :])
        g = np.hypot(gx, gy)
        g /= float(np.nanmax(g) + 1e-12)
        gmaps[pi] = g

    # Compute per-point adaptive fused patch
    eps = 1e-6
    max_i = float(max(i_list))
    alpha = 1.0
    beta = 1.0
    # accumulate numerator/denominator per pi
    num = np.zeros((N, P), dtype=np.float64)
    den = np.zeros((N, P), dtype=np.float64)
    Wmod = (1.0 + 0.5 * gmaps).reshape(N, P)
    for i in i_list:
        ae_vec = per_i_ae[i]
        base_w = np.power((i / max_i), alpha) * np.power(np.maximum(ae_vec, 0.0) + eps, -beta)
        Wpix = (base_w[:, None] * Wmod).astype(np.float64)
        Pi = per_i_patch[i].astype(np.float64)
        num += Wpix * Pi
        den += Wpix
    adapt_patch = (num / np.maximum(den, eps)).astype(np.float32)
    mae_adapt = np.mean(np.abs(adapt_patch - orig_patches), axis=1).astype(np.float32)

    # Save a compact summary JSON (avoid massive CSV)
    summary = {
        'i_list': i_list,
        'n_points': N,
        'window': int(args.window),
        'save_interval': int(save_int),
        'forced_first': pts[0] if pts else None,
        'global_stats': {
            'mae_mean': float(np.mean(mae)),
            'mae_median': float(np.median(mae)),
            'mae_p95': float(np.percentile(mae,95)),
            'bias_mean': float(np.mean(bias)),
            'std_mean': float(np.mean(std)),
        },
        'per_i_mae_mean': {str(i): float(np.mean(per_i_mae[i])) for i in i_list},
        'per_i_mae_median': {str(i): float(np.median(per_i_mae[i])) for i in i_list},
        'baseline_smoothed': {
            'mae_mean': float(np.mean(mae_baseS)),
            'mae_median': float(np.median(mae_baseS)),
            'mae_p95': float(np.percentile(mae_baseS,95))
        },
        'adaptive_patch': {
            'alpha': float(alpha),
            'beta': float(beta),
            'mae_mean': float(np.mean(mae_adapt)),
            'mae_median': float(np.median(mae_adapt)),
            'mae_p95': float(np.percentile(mae_adapt,95)),
            'frac_improved_vs_baseline': float((mae_adapt < mae_baseS).mean())
        }
    }
    # Correlation with approx_error if present
    try:
        corr = {}
        for i in i_list:
            ae = per_i_ae[i]
            m = np.isfinite(ae) & (ae!=0)
            if np.any(m):
                # correlate per-point mae_i with approx_error
                v = per_i_mae[i][m]
                a = ae[m]
                if v.size>1:
                    vm = v - v.mean(); am = a - a.mean()
                    denom = float(np.sqrt((vm*vm).sum()) * np.sqrt((am*am).sum()) + 1e-12)
                    corr[str(i)] = float((vm*am).sum()/denom)
        summary['corr_mae_vs_aprox_error'] = corr
    except Exception:
        pass
    out_json = Path(args.out_dir) / 'summary.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Save a small preview of the first few mean patches
    try:
        import matplotlib.pyplot as plt
        prev_dir = Path(args.out_dir) / 'previews'
        _ensure_dir(prev_dir)
        for pi in range(min(6, N)):
            m = mean_patch[pi, :].reshape(side, side)
            o = orig_patches[pi, :].reshape(side, side)
            fig = plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1); plt.imshow(o, origin='upper', cmap='viridis'); plt.title('orig'); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(m, origin='upper', cmap='viridis'); plt.title('mean rec'); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(m-o, origin='upper', cmap='coolwarm'); plt.title('diff'); plt.axis('off')
            plt.tight_layout()
            plt.savefig(prev_dir / f'pt_{pi}_x{pts[pi][0]}_y{pts[pi][1]}.png')
            plt.close(fig)
    except Exception:
        pass

    print(out_json)


if __name__ == '__main__':
    main()

