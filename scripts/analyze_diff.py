#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


def _load_array(path: str) -> np.ndarray:
    p = str(path)
    if p.lower().endswith('.npy'):
        return np.load(p)
    return np.loadtxt(p)


def _ensure_dir(d: str | os.PathLike) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def _roll_diff_edges(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    # 4-neighborhood boundary
    b = m & (
        (np.roll(m, 1, 0) != m) |
        (np.roll(m, -1, 0) != m) |
        (np.roll(m, 1, 1) != m) |
        (np.roll(m, -1, 1) != m)
    )
    if steps <= 1:
        return b
    out = b.copy()
    for _ in range(steps - 1):
        nb = (
            np.roll(out, 1, 0) | np.roll(out, -1, 0) |
            np.roll(out, 1, 1) | np.roll(out, -1, 1) |
            out
        )
        out = nb
    return out


def _radial_psd(arr: np.ndarray, downsample: int = 4, bins: int = 50) -> dict[str, list[float]]:
    # Downsample for speed
    A = arr
    if downsample and downsample > 1:
        A = A[::downsample, ::downsample]
    A = np.asarray(A, dtype=float)
    A = A - np.nanmean(A)
    A = np.nan_to_num(A, copy=False)
    # 2D FFT and power
    F = np.fft.rfft2(A)
    P = (F * np.conj(F)).real
    H, W_half = P.shape
    # frequency grids
    fy = np.fft.fftfreq(H)
    fx = np.fft.rfftfreq(A.shape[1])
    FY, FX = np.meshgrid(fx, fy)
    R = np.hypot(FX, FY)
    r = R.flatten()
    p = P.flatten()
    # radial binning
    rmax = r.max()
    edges = np.linspace(0.0, rmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=int)
    inds = np.digitize(r, edges) - 1
    for k in range(bins):
        sel = inds == k
        if np.any(sel):
            vals[k] = float(np.mean(p[sel]))
            counts[k] = int(np.count_nonzero(sel))
        else:
            vals[k] = float('nan')
    return {"freq": centers.tolist(), "psd": vals.tolist(), "counts": counts.astype(int).tolist()}


def _autocorr_1d_shifts(arr: np.ndarray, max_shift: int = 16) -> dict[str, list[float]]:
    out_x = []
    out_y = []
    A = np.asarray(arr, dtype=float)
    A = A - np.nanmean(A)
    A = np.nan_to_num(A, copy=False)
    for d in range(1, max_shift + 1):
        # x-shift
        a = A[:, :-d].ravel()
        b = A[:, d:].ravel()
        vx = np.corrcoef(a, b)[0, 1]
        out_x.append(float(vx))
        # y-shift
        a = A[:-d, :].ravel()
        b = A[d:, :].ravel()
        vy = np.corrcoef(a, b)[0, 1]
        out_y.append(float(vy))
    return {"dx": list(range(1, max_shift + 1)), "acf_x": out_x, "acf_y": out_y}


def _robust_stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    mean = float(np.mean(v))
    std = float(np.std(v))
    pcts = {p: float(np.percentile(v, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    skew = float(np.mean(((v - mean) / (std + 1e-12)) ** 3))
    kurt = float(np.mean(((v - mean) / (std + 1e-12)) ** 4)) - 3.0
    th = [0.001, 0.005, 0.01, 0.02, 0.05]
    within = {f"frac_abs<= {t}": float(np.mean(np.abs(v) <= t)) for t in th}
    return {
        "count": int(v.size),
        "mean": mean,
        "median": med,
        "std": std,
        "mad": mad,
        "skew": skew,
        "excess_kurtosis": kurt,
        "pcts": pcts,
        "within": within,
        "frac_pos": float(np.mean(v > 0)),
        "frac_neg": float(np.mean(v < 0)),
    }


def analyze_diff(recon_path: str,
                 functions_path: str = 'data/functions.wave',
                 basis_root: str = 'data', basis_i: Optional[int] = None,
                 out_dir: str = 'output/diff_analysis', downsample_psd: int = 4,
                 support_eps: float = 1e-12) -> str:
    _ensure_dir(out_dir)
    rec = _load_array(recon_path)
    func = _load_array(functions_path)
    if rec.shape != func.shape:
        raise AssertionError(f"Shape mismatch: recon {rec.shape} vs functions {func.shape}")
    diff = rec - func

    # Valid/edges from basis (if requested/available)
    valid_mask = None
    edges_mask = None
    interior_mask = None
    if basis_i is not None:
        try:
            from diser.io.basis import load_basis_dir
            from diser.core.restore import valid_mask_from_bases
            B = load_basis_dir(os.path.join(basis_root, f"basis_{basis_i}"))
            valid_mask = valid_mask_from_bases(B)
            # Internal edges: union of per-basis boundaries
            edges = np.zeros_like(valid_mask, dtype=bool)
            for j in range(B.shape[0]):
                bj = np.isfinite(B[j])
                edges |= _roll_diff_edges(bj, steps=1)
            edges_mask = edges
            interior_mask = valid_mask & (~edges_mask)
        except Exception:
            valid_mask = None

    # Core stats
    overall = _robust_stats(diff)
    stats = {"overall": overall}

    # Support vs outside (by functions amplitude)
    if support_eps is not None and support_eps >= 0:
        supp = np.abs(func) > support_eps
        stats["support_region"] = _robust_stats(diff[supp])
        stats["outside_support_region"] = _robust_stats(diff[~supp])

    if valid_mask is not None:
        stats["valid_region"] = _robust_stats(diff[valid_mask])
        if edges_mask is not None:
            stats["edges_region"] = _robust_stats(diff[edges_mask])
        if interior_mask is not None:
            stats["interior_region"] = _robust_stats(diff[interior_mask])

    # Gradient-conditioned error (by quintiles of |∇T|)
    # Simple central differences
    gx = np.zeros_like(func)
    gy = np.zeros_like(func)
    gx[:, 1:-1] = 0.5 * (func[:, 2:] - func[:, :-2])
    gy[1:-1, :] = 0.5 * (func[2:, :] - func[:-2, :])
    gmag = np.hypot(gx, gy)
    g = gmag[np.isfinite(gmag) & np.isfinite(diff)]
    d = diff[np.isfinite(gmag) & np.isfinite(diff)]
    if g.size:
        qs = np.percentile(g, [0, 20, 40, 60, 80, 100])
        by_grad = []
        for a, b in zip(qs[:-1], qs[1:]):
            sel = (g >= a) & (g <= b)
            if np.any(sel):
                dv = d[sel]
                by_grad.append({
                    "g_range": [float(a), float(b)],
                    "abs_diff_mean": float(np.mean(np.abs(dv))),
                    "abs_diff_median": float(np.median(np.abs(dv))),
                    "count": int(dv.size),
                })
        stats["by_grad_quintiles"] = by_grad

    # ACF along axes (global)
    acf = _autocorr_1d_shifts(np.nan_to_num(diff, copy=False), max_shift=16)
    stats["acf_1d"] = acf

    # PSD (downsampled)
    psd = _radial_psd(diff, downsample=downsample_psd, bins=50)
    stats["psd_radial"] = psd

    # Save histogram figure
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 5))
        v = diff[np.isfinite(diff)]
        plt.hist(v, bins=128, color='#2c7fb8', alpha=0.85)
        plt.xlabel('diff = recon - original')
        plt.ylabel('pixels')
        plt.title('Histogram of difference')
        fig.tight_layout()
        fig_path = os.path.join(out_dir, 'diff_hist.png')
        fig.savefig(fig_path)
        plt.close(fig)
        stats["histogram_png"] = fig_path
    except Exception:
        pass

    # Save report
    out_json = os.path.join(out_dir, 'report.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return out_json


def parse_args():
    p = argparse.ArgumentParser(description='Detailed analysis of (reconstruction - original) difference')
    p.add_argument('--recon', required=True, help='Path to reconstructed field (.npy or text)')
    p.add_argument('--functions', default='data/functions.wave', help='Path to original field')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i} for mask/edges')
    p.add_argument('--basis-i', type=int, default=49, help='i to use for union-of-basis mask/edges')
    p.add_argument('--out-dir', default='output/diff_analysis', help='Where to write report/plots')
    p.add_argument('--downsample-psd', type=int, default=4, help='Downsample factor for PSD speedup')
    p.add_argument('--support-eps', type=float, default=1e-12, help='|functions| > eps defines support region for extra stats')
    return p.parse_args()


def main():
    args = parse_args()
    report = analyze_diff(
        recon_path=args.recon,
        functions_path=args.functions,
        basis_root=args.basis_root,
        basis_i=args.basis_i,
        out_dir=args.out_dir,
        downsample_psd=args.downsample_psd,
        support_eps=args.support_eps,
    )
    print(report)


if __name__ == '__main__':
    main()
