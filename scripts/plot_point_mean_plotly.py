#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from point import reconstruct_mean_over_i


def parse_args():
    p = argparse.ArgumentParser(
        description='Interactive 3D Plotly view for mean reconstruction over multiple i (like point, but Plotly)'
    )
    p.add_argument('--i-list', required=True,
                   help='Comma-separated list of i values, e.g. 4,16,25')
    p.add_argument('--point', nargs=2, type=float, default=[100, 547],
                   help='Point (x y) to pick coefficients for')
    p.add_argument('--folder', default='coefs_process', help='Folder with basis_{i}.json')
    p.add_argument('--basis-root', default='data', help='Root folder with basis_{i}')
    p.add_argument('--functions', default='data/functions.wave', help='Path to functions.wave')
    p.add_argument('--smooth-sigma', type=float, default=None, help='Gaussian sigma for smoothing (optional)')
    p.add_argument('--config', default='data/config.json', help='Path to config.json (for axis ranges)')
    p.add_argument('--save-html', default='output/mean_reconstruction_plotly.html',
                   help='Output HTML file for interactive Plotly figure')
    p.add_argument('--overlay-true', action='store_true',
                   help='Overlay true surface (functions.wave) together with mean reconstruction')
    p.add_argument('--step', type=int, default=None,
                   help='Plot every N-th sample along Y and X to reduce mesh size (e.g., 2,4,8)')
    p.add_argument('--mask-zeros', action='store_true',
                   help='Mask near-zero values in mean surface to avoid giant flat plane')
    p.add_argument('--zero-eps', type=float, default=0.0,
                   help='Threshold for zero masking (|z|<=eps -> NaN when --mask-zeros)')
    p.add_argument('--embed-js', action='store_true',
                   help='Embed plotly.js into HTML (offline viewing without internet)')
    return p.parse_args()


def plot_mean_plotly(mean_Z: np.ndarray,
                     true_Z: np.ndarray | None,
                     point_xy: tuple[float, float] | None,
                     save_html: str,
                     nx_axis: int | None = None,
                     ny_axis: int | None = None,
                     step: int | None = None,
                     mask_zeros: bool = False,
                     zero_eps: float = 0.0,
                     embed_js: bool = False):
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot as plot_offline
    except Exception as e:
        print("[ERROR] Plotly is not installed. Please install it via 'pip install plotly'.", file=sys.stderr)
        raise

    H, W = mean_Z.shape
    # Optional decimation for performance/visibility
    sy = sx = 1
    if step and step > 1:
        sy = sx = int(step)
    mean_Z_p = mean_Z[::sy, ::sx]
    true_Z_p = true_Z[::sy, ::sx] if true_Z is not None else None
    if mask_zeros and zero_eps >= 0.0:
        m = np.isfinite(mean_Z_p)
        z = mean_Z_p
        z_mask = m & (np.abs(z) <= float(zero_eps))
        mean_Z_p = mean_Z_p.copy()
        mean_Z_p[z_mask] = np.nan
        if true_Z_p is not None:
            true_Z_p = np.where(np.isfinite(mean_Z_p), true_Z_p, np.nan)
    H, W = mean_Z_p.shape
    x = np.arange(W) * sx
    y = np.arange(H) * sy

    fig = go.Figure()

    # Mean reconstruction surface (primary)
    # Compute color limits ignoring NaNs and near-zeros if masked
    finite_vals = mean_Z_p[np.isfinite(mean_Z_p)]
    cmin = float(np.nanmin(finite_vals)) if finite_vals.size else None
    cmax = float(np.nanmax(finite_vals)) if finite_vals.size else None

    fig.add_trace(go.Surface(
        z=mean_Z,
        x=x,
        y=y,
        colorscale='Viridis',
        opacity=0.85,
        showscale=True,
        cmin=cmin,
        cmax=cmax,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1),
        name='Mean reconstruction',
    ))

    # Optionally overlay true surface, masked to where mean is finite
    if true_Z_p is not None:
        mask = np.isfinite(mean_Z_p)
        Z_true_masked = np.where(mask, true_Z_p, np.nan)
        fig.add_trace(go.Surface(
            z=Z_true_masked,
            x=x,
            y=y,
            colorscale='Greys',
            opacity=0.5,
            showscale=False,
            name='True (functions.wave)'
        ))

    # Optional point marker
    if point_xy is not None:
        px, py = point_xy
        if 0 <= px < W and 0 <= py < H:
            # Determine Z for marker: prefer mean if finite, else true if provided
            z_val = np.nan
            if np.isfinite(mean_Z[int(py), int(px)]):
                z_val = float(mean_Z[int(py), int(px)])
            elif true_Z is not None and np.isfinite(true_Z[int(py), int(px)]):
                z_val = float(true_Z[int(py), int(px)])
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[z_val],
                mode='markers',
                marker=dict(size=5, color='red'),
                name=f'Point ({int(px)}, {int(py)})'
            ))

    # Determine desired axis ranges matching the full config area (like point.py uses 0..W and 0..H)
    ax_x_max = int(nx_axis) if nx_axis and nx_axis > 0 else W
    ax_y_max = int(ny_axis) if ny_axis and ny_axis > 0 else H

    fig.update_scenes(
        xaxis=dict(title='X', range=[0, ax_x_max]),
        yaxis=dict(title='Y', range=[0, ax_y_max]),
        zaxis=dict(title='Z'),
        aspectmode='data'
    )
    fig.update_layout(
        title='Mean reconstruction (Plotly 3D)',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )

    os.makedirs(os.path.dirname(save_html) or '.', exist_ok=True)
    include_js = True if embed_js else 'cdn'
    plot_offline(fig, filename=save_html, auto_open=False, include_plotlyjs=include_js)
    print(f"Saved interactive Plotly figure to {save_html}")


def main():
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(',') if s.strip()]
    mean_Z, _ = reconstruct_mean_over_i(
        i_list,
        folder=args.folder,
        basis_root=args.basis_root,
        functions_path=args.functions,
        point_xy=tuple(args.point),
        smooth_sigma=args.smooth_sigma,
    )

    true_Z = None
    if args.overlay_true:
        true_Z = np.loadtxt(args.functions)

    # Read config to align axis ranges to full area
    nx_axis = ny_axis = None
    try:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if isinstance(cfg, dict) and 'size' in cfg:
            nx_axis = int(cfg['size'].get('x', 0))
            ny_axis = int(cfg['size'].get('y', 0))
    except FileNotFoundError:
        pass

    plot_mean_plotly(
        mean_Z, true_Z, tuple(args.point), args.save_html,
        nx_axis=nx_axis, ny_axis=ny_axis,
        step=args.step, mask_zeros=args.mask_zeros, zero_eps=args.zero_eps,
        embed_js=args.embed_js,
    )


if __name__ == '__main__':
    main()
