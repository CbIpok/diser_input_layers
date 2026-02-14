#!/usr/bin/env python
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from diser.core.restore import reconstruct_from_bases, valid_mask_from_bases
from diser.io.basis import load_basis_dir
from diser.io.coeffs import resolve_coeffs_dir
from diser.viz.figio import save_figure_bundle
from point import load_basis_coefs, get_coeffs_for_point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 3D reconstruction surfaces for i=100 and 64 at point (2225,1250)."
    )
    p.add_argument("--i-list", default="100,64", help="Comma-separated i list (default: 100,64)")
    p.add_argument("--point", nargs=2, type=float, default=[2225, 1250], help="Point (x y)")
    p.add_argument("--folder", default="coefs_process", help="Folder with basis_{i}.json")
    p.add_argument("--basis-root", default="data", help="Root folder with basis_{i}")
    p.add_argument("--functions", default="data/functions_pow1.wave", help="Path to functions.wave")
    p.add_argument("--out-dir", default="output/point_surfaces", help="Output directory")
    p.add_argument("--img-formats", default="png", help="Comma-separated formats for figure")
    p.add_argument("--step", type=int, default=8, help="Plot every N-th sample for speed")
    p.add_argument("--overlay-true", action="store_true", help="Overlay functions.wave surface")
    return p.parse_args()


def plot_surface(Z_hat: np.ndarray,
                 valid_mask: np.ndarray,
                 point_xy: tuple[float, float],
                 Z_true: np.ndarray | None = None,
                 step: int = 1):
    Z_show = np.where(valid_mask, Z_hat, np.nan)
    if step and step > 1:
        Z_show = Z_show[::step, ::step]
        if Z_true is not None:
            Z_true = Z_true[::step, ::step]

    h, w = Z_show.shape
    x = np.arange(w) * (step if step and step > 1 else 1)
    y = np.arange(h) * (step if step and step > 1 else 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z_show, alpha=0.8, linewidth=0, antialiased=False, shade=True)

    if Z_true is not None:
        Z_true_m = np.where(np.isfinite(Z_show), Z_true, np.nan)
        ax.plot_surface(X, Y, Z_true_m, alpha=0.45, linewidth=0, antialiased=False, shade=True)

    px, py = point_xy
    if 0 <= px < Z_hat.shape[1] and 0 <= py < Z_hat.shape[0]:
        z_val = Z_hat[int(py), int(px)]
        if np.isfinite(z_val):
            ax.scatter(px, py, z_val, s=40, color="red")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig


def main() -> None:
    args = parse_args()
    i_list = [int(s) for s in args.i_list.split(",") if s.strip()]
    x_sel, y_sel = map(float, args.point)

    true_Z = None
    if args.overlay_true:
        if str(args.functions).lower().endswith(".npy"):
            true_Z = np.load(args.functions)
        else:
            true_Z = np.loadtxt(args.functions)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for i_val in i_list:
        coefs_dir = resolve_coeffs_dir(args.folder, args.functions)
        coefs_json = coefs_dir / f"basis_{i_val}.json"
        basis_dir = os.path.join(args.basis_root, f"basis_{i_val}")

        xs, ys, coefs = load_basis_coefs(coefs_json)
        bases = load_basis_dir(basis_dir)
        c, _ = get_coeffs_for_point(x_sel, y_sel, xs, ys, coefs)
        Z_hat = reconstruct_from_bases(c, bases)
        valid_mask = valid_mask_from_bases(bases)

        fig = plot_surface(
            Z_hat,
            valid_mask,
            (x_sel, y_sel),
            Z_true=true_Z,
            step=args.step,
        )

        suffix = f"i-{i_val}_point-{int(round(x_sel))}_{int(round(y_sel))}"
        base = os.path.join(out_dir, f"surface_3d__{suffix}")
        fmts = tuple([s.strip() for s in str(args.img_formats).split(",") if s.strip()]) or ("png",)
        save_figure_bundle(fig, base, formats=fmts, with_pickle=True)
        plt.close(fig)
        print(f"Saved: {base}.*")


if __name__ == "__main__":
    main()
