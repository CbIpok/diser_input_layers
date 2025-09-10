#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

from diser.io.coeffs import read_coef_json
from diser.io.basis import load_basis_dir
from diser.core.restore import (
    reconstruct_from_bases,
    valid_mask_from_bases,
    mse_on_valid_region,
)


def load_basis_coefs(path: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    samples = read_coef_json(path)
    xs_arr = np.asarray(samples.xs, dtype=float)
    ys_arr = np.asarray(samples.ys, dtype=float)
    coef_arrays = [np.asarray(c, dtype=float) for c in samples.coefs]
    return xs_arr, ys_arr, coef_arrays


def get_coeffs_for_point(x_sel: float, y_sel: float,
                         xs: np.ndarray, ys: np.ndarray,
                         coefs: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    xs_f = np.asarray(xs, dtype=float)
    ys_f = np.asarray(ys, dtype=float)
    mask = (np.isclose(xs_f, x_sel) & np.isclose(ys_f, y_sel))
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        d2 = (xs_f - x_sel) ** 2 + (ys_f - y_sel) ** 2
        idx = int(np.argmin(d2))
        print(f"[!] Точка ({x_sel}, {y_sel}) не найдена. "
              f"Ближайшая использована: ({xs_f[idx]:.0f}, {ys_f[idx]:.0f}), index={idx}")
    else:
        idx = int(idxs[0])
    c = np.array([coef[idx] for coef in coefs], dtype=float)
    return c, idx


def plot_overlayed_surfaces_3d(Z_true: np.ndarray, Z_hat: np.ndarray,
                               valid_mask: np.ndarray, point: Tuple[int, int] | None = None):
    H, W = Z_true.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    Z_true_m = np.where(valid_mask, Z_true, np.nan)
    Z_hat_m = np.where(valid_mask, Z_hat, np.nan)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    alpha_true, alpha_hat = 0.7, 0.6
    ax.plot_surface(X, Y, Z_true_m, alpha=alpha_true, linewidth=0, antialiased=False, shade=True)
    ax.plot_surface(X, Y, Z_hat_m, alpha=alpha_hat, linewidth=0, antialiased=False, shade=True)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Сравнение поверхностей: functions.wave vs восстановление')

    if point is not None:
        px, py = point
        if 0 <= py < H and 0 <= px < W and valid_mask[int(py), int(px)]:
            ax.scatter(px, py, Z_true[int(py), int(px)], s=50)

    legend_elems = [
        Patch(facecolor='gray', alpha=alpha_true, label='истинная (functions.wave)'),
        Patch(facecolor='C0', alpha=alpha_hat, label='восстановленная'),
    ]
    ax.legend(handles=legend_elems, loc='upper right')
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Сравнение поверхностей и точечная проверка MSE на базе коэффициентов.'
    )
    p.add_argument('--i', type=int, default=4,
                   help='Количество базисных: coefs_process/basis_{i}.json и data/basis_{i}')
    p.add_argument('--folder', default='coefs_process',
                   help='Папка с JSON-файлом коэффициентов (по умолчанию coefs_process)')
    p.add_argument('--basis-root', default='data',
                   help='Корневой путь к basis_{i} (по умолчанию data)')
    p.add_argument('--functions', default='data/functions.wave',
                   help='Путь к полю функций (по умолчанию data/functions.wave)')
    p.add_argument('--point', nargs=2, type=float, default=[100, 547],
                   help='Координата (x y) для подстановки коэффициентов. По умолчанию 100 547')
    return p.parse_args()


def main():
    args = parse_args()
    coefs_json = os.path.join(args.folder, f"basis_{args.i}.json")
    basis_dir = os.path.join(args.basis_root, f"basis_{args.i}")

    print('Входные пути:')
    print(f"  --coefs-json : {coefs_json}")
    print(f"  --basis-dir  : {basis_dir}")
    print(f"  --functions  : {args.functions}")

    xs, ys, coefs = load_basis_coefs(coefs_json)
    bases = load_basis_dir(basis_dir)
    to_restore = np.loadtxt(args.functions)

    if to_restore.ndim != 2 or bases.ndim != 3:
        raise ValueError('Неверные формы: to_restore -> (H, W), bases -> (k, H, W)')

    k, H_b, W_b = bases.shape
    H_f, W_f = to_restore.shape
    if (H_b, W_b) != (H_f, W_f):
        raise AssertionError(f'Несогласование размеров: bases: {(H_b, W_b)}, functions: {(H_f, W_f)}')

    if len(coefs) != k:
        print(f"[!] Предупреждение: число коэф. в JSON ({len(coefs)}) != число базисов ({k}). "
              f"Будет использовано min(k_json, k).")
        k_use = min(len(coefs), k)
        bases = bases[:k_use]
        coefs = coefs[:k_use]
        k = k_use

    valid_mask = valid_mask_from_bases(bases)
    if not np.any(valid_mask):
        raise RuntimeError('Нет валидной области: все значения NaN.')

    x_sel, y_sel = args.point
    c, used_idx = get_coeffs_for_point(x_sel, y_sel, xs, ys, coefs)
    if c.shape[0] != k:
        raise AssertionError(f'Размерность коэффициентов ({c.shape[0]}) не совпадает с числом базисов ({k}).')

    Z_hat = reconstruct_from_bases(c, bases)
    mse_val = mse_on_valid_region(to_restore, Z_hat, valid_mask)
    print(f"MSE (по валидной области) для точки {used_idx} "
          f"(~({xs[used_idx]}, {ys[used_idx]})): {mse_val:.6g}")

    plot_overlayed_surfaces_3d(to_restore, Z_hat, valid_mask, point=(int(x_sel), int(y_sel)))


if __name__ == '__main__':
    main()
