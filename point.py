#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch


# ---------- Загрузка данных ----------

def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_basis_coefs(path: str) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Загружает из JSON координаты (x, y) и списки коэфов по каждому базису.
    Ожидается структура:
        {
          "(x, y)": {"coefs": [c1, c2, ...], "aprox_error": ...},
          ...
        }
    Возвращает:
        xs (M,), ys (M,), coefs: список длины k, каждый массив формы (M,)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    xs, ys = [], []
    coefs: List[List[float]] = []

    for key, val in data.items():
        x, y = ast.literal_eval(key)
        xs.append(float(x))
        ys.append(float(y))
        coef_vals = [_to_float(c) for c in val.get("coefs", [])]
        if not coefs:
            coefs = [[] for _ in coef_vals]
        for j, c in enumerate(coef_vals):
            coefs[j].append(c)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    coef_arrays = [np.asarray(c, dtype=float) for c in coefs]
    return xs_arr, ys_arr, coef_arrays


def load_basis_dir(path: str) -> np.ndarray:
    """
    Читает все файлы basis_*.wave из папки и сортирует по индексу.
    Возвращает массив формы (k, H, W). Нули заменяются на NaN.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Папка не найдена: {path}")

    arrays = []
    indices = []
    for filename in os.listdir(path):
        if filename.startswith("basis_") and filename.endswith(".wave"):
            try:
                idx = int(filename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            full_path = os.path.join(path, filename)
            data = np.loadtxt(full_path).astype(float, copy=False)
            data[data == 0] = np.nan
            arrays.append(data)
            indices.append(idx)

    if not arrays:
        raise RuntimeError(f"В {path} не найдено файлов basis_*.wave")

    arrays_sorted = [arr for _, arr in sorted(zip(indices, arrays))]
    B = np.stack(arrays_sorted, axis=0)  # (k, H, W)
    return B


# ---------- Логика выбора коэфов и реконструкции ----------

def get_coeffs_for_point(x_sel: float, y_sel: float,
                         xs: np.ndarray, ys: np.ndarray,
                         coefs: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    """
    Возвращает вектор коэффициентов c (k,) для точки (x_sel, y_sel).
    Если точной точки нет — берёт ближайшую по евклидовой метрике.
    Также возвращает индекс выбранной записи.
    """
    xs_f = np.asarray(xs, dtype=float)
    ys_f = np.asarray(ys, dtype=float)
    mask = (np.isclose(xs_f, x_sel) & np.isclose(ys_f, y_sel))
    idxs = np.flatnonzero(mask)

    if idxs.size == 0:
        d2 = (xs_f - x_sel) ** 2 + (ys_f - y_sel) ** 2
        idx = int(np.argmin(d2))
        print(f"[!] Точка ({x_sel}, {y_sel}) не найдена. "
              f"Использую ближайшую: ({xs_f[idx]:.0f}, {ys_f[idx]:.0f}), index={idx}")
    else:
        idx = int(idxs[0])

    c = np.array([coef[idx] for coef in coefs], dtype=float)  # (k,)
    return c, idx


def reconstruct_from_bases(c: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """
    Реконструкция поверхности: Z_hat = sum_j c_j * basis_j.
    NaN в базисах трактуем как 0 (отсутствие вклада).
    bases: (k, H, W), c: (k,)
    return: (H, W)
    """
    B = np.nan_to_num(bases, nan=0.0, copy=False)
    Z_hat = np.tensordot(c, B, axes=(0, 0))  # (H, W)
    return Z_hat


def valid_mask_from_bases(bases: np.ndarray) -> np.ndarray:
    """
    Маска валидной области: True там, где хотя бы один базис не NaN,
    как в исходной calc_mse.
    """
    return np.any(np.isfinite(bases), axis=0)  # (H, W)


def mse_on_valid_region(Z_true: np.ndarray, Z_hat: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    MSE только на валидной области.
    """
    if not np.any(valid_mask):
        return float("nan")
    diff = (Z_hat - Z_true)[valid_mask]
    return float(np.mean(diff ** 2))


# ---------- Визуализация (наложение) ----------

def plot_overlayed_surfaces_3d(Z_true: np.ndarray, Z_hat: np.ndarray,
                               valid_mask: np.ndarray,
                               point: Tuple[int, int] | None = None,
                               alpha_true: float = 0.35,
                               alpha_hat: float = 1.0) -> None:
    """
    Рисует ОДНУ 3D-сцену с наложением двух поверхностей:
      - Z_true (полупрозрачная, серого цвета),
      - Z_hat (поверх, почти непрозрачная).
    Показываем ТОЛЬКО валидную область (остальное маскируем).
    """
    H, W = Z_true.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))  # X — столбцы, Y — строки

    # Маскируем невалидные области
    Z_true_m = np.ma.array(Z_true, mask=~valid_mask)
    Z_hat_m = np.ma.array(Z_hat, mask=~valid_mask)

    # Границы валидной области для рамки осей
    yy, xx = np.where(valid_mask)
    xmin, xmax = int(xx.min()), int(xx.max())
    ymin, ymax = int(yy.min()), int(yy.max())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Оригинальная поверхность — полупрозрачная "плоскость" функции
    ax.plot_surface(X, Y, Z_true_m, color="gray", alpha=alpha_true,
                    linewidth=0, antialiased=False, shade=True)

    # Аппроксимация — поверх, почти непрозрачная
    ax.plot_surface(X, Y, Z_hat_m, alpha=alpha_hat,
                    linewidth=0, antialiased=False, shade=True)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Наложение: аппроксимация поверх полупрозрачной функции")

    # Маркер интересующей точки (если она в валидной зоне)
    if point is not None:
        px, py = point
        if 0 <= py < H and 0 <= px < W and valid_mask[int(py), int(px)]:
            ax.scatter(px, py, Z_true[int(py), int(px)], s=50)

    # Легенда через прокси-объекты
    legend_elems = [
        Patch(facecolor="gray", alpha=alpha_true, label="Оригинал (functions.wave)"),
        Patch(facecolor="C0", alpha=alpha_hat, label="Аппроксимация"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    plt.tight_layout()
    plt.show()


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Наложение аппроксимации на полупрозрачную поверхность исходной функции. Рисуем только валидную область."
    )
    p.add_argument("--i", type=int, default=4,
                   help="Номер набора: используем coefs_process/basis_{i}.json и data/basis_{i}")
    p.add_argument("--folder", default="coefs_process",
                   help="Папка с JSON-файлами коэффициентов (по умолчанию coefs_process)")
    p.add_argument("--basis-root", default="data",
                   help="Корень с папками basis_{i} (по умолчанию data)")
    p.add_argument("--functions", default="data/functions.wave",
                   help="Путь к исходной поверхности (по умолчанию data/functions.wave)")
    p.add_argument("--point", nargs=2, type=float, default=[100, 547],
                   help="Координаты (x y) для набора коэффициентов. По умолчанию 100 547")
    return p.parse_args()


def main():
    args = parse_args()

    # Формируем пути из --i
    coefs_json = os.path.join(args.folder, f"basis_{args.i}.json")
    basis_dir = os.path.join(args.basis_root, f"basis_{args.i}")

    # Печатаем используемые пути
    print("Используемые пути:")
    print(f"  --coefs-json : {coefs_json}")
    print(f"  --basis-dir  : {basis_dir}")
    print(f"  --functions  : {args.functions}")

    # Загрузка
    xs, ys, coefs = load_basis_coefs(coefs_json)    # xs:(M,), ys:(M,), coefs: list k x (M,)
    bases = load_basis_dir(basis_dir)               # (k, H, W)
    to_restore = np.loadtxt(args.functions)         # (H, W)

    if to_restore.ndim != 2 or bases.ndim != 3:
        raise ValueError("Ожидаются формы: to_restore -> (H, W), bases -> (k, H, W)")

    k, H_b, W_b = bases.shape
    H_f, W_f = to_restore.shape
    if (H_b, W_b) != (H_f, W_f):
        raise AssertionError(f"Размерности не совпадают: bases: {(H_b, W_b)}, functions: {(H_f, W_f)}")

    if len(coefs) != k:
        print(f"[!] Предупреждение: число коэфов в JSON ({len(coefs)}) != числу базисов ({k}). "
              f"Будет использовано min(k_json, k).")
        k_use = min(len(coefs), k)
        bases = bases[:k_use]
        coefs = coefs[:k_use]
        k = k_use

    # Маска валидной области — как в calc_mse
    valid_mask = valid_mask_from_bases(bases)
    if not np.any(valid_mask):
        raise RuntimeError("Валидная область пуста: во всех базисах только NaN.")

    # Выбор коэффициентов для точки
    x_sel, y_sel = args.point
    c, used_idx = get_coeffs_for_point(x_sel, y_sel, xs, ys, coefs)  # c:(k,)
    if c.shape[0] != k:
        raise AssertionError(f"Длина вектора коэфов ({c.shape[0]}) должна равняться числу базисов ({k}).")

    # Реконструкция
    Z_hat = reconstruct_from_bases(c, bases)

    # MSE только по валидной области
    mse_val = mse_on_valid_region(to_restore, Z_hat, valid_mask)
    print(f"MSE (валидная область) для точки индекса {used_idx} "
          f"(коорд ~({xs[used_idx]}, {ys[used_idx]})): {mse_val:.6g}")

    # Наложение поверхностей
    plot_overlayed_surfaces_3d(to_restore, Z_hat, valid_mask, point=(int(x_sel), int(y_sel)))


if __name__ == "__main__":
    main()
