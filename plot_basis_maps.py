"""Plot 2D height maps for approximation error and coefficients.

This script reads files of the form ``coefs_process/basis_{i}.json``
where each entry has a key ``"[x,y]"`` describing coordinates and a
value with ``"aprox_error"`` and a list of ``i`` coefficients.
The coordinates may form a non-rectangular domain, so plots are built
using triangulation.

Example usage for ``i=4``:

    python plot_basis_maps.py --i 4 --save-dir output

This will create separate PNG files for the approximation error and
each coefficient in the ``output`` directory.
"""
import argparse
import ast
import json
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from tqdm import tqdm

from loader import load_bath_grid


def _to_float(value) -> float:
    """Return float(value) or NaN if conversion fails or value is None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def plot_aprox_error_raw(xs: np.ndarray,
                         ys: np.ndarray,
                         approx_err: np.ndarray,
                         out_dir: str | None = None):
    """
    Кладёт значения 'aprox_error' строго в их ячейки (xs, ys) без триангуляции
    и показывает/сохраняет картинку. Остальные пиксели — NaN.
    """
    # Берём форму сетки, чтобы знать финальный размер
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])
    H, W = grid.shape

    # Готовим "полотно" и индексы
    raw = np.full((H, W), np.nan, dtype=float)
    xi = np.asarray(xs, dtype=int)
    yi = np.asarray(ys, dtype=int)

    # Защитная маска: только валидные и попадающие в границы точки
    mask = (
        (xi >= 0) & (xi < H) &
        (yi >= 0) & (yi < W) &
        np.isfinite(approx_err)
    )

    raw[xi[mask], yi[mask]] = approx_err[mask]

    # Рисуем "как есть" (как и в calc_mse — транспонируем и origin='lower')
    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        raw.T,
        origin='lower',
        interpolation='nearest',
        cmap='viridis'
    )
    plt.colorbar(im, label='aprox_error (raw)')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title('aprox_error (raw)')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'aprox_error_raw.png')
        plt.savefig(path)
        plt.close()
        return [path]
    else:
        plt.show()
        return []

def load_basis_coofs(path: str):
    """Load coordinate and coefficient data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    xs: List[float] = []
    ys: List[float] = []
    approx_errors: List[float] = []
    coefs: List[List[float]] = []

    for key, val in data.items():
        x, y = ast.literal_eval(key)
        xs.append(float(x))
        ys.append(float(y))
        approx_errors.append(_to_float(val.get("aprox_error")))
        coef_vals = [_to_float(c) for c in val.get("coefs", [])]
        if not coefs:
            coefs = [[] for _ in coef_vals]
        for idx, c in enumerate(coef_vals):
            coefs[idx].append(c)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    approx_arr = np.asarray(approx_errors, dtype=float)
    coef_arrays = [np.asarray(c, dtype=float) for c in coefs]
    return xs_arr, ys_arr, approx_arr, coef_arrays


def load_basis(path: str):
    arrays = []
    indices = []

    # Проверяем, что путь существует
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist")

    # Перебираем файлы в директории
    for filename in os.listdir(path):
        if filename.startswith("basis_") and filename.endswith(".wave"):
            # Извлекаем числовую часть между 'basis_' и '.wave'
            try:
                index = int(filename.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                continue  # Пропускаем файлы с неправильным форматом

            full_path = os.path.join(path, filename)
            # Загружаем данные из файла
            data = np.loadtxt(full_path)
            # Заменяем 0 на NaN
            data[data == 0] = np.nan
            arrays.append(data)
            indices.append(index)

    # Сортируем массивы по индексам
    sorted_arrays = [array for _, array in sorted(zip(indices, arrays))]
    return sorted_arrays

def calc_mse(to_restore, bases, xs_arr, ys_arr, coef_arrays, block=512, dtype=np.float32):
    """
        Быстрая версия:
        - игнорирует область, где все базы = NaN
        - предвычисляет грам-матрицу и линейный член
        - векторно считает MSE для всех (x,y)

        to_restore: (H, W), без NaN
        bases:      (k, H, W), возможны NaN
        xs_arr, ys_arr: индексы длиной M (куда класть MSE)
        coef_arrays: список длиной k с массивами длиной M, либо массив (k, M) / (M, k)
        """
    # Оставил загрузку как в оригинале (можно вынести наружу ради ещё +скорости)
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])

    k, Hb, Wb = bases.shape
    Hg, Wg = grid.shape
    assert (Hb, Wb) == (Hg, Wg), "bases и grid должны совпадать по форме"
    assert to_restore.shape == (Hg, Wg), "to_restore должен совпадать по форме с grid"

    # Приведём индексы и проверим размеры
    xs = np.asarray(xs_arr, dtype=int)
    ys = np.asarray(ys_arr, dtype=int)
    assert xs.shape == ys.shape, "xs_arr и ys_arr должны быть одинаковой длины"
    M = xs.size

    # Маска полезной области: где хотя бы одна база не NaN
    valid_mask = np.any(np.isfinite(bases), axis=0)  # (H, W)
    idx = np.flatnonzero(valid_mask)
    # Если полезных пикселей нет — вернём NaN-поле
    if idx.size == 0:
        out = np.full_like(grid, np.nan, dtype=float)
        return out

    # Сжимаем в матрицу B (k, N) и вектор t (N,)
    B = bases.reshape(k, -1)[:, idx]
    # NaN в базах трактуем как 0 (вклад отсутствует)
    B = np.nan_to_num(B, nan=0.0, copy=False)
    t = to_restore.reshape(-1)[idx].astype(np.float64, copy=False)

    # Предвычисляем компоненты квадратичной формы
    # G = B B^T (k×k), h = B t (k,), tt = t·t (скаляр), N = |валид|
    G = B @ B.T
    h = B @ t
    tt = float(t @ t)
    N = t.size

    # Нормализуем форму коэффициентов к (M, k)
    C = np.asarray(coef_arrays, dtype=np.float64)
    if C.ndim == 1:
        C = C[None, :]  # (1, k)
    if C.shape[0] == k:
        C = C.T  # (M, k)

    if C.shape != (M, k):
        raise ValueError(f"Ожидалось coef_arrays формы (M={M}, k={k}), а получено {C.shape}")

    # Векторно считаем MSE для всех M точек:
    # mse(c) = (c^T G c - 2 c^T h + tt) / N
    term1 = np.einsum('ik,kl,il->i', C, G, C, optimize=True)  # diag(C G C^T)
    term2 = 2.0 * (C @ h)
    mse_vals = (term1 - term2 + tt) / N

    # Выкладываем результаты в нужные координаты, остальное — NaN
    out = np.full_like(grid, np.nan, dtype=np.float64)
    out[xs, ys] = np.sqrt(mse_vals)
    mse = out
    plt.figure(figsize=(10, 8))
    # Используем псевдоцветное изображение с учетом NaN значений
    im = plt.imshow(mse.T,  # Транспонируем для правильной ориентации осей
                    origin='lower',  # Начало координат снизу
                    cmap='viridis',  # Цветовая схема
                    interpolation='nearest')
    plt.colorbar(im, label='MSE')  # Добавляем цветовую шкалу
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title('MSE Distribution')
    plt.show()

    return mse  # Возвращаем вычисленные значения MSE

def plot_maps(xs: np.ndarray, ys: np.ndarray, approx_err: np.ndarray,
              coefs: List[np.ndarray], out_dir: str | None = None):
    """Generate height maps for approximation error and coefficients."""
    variables = [("approx_error", approx_err)]
    variables += [(f"coef_{i+1}", arr) for i, arr in enumerate(coefs)]

    output_files = []
    for title, arr in variables:
        mask = np.isfinite(arr)
        if mask.sum() < 3:
            # Not enough points to build triangulation
            continue
        triang = tri.Triangulation(xs[mask], ys[mask])
        fig, ax = plt.subplots()
        tpc = ax.tricontourf(triang, arr[mask], levels=100)
        ax.set_title(title.replace("_", " "))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        fig.colorbar(tpc, ax=ax)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{title}.png"
            path = os.path.join(out_dir, filename)
            fig.savefig(path)
            output_files.append(path)
            plt.close(fig)
        else:
            plt.show()
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D height maps from basis_i.json files"
    )
    parser.add_argument(
        "--folder", default="coefs_process",
        help="Directory containing basis_{i}.json"
    )
    parser.add_argument(
        "--i", type=int, default=4,
        help="Number of coefficients (file basis_{i}.json)"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save PNG plots. If omitted, plots are shown"
    )
    args = parser.parse_args()

    file_path = os.path.join(args.folder, f"basis_{args.i}.json")
    xs, ys, approx_err, coefs = load_basis_coofs(file_path)

    basis = np.array(load_basis(f"data/basis_{args.i}"))
    to_restore = np.loadtxt("data/functions.wave")

    # Считаем и показываем RMSE-карту
    calc_mse(to_restore, basis, xs, ys, coefs)

    # Показываем "сырые" значения aprox_error — как есть, без триангуляции
    plot_aprox_error_raw(xs, ys, approx_err, args.save_dir)

    # Плюс прежние карты (в т.ч. сглаженная карта approx_error через триангуляцию)
    plot_maps(xs, ys, approx_err, coefs, args.save_dir)


if __name__ == "__main__":
    main()
