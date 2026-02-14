import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


FILES_PATTERN = r"output/rmse_mean/rmse_mean__i_*__func_functions_pow1__recon_sigma_5_0.npy"
RX = re.compile(
    r"rmse_mean__i_(?P<i>\d+)__func_(?P<func>.+)__recon_sigma_(?P<sigma>[^.]+)\.npy$"
)

X0 = 2225
Y0 = 1250
K = 100


def mean_of_k_nearest(arr: np.ndarray, x0: int, y0: int, k: int) -> tuple[float, int, int]:
    h, w = arr.shape
    if not (0 <= x0 < w and 0 <= y0 < h):
        raise ValueError(f"Point (x={x0}, y={y0}) is outside array shape {arr.shape}")

    max_r = max(x0, y0, w - 1 - x0, h - 1 - y0)
    r = 1
    while True:
        x_min = max(0, x0 - r)
        x_max = min(w - 1, x0 + r)
        y_min = max(0, y0 - r)
        y_max = min(h - 1, y0 + r)

        sub = arr[y_min : y_max + 1, x_min : x_max + 1]
        valid = np.isfinite(sub)
        count = int(valid.sum())
        full = x_min == 0 and x_max == w - 1 and y_min == 0 and y_max == h - 1

        if count >= k or full:
            if count == 0:
                return float("nan"), 0, r
            ys, xs = np.nonzero(valid)
            ys = ys + y_min
            xs = xs + x_min
            vals = sub[valid]
            dist2 = (xs - x0) ** 2 + (ys - y0) ** 2
            if count > k:
                idx = np.argpartition(dist2, k - 1)[:k]
                vals = vals[idx]
            return float(np.mean(vals)), min(count, k), r

        if r >= max_r:
            return float("nan"), 0, r
        r = min(max_r, max(r * 2, r + 1))


def main() -> None:
    files = sorted(glob.glob(FILES_PATTERN))
    if not files:
        raise SystemExit(f"No files matched: {FILES_PATTERN}")

    rows = []
    for path in files:
        name = os.path.basename(path)
        m = RX.match(name)
        if not m:
            continue
        i_val = int(m.group("i"))
        arr = np.load(path)
        mean_val, used_count, radius = mean_of_k_nearest(arr, X0, Y0, K)
        rows.append((i_val, mean_val, used_count, radius, name))

    rows.sort(key=lambda r: r[0])
    if not rows:
        raise SystemExit("No valid rows to plot.")

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.title(f"Mean of {K} nearest points to ({X0}, {Y0})")
    plt.xlabel("i")
    plt.ylabel("mean rmse")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join("output", "rmse_mean", "plot_2225_1250.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Saved plot to: {out_path}")
    for i_val, mean_val, used_count, radius, name in rows:
        print(
            f"i={i_val:>3} mean={mean_val:.6f} used={used_count:>3} "
            f"radius={radius:>4} file={name}"
        )


if __name__ == "__main__":
    main()
