from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

BASE_DIR = Path("output/rmse_mean")
OUTPUT_PATH = Path("output/rmse_mean_stats.json")
CENTER_X = 2000
CENTER_Y = 1400
WINDOW_SIDE = 250
HALF_SIDE = WINDOW_SIDE // 2

def parse_pow(raw: str) -> float:
    return float(raw.replace("_", "."))

def sigma_sort_key(raw: str) -> float:
    if raw == "none":
        return math.inf
    return float(raw.replace("_", "."))

def gather_stats(array: np.ndarray) -> tuple[float | None, float | None, float | None]:
    flat = np.asarray(array).ravel()
    if flat.size == 0:
        return (None, None, None)
    mask = ~np.isnan(flat)
    if not mask.any():
        return (None, None, None)
    valid = flat[mask]
    mean_value = float(valid.mean())
    min_value = float(valid.min())
    max_value = float(valid.max())
    return (mean_value, min_value, max_value)

def region_mean(array: np.ndarray) -> float | None:
    region = compute_region(array)
    if region.size == 0:
        return None
    flat = np.asarray(region).ravel()
    if flat.size == 0:
        return None
    mask = ~np.isnan(flat)
    if not mask.any():
        return None
    return float(flat[mask].mean())

def compute_region(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    if not (0 <= CENTER_X < width and 0 <= CENTER_Y < height):
        return array[0:0, 0:0]
    x0 = max(0, CENTER_X - HALF_SIDE)
    x1 = min(width, CENTER_X + HALF_SIDE)
    y0 = max(0, CENTER_Y - HALF_SIDE)
    y1 = min(height, CENTER_Y + HALF_SIDE)
    return array[y0:y1, x0:x1]

def iter_targets() -> Iterable[Path]:
    pattern = "rmse_mean__i_*__func_functions_pow*__recon_sigma_*.npy"
    for path in sorted(BASE_DIR.glob(pattern)):
        if "__recon_sigma_11_0" in path.name:
            continue
        yield path

def main() -> None:
    records = []
    for path in iter_targets():
        name = path.name
        parts = name.split("__")
        i_value = parts[1].split("_")[1]
        pow_value = parts[2].split("pow")[1]
        sigma_value = parts[3].split("sigma_")[1].removesuffix(".npy")

        array = np.load(path, allow_pickle=False)
        mean_val, min_val, max_val = gather_stats(array)
        region_val = region_mean(array)

        records.append(
            {
                "file": str(path.as_posix()),
                "i": int(i_value),
                "pow": parse_pow(pow_value),
                "pow_raw": pow_value,
                "sigma": sigma_value,
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
                "mean_region_center_2000_1400_side_250": region_val,
            }
        )

    records.sort(
        key=lambda r: (
            r["i"],
            r["pow"],
            sigma_sort_key(r["sigma"]),
            r["sigma"],
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"stats": records}, fh, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
