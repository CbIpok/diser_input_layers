#!/usr/bin/env python
"""Visualize .npy arrays with NaN interpolation and optional drag & drop GUI."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def fill_nan_with_neighbors(arr: np.ndarray, radius: int) -> Tuple[np.ndarray, int]:
    """Fill NaNs using the mean of finite neighbours within the given radius."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    if radius < 1:
        raise ValueError("Radius must be >= 1")

    data = arr.astype(float, copy=True)
    nan_mask = np.isnan(data)
    if not nan_mask.any():
        return data, 0

    height, width = data.shape
    sum_values = np.zeros_like(data, dtype=np.float64)
    counts = np.zeros_like(data, dtype=np.int32)
    radius_sq = radius * radius

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy > radius_sq:
                continue

            if dy >= 0:
                y_src_start = 0
                y_src_end = height - dy
                y_dst_start = dy
                y_dst_end = height
            else:
                y_src_start = -dy
                y_src_end = height
                y_dst_start = 0
                y_dst_end = height + dy
            if y_src_start >= y_src_end:
                continue

            if dx >= 0:
                x_src_start = 0
                x_src_end = width - dx
                x_dst_start = dx
                x_dst_end = width
            else:
                x_src_start = -dx
                x_src_end = width
                x_dst_start = 0
                x_dst_end = width + dx
            if x_src_start >= x_src_end:
                continue

            src = data[y_src_start:y_src_end, x_src_start:x_src_end]
            valid = np.isfinite(src)
            if not valid.any():
                continue

            dst_sum = sum_values[y_dst_start:y_dst_end, x_dst_start:x_dst_end]
            dst_cnt = counts[y_dst_start:y_dst_end, x_dst_start:x_dst_end]
            dst_sum[valid] += src[valid]
            dst_cnt[valid] += 1

    filled = data.copy()
    fill_mask = nan_mask & (counts > 0)
    filled[fill_mask] = sum_values[fill_mask] / counts[fill_mask]
    return filled, int(fill_mask.sum())


def finite_limits(arrays: Iterable[np.ndarray]) -> Tuple[float, float]:
    values = []
    for arr in arrays:
        masked = np.ma.masked_invalid(arr)
        compressed = masked.compressed()
        if compressed.size:
            values.append(compressed)
    if not values:
        return 0.0, 1.0
    stacked = np.concatenate(values)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    if vmin == vmax:
        vmax = vmin + 1.0 or 1.0
    return vmin, vmax


def plot_arrays(original: np.ndarray, filled: np.ndarray, title: str, dpi: int) -> None:
    masked_original = np.ma.masked_invalid(original)
    masked_filled = np.ma.masked_invalid(filled)
    vmin, vmax = finite_limits((masked_original, masked_filled))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi, sharex=True, sharey=True)
    for ax, data, label in zip(axes, (masked_original, masked_filled), ("Original", "Interpolated")):
        im = ax.imshow(data, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def load_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        size = int(np.sqrt(arr.size))
        if size * size == arr.size:
            arr = arr.reshape(size, size)
        else:
            raise ValueError(f"Cannot reshape 1D array of length {arr.size} into square grid")
    elif arr.ndim > 2:
        raise ValueError(f"Only 2D arrays are supported, got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def launch_gui(default_radius: int, dpi: int, save_default: bool) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
        has_dnd = True
    except Exception:
        TkinterDnD = tk  # type: ignore
        DND_FILES = None  # type: ignore
        has_dnd = False

    root = TkinterDnD.Tk()  # type: ignore
    root.title("Plot .npy with NaN interpolation")
    root.geometry("520x320")

    radius_var = tk.IntVar(value=default_radius)
    save_var = tk.BooleanVar(value=save_default)
    path_var = tk.StringVar(value="(file not selected)")

    tk.Label(root, text="Load .npy file (drag & drop or button)", font=("Segoe UI", 11)).pack(pady=10)

    info_frame = tk.Frame(root)
    info_frame.pack(pady=4)
    tk.Label(info_frame, text="Radius (pixels):").grid(row=0, column=0, padx=4, sticky="e")
    radius_entry = tk.Spinbox(info_frame, from_=1, to=64, textvariable=radius_var, width=5)
    radius_entry.grid(row=0, column=1, padx=4)
    tk.Checkbutton(info_frame, text="Save filled *.npy", variable=save_var).grid(row=1, column=0, columnspan=2, pady=6)

    tk.Label(root, textvariable=path_var, fg="#444").pack(pady=4)

    def handle_file(path_str: str) -> None:
        file_path = Path(path_str.strip())
        if not file_path.exists():
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        try:
            radius = int(radius_var.get())
        except Exception:
            messagebox.showerror("Error", "Radius must be an integer")
            return
        if radius < 1:
            messagebox.showerror("Error", "Radius must be >= 1")
            return
        try:
            arr = load_array(file_path)
            filled, replaced = fill_nan_with_neighbors(arr, radius)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        path_var.set(f"{file_path.name} (filled pixels: {replaced})")
        title = f"{file_path.name} (filled pixels: {replaced})"
        plot_arrays(arr, filled, title, dpi)

        if save_var.get():
            out_path = file_path.with_name(f"{file_path.stem}__filled.npy")
            np.save(out_path, filled)
            messagebox.showinfo("Saved", f"Filled array saved to: {out_path}")

    def browse() -> None:
        path = filedialog.askopenfilename(title="Select .npy", filetypes=[("NumPy", "*.npy"), ("All files", "*.*")])
        if path:
            handle_file(path)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=6)
    tk.Button(btn_frame, text="Open file", command=browse).grid(row=0, column=0, padx=6)

    if has_dnd and DND_FILES is not None:
        drop = tk.Label(root, text="Drop .npy here", relief="groove", borderwidth=2, width=38, height=6)
        drop.pack(pady=12)

        def _handle_drop(event):  # type: ignore
            data = event.data
            if data.startswith("{") and data.endswith("}"):
                data = data[1:-1]
            path = data.split()[0]
            handle_file(path)

        drop.drop_target_register(DND_FILES)
        drop.dnd_bind("<<Drop>>", _handle_drop)
    else:
        tk.Label(root, text="Drag & drop not available (tkinterdnd2 missing)").pack(pady=12)

    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot .npy arrays with NaNs interpolated from nearby pixels.")
    parser.add_argument("paths", nargs="*", type=Path, help="Input .npy files")
    parser.add_argument("--radius", type=int, default=4, help="Neighbourhood radius in pixels")
    parser.add_argument("--dpi", type=int, default=96, help="Figure DPI")
    parser.add_argument("--save-filled", action="store_true", help="Save filled arrays next to the originals")
    parser.add_argument("--gui", action="store_true", help="Launch drag & drop GUI")
    return parser.parse_args()


def run_cli(paths: Iterable[Path], radius: int, dpi: int, save_filled: bool) -> None:
    if radius < 1:
        raise ValueError("Radius must be >= 1")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        arr = load_array(path)
        filled, replaced = fill_nan_with_neighbors(arr, radius)
        title = f"{path.name} (filled pixels: {replaced})"
        plot_arrays(arr, filled, title, dpi)
        if save_filled:
            out_path = path.with_name(f"{path.stem}__filled.npy")
            np.save(out_path, filled)
            print(f"Saved filled array to {out_path}")


def main() -> None:
    args = parse_args()

    launched = False
    if args.gui:
        launch_gui(args.radius, args.dpi, args.save_filled)
        launched = True

    if args.paths:
        run_cli(args.paths, args.radius, args.dpi, args.save_filled)
        launched = True

    if not launched:
        # No paths and GUI flag not provided: fall back to GUI for convenience.
        launch_gui(args.radius, args.dpi, args.save_filled)


if __name__ == "__main__":
    main()