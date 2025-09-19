#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def launch_drag_and_drop_gui():
    """GUI to view .npy/.wave arrays (A, B) and their difference (B - A).

    - Buttons: open A, open B, show A, show B, show difference.
    - With tkinterdnd2 installed: drag-and-drop zones for files A and B.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
        has_dnd = True
    except Exception:
        # Fallback when tkinterdnd2 is not available
        import tkinter as tk  # noqa: F401 (tk symbol is already imported; keep for clarity)
        TkinterDnD = tk  # type: ignore
        DND_FILES = None  # type: ignore
        has_dnd = False

    # Window: use TkinterDnD.Tk() if available, otherwise fall back to tk.Tk()
    root = TkinterDnD.Tk()  # type: ignore
    root.title("Height Map (.npy/.wave): A/B and difference (B - A)")
    root.geometry("840x440")

    instr = tk.Label(
        root,
        text=(
            "Load TWO .npy/.wave files: A and B (difference = B - A).\n"
            "Use drag & drop (if available) or the buttons."
        ),
        font=("Segoe UI", 11),
        fg="#222",
        wraplength=800,
        justify="center",
    )
    instr.pack(pady=10)

    # Internal state
    state = {
        "A": {"arr": None, "path": None},
        "B": {"arr": None, "path": None},
    }

    # Labels for selected files
    files_frame = tk.Frame(root)
    files_frame.pack(pady=4, fill="x", padx=10)
    label_a_var = tk.StringVar(value="A: (not selected)")
    label_b_var = tk.StringVar(value="B: (not selected)")
    tk.Label(files_frame, textvariable=label_a_var, anchor="w").pack(fill="x")
    tk.Label(files_frame, textvariable=label_b_var, anchor="w").pack(fill="x")

    def _load_array(path: str) -> np.ndarray:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".npy":
            loaders = (np.load,)
        elif suffix == ".wave":
            loaders = (np.loadtxt,)
        else:
            loaders = (np.load, np.loadtxt)
        arr = None
        last_error = None
        for loader in loaders:
            try:
                arr = loader(file_path)
                break
            except Exception as exc:
                last_error = exc
        if arr is None:
            raise ValueError(f"�� ������� ��������� ������ {file_path.name}: {last_error}")
        arr = np.asarray(arr)
        arr = np.atleast_2d(arr)
        if arr.ndim != 2:
            raise ValueError(f"�������� 2D ������, �������� {arr.ndim}D")
        return arr

    def _plot_array(arr: np.ndarray, title: str):
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(arr, origin="upper", cmap="viridis")
        plt.colorbar(label="Value")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()

    def _set_file(slot: str, path: str):
        try:
            arr = _load_array(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load array:\n{e}")
            return
        state[slot]["arr"] = arr
        state[slot]["path"] = path
        name = Path(path).name
        if slot == "A":
            label_a_var.set(f"A: {name}  shape={arr.shape}")
        else:
            label_b_var.set(f"B: {name}  shape={arr.shape}")
        # Update button states
        btn_show_a.configure(state=("normal" if state["A"]["arr"] is not None else "disabled"))
        btn_show_b.configure(state=("normal" if state["B"]["arr"] is not None else "disabled"))
        both = state["A"]["arr"] is not None and state["B"]["arr"] is not None
        btn_show_diff.configure(state=("normal" if both else "disabled"))

    def _browse(slot: str):
        path = filedialog.askopenfilename(
            title=f"Select .npy/.wave for {slot}",
            filetypes=[
                ("Supported files", "*.npy *.wave"),
                ("NumPy array", "*.npy"),
                ("Wave text", "*.wave"),
                ("All files", "*.*"),
            ],
        )
        if path:
            _set_file(slot, path)

    # Action buttons
    btns = tk.Frame(root)
    btns.pack(pady=6)
    tk.Button(btns, text="Open A", command=lambda: _browse("A")).grid(row=0, column=0, padx=4)
    tk.Button(btns, text="Open B", command=lambda: _browse("B")).grid(row=0, column=1, padx=4)

    def _show_a():
        arr, path = state["A"]["arr"], state["A"]["path"]
        if arr is None:
            messagebox.showinfo("Info", "File A is not selected")
            return
        _plot_array(arr, f"A — {Path(path).name}")

    def _show_b():
        arr, path = state["B"]["arr"], state["B"]["path"]
        if arr is None:
            messagebox.showinfo("Info", "File B is not selected")
            return
        _plot_array(arr, f"B — {Path(path).name}")

    def _show_diff():
        arr_a, arr_b = state["A"]["arr"], state["B"]["arr"]
        path_a, path_b = state["A"]["path"], state["B"]["path"]
        if arr_a is None or arr_b is None:
            messagebox.showinfo("Info", "Select both files: A and B")
            return
        if arr_a.shape != arr_b.shape:
            messagebox.showerror("Error", f"Shapes do not match: A{arr_a.shape} vs B{arr_b.shape}")
            return
        diff = arr_b - arr_a
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(diff, origin="upper", cmap="coolwarm")
        plt.colorbar(label="Difference (B - A)")
        plt.title(f"Difference: B - A — {Path(path_b).name} - {Path(path_a).name}")
        plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout(); plt.show()

    btn_show_a = tk.Button(btns, text="Show A", command=_show_a, state="disabled")
    btn_show_a.grid(row=0, column=2, padx=4)
    btn_show_b = tk.Button(btns, text="Show B", command=_show_b, state="disabled")
    btn_show_b.grid(row=0, column=3, padx=4)
    btn_show_diff = tk.Button(btns, text="Show difference (B - A)", command=_show_diff, state="disabled")
    btn_show_diff.grid(row=0, column=4, padx=4)

    # Drag-and-drop zones
    if has_dnd and DND_FILES is not None:
        dnd = tk.Frame(root)
        dnd.pack(padx=16, pady=12, fill="x")

        drop_a = tk.Label(dnd, text="Drop .npy/.wave here (A)", relief="groove", borderwidth=2, width=35, height=6)
        drop_b = tk.Label(dnd, text="Drop .npy/.wave here (B)", relief="groove", borderwidth=2, width=35, height=6)
        drop_a.grid(row=0, column=0, padx=6)
        drop_b.grid(row=0, column=1, padx=6)

        def _handle_drop(slot: str):
            def _inner(event):  # type: ignore
                data = event.data
                if data.startswith("{") and data.endswith("}"):
                    data = data[1:-1]
                path = data.split()[0]
                _set_file(slot, path)
            return _inner

        for widget, slot in ((drop_a, "A"), (drop_b, "B")):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", _handle_drop(slot))

    root.mainloop()


def parse_args():
    p = argparse.ArgumentParser(description="GUI for .npy/.wave A/B and their difference (B - A)")
    return p.parse_args()


def main():
    parse_args()
    launch_drag_and_drop_gui()


if __name__ == "__main__":
    main()

