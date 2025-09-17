#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def launch_drag_and_drop_gui():
    """GUI для просмотра .npy (A, B) и их разницы (B - A).

    - Кнопки: открыть A, открыть B, показать A, показать B, показать разницу.
    - Если установлен tkinterdnd2: две зоны для перетаскивания файлов A и B.
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
        # Fallback без tkinterdnd2
        import tkinter as tk  # noqa: F401  (символ tk уже есть, оставляем для ясности)
        TkinterDnD = tk  # type: ignore
        DND_FILES = None  # type: ignore
        has_dnd = False

    # Окно: с tkinterdnd2: TkinterDnD.Tk(), иначе обычный tk.Tk()
    root = TkinterDnD.Tk()  # type: ignore
    root.title("Карта высот (.npy): A/B и разница (B - A)")
    root.geometry("840x440")

    instr = tk.Label(
        root,
        text=(
            "Загрузите ДВА файла .npy: A и B (разница = B - A).\n"
            "Используйте перетаскивание (если доступно) или кнопки."
        ),
        font=("Segoe UI", 11),
        fg="#222",
        wraplength=800,
        justify="center",
    )
    instr.pack(pady=10)

    # Состояние
    state = {
        "A": {"arr": None, "path": None},
        "B": {"arr": None, "path": None},
    }

    # Подписи выбранных файлов
    files_frame = tk.Frame(root)
    files_frame.pack(pady=4, fill="x", padx=10)
    label_a_var = tk.StringVar(value="A: (не выбран)")
    label_b_var = tk.StringVar(value="B: (не выбран)")
    tk.Label(files_frame, textvariable=label_a_var, anchor="w").pack(fill="x")
    tk.Label(files_frame, textvariable=label_b_var, anchor="w").pack(fill="x")

    def _load_array(path: str) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Ожидался 2D массив, получено {arr.ndim}D")
        return arr

    def _plot_array(arr: np.ndarray, title: str):
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(arr, origin="upper", cmap="viridis")
        plt.colorbar(label="Высота")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()

    def _set_file(slot: str, path: str):
        try:
            arr = _load_array(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить массив:\n{e}")
            return
        state[slot]["arr"] = arr
        state[slot]["path"] = path
        name = Path(path).name
        if slot == "A":
            label_a_var.set(f"A: {name}  shape={arr.shape}")
        else:
            label_b_var.set(f"B: {name}  shape={arr.shape}")
        # Обновить доступность кнопок
        btn_show_a.configure(state=("normal" if state["A"]["arr"] is not None else "disabled"))
        btn_show_b.configure(state=("normal" if state["B"]["arr"] is not None else "disabled"))
        both = state["A"]["arr"] is not None and state["B"]["arr"] is not None
        btn_show_diff.configure(state=("normal" if both else "disabled"))

    def _browse(slot: str):
        path = filedialog.askopenfilename(
            title=f"Выберите .npy для {slot}",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
        )
        if path:
            _set_file(slot, path)

    # Кнопки действий
    btns = tk.Frame(root)
    btns.pack(pady=6)
    tk.Button(btns, text="Открыть A", command=lambda: _browse("A")).grid(row=0, column=0, padx=4)
    tk.Button(btns, text="Открыть B", command=lambda: _browse("B")).grid(row=0, column=1, padx=4)

    def _show_a():
        arr, path = state["A"]["arr"], state["A"]["path"]
        if arr is None:
            messagebox.showinfo("Информация", "Файл A не выбран")
            return
        _plot_array(arr, f"A — {Path(path).name}")

    def _show_b():
        arr, path = state["B"]["arr"], state["B"]["path"]
        if arr is None:
            messagebox.showinfo("Информация", "Файл B не выбран")
            return
        _plot_array(arr, f"B — {Path(path).name}")

    def _show_diff():
        arr_a, arr_b = state["A"]["arr"], state["B"]["arr"]
        path_a, path_b = state["A"]["path"], state["B"]["path"]
        if arr_a is None or arr_b is None:
            messagebox.showinfo("Информация", "Выберите оба файла: A и B")
            return
        if arr_a.shape != arr_b.shape:
            messagebox.showerror("Ошибка", f"Размеры не совпадают: A{arr_a.shape} vs B{arr_b.shape}")
            return
        diff = arr_b - arr_a
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(diff, origin="upper", cmap="coolwarm")
        plt.colorbar(label="Разница (B - A)")
        plt.title(f"Разница: B - A — {Path(path_b).name} − {Path(path_a).name}")
        plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout(); plt.show()

    btn_show_a = tk.Button(btns, text="Показать A", command=_show_a, state="disabled")
    btn_show_a.grid(row=0, column=2, padx=4)
    btn_show_b = tk.Button(btns, text="Показать B", command=_show_b, state="disabled")
    btn_show_b.grid(row=0, column=3, padx=4)
    btn_show_diff = tk.Button(btns, text="Показать разницу (B - A)", command=_show_diff, state="disabled")
    btn_show_diff.grid(row=0, column=4, padx=4)

    # DnD зоны
    if has_dnd and DND_FILES is not None:
        dnd = tk.Frame(root)
        dnd.pack(padx=16, pady=12, fill="x")

        drop_a = tk.Label(dnd, text="Перетащите .npy сюда (A)", relief="groove", borderwidth=2, width=35, height=6)
        drop_b = tk.Label(dnd, text="Перетащите .npy сюда (B)", relief="groove", borderwidth=2, width=35, height=6)
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
    p = argparse.ArgumentParser(description="GUI для .npy A/B и их разницы (B - A)")
    return p.parse_args()


def main():
    parse_args()
    launch_drag_and_drop_gui()


if __name__ == "__main__":
    main()

