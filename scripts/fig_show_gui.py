#!/usr/bin/env python
from __future__ import annotations

"""
Small GUI to view saved Matplotlib figures (.mplfig.pkl).

Features
- Drag & drop .mplfig.pkl onto the window (if tkinterdnd2 is available).
- Or click "Open..." to choose one or multiple .mplfig.pkl files.
- Also accepts file paths as CLI args: `python -m scripts.fig_show_gui fig1.mplfig.pkl`.

Notes
- Uses diser.viz.figio.load_figure_pickle to restore figures.
- Shows each figure in a native Matplotlib window (non‑blocking).
- Drag&drop support is optional; without tkinterdnd2, the Open dialog still works.
"""

import argparse
import os
import sys
import threading
from pathlib import Path

import matplotlib.pyplot as plt

try:
    # Local helper for loading pickled figures
    from diser.viz.figio import load_figure_pickle
except Exception as e:  # pragma: no cover
    print("[ERROR] Could not import diser.viz.figio.load_figure_pickle:", e, file=sys.stderr)
    raise


def _show_figure_nonblocking(fig, title: str | None = None):
    try:
        if title and hasattr(fig.canvas, 'manager') and fig.canvas.manager:
            try:
                fig.canvas.manager.set_window_title(title)
            except Exception:
                pass
        # Non-blocking show
        plt.show(block=False)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Failed to show figure: {e}")


def _open_and_show(path: str | os.PathLike):
    try:
        fig = load_figure_pickle(path)
        _show_figure_nonblocking(fig, title=os.path.basename(str(path)))
    except Exception as e:
        print(f"[ERROR] Failed to open {path}: {e}")


def _handle_paths(paths: list[str]):
    # Open each path in a separate thread so GUI doesn't block on plt.show
    for p in paths:
        if not p:
            continue
        ext = Path(p).suffix.lower()
        if ext != '.pkl' and not str(p).endswith('.mplfig.pkl'):
            # Accept both .mplfig.pkl and any .pkl (best effort)
            pass
        t = threading.Thread(target=_open_and_show, args=(p,), daemon=True)
        t.start()


def _parse_dnd_file_list(data: str) -> list[str]:
    # tkinterdnd2 passes a string that may contain braces around paths with spaces
    # e.g., {C:\path with space\file.mplfig.pkl} C:\plain\file2.mplfig.pkl
    items: list[str] = []
    token = ''
    in_brace = False
    for ch in data:
        if ch == '{':
            in_brace = True
            token = ''
            continue
        if ch == '}':
            in_brace = False
            items.append(token)
            token = ''
            continue
        if ch.isspace() and not in_brace:
            if token:
                items.append(token)
                token = ''
            continue
        token += ch
    if token:
        items.append(token)
    return items


def build_gui(initial_paths: list[str] | None = None):
    # Try optional tkinterdnd2 for drag&drop
    use_dnd = False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        root = TkinterDnD.Tk()
        use_dnd = True
    except Exception:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        root = tk.Tk()
        DND_FILES = None  # type: ignore

    root.title('Figure Viewer (.mplfig.pkl)')
    root.geometry('520x220')

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill='both', expand=True)

    title = ttk.Label(frm, text='Drop .mplfig.pkl files here or use Open...')
    title.pack(anchor='w')

    status_var = tk.StringVar(value='Ready')

    drop_area = ttk.Label(frm, relief='ridge', padding=16,
                          text='Drag & drop here' if use_dnd else 'Drag & drop not available')
    drop_area.pack(fill='both', expand=True, pady=8)

    def on_open_clicked():
        paths = filedialog.askopenfilenames(
            title='Open .mplfig.pkl files',
            filetypes=[('Matplotlib Figure Pickle', '*.mplfig.pkl'), ('Pickle', '*.pkl'), ('All files', '*.*')]
        )
        if not paths:
            return
        status_var.set(f'Opening {len(paths)} file(s)...')
        _handle_paths(list(paths))

    btn_bar = ttk.Frame(frm)
    btn_bar.pack(fill='x')
    ttk.Button(btn_bar, text='Open…', command=on_open_clicked).pack(side='left')
    ttk.Button(btn_bar, text='Quit', command=root.destroy).pack(side='right')

    status = ttk.Label(frm, textvariable=status_var)
    status.pack(anchor='w', pady=(6, 0))

    if use_dnd:
        def on_drop(event):
            try:
                files = _parse_dnd_file_list(event.data)
                status_var.set(f'Dropped {len(files)} file(s)')
                _handle_paths(files)
            except Exception as e:  # pragma: no cover
                status_var.set(f'Error: {e}')
        drop_area.drop_target_register(DND_FILES)
        drop_area.dnd_bind('<<Drop>>', on_drop)

    # Open initial files passed via CLI
    if initial_paths:
        status_var.set(f'Opening {len(initial_paths)} file(s) from CLI...')
        _handle_paths(initial_paths)

    root.mainloop()


def parse_args():
    p = argparse.ArgumentParser(description='GUI viewer for .mplfig.pkl files (drag & drop or Open…)')
    p.add_argument('paths', nargs='*', help='Optional list of .mplfig.pkl files to open on start')
    return p.parse_args()


def main():
    args = parse_args()
    build_gui(args.paths)


if __name__ == '__main__':
    main()

