from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure_bundle(fig, base_path: str | os.PathLike,
                       formats: Iterable[str] = ("png",),
                       with_pickle: bool = True) -> list[Path]:
    """Save a matplotlib figure in convenient bundle formats.

    - formats: iterable of extensions without dot (e.g., ["png", "svg"])
    - with_pickle: additionally save a .mplfig.pkl with the full figure object
    Returns list of saved file paths.
    """
    base = Path(base_path)
    _ensure_parent(base)
    saved = []
    for ext in formats:
        out = base.with_suffix('.' + ext)
        fig.savefig(out)
        saved.append(out)
    if with_pickle:
        pkl = base.with_suffix('.mplfig.pkl')
        with open(pkl, 'wb') as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved.append(pkl)
    return saved


def load_figure_pickle(path: str | os.PathLike):
    """Load a matplotlib figure from .mplfig.pkl and return it (does not show)."""
    with open(path, 'rb') as f:
        fig = pickle.load(f)
    return fig


def show_figure_pickle(path: str | os.PathLike):
    """Load a .mplfig.pkl figure and show it."""
    fig = load_figure_pickle(path)
    plt.show()
    return fig

