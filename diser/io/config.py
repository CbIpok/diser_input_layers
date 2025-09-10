from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from diser.coords import Size


@dataclass(frozen=True)
class Config:
    size: Size
    bath_path: str
    areas_dir: str | None = None


def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    sx = int(data["size"]["x"])  # width
    sy = int(data["size"]["y"])  # height
    return Config(size=Size(sx, sy), bath_path=data["bath_path"],
                  areas_dir=data.get("areas_dir"))

