import os
import json
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.path import Path


def apply_affine_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Применяет аффинное преобразование к точкам (гомогенные координаты).
    matrix: 3x3
    """
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (matrix @ pts_h.T).T
    return transformed[:, :2]


class Area:
    """
    Базовый класс зоны с собственной системой координат и визуализацией.
    """
    def __init__(self, name: str, points: np.ndarray, config: dict):
        self.name = name
        self.config = config
        self.raw_points = points  # локальные координаты
        self.transform_cfg = config.get('transform', None)
        self.points = self._to_global(points)

    def _to_global(self, pts: np.ndarray) -> np.ndarray:
        if self.transform_cfg and self.transform_cfg.get('type') == 'affine':
            matrix = np.array(self.transform_cfg['matrix'], dtype=float)
            return apply_affine_transform(pts, matrix)
        return pts

    def compute_mask(self, nx: int, ny: int) -> np.ndarray:
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        path = Path(self.points)
        mask = path.contains_points(np.vstack((X.ravel(), Y.ravel())).T)
        return mask.reshape((ny, nx))

    def plot(self, ax, **kwargs):
        raise NotImplementedError


class PolygonArea(Area):
    """Простая заливка полигона"""
    def plot(self, ax, color: str = 'red'):
        poly = Polygon(
            self.points,
            facecolor=color,
            edgecolor='black',
            alpha=0.4,
            linewidth=1.5,
            label=self.name
        )
        ax.add_patch(poly)


class CosineWaveArea(Area):
    """Зона с косинусной волной"""
    def __init__(self, name: str, points: np.ndarray, config: dict):
        super().__init__(name, points, config)
        self.eta0 = float(config.get('eta0', 1.0))

    def compute_wave(self, mask: np.ndarray) -> np.ndarray:
        yy, xx = np.nonzero(mask)
        cy, cx = self.points[:,1].mean(), self.points[:,0].mean()
        dy = yy - cy
        dx = xx - cx
        r1 = (self.points[:,0].max() - self.points[:,0].min()) / 3
        r2 = (self.points[:,1].max() - self.points[:,1].min()) / 3
        rad = np.sqrt((dx/r1)**2 + (dy/r2)**2)
        valid = rad <= 1.0
        wave = np.full(mask.shape, np.nan)
        wave[yy[valid], xx[valid]] = self.eta0/2 * (1 + np.cos(np.pi * rad[valid]))
        return wave

    def plot(self, ax, **kwargs):
        mask = self.compute_mask(kwargs['nx'], kwargs['ny'])
        wave = self.compute_wave(mask)
        im = ax.imshow(
            wave,
            origin='upper',
            cmap='viridis',
            norm=Normalize(vmin=0, vmax=np.nanmax(wave)),
            extent=(0, kwargs['nx'], kwargs['ny'], 0),
            alpha=1.0
        )
        return im


def load_areas(areas_dir: str) -> list:
    """Загружает и инициирует зоны из JSON-конфигов"""
    areas = []
    for fname in sorted(os.listdir(areas_dir)):
        if not fname.lower().endswith('.json'):
            continue
        path = os.path.join(areas_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        pts = np.array(cfg.get('points', []), dtype=float)
        if pts.size == 0:
            continue
        name = os.path.splitext(fname)[0]
        func = cfg.get('function')
        if func == 'cos':
            area = CosineWaveArea(name, pts, cfg)
        else:
            area = PolygonArea(name, pts, cfg)
        areas.append(area)
    return areas