import os
import json
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.path import Path


def apply_affine_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transform to 2D points expressed in local coordinates.

    The matrix must be 3x3 (homogeneous coordinates)."""
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (matrix @ pts_h.T).T
    return transformed[:, :2]


class Area:
    """Base class for geometric areas with optional affine transforms."""
    def __init__(self, name: str, points: np.ndarray, config: dict):
        self.name = name
        self.config = config
        self.raw_points = points  # Original points before applying any transform
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
    """Polygonal area rendered as a filled patch."""
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
    """Area that places a raised cosine wave inside the polygon."""
    def __init__(self, name: str, points: np.ndarray, config: dict):
        super().__init__(name, points, config)
        self.eta0 = float(config.get('eta0', 1.0))
        self.power = float(config.get('power', 1.0))

    def compute_wave(self, mask: np.ndarray) -> np.ndarray:
        wave = np.full(mask.shape, np.nan)
        yy, xx = np.nonzero(mask)
        if yy.size == 0:
            return wave
        cy, cx = self.points[:, 1].mean(), self.points[:, 0].mean()
        dy = yy - cy
        dx = xx - cx
        width = self.points[:, 0].max() - self.points[:, 0].min()
        height = self.points[:, 1].max() - self.points[:, 1].min()
        r1 = max(width / 3, 1e-9)
        r2 = max(height / 3, 1e-9)
        rad = np.sqrt((dx / r1)**2 + (dy / r2)**2)
        base = 0.5 * (1 + np.cos(np.pi * np.clip(rad, 0.0, 1.0)))
        valid = rad <= 1.0
        values = base[valid]
        if self.power != 1.0:
            values = np.power(values, self.power)
        wave[yy[valid], xx[valid]] = self.eta0 * values
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
    """Load area definitions from JSON files."""
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
