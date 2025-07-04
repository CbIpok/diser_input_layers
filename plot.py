import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.path import Path


def load_bath_grid(path):
    """
    Загружает MOST-файл: первые два числа - nx, ny;
    пропускает nx и ny координатных значений;
    возвращает массив shape=(ny, nx).
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    nx, ny = map(int, tokens[:2])
    idx = 2 + nx + ny
    data = np.array(tokens[idx:idx + nx * ny], dtype=float)
    return data.reshape((ny, nx))


def make_land_ocean_cmap(vmin, vmax, land_color='saddlebrown'):
    """
    Комбинированный colormap: ниже 0 — land_color, выше — Blues.
    """
    zero_frac = -vmin / (vmax - vmin)
    cb = plt.cm.Blues
    light, dark = cb(0.2), cb(1.0)
    cdict = [
        (0.0, land_color),
        (zero_frac, land_color),
        (zero_frac, light),
        (1.0, dark),
    ]
    cmap = LinearSegmentedColormap.from_list('land_ocean', cdict)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def compute_cos_wave(mask, center, r1, r2, eta0=1.0):
    """
    Вычисляет высоту волны по формуле:
      η = η0/2 * (1 + cos(pi * sqrt((dx/r1)^2 + (dy/r2)^2)))
    Применяется только для точек внутри эллипса (mask).
    """
    yy, xx = np.nonzero(mask)
    dy = yy - center[1]
    dx = xx - center[0]
    rad = np.sqrt((dx/r1)**2 + (dy/r2)**2)
    valid = rad <= 1.0
    wave = np.full_like(mask, np.nan, dtype=float)
    wave[yy[valid], xx[valid]] = eta0/2 * (1 + np.cos(np.pi * rad[valid]))
    return wave


def plot_with_polygons(grid, areas_dir='data/areas'):
    """
    Рисует базовый рельеф и обрабатывает полигоны:
      - polygon с function == 'cos': непрозрачная карта волны + отдельный colobar
      - остальные: заполненные полигоны
    """
    ny, nx = grid.shape
    fig, ax = plt.subplots(figsize=(10, 7))

    # Colormap land/ocean для рельефа
    vmin, vmax = grid.min(), grid.max()
    cmap1, norm1 = make_land_ocean_cmap(vmin, vmax)
    im = ax.imshow(grid, origin='upper', cmap=cmap1, norm=norm1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Bathymetry')

    base_colors = ['red', 'green', 'blue']
    normal_idx = 0

    for fname in sorted(os.listdir(areas_dir)):
        if not fname.lower().endswith('.json'):
            continue
        with open(os.path.join(areas_dir, fname), 'r', encoding='utf-8') as f:
            cfg_area = json.load(f)
        pts = np.array(cfg_area.get('points', []))
        if pts.size == 0:
            continue

        func = cfg_area.get('function', None)
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        r1 = (pts[:,0].max() - pts[:,0].min()) / 2
        r2 = (pts[:,1].max() - pts[:,1].min()) / 2
        poly_path = Path(pts)
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        mask = poly_path.contains_points(np.vstack((X.ravel(), Y.ravel())).T)
        mask = mask.reshape((ny, nx))

        if func == 'cos':
            wave = compute_cos_wave(mask, (cx, cy), r1, r2, eta0=cfg_area.get('eta0', 1.0))
            im2 = ax.imshow(
                wave, origin='upper', cmap='viridis',
                norm=Normalize(vmin=0, vmax=np.nanmax(wave)), alpha=1.0,
                extent=(0, nx, ny, 0)
            )
            cbar2 = fig.colorbar(im2, ax=ax, pad=0.02)
            cbar2.set_label('Wave Height')
        else:
            poly = Polygon(
                pts, facecolor=base_colors[normal_idx % len(base_colors)],
                edgecolor='black', alpha=0.4, linewidth=1.5, label=fname
            )
            ax.add_patch(poly)
            normal_idx += 1

    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Bathymetry with Cosine Wave and Polygons')
    ax.legend(loc='upper right')
    plt.show()


def main():
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])
    plot_with_polygons(grid)


if __name__ == '__main__':
    main()
