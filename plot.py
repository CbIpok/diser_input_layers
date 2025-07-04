import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize

def load_bath_grid(path):
    with open(path, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    nx, ny = map(int, tokens[:2])
    idx = 2 + nx + ny
    data = np.array(tokens[idx:idx + nx * ny], dtype=float)
    return data.reshape((ny, nx))

def make_land_ocean_cmap(vmin, vmax, land_color='saddlebrown'):
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

def plot_areas_on_bath(grid, areas_dir='data/areas', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Построение, так чтобы (0,0) был в верхнем левом углу
    vmin, vmax = grid.min(), grid.max()
    cmap, norm = make_land_ocean_cmap(vmin, vmax)
    im = ax.imshow(grid, origin='upper', cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value (+ ocean, – land)')

    # Рисуем полигоны
    colors = ['red', 'green', 'blue']
    for i, fname in enumerate(sorted(os.listdir(areas_dir))):
        if not fname.lower().endswith('.json'):
            continue
        with open(os.path.join(areas_dir, fname), 'r', encoding='utf-8') as f:
            area = json.load(f)
        pts = area.get('points', [])
        if not pts:
            continue
        # Учитываем, что y-координата растет вниз
        poly = Polygon(pts,
                       facecolor=colors[i % len(colors)],
                       edgecolor='black',
                       alpha=0.4,
                       linewidth=1.5,
                       label=fname)
        ax.add_patch(poly)

    ax.set_xlim(0, grid.shape[1] - 1)
    ax.set_ylim(grid.shape[0] - 1, 0)
    ax.set_xlabel('X index →')
    ax.set_ylabel('← Y index (downward)')
    ax.set_title('Bath with Land (–) and Ocean (+), origin (0,0) at top-left')
    ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])
    plot_areas_on_bath(grid)

if __name__ == '__main__':
    main()
