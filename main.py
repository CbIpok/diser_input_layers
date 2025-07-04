import json
from loader import load_bath_grid
from colormap import make_land_ocean_cmap
from area import load_areas
from plotter import Plotter


def main():
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    grid = load_bath_grid(cfg['bath_path'])
    cmap, norm = make_land_ocean_cmap(grid.min(), grid.max())
    areas = load_areas(cfg.get('areas_dir', 'data/areas'))
    plotter = Plotter(grid)
    plotter.plot(cmap, norm, areas)


if __name__ == '__main__':
    main()