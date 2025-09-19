import matplotlib.pyplot as plt
import numpy as np

from area import CosineWaveArea


class Plotter:
    """Utility for plotting bathymetry grid with configured areas."""
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.ny, self.nx = grid.shape
        self.fig, self.ax = plt.subplots(figsize=(10, 7))

    def plot_grid(self, cmap, norm):
        im = self.ax.imshow(
            self.grid, origin='upper', cmap=cmap, norm=norm
        )
        cbar = self.fig.colorbar(im, ax=self.ax)
        cbar.set_label('Bathymetry')

    def plot_areas(self, areas: list):
        base_colors = ['red', 'green', 'blue']
        idx = 0
        for area in areas:
            if isinstance(area, CosineWaveArea):
                im2 = area.plot(
                    self.ax, nx=self.nx, ny=self.ny
                )
                cbar2 = self.fig.colorbar(im2, ax=self.ax, pad=0.02)
                cbar2.set_label('Wave Height')
            else:
                color = base_colors[idx % len(base_colors)]
                area.plot(self.ax, color=color)
                idx += 1

    def finalize(self):
        self.ax.set_xlim(0, self.nx)
        self.ax.set_ylim(self.ny, 0)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Bathymetry with Configurable Areas')
        self.ax.legend(loc='upper right')
        plt.show()

    def plot(self, cmap, norm, areas: list):
        self.plot_grid(cmap, norm)
        self.plot_areas(areas)
        self.finalize()
