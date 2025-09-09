import unittest
import numpy as np
import matplotlib

matplotlib.use('Agg')

from plotter import Plotter
from colormap import make_land_ocean_cmap


class TestPlotter(unittest.TestCase):
    def test_plotter_runs_without_gui(self):
        grid = np.arange(4*5, dtype=float).reshape(4, 5)
        cmap, norm = make_land_ocean_cmap(grid.min(), grid.max())
        p = Plotter(grid)
        # No areas passed — should still render without errors
        p.plot(cmap, norm, areas=[])


if __name__ == '__main__':
    unittest.main()

