import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

def make_land_ocean_cmap(vmin: float, vmax: float, land_color: str = 'saddlebrown'):
    """Create a land/ocean colormap with a custom land color at zero.

    Values below zero use the Blues colormap; values at or above zero use the provided land color."""
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