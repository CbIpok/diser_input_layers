import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ------------------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------------------
n = 4                 # number of cells per side inside the polygon (n x n grid)
theta_deg = -30        # rotation angle applied to the local grid (degrees)
basis_value = 1.1       # value assigned to basis cells inside each patch
basis_dir = 'data/basis'

os.makedirs(basis_dir, exist_ok=True)

# Paths to configuration and area definition
config_path = 'data/config.json'
area_path   = 'data/areas/source.json'

# ------------------------------------------------------------------------
# 1) Load scene size and polygon
# ------------------------------------------------------------------------
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
nx, ny = cfg['size']['x'], cfg['size']['y']

with open(area_path, 'r', encoding='utf-8') as f:
    area = json.load(f)
verts = np.array(area['points'])              # Nx2 polygon vertices

# Use the minimum corner as the local origin
x0, y0 = verts[:, 0].min(), verts[:, 1].min()

# ------------------------------------------------------------------------
# 2) Build a mask for the polygon on the original grid
# ------------------------------------------------------------------------
xs = np.arange(0.5, nx, 1.0)
ys = np.arange(0.5, ny, 1.0)
X, Y = np.meshgrid(xs, ys)

polygon_path = Path(verts)
mask_poly = polygon_path.contains_points(np.vstack((X.ravel(), Y.ravel())).T)
mask_poly = mask_poly.reshape(ny, nx)

# ------------------------------------------------------------------------
# 3) Rotate coordinates into the local reference frame
# ------------------------------------------------------------------------
rad = np.deg2rad(theta_deg)
cos_t, sin_t = np.cos(rad), np.sin(rad)

Xc = X - x0
Yc = Y - y0

# Rotation by -theta: [Xr; Yr] = R(-theta) * [Xc; Yc]
Xr =  cos_t * Xc + sin_t * Yc
Yr = -sin_t * Xc + cos_t * Yc

Xr_in = Xr[mask_poly]
Yr_in = Yr[mask_poly]

# ------------------------------------------------------------------------
# 4) Determine cell size that fits an n x n grid inside the polygon bounds
# ------------------------------------------------------------------------
xr_min, xr_max = Xr_in.min(), Xr_in.max()
yr_min, yr_max = Yr_in.min(), Yr_in.max()

cell = max((xr_max - xr_min) / n, (yr_max - yr_min) / n)

# Center the grid inside the rotated polygon bounds
xr0 = xr_min + ((xr_max - xr_min) - cell * n) / 2
yr0 = yr_min + ((yr_max - yr_min) - cell * n) / 2

# Grid edges and centers in rotated coordinates
i_edges = xr0 + cell * np.arange(n + 1)
j_edges = yr0 + cell * np.arange(n + 1)
i_centers = (i_edges[:-1] + i_edges[1:]) / 2
j_centers = (j_edges[:-1] + j_edges[1:]) / 2

# ------------------------------------------------------------------------
# 5) Transform helper back to global coordinates
# ------------------------------------------------------------------------
R = np.array([[ cos_t, -sin_t],
              [ sin_t,  cos_t]])
translation = np.array([x0, y0])


def to_global(pts_r):
    """Convert Nx2 points from rotated coordinates [Xr, Yr] to global [X, Y]."""
    pts_c = (R.dot(pts_r.T)).T
    pts = pts_c + translation
    return pts

# ------------------------------------------------------------------------
# 6) Visualize the rotated grid overlay (optional)
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(mask_poly, origin='lower', cmap='gray', alpha=0.3)

L = np.hypot(nx, ny) * 2
for xr in i_edges:
    base_r = np.array([xr, yr0 - cell])
    dir_r  = np.array([0, 1])
    seg = np.vstack([base_r + t * dir_r for t in (-L, L)])
    segg = to_global(seg)
    ax.plot(segg[:, 0], segg[:, 1], '--', color='k', linewidth=0.5)

for yr in j_edges:
    base_r = np.array([xr0 - cell, yr])
    dir_r  = np.array([1, 0])
    seg = np.vstack([base_r + t * dir_r for t in (-L, L)])
    segg = to_global(seg)
    ax.plot(segg[:, 0], segg[:, 1], '--', color='k', linewidth=0.5)

ax.set_aspect('equal')
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_title(f'{n}x{n} grid, theta={theta_deg} deg')
plt.show()

# ------------------------------------------------------------------------
# 7) Save n x n basis masks
# ------------------------------------------------------------------------
count = 0
for i0 in i_edges[:-1]:
    for j0 in j_edges[:-1]:
        cell_mask = ((Xr >= i0) & (Xr < i0 + cell) &
                     (Yr >= j0) & (Yr < j0 + cell))
        mask = mask_poly & cell_mask
        if not mask.any():
            continue
        fn = os.path.join(basis_dir, f'basis_{count}.wave')
        np.savetxt(fn, mask.astype(float) * basis_value, fmt='%.6f')
        count += 1

print(f'Generated {count} patches (expected up to {n * n}).')
