import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ------------------------------------------------------------------------
#  Настройки
# ------------------------------------------------------------------------
n = 8                 # число делений по каждой стороне → получится n×n клеток
theta_deg = -30        # угол наклона сетки в градусах (по часовой)
basis_dir = 'data/basis'

os.makedirs(basis_dir, exist_ok=True)

# Пути к данным
config_path = 'data/config.json'
area_path   = 'data/areas/source.json'

# ------------------------------------------------------------------------
#  1) Загрузка полигона и сцены
# ------------------------------------------------------------------------
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
nx, ny = cfg['size']['x'], cfg['size']['y']

with open(area_path, 'r', encoding='utf-8') as f:
    area = json.load(f)
verts = np.array(area['points'])              # Nx2 массив вершин полигона

# Опорная точка (центр) — просто минимум по x,y (можно centroid)
x0, y0 = verts[:,0].min(), verts[:,1].min()

# ------------------------------------------------------------------------
#  2) Строим булеву маску полигона в исходных координатах
# ------------------------------------------------------------------------
xs = np.arange(0.5, nx, 1.0)
ys = np.arange(0.5, ny, 1.0)
X, Y = np.meshgrid(xs, ys)

polygon_path = Path(verts)
mask_poly = polygon_path.contains_points(np.vstack((X.ravel(), Y.ravel())).T)
mask_poly = mask_poly.reshape(ny, nx)

# ------------------------------------------------------------------------
#  3) Прямое преобразование в «развернутую» систему
#     (сдвиг центра → поворот на -theta)
# ------------------------------------------------------------------------
rad = np.deg2rad(theta_deg)
cos_t, sin_t = np.cos(rad), np.sin(rad)

# центрирование
Xc = X - x0
Yc = Y - y0

# поворот на -theta: [Xr; Yr] = R(-θ) · [Xc; Yc]
Xr =  cos_t * Xc + sin_t * Yc
Yr = -sin_t * Xc + cos_t * Yc

# берём только те Xr, Yr, что внутри полигона
Xr_in = Xr[mask_poly]
Yr_in = Yr[mask_poly]

# ------------------------------------------------------------------------
#  4) Делаем равномерное разбиение диапазонов Xr_in, Yr_in на n частей
# ------------------------------------------------------------------------
xr_min, xr_max = Xr_in.min(), Xr_in.max()
yr_min, yr_max = Yr_in.min(), Yr_in.max()

# длина ячейки так, чтобы ровно n клеток покрыло диапазон
cell = max((xr_max-xr_min)/n, (yr_max-yr_min)/n)

# смещаем начало сетки чуть внутрь, чтобы центральный полигон точно покрылся
xr0 = xr_min + ((xr_max-xr_min) - cell*n)/2
yr0 = yr_min + ((yr_max-yr_min) - cell*n)/2

# координаты границ всех клеток
i_edges = xr0 + cell * np.arange(n+1)
j_edges = yr0 + cell * np.arange(n+1)

# для упрощения — центры клеток (необязательно, но удобно)
i_centers = (i_edges[:-1] + i_edges[1:]) / 2
j_centers = (j_edges[:-1] + j_edges[1:]) / 2

# ------------------------------------------------------------------------
#  5) Обратное преобразование (для рисования и для сохранения масок)
#     сначала R(θ), затем +[x0, y0]
# ------------------------------------------------------------------------
R = np.array([[ cos_t, -sin_t],
              [ sin_t,  cos_t]])
translation = np.array([x0, y0])

def to_global(pts_r):
    """ из массива Nx2 точек [Xr,Yr] в глобальные [X,Y] """
    pts_c = (R.dot(pts_r.T)).T   # поворот
    pts   = pts_c + translation  # смещение
    return pts

# ------------------------------------------------------------------------
#  6) Рисуем полигона и сетку для контроля
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(mask_poly, origin='lower', cmap='gray', alpha=0.3)

# рисуем линии i_edges (горизонтальные в развернутой системе)
L = np.hypot(nx, ny)*2
for xr in i_edges:
    # в развернутой системе это прямая Yr любая, Xr=xr
    # базовая точка и направление
    base_r = np.array([xr, yr0 - cell])    # начинаем чуть ниже
    dir_r  = np.array([0, 1])             # вдоль Yr
    seg = np.vstack([base_r + t*dir_r for t in (-L, L)])
    segg = to_global(seg)
    ax.plot(segg[:,0], segg[:,1], '--', color='k', linewidth=0.5)

# рисуем линии j_edges (вертикальные в развернутой системе)
for yr in j_edges:
    base_r = np.array([xr0 - cell, yr])
    dir_r  = np.array([1, 0])
    seg = np.vstack([base_r + t*dir_r for t in (-L, L)])
    segg = to_global(seg)
    ax.plot(segg[:,0], segg[:,1], '--', color='k', linewidth=0.5)

ax.set_aspect('equal')
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_title(f'{n}×{n} сетка, θ={theta_deg}°')
plt.show()

# ------------------------------------------------------------------------
#  7) Формируем и сохраняем маски n×n клеток
# ------------------------------------------------------------------------
count = 0
for i0 in i_edges[:-1]:
    for j0 in j_edges[:-1]:
        # маска «клетки» в развернутой системе
        cell_mask = ((Xr >= i0) & (Xr < i0 + cell) &
                     (Yr >= j0) & (Yr < j0 + cell))
        mask = mask_poly & cell_mask
        if not mask.any():
            continue
        # сохраняем
        fn = os.path.join(basis_dir, f'basis_{count}.wave')
        np.savetxt(fn, mask.astype(float), fmt='%.1f')
        count += 1

print(f'Создано {count} базисных клеток (из максимальных {n*n})')
