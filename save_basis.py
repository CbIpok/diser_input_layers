import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path

# Параметры (жестко закодированы)
w = 170               # сторона квадрата
rotation = -30          # угол поворота сетки по часовой стрелке, градусы
offset_x = -5         # смещение сетки по X в повёрнутых координатах
offset_y = 5          # смещение сетки по Y в повёрнутых координатах

# Пути
config_path = os.path.join('data', 'config.json')
area_json = os.path.join('data', 'areas', 'source.json')
basis_dir = os.path.join('data', 'basis')

# Создаем папку для базисных функций
os.makedirs(basis_dir, exist_ok=True)

# Загружаем размеры сцены
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
nx, ny = config['size']['x'], config['size']['y']

# Создаем координаты центров ячеек
x = np.arange(0.5, nx, 1.0)
y = np.arange(0.5, ny, 1.0)
X, Y = np.meshgrid(x, y)

# Загружаем полигон области из source.json и вычисляем его центр
with open(area_json, 'r', encoding='utf-8') as f:
    area_data = json.load(f)
verts = np.array(area_data['points'])
# Центр области (среднее по вершинам)
cx, cy = verts[:,0].mean(), verts[:,1].mean()

# Сдвигаем координаты в центр области
Xc = X - cx
Yc = Y - cy

# Вычисляем поворот
rad = np.deg2rad(rotation)
cos_r, sin_r = np.cos(rad), np.sin(rad)
# Поворотные координаты для разбиения
Xr = cos_r * Xc + sin_r * Yc + offset_x
Yr = -sin_r * Xc + cos_r * Yc + offset_y

# Булева маска области без учёта смещения/поворота
polygon_path = Path(verts)
points = np.vstack((X.flatten(), Y.flatten())).T
mask_poly = polygon_path.contains_points(points).reshape(ny, nx)

# Определяем диапазон шагов
xr_min, xr_max = Xr.min(), Xr.max()
yr_min, yr_max = Yr.min(), Yr.max()
i_start = np.floor(xr_min / w) * w
i_end   = np.ceil(xr_max / w) * w
j_start = np.floor(yr_min / w) * w
j_end   = np.ceil(yr_max / w) * w
i_vals = np.arange(i_start, i_end + w, w)
j_vals = np.arange(j_start, j_end + w, w)

# Рисуем область и повернутую сетку
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(mask_poly, origin='lower', cmap='gray', alpha=0.3)

# Длина отрезков для прорисовки линий
L = np.hypot(nx, ny) * 2
# Горизонтальные линии (Xr = const)
for i_rot in i_vals:
    # точка на линии в повернутых-центр координатах
    if abs(cos_r) > 1e-6:
        x0c = i_rot / cos_r
        y0c = 0
    else:
        x0c = 0
        y0c = i_rot / sin_r
    direction = np.array([-sin_r, cos_r])
    pts = np.array([[x0c, y0c] + t * direction for t in (-L, L)])
    # переводим обратно к глобальным координатам
    pts[:,0] += cx
    pts[:,1] += cy
    ax.plot(pts[:,0], pts[:,1], linestyle='--', linewidth=0.5, color='gray')
# Вертикальные линии (Yr = const)
for j_rot in j_vals:
    if abs(cos_r) > 1e-6:
        x0c = 0
        y0c = j_rot / cos_r
    else:
        x0c = j_rot / sin_r
        y0c = 0
    direction = np.array([cos_r, sin_r])
    pts = np.array([[x0c, y0c] + t * direction for t in (-L, L)])
    pts[:,0] += cx
    pts[:,1] += cy
    ax.plot(pts[:,0], pts[:,1], linestyle='--', linewidth=0.5, color='gray')

# Накладываем квадраты и сохраняем функции
idx = 0
for i_rot in i_vals:
    for j_rot in j_vals:
        # маска квадрата в повёрнутых-центр координатах
        mask_square = (
            (Xr >= i_rot) & (Xr < i_rot + w) &
            (Yr >= j_rot) & (Yr < j_rot + w)
        )
        mask = mask_poly & mask_square
        if not mask.any():
            continue
        # координаты нижнего-левого угла в центр-коорд
        x0c = i_rot
        y0c = j_rot
        # сохраняем базисную функцию
        basis = mask.astype(float)
        out_path = os.path.join(basis_dir, f'basis_{idx}.wave')
        np.savetxt(out_path, basis, fmt='%.1f')
        idx += 1

ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_aspect('equal')
ax.set_title(
    f'Разбиение: w={w}, rotation={rotation}°, offset=({offset_x},{offset_y}), всего {idx} функций'
)
plt.show()
