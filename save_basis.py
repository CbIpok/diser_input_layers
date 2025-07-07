import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path

# Параметры (жестко закодированы)
w = 150  # сторона квадрата

# Пути
config_path = os.path.join('data', 'config.json')
area_json = os.path.join('data', 'areas', 'source.json')
basis_dir = os.path.join('data', 'basis')

# Создаем папку для базисных функций
os.makedirs(basis_dir, exist_ok=True)

# Загружаем размеры сцены
with open(config_path, 'r') as f:
    config = json.load(f)
nx, ny = config['size']['x'], config['size']['y']

# Задаем координаты узлов сетки (центры ячеек)
x = np.arange(0.5, nx, 1.0)
y = np.arange(0.5, ny, 1.0)
X, Y = np.meshgrid(x, y)

# Загружаем полигон области из source.json
with open(area_json, 'r') as f:
    area_data = json.load(f)
# Ожидаем, что в JSON есть ключ 'raw_points': список [ [x1,y1], [x2,y2], ... ]
verts = area_data['points']

# Создаем Path для определения принадлежности точек полигону
polygon_path = Path(verts)
points = np.vstack((X.flatten(), Y.flatten())).T
mask_poly = polygon_path.contains_points(points).reshape(ny, nx)

# Рисуем разбиение квадратиками
fig, ax = plt.subplots(figsize=(8, 6))
# Область
ax.imshow(mask_poly, origin='lower', cmap='gray', alpha=0.3)

idx = 0
# Проходим по всем квадратам
for i in range(0, nx, w):
    for j in range(0, ny, w):
        # Определим квадрат [i, i+w) x [j, j+w)
        sq_mask = (
            (X >= i) & (X < i + w) &
            (Y >= j) & (Y < j + w)
        )
        # Пересечение с областью
        mask = mask_poly & sq_mask
        if not mask.any():
            continue
        # Рисуем контур квадрата
        rect = Rectangle((i, j), w, w, fill=False, edgecolor='blue', linewidth=0.5)
        ax.add_patch(rect)
        # Сохраняем базисную функцию
        basis = mask.astype(float)
        out_path = os.path.join(basis_dir, f'basis_{idx}.wave')
        np.savetxt(out_path, basis, fmt='%.1f')
        idx += 1

ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_aspect('equal')
ax.set_title(f'Разбиение области на квадраты w={w}, всего {idx} функций')
plt.show()
