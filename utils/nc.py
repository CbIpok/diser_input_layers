import xarray as xr
import os
import glob
import re
from tqdm import tqdm
import numpy as np



### PUBLIC ###

# Загрузка данных mariogramm в точке
def load_mariogramm(nc_file,x,y):
    ds = xr.open_dataset(nc_file)
    # Извлечь данные переменной 'height'
    height_data = ds['height']
    # Вернуть данные по координатам x и y
    return height_data.isel(lon=x, lat=y).values

# Загрузка данных fk в точке
def load_fk(basis_path,x,y):
    path = basis_path
    basis_files = _get_sorted_file_list(path)
    fk = []
    for basis_file in tqdm(basis_files,desc="fk loading"):
        fk.append(load_mariogramm(basis_file,x,y))
    return fk

# Загрузка данных mariogramm для региона вдоль y
def load_mariogramm_by_region(nc_file, y_start, y_end):
    ds = xr.open_dataset(nc_file)
    # Извлекаем данные переменной 'height' для диапазона строк y
    height_data = ds['height']
    data =  height_data.isel(y=slice(y_start, y_end)).values
    return data

# Функция загрузки fk для региона вдоль y
def load_fk_by_region(basis_path, y_start, y_end):
    path = basis_path
    basis_files = _get_sorted_file_list(path)
    fk = []
    for basis_file in tqdm(basis_files, desc="fk loading"):
        # Загружаем данные для региона вдоль y (по 10 строк)
        fk.append(load_mariogramm_by_region(basis_file, y_start, y_end))
    return np.array(fk)

### PRIVATE ###

def _get_sorted_file_list(directory):
    # Паттерн для поиска файлов
    pattern = os.path.join(directory, '*_*.nc')

    # Находим все файлы, соответствующие паттерну
    files = glob.glob(pattern)

    # Регулярное выражение для извлечения значения i из имени файла
    regex = re.compile(r'_(\d+)\.nc')

    # Функция для сортировки, которая извлекает значение i
    def extract_index(filename):
        match = regex.search(os.path.basename(filename))
        if match:
            return int(match.group(1))  # Возвращаем i как целое число
        return float('inf')  # На случай, если не удастся извлечь индекс (файл будет в конце)

    # Сортируем файлы по значению i
    sorted_files = sorted(files, key=extract_index)

    return sorted_files

