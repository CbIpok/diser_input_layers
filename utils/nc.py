import xarray as xr
import os
import glob
import re
from tqdm import tqdm
import numpy as np


### PUBLIC ###

# Load a single mariogram (height) value at lon/lat indices
def load_mariogramm(nc_file, x, y):
    ds = xr.open_dataset(nc_file)
    height_data = ds['height']
    return height_data.isel(lon=x, lat=y).values

# Load mariogram values across all basis NetCDF files for a specific lon/lat
def load_fk(basis_path, x, y):
    path = basis_path
    basis_files = _get_sorted_file_list(path)
    fk = []
    for basis_file in tqdm(basis_files, desc="fk loading"):
        fk.append(load_mariogramm(basis_file, x, y))
    return fk

# Load a vertical strip of mariogram data (range of y indices)
def load_mariogramm_by_region(nc_file, y_start, y_end):
    ds = xr.open_dataset(nc_file)
    height_data = ds['height']
    data = height_data.isel(y=slice(y_start, y_end)).values
    return data

# Load fk values for a strip of y indices across all basis files
def load_fk_by_region(basis_path, y_start, y_end):
    path = basis_path
    basis_files = _get_sorted_file_list(path)
    fk = []
    for basis_file in tqdm(basis_files, desc="fk loading"):
        fk.append(load_mariogramm_by_region(basis_file, y_start, y_end))
    return np.array(fk)

### PRIVATE ###

def _get_sorted_file_list(directory):
    pattern = os.path.join(directory, '*_*.nc')
    files = glob.glob(pattern)
    regex = re.compile(r'_(\d+)\.nc')

    def extract_index(filename):
        match = regex.search(os.path.basename(filename))
        if match:
            return int(match.group(1))  # numerical suffix after underscore
        return float('inf')  # place unmatched files at the end

    sorted_files = sorted(files, key=extract_index)
    return sorted_files
