#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np


def _ensure_parent(path: os.PathLike | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_max_height_2d(nc_path: str, var_name: str = "max_height") -> np.ndarray:
    """Load variable as a 2D numpy array from a NetCDF file.

    If the variable has more than 2 dimensions (e.g., time, level),
    the first index along extra dimensions is taken until a 2D slice remains.
    """
    try:
        import xarray as xr
    except Exception as e:
        raise RuntimeError(
            "xarray is required to read NetCDF files. Please install it (and netCDF4)"
        ) from e

    ds = xr.open_dataset(nc_path)
    if var_name not in ds:
        available = ", ".join(list(ds.data_vars))
        raise KeyError(f"Variable '{var_name}' not found. Available: {available}")
    da = ds[var_name]

    # Reduce to 2D by selecting the first index along extra dims, if needed
    while da.ndim > 2:
        dim0 = da.dims[0]
        da = da.isel({dim0: 0})

    if da.ndim != 2:
        raise ValueError(f"Variable '{var_name}' is {da.ndim}D after reduction; expected 2D")

    arr = da.values
    # Ensure ndarray, not masked array
    arr = np.asarray(arr)
    return arr


def save_npy(array: np.ndarray, out_path: str) -> str:
    _ensure_parent(out_path)
    np.save(out_path, array)
    return out_path


def extract_cli(nc: str, out: Optional[str], var: str) -> str:
    arr = load_max_height_2d(nc, var)
    if out is None:
        base = Path(nc).stem
        out = Path("output") / f"{base}_{var}.npy"
        out = str(out)
    return save_npy(arr, out)



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract 'max_height' 2D array from NetCDF and save as NumPy,"
            " plus a simple drag-and-drop GUI to view .npy height maps."
        )
    )

    sub = p.add_subparsers(dest="cmd", required=False)

    # extract subcommand
    p_ext = sub.add_parser("extract", help="Extract variable to .npy")
    p_ext.add_argument(
        "--nc",
        default="data/restored_distribution/functions.nc",
        help="Path to input .nc file",
    )
    p_ext.add_argument(
        "--var",
        default="max_height",
        help="Variable name inside NetCDF (default: max_height)",
    )
    p_ext.add_argument(
        "--save-npy",
        dest="save_npy",
        default=None,
        help="Output .npy path (default derives from input)",
    )
    # Convenience flag to print example arguments for PyCharm
    p.add_argument(
        "--print-example-args",
        action="store_true",
        help=(
            "Print example program arguments for PyCharm using \n"
            "data/restored_distribution/functions.nc"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    if getattr(args, "print_example_args", False):
        example = (
            "extract --nc data/restored_distribution/functions.nc "
            "--var max_height --save-npy output/max_height.npy"
        )
        print(example)
        return

    cmd = getattr(args, "cmd", None) or "extract"
    if cmd == "extract":
        out_path = extract_cli(
            nc=getattr(args, "nc", "data/restored_distribution/functions.nc"),
            out=getattr(args, "save_npy", None),
            var=getattr(args, "var", "max_height"),
        )
        print(f"Saved: {out_path}")
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
