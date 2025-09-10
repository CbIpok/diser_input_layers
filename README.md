Project: diser_input_layers

Overview
- Purpose: prepare input layers (functions and basis masks), store decomposition coefficients, and analyze/visualize approximation error (RMSE) for tsunami modeling experiments.
- The codebase is split into: internal library (`diser/*`), CLI scripts (`scripts/*` and root tools), and analysis/plotting scripts.

Coordinate Convention (very important)
- Points are Cartesian `(x, y)`, where `x` is column (width), `y` is row (height).
- Numpy arrays are indexed as `[row, col] == [y, x]` everywhere.
- Helpers live in `diser/coords.py` (`xy_to_rc`, `rc_to_xy`).

Config and Data
- `data/config.json` minimal schema:
  - `size`: `{ "x": width, "y": height }`
  - `bath_path`: path to MOST bath file used by `loader.load_bath_grid`
  - `save_interval` (int, optional): spacing of saved sample coordinates in `coefs_process/basis_{i}.json`.
    - Only affects where to PLACE values from the JSON (rendering), not the underlying computations.
    - Example: if `save_interval=4`, values from `(x, y)` in JSON must be written into the full grid at `[y*4, x*4]`.
- Areas (polygons) in `data/areas/*.json` (e.g., `source.json`, `mariogramm.json`).

Files and Formats
- `data/functions.wave`: 2D text grid (H×W) with the field to approximate/compare.
- Basis masks in `data/basis_{i}/basis_*.wave`: one mask per basis function, numeric, zeros treated as NaN when loading for analysis.
- Coefficients in `coefs_process/basis_{i}.json`:
  - JSON mapping string coordinates "(x, y)" -> `{ "coefs": [c1, c2, ...], "aprox_error": float }`.
  - Coordinates in this JSON may be downsampled according to `save_interval` in config.

Internal Library (diser/*)
- `diser/coords.py`: coordinate utilities and `Size` type.
- `diser/io/config.py`: `Config` loader (not mandatory in scripts; many use raw json for now).
- `diser/io/basis.py`: load/save of basis masks and functions.wave.
- `diser/io/coeffs.py`: `read_coef_json` -> `CoefSamples(xs, ys, coefs, approx_error)`.
- `diser/core/builder.py`: build n×n masks inside polygon with rotation (used by scripts/compute_basis.py).
- `diser/core/restore.py`:
  - `reconstruct_from_bases(c, bases)` → `Z_hat`.
  - `valid_mask_from_bases(bases)` → where any basis is valid.
  - `mse_on_valid_region(Z_true, Z_hat, mask)`.
  - `pointwise_rmse_from_coefs(to_restore, bases, xs, ys, coefs)` returns grid with RMSE only at (y, x) = (ys, xs).
- `diser/viz/maps.py`: plotting helpers (no implicit `plt.show`).

Root/CLI Scripts
- `save_subduction.py`: computes `data/functions.wave` from config and areas (uses numpy.savetxt).
- `scripts/compute_functions.py`: thin wrapper over `save_subduction.save_functions`.
- `scripts/compute_basis.py`: builds rectangular basis masks (`data/basis_{i}`) from a polygon with rotation.
- `plot_basis_maps.py`:
  - `--mode compute`: builds RMSE grid using `coefs_process/basis_{i}.json` and basis/functions, writes `.npy` if `--rmse-out`.
  - `--mode plot`: loads `.npy` if `--rmse-in` and renders an interpolated RMSE-only PNG (no background/zones) as `rmse_interpolated.png` in `--save-dir`.
  - `--mode both`: compute and then plot (saves both `.npy` and PNG).
  - The JSON coordinates are placed at `[y*save_interval, x*save_interval]` when building the RMSE grid; computations do not scale coordinates.
- `point.py`: point-wise reconstruction flow; uses new `diser/*` but keeps prior CLI.

Usage (common)
- Compute functions (PowerShell): `python scripts/compute_functions.py -c data/config.json -o data/functions.wave`
- Compute basis masks (example 16 = 4×4, rotate -30°):
  - `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 16 --theta -30`
- Compute + plot RMSE (i=16):
  - `python plot_basis_maps.py --i 16 --mode both --save-dir output`
  - Outputs: `output/rmse_16.npy`, `output/rmse_interpolated.png` (and per-coef maps if enabled elsewhere).

Save Interval (formerly stride)
- Config key: `save_interval`.
- Meaning: the sampling interval of coordinates that were written to `basis_{i}.json` (e.g., every 4th point).
- Application:
  - When building the RMSE grid from JSON samples, place at `[y*save_interval, x*save_interval]`.
  - Do NOT scale during basis/functions math; only when placing values on the final grid or triangulation coords.

Testing
- Unit tests: `python -m unittest discover -s tests -p "test_*.py" -v`
- Integration tests: `python -m unittest discover -s tests/integration -p "test_*.py" -v`
- Headless runs (no GUI): set `MPLBACKEND=Agg` (PowerShell: `$env:MPLBACKEND='Agg'`).

Large Data Notes
- Repo expects large files under `data/` (e.g., MANY `.wave` or `.nc`). Interpolations and RMSE builds may take time and memory.
- Some integration tests read large files and can take ~50s.

Tips for Future Sessions
- If RMSE overlay on bathymetry is needed again, re-enable the background section in `plot_basis_maps.py` (currently plotting RMSE only, for clarity). A saved example from earlier exists: `output/overlay_rmse.png`.
- When results look “shrunk”, verify that `save_interval` matches how the JSON was generated; wrong interval misplaces values.
- Consistently use array indexing `[y, x]` when placing values into grids.

Gotchas (common pitfalls)
- Indexing order: Always place values into arrays as `[row=y, col=x]`.
- `save_interval`: applies only to coordinates read from `basis_{i}.json`. Never rescale during math with basis/functions — only when placing or triangulating points.
- Shape checks: `functions.wave.shape == basis[j].shape` for all j; mismatches cause assertions.
- Interpolation: triangulation requires ≥3 finite points; otherwise RMSE falls back to the sparse grid.
- NaN semantics: masks and missing values are NaN; down‑stream code assumes NaN=invalid and suppresses drawing.
- Large outputs: `rmse_*.npy` can be tens of MB; keep `--save-dir` on a fast disk.

Quick Pipelines (PowerShell examples)
- Compute functions:
  - `python scripts/compute_functions.py -c data/config.json -o data/functions.wave`

- Build basis masks (N must be a perfect square; examples below):
  - 4 (=2×2):  `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 4  --theta -30`
  - 16 (=4×4): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 16 --theta -30`
  - 25 (=5×5): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 25 --theta -30`
  - 36 (=6×6): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 36 --theta -30`
  - 49 (=7×7): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 49 --theta -30`

- RMSE (separate steps):
  - Compute grid only (writes `.npy`):
    - `python plot_basis_maps.py --i 16 --mode compute --rmse-out output/rmse_16.npy`
  - Plot from precomputed grid (interpolated RMSE only):
    - `python plot_basis_maps.py --i 16 --mode plot --rmse-in output/rmse_16.npy --save-dir output`

- RMSE (one shot):
  - `python plot_basis_maps.py --i 16 --mode both --save-dir output`

- Headless (no GUI):
  - `$env:MPLBACKEND='Agg'; python plot_basis_maps.py --i 16 --mode both --save-dir output`
