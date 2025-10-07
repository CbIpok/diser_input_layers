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
  - Coefficient JSON directories may be split per functions dataset (e.g., `coefs_process/functions_pow1`). Scripts automatically resolve the matching folder based on `--functions`, but you can still point `--folder` to a specific subdirectory if needed.
- Areas (polygons) in `data/areas/*.json` (e.g., `source.json`, `mariogramm.json`).

Files and Formats
- `data/functions.wave`: 2D text grid (H-W) with the field to approximate/compare.
- Basis masks in `data/basis_{i}/basis_*.wave`: one mask per basis function, numeric, zeros treated as NaN when loading for analysis.
- Coefficients in `coefs_process/basis_{i}.json`:
  - JSON mapping string coordinates "(x, y)" -> `{ "coefs": [c1, c2, ...], "aprox_error": float }`.
  - Coordinates in this JSON may be downsampled according to `save_interval` in config.

Internal Library (diser/*)
- `diser/coords.py`: coordinate utilities and `Size` type.
- `diser/io/config.py`: `Config` loader (not mandatory in scripts; many use raw json for now).
- `diser/io/basis.py`: load/save of basis masks and functions.wave.
- `diser/io/coeffs.py`: `read_coef_json` -> `CoefSamples(xs, ys, coefs, approx_error)`.
- `diser/core/builder.py`: build n-n masks inside polygon with rotation (used by scripts/compute_basis.py).
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
- Compute basis masks (example 16 = 4-4, rotate -30 deg):
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
- `save_interval`: applies only to coordinates read from `basis_{i}.json`. Never rescale during math with basis/functions " only when placing or triangulating points.
- Shape checks: `functions.wave.shape == basis[j].shape` for all j; mismatches cause assertions.
- Interpolation: triangulation requires ≥3 finite points; otherwise RMSE falls back to the sparse grid.
- NaN semantics: masks and missing values are NaN; down‑stream code assumes NaN=invalid and suppresses drawing.
- Large outputs: `rmse_*.npy` can be tens of MB; keep `--save-dir` on a fast disk.

Quick Pipelines (PowerShell examples)
- Compute functions:
  - `python scripts/compute_functions.py -c data/config.json -o data/functions.wave`

- Build basis masks (N must be a perfect square; examples below):
  - 4 (=2-2):  `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 4  --theta -30`
  - 16 (=4-4): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 16 --theta -30`
  - 25 (=5-5): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 25 --theta -30`
  - 36 (=6-6): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 36 --theta -30`
  - 49 (=7-7): `python scripts/compute_basis.py -c data/config.json -a data/areas/source.json -i 49 --theta -30`

- RMSE (separate steps):
  - Compute grid only (writes `.npy`):
    - `python plot_basis_maps.py --i 16 --mode compute --rmse-out output/rmse_16.npy`
  - Plot from precomputed grid (interpolated RMSE only):
    - `python plot_basis_maps.py --i 16 --mode plot --rmse-in output/rmse_16.npy --save-dir output`

- RMSE (one shot):
  - `python plot_basis_maps.py --i 16 --mode both --save-dir output`

- Headless (no GUI):
  - `$env:MPLBACKEND='Agg'; python plot_basis_maps.py --i 16 --mode both --save-dir output`

CLI for new averaging functionality
- Mean reconstruction over multiple i (point module):
  - `python scripts/point_mean.py --i-list 4,16,25,36,49 --point 100 547 --save-mean output/mean_reconstruction.npy --save-smooth output/mean_reconstruction_smoothed.npy --smooth-sigma 2.0 --folder coefs_process --basis-root data --functions data/functions.wave --save-dir output`
  - Outputs: `.npy` arrays and PNGs (`mean_reconstruction.png`, `mean_reconstruction_smoothed.png`) in `output`.

- Mean RMSE over multiple i (plot_basis_maps module):
  - `python scripts/rmse_mean.py --i-list 4,16,25,36,49 --smooth-sigma 1.5 --folder coefs_process --basis-root data --functions data/functions.wave --out-dir output/rmse_mean`
  - Writes raw (`output/rmse_mean/rmse_mean__i_4_16_25_36_49__func_functions__recon_sigma_none.npy`) and smoothed (`...__recon_sigma_1_5.npy`) grids plus matching PNG/SVG/MPL figures. `--smooth-sigma` controls the pre-smoothing of reconstructed forms before RMSE.

Plotting scripts for saved means
- Forms (reconstruction):
  - `python scripts/plot_point_mean.py --mean output/mean_reconstruction.npy --smooth output/mean_reconstruction_smoothed.npy --save-dir output`
- RMSE grids:
  - `python scripts/plot_rmse_mean.py --rmse output/rmse_mean.npy --rmse-smooth output/rmse_mean_smoothed.npy --save-dir output`

Research pipeline (forms, NC, adaptive fusion)
- One‑off NC reconstruction for specific `i` (fits coeffs on max_height in NC, synthesizes full form):
  - `python scripts/reconstruct_from_nc.py --nc-root T:\tsumami_temp_shared_folder\res\Tokai_most --i 16 --basis-root data --functions-wave data/functions.wave --save-coeffs output/nc_fit/coeffs_i16.json --save-form output/nc_fit/form_i16.npy --save-diff output/nc_fit/diff_i16.npy --save-preview output/nc_fit/preview_i16.png`

- Mass local analysis on K points inside `source.json` (per‑i patches, mean+smooth baseline, adaptive fusion on patches):
  - `python scripts/research_5000.py --i-list 16,25,36,49 --n-points 5000 --window 12 --seed 0 --out-dir output/research_5000`
  - Output: `summary.json` with global stats, per‑i MAE, baseline (mean+σ=1.5), adaptive patch fusion (α,β), and correlation of MAE vs aprox_error.

- Adaptive multi‑i reconstruction (global, full grid) driven by NC + approx_error:
  - `python scripts/reconstruct_adaptive.py --nc-root T:\tsumami_temp_shared_folder\res\Tokai_most --basis-root data --i-list 16,25,36,49 --alpha 0.5 --beta 1.0 --smooth-sigma 1.5 --save output/adaptive/reconstruction.npy --save-png output/adaptive/reconstruction.png --save-json output/adaptive/summary.json`
  - If `--i-list` omitted, the script uses all i present in both NC root and `data/basis_i`.

- Compare adaptive vs baseline (mean{16,25,36,49}+σ=1.5) on the same K points:
  - `python scripts/compare_methods.py --nc-root T:\tsumami_temp_shared_folder\res\Tokai_most --basis-root data --i-list 16,25,36,49 --smooth-sigma 1.5 --adaptive-form output/adaptive/reconstruction.npy --functions data/functions.wave --n-points 5000 --seed 0 --force-first 2000,1400 --out-dir output/compare_adaptive`
  - Output: `summary.json` with MAE/bias quantiles and fraction of points improved by adaptive method.

Profiles and GUI tools
- Point‑mean with 1D profiles and signed filenames:
  - `python scripts/point_mean.py --i-list 16,25,36,49 --point 500 350 --save-mean output/mean.npy --save-dir output --profile-rows 1300,1400 --profile-cols 1900,2000`
  - Saves `profile_row_*.csv/.png` and `profile_col_*.csv/.png`. Arrays/images carry suffix `__i-list-*_point-X_Y`.

- View and diff `.npy` height maps (drag&drop):
  - `python scripts/plot_npy_gui.py` — two slots A/B plus "Show difference (B - A)".

NC utilities
- Extract `max_height` 2D array from NC to `.npy`:
  - `python scripts/extract_max_height.py extract --nc data/restored_distribution/functions.nc --var max_height --save-npy output/max_height.npy`

Universal figure save/load (with labels/annotations)
- Any script that saves figures now uses `diser.viz.figio.save_figure_bundle`, which writes:
  - `.png` (raster), `.svg` (vector), and `.mplfig.pkl` (full matplotlib figure object) with the same base name.
- You can reopen any saved figure with labels/annotations intact:
  - `python scripts/fig_show.py output/rmse_interpolated.mplfig.pkl`
  - Works for any figure saved via the bundle (RMSE, mean forms, etc.).

