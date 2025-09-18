Agent Guide for diser_input_layers

Scope
- This document explains how to run the end‑to‑end research pipeline the repository supports: reading NC, fitting coefficients, reconstructing forms, sampling points, running local analyses, building adaptive reconstructions, and comparing against the baseline (mean over i with smoothing).
- Applies to the entire repository.

Prerequisites
- Python 3.10+ with numpy, matplotlib, xarray. netCDF4 is optional (xarray works without it with default engines).
- Large data files are expected under `data/` and an external NC root may be mounted (e.g., `T:\tsumami_temp_shared_folder\res\Tokai_most`).
- Be mindful of `data/config.json` and its `save_interval` (commonly 4). JSON coordinates in `coefs_process/basis_{i}.json` are in the decimated grid and must be mapped back by `* save_interval`.
- Array indexing is `[y, x]` (rows, cols).

Key scripts (quick index)
- `scripts/reconstruct_from_nc.py` — fit coeffs from NC max_height for specific i, reconstruct full form.
- `scripts/reconstruct_adaptive.py` — adaptive multi‑i fusion on the full grid (NC fit per i + approx_error‑driven weights + optional smoothing).
- `scripts/research_5000.py` — sample K points in `source.json`, compute per‑i patches, baseline (mean+σ), adaptive patch fusion, and correlations.
- `scripts/compare_methods.py` — compare adaptive full‑grid vs baseline (mean+σ) on the same K points (MAE/bias quantiles, fraction improved).
- `scripts/point_mean.py` — mean reconstruction over i; supports saving arrays/images with i‑list/point suffixes and exporting 1D profiles.
- `scripts/plot_npy_gui.py` — drag&drop GUI to view A/B `.npy` maps and show B − A difference.
- `scripts/extract_max_height.py` — extract `max_height` from a NetCDF into `.npy`.

NC → coefficients → form (single i)
- Example (PowerShell):
  - `python scripts/reconstruct_from_nc.py --nc-root T:\tsumami_temp_shared_folder\res\Tokai_most --i 16 --basis-root data --functions-wave data/functions.wave --save-coeffs output/nc_fit/coeffs_i16.json --save-form output/nc_fit/form_i16.npy --save-diff output/nc_fit/diff_i16.npy --save-preview output/nc_fit/preview_i16.png`

Adaptive full‑grid reconstruction
- Use all i present in both NC root and `data/basis_i`, or specify `--i-list` (e.g., `16,25,36,49`).
- Example:
  - `python scripts/reconstruct_adaptive.py --nc-root T:\tsumami_temp_shared_folder\res\Tokai_most --basis-root data --i-list 16,25,36,49 --alpha 0.5 --beta 1.0 --smooth-sigma 1.5 --save output/adaptive/reconstruction.npy --save-png output/adaptive/reconstruction.png --save-json output/adaptive/summary.json`
- Weights: per‑pixel weight for i is `(i/max_i)^alpha * (approx_error_i + eps)^(-beta) * (1 + 0.5 * |∇Z̄|)`. Tuning `alpha, beta, smooth-sigma` trades average vs tail errors.

Research on K points (≥5000)
- Sample inside `data/areas/source.json`. First point can be forced (e.g., 2000,1400 corresponds to 500,350 when `save_interval=4`).
- Example:
  - `python scripts/research_5000.py --i-list 16,25,36,49 --n-points 5000 --window 12 --seed 0 --out-dir output/research_5000`
- Output: `summary.json` with:
  - Global stats of mean‑patch vs original (MAE/bias/std quantiles).
  - Per‑i MAE stats.
  - Baseline (mean over i + σ≈1.5) stats.
  - Adaptive patch fusion stats and fraction of points improved vs baseline.
  - Correlation of per‑i MAE vs `aprox_error` from `coefs_process/basis_{i}.json`.

Compare methods (adaptive vs baseline)
- Baseline: mean over i `16,25,36,49` followed by Gaussian smoothing (σ≈1.5).
- Adaptive: result of `scripts/reconstruct_adaptive.py`.
- Example:
  - `python scripts/compare_methods.py --nc-root T:\tsumami_temp_shared_folder\res\Tokai_most --basis-root data --i-list 16,25,36,49 --smooth-sigma 1.5 --adaptive-form output/adaptive/reconstruction.npy --functions data/functions.wave --n-points 5000 --seed 0 --force-first 2000,1400 --out-dir output/compare_adaptive`
- Output: `summary.json` with MAE/bias quantiles and fraction of improved points.

Profiles & visualization
- Mean over i with 1D profiles:
  - `python scripts/point_mean.py --i-list 16,25,36,49 --point 500 350 --save-mean output/mean.npy --save-dir output --profile-rows 1300,1400 --profile-cols 1900,2000`
- Drag&drop `.npy` viewer for A/B and B − A:
  - `python scripts/plot_npy_gui.py`

Conventions & pitfalls
- Always map sample coords (xs,ys) to full grid (x,y) via `* save_interval` from `data/config.json` when needed.
- Arrays are `[y, x]`. Coarse grids from JSON are placed at `[y*save_interval, x*save_interval]`.
- Large arrays: keep `--out-dir` on a fast disk. Avoid storing enormous CSVs; prefer `summary.json` and a few targeted previews.

Troubleshooting
- If NetCDF fails to load with `netCDF4` missing, xarray will still open via default engines in most environments. Install `netCDF4` if necessary for your setup.
- If you see zero error outside source area: ensure you sample points inside the polygon (`data/areas/source.json`).
- If baseline “wins” on MAE/median, but tails are large: try adaptive patch fusion (α≈0.5–1.0, β≈1.0) to reduce p95 while keeping median near baseline.

