# CASS LES Analysis

## Layout (2026-10-05)

| Path | What |
|---|---|
| `cass_analysis.py`, `catalog.py`, `test_shared_fixes.py` | shared library and experiment registry; the notebooks import them from this directory |
| `base_comparison.ipynb`, `rs_scale_comparison.ipynb`, `wind_geo_comparison.ipynb` | notebooks (v2 era; run from this directory) |
| `characterize/` | the v3 analysis pipeline (snapshots, population, composites) and all its figures, including the `fig_*.py` time series |

All figures go to one folder, `$SCRATCH/CASS_LES/analysis/figures/<expt>/`, with unsettled ones in
`in_progress/`. File names carry no figure numbers.
| `seb/` | conditioned surface-energy-balance cache (`compute_seb_cache.py`, `submit_seb_cache.sh`) |
| `sweeps/` | `_gen_sweep_notebooks.py`, generates sweep notebooks into this directory |
| `archive/` | superseded: `OLD/` notebooks, `cloud_roots/` (shadow composites, replaced by `characterize/composite.py`), `updrafts/`, `t_scale/` |

`base_comparison.ipynb` section 3 imports from `cloud_roots/`, now `archive/cloud_roots/` (one path line in the notebook).

## Architecture

`cass_analysis.py` is the shared library. All notebooks and pipeline scripts import from it.
`catalog.py` is the experiment registry (paths, labels, colors). Never hardcode paths in notebooks.
`sweeps/_gen_sweep_notebooks.py` generates sweep comparison notebooks from shared templates.

## Key conventions

- `load_stats()` returns `xr.Dataset`, not a dict. Access variables as `ds["H"]`, coordinates as `ds["z"]` or `ds.coords["z"]`. Use `.values` when you need numpy arrays.
- `load_stats_ensemble()` returns `(mean_ds, std_ds)` — both `xr.Dataset`. The `t_local` variable is preserved (datetimes dropped before concat to avoid NaT, re-attached after).
- Use `RT_STYLE[rt]` and `RT_LABEL[rt]` for colors/labels, not hardcoded values or `case["color"]`.
- Use `CASS_ROOT` from `catalog.py` for scratch paths, not hardcoded `/pscratch/...`.
- Use `LST_OFFSET`, `XL_GRID`, `ZND_GRID`, `eps_v`, `rho`, `cp`, `Lv` from `cass_analysis.py` — never redefine.
- f-strings with dict access inside: use double quotes for the f-string when the dict key uses single quotes: `f"...{RT_LABEL[case['rt']]}..."`, NOT `f'...{RT_LABEL[case['rt']]}...'`.

## Shared functions (cass_analysis.py)

**Data loading:**
- `load_stats(run_dir, mask="default")` → `xr.Dataset` (domain-mean stats from all groups). Pass `mask="couvreux"`, `mask="wplus"`, etc. to open the mask-conditioned stats file `cass.{mask}.0000000.nc`.
- `load_stats_ensemble(rep_dirs)` → `(mean, std)` xr.Datasets
- `load_xy_files(run_dir, variables)` → `xr.Dataset` (surface xy fields, lazy/dask)
- `load_sfc_xy(run_dir, varname)` → `(arr, t_sec)` numpy arrays
- `load_sfc_sw_dn(run_dir)` → `(arr, t_sec)` (handles 2stream vs raytracer naming)
- `load_3d_nc(run_dir, variables)` → `xr.Dataset` (3D dump fields)

**Analysis:**
- `seb_residual(ds)` — Rnet - (H + LE + G + S)
- `bowen_ratio(ds)` — H / LE
- `evaporative_fraction(ds)` — LE / (H + LE)
- `compute_z_sl(ds, time_idx)` — lowest level with any cloud in the domain mean. Biased low and jumpy; for cloud base use `characterize/masks.cloud_base_index` on a core or cloudy-updraft fraction
- `cloud_top_za(z, ql_frac)` — cloud top (highest ql_frac > 10% of peak)
- `cloud_mask_2d(ds)` — boolean mask where qlqi_path > 0
- `find_cloud_objects(mask, dx, dy, min_L)` — 8-connected, doubly periodic labelling; circular-mean centroids
- `chord_length_1d`, `chord_run_1d`, `centre_on_chord` — contiguous periodic chord through a cloud and slice centring
- `conditioned_means_ensemble(rep_dirs)` — cloud-root vs cloud-free flux means
- `lwp_integral(diff, dt_s)` — time-integrated LWP difference
- `sim_time_to_lst(t_sec)` — simulation seconds → local apparent solar time, hours
- `lst_window_mask(t_sec, lo, hi)` — boolean mask for LST window
- `dump_t_to_lst(dump_t_ns)` — ns-epoch float → LST hour

**Plotting:**
- `plot_cloud_root_composite(ax_thl, ax_qt, ds)` — L&P-style composite panels
- `plot_circulation_composite(ax, ds)` — b' + (u', w') quiver
- `plot_seb_timeseries(ax, stats_mean, stats_std, rt_type)`
- `plot_seb_residual(ax, stats_mean, rt_type)`
- `plot_flux_conditioned(ax, t, shaded, unshaded, ...)`

**Constants:**
- `eps_v = 0.608`, `Lv = 2.5e6`, `cp = 1005`, `rho = 1.2` (surface-layer value only; mass fluxes use `rhoref(z)`)
- `LST_OFFSET = 3.899` (solar, for runs starting 10:30 UTC), `THETA_REF = 300.0`
- `XL_GRID`, `ZND_GRID` — composite grids
- `RT_STYLE`, `RT_LABEL`, `FLUX_COLORS` — plot styling

## Data file layout

Stats: `$RUN_DIR/cass.{mask}.0000000.nc` (masks: `default`, `couvreux`, `wplus`, `ql`; groups: land_surface, radiation, thermo, default)
XY snapshots: `$RUN_DIR/<var>.xy.nc`
3D dumps: `$RUN_DIR/<var>.nc` (after `3d_to_nc.py`)
Composites: `$SCRATCH/CASS_LES/analysis/cloud_root_composite/<expt>/<rt>/rep_NN/events_{xz,yz}.nc`
SW composites: same path, `sw_composite.nc`
Caches: `$SCRATCH/CASS_LES/analysis/*.pkl` (regenerable, not committed)

## Cache stamp

Analysis output written by the scripts carries `cache_stamp = CACHE_STAMP` (netCDF attribute, or a `.stamp`
sidecar for pickles). Readers call `require_current_cache(path)` and skip-if-exists checks call
`cache_is_current(path)`, so output made under an older convention is recomputed or refused, never reused.
Bump `CACHE_STAMP` in `cass_analysis.py` whenever a definition behind cached output changes. Everything archived
on HPSS in `analysis.tar` predates the stamp. The pickle caches written from notebook cells generated by
`_gen_sweep_notebooks.py` are not stamped.

## archive/cloud_roots/ pipeline (superseded)

1. `3d_to_nc.py` (upstream) — converts binary dumps to netCDF
2. `cloud_root_composite_prep.py` — extracts per-event L&P slices → `events_{xz,yz}.nc`
3. `cloud_root_composite_average.py` — averages events across reps → composites
   - `average_reps()` — full-period averages
   - `average_reps_windowed()` — per-SZA-window averages
   - `average_reps_azimuth()` — azimuth-blended parallel/perpendicular composites
4. `sw_surface_composite.py` — chord-normalised surface SW profiles → `sw_composite.nc`
5. `submit_cloud_root_mega.sh` — SLURM submission for the full pipeline

## Notebooks

- `base_comparison.ipynb` — headline figures for no_aerosols_zero_wind: LWP + cloud population (§1), sw_scale comparison, SEB domain-mean + cloud-conditioned (§2), q_l time-height (§2), azimuth-rotated shadow composites with 1D / 3D / diff / SW% per SZA window (§3)
- `wind_geo_comparison.ipynb` — geostrophic-wind sweep: LWP summary + Bowen ratio vs u_g
- `OLD/` — retired, unedited: `radiative_coupling.ipynb`, `delta_sw_prediction.ipynb`, and since 2026-09-28
  `tmp_exploration.ipynb` and `mechanism.ipynb`, whose cells read caches that predate the stamp

## archive/updrafts/ pipeline (superseded)

H1 (mass flux) + H2 (tracer-dilution entrainment) via the offline Couvreux mask.

- `diagnostics.py` — `compute_sigma_min`, `build_couvreux_mask`, `updraft_mass_flux` (uses `rhoref(z)`),
  `entrainment_rate_tracer`, `entrainment_rate_siebesma` (contrast below 5 % of its profile maximum is masked),
  `compute_h1_h2_ensemble`
- `sanity_check.py` — CLI runner; Level-1 (stats-only) and Level-2 (needs `3d_to_nc.py` first) checks

## characterize/ (2026-09-28)

Characterization of existing 1D vs 3D runs from RAW hourly 3D snapshots, with these definitions:
core = q_l > 0, w > 0 and theta_v above the slab mean; environment = mean over columns with no condensate at any
level; cloud base = lowest level where core fraction exceeds a threshold (1e-4, 5e-4, 1e-3 all reported);
times in local apparent solar time from each run's own `cass.ini`.

| File | Role |
|---|---|
| `thermo.py` | MicroHH constants, saturation polynomial and `sat_adjust`, ported from the model source |
| `les_io.py` | readers for `var.NNNNNNN` binaries, `grid.0000000`, `thermo_basestate.NNNNNNN` |
| `masks.py` | core, clear columns, cloud base, periodic labelling |
| `snapshot.py` | per-snapshot profiles, cloud-base means for core and cloudy updraft, root anomalies, root width |
| `parcel.py` | bulk entrainment, undilute and entraining parcels, CIN, w_crit |
| `activation.py` | per-parcel active fraction at cloud base and its response to boosts in w, moisture, heat |
| `collect.py` | tables with ensemble mean, sample sd, 3D minus 1D and its standard error |
| `figures.py`, `style.py` | figures, written to scratch |
| `closure_check.py` | offline thermodynamics vs dumped T, q_l, b; snapshot means vs 5-min statistics |
| `restore_v2.sh` | HPSS subset restore with size check (NERSC only) |
| `submit_characterize.sh` | CPU batch driver; `STAGES="snapshot parcel activation"` |

Run: `bash characterize/submit_characterize.sh <expt>`, then `python characterize/figures.py --expt <expt>`.
Tests: `python characterize/test_*.py` and `python test_shared_fixes.py`. Output goes to
`$SCRATCH/CASS_LES/analysis/characterize/<expt>/`.

## Removed 2026-09-28

Superseded by `characterize/`; last present in commit c3b6b67c8:
`entrainment/compute_entrainment.py`, `entrainment/compute_shell.py`, `updrafts/compute_couvreux_columns.py`,
`t_scale/compute_cloudbase_mf.py`, each with its submit script. Their cached outputs (`entrainment.nc`, `shell.nc`,
couvreux-column pickles, `cb_mf.nc`) should not be used: "core" had no buoyancy test, the environment was not
clear-column, cloud base was the lowest level with any cloud, and stored times were on the clock convention.

Known and not yet fixed: `t_scale/compute_T_scale.py` sums the autocorrelation from lag 0 with a rectangle rule,
which overestimates the timescale by about half a time step.
