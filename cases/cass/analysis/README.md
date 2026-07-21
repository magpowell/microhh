# CASS LES Analysis

## Architecture

`cass_analysis.py` is the shared library. All notebooks and pipeline scripts import from it.
`catalog.py` is the experiment registry (paths, labels, colors). Never hardcode paths in notebooks.
`_gen_sweep_notebooks.py` generates sweep comparison notebooks from shared templates.

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
- `compute_z_sl(ds, time_idx)` — cloud base height (lowest ql_frac > 0)
- `cloud_top_za(z, ql_frac)` — cloud top (highest ql_frac > 10% of peak)
- `cloud_mask_2d(ds)` — boolean mask where qlqi_path > 0
- `conditioned_means_ensemble(rep_dirs)` — cloud-root vs cloud-free flux means
- `lwp_integral(diff, dt_s)` — time-integrated LWP difference
- `sim_time_to_lst(t_sec)` — simulation seconds → LST hours
- `lst_window_mask(t_sec, lo, hi)` — boolean mask for LST window
- `dump_t_to_lst(dump_t_ns)` — ns-epoch float → LST hour

**Plotting:**
- `plot_cloud_root_composite(ax_thl, ax_qt, ds)` — L&P-style composite panels
- `plot_circulation_composite(ax, ds)` — b' + (u', w') quiver
- `plot_seb_timeseries(ax, stats_mean, stats_std, rt_type)`
- `plot_seb_residual(ax, stats_mean, rt_type)`
- `plot_flux_conditioned(ax, t, shaded, unshaded, ...)`

**Constants:**
- `eps_v = 0.608`, `Lv = 2.5e6`, `cp = 1005`, `rho = 1.2`
- `LST_OFFSET = 5.5`, `THETA_REF = 300.0`
- `XL_GRID`, `ZND_GRID` — composite grids
- `RT_STYLE`, `RT_LABEL`, `FLUX_COLORS` — plot styling

## Data file layout

Stats: `$RUN_DIR/cass.{mask}.0000000.nc` (masks: `default`, `couvreux`, `wplus`, `ql`; groups: land_surface, radiation, thermo, default)
XY snapshots: `$RUN_DIR/<var>.xy.nc`
3D dumps: `$RUN_DIR/<var>.nc` (after `3d_to_nc.py`)
Composites: `$SCRATCH/CASS_LES/analysis/cloud_root_composite/<expt>/<rt>/rep_NN/events_{xz,yz}.nc`
SW composites: same path, `sw_composite.nc`
Caches: `$SCRATCH/CASS_LES/analysis/*.pkl` (regenerable, not committed)

## cloud_roots/ pipeline

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
- `tmp_exploration.ipynb` — exploratory diagnostics: shell decomposition (Heus & Jonker), w*, T_tau, ε(z), archived xz/yz composites, surface-field distributions
- `wind_geo_comparison.ipynb` — geostrophic-wind sweep: LWP summary + Bowen ratio vs u_g
- `radiative_coupling.ipynb` — alpha, gamma_s, Q_rho profiles (core theoretical framework)
- `delta_sw_prediction.ipynb` — full α closure chain: f_shadow, SW_out, and α_3D predictions from observable quantities (validated on CASS + Tijhuis Zenodo 15649286)

## updrafts/ pipeline

H1 (mass flux) + H2 (tracer-dilution entrainment) via the offline Couvreux mask.
See `updrafts/PLAN.md` for the full spec.

- `diagnostics.py` — `compute_sigma_min`, `build_couvreux_mask`,
  `updraft_mass_flux`, `entrainment_rate_tracer`, `compute_h1_h2_ensemble`
- `sanity_check.py` — CLI runner; Level-1 (stats-only, runs immediately after a
  sim finishes) and Level-2 (needs `3d_to_nc.py` first) checks with graceful skip
- `verify_and_cleanup.py` — verify each `{var}.nc` against its binary dumps,
  then (with `--delete`) reclaim disk while preserving the final-timestamp
  restart files (~45 GB per 256³ rep)
