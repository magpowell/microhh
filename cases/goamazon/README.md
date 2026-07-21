# GoAmazon single-pulse — 1D vs 3D radiative transfer (MicroHH)

Deep-convection companion to the CASS shallow-cumulus study: the GoAmazon
day 278 "early single peak convection" case (2014-10-05, Tian & Zhang 2025
forcing), run in MicroHH with 2-stream RRTMGP vs the coupled 3D ray tracer.
Same case as the DP-SCREAM runs analysed offline in
`~/repos/3D_RT_DPSCREAM`, now with interactive radiation feedback.

No experiments layer: this case is the base 1D-vs-3D comparison only.

## Case design

| Element | Choice |
|---|---|
| Site / time | GoAmazon T3 (-3.2, -60.6); t=0 at 2014-10-05 12:00 UTC (08:00 LT); 12 h |
| Forcing | IOP divT/divq as timedep LS thl/qt tendencies; no subsidence (divT/divq are total advective forcings, `iop_dosubsidence=false`); no Coriolis |
| Winds | Nudged to IOP `u_ls`/`v_ls` (the EAMxx target when `*_ls` present), tau=10800 s (EAMxx `iop_nudge_tscale` default) |
| Surface | Interactive HTESSEL LSM (the intland-analog, NOT the prescribed-flux cntl); ERA5 soil init; IFS evergreen-broadleaf vegetation; IOP shflx/lhflx kept in the input `validation` group for LSM checks |
| Microphysics | `nsw6` (Tomita 6-class, ice-capable; 2mom_warm cannot do deep convection) |
| Radiation | rrtmgp vs rrtmgp_rt; aerosols OFF; upstream cloud LUTs (both RT modes share them, internally consistent) |
| Background | **HYBRID** (`make_hybrid_ls2d_input.py`): soil = real ERA5 surface_an (12 UTC 2014-10-05: slt=3, theta 0.27-0.37; veg validated tvh=6, lai 5.35, fsr 1.30 m, cvh 0.91 -> ini values); radiation background / o3 / above-IOP padding = RCEMIP analytic tropical sounding (common-mode between RT legs, negligible for the comparison). Full model-level ERA5 (`goamazon_ls2d_input.py`) optional upgrade if MARS ever delivers |
| Vertical grid | **Uniform dz** (hard `radiation_rrtmgp_rt` requirement, `src/radiation_rrtmgp_rt.cxx` create()); zsize=25.6 km, sponge above 20 km; IOP p-levels mapped hydrostatically to z; background fills above the IOP top (~110 hPa / ~16.5 km, 1 km blend) |
| Grid | **512x512x384: dx=200 m, dz=66.7 m, 102.4 km domain** (production, `dz67_k384`) |

## Grid selection (fit tests, 2026-07-10)

Raytracer GPU memory scales linearly at ~0.60 GB/Mcell (measured 67M->40.4,
84M->49.9, 105M->61.6, 76M->46.4, 134M->80.5 GB). 512^3 (dz=50, full domain)
peaks at 98% of hbm80g -> rejected; dz=66.7 at full domain interpolates to
~60 GB. Wall (clear-sky morning): ~0.35 us/cell/step incl. dt_rad=60 s.

BL-resolution ladder (64^2 mini-domain to 11:10 LT, dz=100/66.7/50): mixed
layer fully resolved at all three (SGS flux fraction <= 0.14 above the first
level); entrainment zone: dz=66.7 gives 0.10-0.16 vs 0.10-0.13 at dz=50 and
0.15-0.19 at dz=100 -> dz=66.7 captures nearly all of the dz=50 improvement.
First model level and EZ flux zero-crossing are SGS-heavy at any affordable
dz (standard LES caveats, common-mode across the RT pair).

## Known biases (accepted 2026-07-10)

LSM H runs ~2.5x the IOP obs (255 vs ~105 W/m2 peak) while LE matches
(~380-400 vs ~385): the SEB closes exactly, so this is surplus Rnet, not a
partition bug — (a) aerosols off vs the polluted GoAmazon IOP2 dry season
(~60-80 W/m2 clear-sky SWdn excess already at 08:00 LT), (b) the LES morning
is sunnier than the observed day (20-33% shallow-Cu cover). The DP-SCREAM
interactive-ELM runs show the same H excess (ELM offline 1D/3D and intland
peak 220-250 W/m2; see 3D_RT_DPSCREAM thoughts.ipynb), so MicroHH-HTESSEL
sits inside the ELM family — comparability with the SCREAM runs is intact,
and the bias is common-mode across the 1D/3D pair. If onset-vs-obs timing
ever matters: CAMS EAC4 aerosol variant is the clean fix.

## Workflow

```bash
XR_PY=/global/homes/m/mpowell/.conda/envs/xr_env/bin/python

# 1. ERA5 via LS2D (CDS may queue; re-run until it completes), then symlink
$XR_PY preprocessing/goamazon_ls2d_input.py
ln -sf $SCRATCH/GOAMAZON_LES/shared_data/goamazon_ls2d_input.nc data/

# 2. Raytracer grid/memory fit test (4 candidates, one hbm80g debug node)
python setup_runs.py --fit-test
./submit_fit_test.sh
# then: peak memory in fit_test/*/raytracer/gpu_mem.log; pick grid,
# update GRID_PRESETS default / --grid accordingly

# 3. Production: 4 reps x {2stream, raytracer}
python setup_runs.py [--grid PRESET]
./submit_production.sh
```

Runs land in `$SCRATCH/GOAMAZON_LES/{fit_test,debug,base}/`.
Raytracer needs `--constraint="gpu&hbm80g"`; 2stream plain `gpu`.
Long raytracer runs are preempt-QOS candidates (>2 h single-GPU jobs).

## Caveats / known physics gaps

- **Radiation hydrometeor categories mimic SCREAM/P3** via
  `[thermo] swqsqg_to_rad=1` (mpowell-local feature, this case): the
  radiation ciwp is qi (saturation-adjustment ice) + qs + qg, matching
  P3's single all-frozen ice category being radiatively active. Rain is
  invisible in both models. Implemented in `calc_radiation_fields` /
  `calc_radiation_columns` (thermo_moist.cxx + .cu), the single choke
  point feeding 2stream, raytracer, and column stats — so both RT modes
  see identical clouds by construction. Default is 0 (upstream behavior);
  requires swmicro=nsw6.
- Residual difference vs SCREAM: effective radii. SCREAM uses P3's
  prognostic-PSD rel/rei; MicroHH derives rei internally from ciwp with
  a monodisperse assumption capped at 10-180 um
  (`effective_radius_and_ciwp_to_gm2`). Lumping qs/qg into ciwp pushes
  anvil rei toward the (snow-appropriate) large end of that range.
- Nc0=200e6 in nsw6 — GoAmazon IOP2 (Sep-Oct) is the polluted dry-to-wet
  transition; revisit if droplet number matters.
- Nudge targets above the IOP top follow ERA5; LS tendencies taper to
  zero over the top 1 km of IOP coverage.
- Fit-test memory is allocation-dominated (mostly clear sky at t<1800 s);
  wall-time under storm clouds only comes from a production rep.
- The 30-min fit test cannot reach convective onset (~early afternoon);
  a timed-out production rep still gives the timing calibration.

## Provenance

- IOP file: `/global/cfs/cdirs/e3sm/inputdata/atm/cam/scam/iop/GOAMAZON_singlepulse_iopfile_4scam.nc`
- DP-SCREAM case scripts: `~/repos/3D_RT_DPSCREAM/scripts/case_config/`
- ERA5 cache: `/pscratch/sd/m/mpowell/LS2D_ERA5/goamazon/`
- van Genuchten table: `microhh/misc/van_genuchten_parameters.nc`
  (the CASS copy on scratch was archived to HPSS and deleted)

## Setup on a new machine

### 1. MicroHH itself

```bash
git clone --recursive -b mpowell-local git@github.com:magpowell/microhh.git
```

`--recursive` pulls `rte-rrtmgp-cpp` and its own `rte-rrtmgp`/`rrtmgp-data`
submodules — no separate radiation-data download needed. Build the GPU
target per the root `README.md` / `config/` (Perlmutter uses
`config/default.cmake`; write a new `config/<machine>.cmake` for anywhere
else — paths to CUDA, NetCDF, HDF5, cuRAND/cuFFT are machine-specific).

This branch includes the `[thermo] swqsqg_to_rad` radiation feature (adds
nsw6 qs+qg to the radiation cloud ice, SCREAM/P3 category parity) that both
`config/goamazon_base.ini` and the ARM97SD sibling case require — it's in
`include/thermo_moist.h` + `src/thermo_moist.cxx`/`.cu`, already on this
branch, no extra patching needed after clone + build.

### 2. LS2D (ERA5 preprocessing)

```bash
git clone https://github.com/LS2D/LS2D.git
```

Then edit the hardcoded path in **`preprocessing/goamazon_ls2d_input.py:28`**
(`sys.path.insert(0, '...')`) to point at your clone.

### 3. CDS/Copernicus API credentials

LS2D needs `~/.cdsapirc` with a valid Copernicus Climate Data Store API key
(free account at https://cds.climate.copernicus.eu). The model-level ERA5
request (`reanalysis-era5-complete`, MARS-backed) can take minutes to
hours to queue; surface/pressure-level requests are usually fast. See the
case's dev history for a direct-pickle fallback if `ls2d.download_era5`
raises on a still-pending request (LS2D bails on the first pending request
instead of checking the others).

### 4. Python environment

`netCDF4`, `xarray`, `numpy` for the input generator; LS2D additionally
needs `cdsapi`, `matplotlib`. This project ran everything through a conda
env — no `environment.yml` shipped, just install the above into whatever
env you point `XR_PY` at (see below).

### 5. IOP forcing file

`GOAMAZON_singlepulse_iopfile_4scam.nc` (Tian & Zhang 2025) is an E3SM
inputdata file, relative path `atm/cam/scam/iop/` under whatever
`DIN_LOC_ROOT` your E3SM checkout uses. On Perlmutter it's mounted at
`/global/cfs/cdirs/e3sm/inputdata/...`; elsewhere, fetch it through your
own E3SM inputdata mechanism (e.g. `check_input_data` in an E3SM
case, or ask whoever supplied the forcing).

### 6. Hardcoded paths to edit

All NERSC/personal-account absolute paths, by file and line:

| File | Line | What |
|---|---|---|
| `preprocessing/goamazon_ls2d_input.py` | 28 | LS2D clone location |
| `preprocessing/goamazon_ls2d_input.py` | 53 | `era5_path` (LS2D's ERA5 cache dir) |
| `setup_runs.py` | 26 | `MICROHH_DIR` |
| `setup_runs.py` | 31 | `XR_PY` (python interpreter with netCDF4/xarray) |
| `setup_runs.py` | 33 | `IOP_FILE` |

`setup_runs.py`'s `SCRATCH` (line ~28) already reads `$SCRATCH` with a
personal-path fallback — just `export SCRATCH=...` rather than editing it.
All `sbatch_*.sh` / `submit_*.sh` scripts assume Slurm with `gpu_shared`
(2 GPUs/job, `--cpus-per-task=32`, 48 h `MaxWall`) and `gpu_debug`/`hbm80g`
constraints — adjust QOS/constraint names for a non-Perlmutter Slurm
cluster, or the equivalent for a different scheduler entirely.
