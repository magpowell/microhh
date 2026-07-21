# CASS LES — Launch Guide

**Goal**: understand how 3D (raytracer) vs 1D (2stream) radiative transfer affects
shallow cumulus cloud LWP over land. See `project_directive.md` for full context.

---

## Prerequisites

Preprocessing (run once before any setup script):

| File | How to generate |
|------|----------------|
| `shared/data/cass_ls2d_input.nc` | `python shared/preprocessing/cass_ls2d_input.py` (ERA5 composite) |
| `shared/data/cass_cams_composite.nc` | `python shared/preprocessing/compute_cams_composite.py` (CAMS aerosol composite) |

Both live on scratch, symlinked into `shared/data/`.
Run `bash shared/setup_shared_data.sh` to repair broken symlinks.

---

## Workflow

### 1. Set up run directories (login node)

```bash
python experiments/no_aerosols/setup_no_aerosols.py
python experiments/no_aerosols_zero_wind/setup_no_aerosols_zero_wind.py
python experiments/rs_scale/setup_rs_scale.py
python experiments/sw_scale/setup_sw_scale.py
python experiments/wind_geo/setup_wind_geo.py
```

All scripts support `--dry-run` and `--debug` (64x64 grid, single rep).

### 2. Submit jobs

```bash
bash experiments/<name>/submit_<name>.sh        # production
bash experiments/<name>/submit_debug_<name>.sh   # debug QOS
```

### 3. Debug runs

Debug uses `shared/sbatch_debug.sh` (debug QOS, single GPU, ~15-30 min for 64x64).
Log: `$SCRATCH/CASS_LES/logs/debug-<JOBID>.out`

### 4. Restart after a TIMEOUT

If a run hits the wall before reaching the completion timestamp:

```bash
bash experiments/<name>/submit_restart_<name>.sh [2stream|raytracer]
```

The restart script auto-detects the latest savetime, verifies all
prognostic vars are present, and patches `[time] starttime` in `cass.ini`.
It does **not** wipe restart files or call `microhh init`.

**Post-processing after restart**: when converting binary dumps with
`3d_to_nc.py` or `cross_to_nc.py`, pass `-t0 0` so the full simulation
timeline is converted (the patched `cass.ini` would otherwise start the
output time axis at the restart point).

> **Tracer overhead**: experiments with the Couvreux passive tracer
> (e.g. `no_aerosols_zero_wind_v2`) run ~25 % slower than pre-tracer
> versions.  Either bump 2stream wall to 12 h / raytracer to 28 h, or
> use the restart workflow.

---

## Constraints

| RT type | `--constraint` | Reason |
|---------|---------------|--------|
| 2stream | `gpu` | Standard GPU node |
| raytracer | `gpu&hbm80g` | Requires HBM memory |

---

## Key directories

| Path | Contents |
|------|----------|
| `shared/config/` | INI templates (`cass_base.ini`, `cass_2stream.ini`, `cass_raytracer.ini`) |
| `shared/data/` | Symlinks to shared NetCDF files on scratch |
| `shared/preprocessing/` | Run-once scripts (ERA5, CAMS) |
| `$SCRATCH/CASS_LES/experiments/` | Experiment run dirs |
| `$SCRATCH/CASS_LES/debug/` | Debug run dirs |
| `$SCRATCH/CASS_LES/logs/` | Slurm stdout/stderr |

---

## After runs complete

Stats: `$RUN_DIR/cass.default.0000000.nc` (sampled every 300 s).
Key groups: `lsm` (land surface), `thermo`, `radiation`.
See `analysis/` for plotting and analysis tools.

---

## Archived experiments

Removed from repo 2026-04-08 (recoverable from git history).

| Experiment | Data |
|---|---|
| base | Deleted from scratch (superseded by no_aerosols) |
| cs_veg | HPSS at `/home/m/mpowell/CASS_LES/cs_veg/` (verified), scratch safe to delete |
| soil_moisture | HPSS at `/home/m/mpowell/CASS_LES/soil_moisture/` (verified) |
| mean_state_nudge | Deleted (re-derivable) |
| wind_u | HPSS at `/home/m/mpowell/CASS_LES/wind_u/` (verified), scratch safe to delete |
