# CASS LES — Launch Guide

**Goal**: understand when 3D (raytracer) vs 1D (2stream) radiative transfer produces the
largest differences in cloud LWP. See `project_directive.md` for full scientific context.

---

## Prerequisites

All preprocessing must be done once before any setup script will work:

| File | How to generate |
|------|----------------|
| `shared/data/cass_ls2d_input.nc` | `python shared/preprocessing/cass_ls2d_input.py` (ERA5 composite) |
| `shared/data/cass_cams_composite.nc` | `python shared/preprocessing/compute_cams_composite.py` (CAMS aerosol composite) |

Both files live on scratch and are symlinked into `shared/data/`. Run
`bash shared/setup_shared_data.sh` to repair broken symlinks.

---

## Typical workflow

### 1. Set up run directories (login node, fast)

```bash
# Base case
python base/setup_base.py

# Active experiments
python experiments/no_aerosols/setup_no_aerosols.py
python experiments/no_aerosols_zero_wind/setup_no_aerosols_zero_wind.py
python experiments/sw_scale/setup_sw_scale.py
python experiments/rs_scale/setup_rs_scale.py
python experiments/wind_geo/setup_wind_geo.py
```

Each script creates `$SCRATCH/CASS_LES/.../rep_01..04/` with:
- `cass.ini` — merged ini for that run
- `cass_input.nc` — initial/forcing/aerosol data (fast: reads cached composites)
- Symlinks to `microhh`, radiation coefficients, shared data files

All scripts support `--dry-run` to preview without writing anything.

### 2. Submit jobs (Slurm)

```bash
bash base/submit_base.sh
bash experiments/no_aerosols/submit_no_aerosols.sh
bash experiments/no_aerosols_zero_wind/submit_no_aerosols_zero_wind.sh
bash experiments/sw_scale/submit_sw_scale.sh
bash experiments/rs_scale/submit_rs_scale.sh
bash experiments/wind_geo/submit_wind_geo.sh
```

### 3. Single debug run (debug QOS, ~15-30 min for 64x64 grid)

```bash
bash experiments/no_aerosols/submit_debug_no_aerosols.sh
bash experiments/no_aerosols_zero_wind/submit_debug_no_aerosols_zero_wind.sh
bash experiments/sw_scale/submit_debug_sw_scale.sh
bash experiments/rs_scale/submit_debug_rs_scale.sh
bash experiments/wind_geo/submit_debug_wind_geo.sh
```

Check the log: `$SCRATCH/CASS_LES/logs/debug-<JOBID>.out`

---

## sw_scale experiment

Runs the raytracer with `swscalesfc_to_2str=true`. At each radiation timestep the
surface SW downwelling flux is multiplied by `mean(sw_2stream_sfc) / mean(sw_raytracer_sfc)`,
preserving 3D heterogeneity but enforcing the same domain-mean forcing as the 1D solution.
No separate 2stream runs are needed — the 1D fluxes are computed internally every timestep.
The scale factor is output as `sw_scale_factor` in the `radiation` stats group.

Requires the `swscalesfc_to_2str` feature on branch `mpowell-local`.
4 raytracer reps only (no 2stream runs needed for this experiment).

---

## rs_scale experiment (Bowen ratio sweep)

Varies surface resistance via `[land_surface] rs_scale` to sweep the Bowen ratio.
Tests the prediction that alpha ~ (Bo + eps)/(1 + Bo). Bo diagnosed post-hoc from H/LE.
Zero winds, no aerosols, 5 values x 4 reps x 2 RT = 40 runs.

Requires `rs_scale` feature on branch `mpowell-local`.

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
| `$SCRATCH/CASS_LES/base/` | Base case run dirs |
| `$SCRATCH/CASS_LES/experiments/` | Experiment run dirs |
| `$SCRATCH/CASS_LES/debug/` | Debug/validation run dirs |
| `$SCRATCH/CASS_LES/logs/` | Slurm stdout/stderr |
| `$SCRATCH/CASS_LES/shared_data/` | Large files (cass_ls2d_input.nc, cass_cams_composite.nc) |

---

## After runs complete

Stats are in `$RUN_DIR/cass.default.0000000.nc` (sampled every 300 s).
Key groups: `lsm` (land surface, including soil `theta`), `thermo`, `radiation`.

See `analysis/explore.ipynb` for plotting.

---

## Archived experiments (removed 2026-04-08, recoverable from git history)

The following experiments were removed from the repo. Data status:

| Experiment | Data |
|---|---|
| cs_veg | HPSS TODO (4.7 TB on scratch) |
| soil_moisture | HPSS at `/home/m/mpowell/CASS_LES/soil_moisture/` (verified) |
| mean_state_nudge | Deleted (re-derivable from no_aerosols_zero_wind 2stream) |
| wind_u | HPSS TODO (production runs on scratch) |
