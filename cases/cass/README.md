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

# Experiments
python experiments/no_aerosols/setup_no_aerosols.py
python experiments/no_aerosols_zero_wind/setup_no_aerosols_zero_wind.py
python experiments/cs_veg/setup_cs_veg.py
python experiments/soil_moisture/setup_soil_moisture.py
python experiments/wind_u/setup_wind_u.py
python experiments/sw_scale/setup_sw_scale.py

# mean_state_nudge: sequential — requires no_aerosols_zero_wind 2stream to have completed first
# (see Mean-state nudge section below)
```

Each script creates `$SCRATCH/CASS_LES/.../rep_01..04/` with:
- `cass.ini` — merged ini for that run
- `cass_input.nc` — initial/forcing/aerosol data (fast: reads cached composites)
- Symlinks to `microhh`, radiation coefficients, shared data files

All scripts support `--dry-run` to preview without writing anything.

### 2. Submit jobs (Slurm)

```bash
# All reps of an experiment at once
bash base/submit_base.sh
bash experiments/no_aerosols/submit_no_aerosols.sh
bash experiments/no_aerosols_zero_wind/submit_no_aerosols_zero_wind.sh
bash experiments/cs_veg/submit_cs_veg.sh
bash experiments/soil_moisture/submit_soil_moisture.sh
bash experiments/wind_u/submit_wind_u.sh
bash experiments/sw_scale/submit_sw_scale.sh

# mean_state_nudge (after extracting nudge profiles — see below)
bash experiments/mean_state_nudge/submit_mean_state_nudge.sh --timescale 3600
```

### 3. Single debug run (debug QOS, ~15-30 min for 64×64 grid)

```bash
# Most experiments: setup + submit in one step
bash experiments/no_aerosols/submit_debug_no_aerosols.sh
bash experiments/no_aerosols_zero_wind/submit_debug_no_aerosols_zero_wind.sh
bash experiments/soil_moisture/submit_debug_soil_moisture.sh
bash experiments/wind_u/submit_debug_wind_u.sh         # all 5 values
bash experiments/wind_u/submit_debug_wind_u.sh 5.0     # single value
bash experiments/sw_scale/submit_debug_sw_scale.sh

# mean_state_nudge debug (requires nudge_profiles.nc first — see below)
bash experiments/mean_state_nudge/submit_debug_mean_state_nudge.sh \
    --timescale 3600 \
    --nudge-profiles $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc
```

Check the log: `$SCRATCH/CASS_LES/logs/debug-<JOBID>.out`

---

## Mean-state nudge experiment

This experiment keeps the raytracer on the same mean thermodynamic state as the
no_aerosols_zero_wind 2stream runs, to isolate the direct 3D RT effect from mean-state
divergence. It has a **sequential dependency**: the 2stream runs must finish before the
raytracer is set up.

```
Step 1: Run no_aerosols_zero_wind 2stream (all 4 reps)
        bash experiments/no_aerosols_zero_wind/submit_no_aerosols_zero_wind.sh
        # the submit script submits both 2stream and raytracer; only 2stream is needed here

Step 2: Extract nudge profiles (login node, fast)
        python experiments/mean_state_nudge/extract_nudge_profiles.py \
            --run-dir $SCRATCH/CASS_LES/experiments/no_aerosols_zero_wind/2stream \
            --output  $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc

Step 3: Debug run to check stability (try 3600 s first, then shorter)
        bash experiments/mean_state_nudge/submit_debug_mean_state_nudge.sh \
            --timescale 3600 \
            --nudge-profiles $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc

Step 4: Production runs (once stable timescale is confirmed)
        python experiments/mean_state_nudge/setup_mean_state_nudge.py \
            --timescale 3600 \
            --nudge-profiles $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc
        bash experiments/mean_state_nudge/submit_mean_state_nudge.sh --timescale 3600
```

**What `extract_nudge_profiles.py` does**: reads `cass.column.*.0000000.nc` from each of the
4 no_aerosols_zero_wind 2stream rep dirs, averages `thl(z, t)` and `qt(z, t)` across reps,
interpolates to the `time_ls` grid (15 hourly points), and writes `nudge_profiles.nc`.

**Nudgefac note**: the nudging timescale applies to u, v, thl, and qt simultaneously — there is
no per-variable timescale. Start at 3600 s and decrease carefully; the code has no tendency
limiter (Bart van Stratum, pers. comm.), so overshooting is possible at short timescales.

---

## Wind-u sweep experiment

Varies the constant domain-mean wind speed u ∈ {0, 2.5, 5, 7.5, 10} m/s with aerosols off and
v=0, to isolate the effect of wind speed on the 3D vs 1D RT LWP difference.

**Key design choice — `swlspres=uflux`**: the geostrophic forcing (`swlspres=geo`) is replaced
with `swlspres=uflux` which applies a body force each timestep to enforce the exact domain-mean
u. With `swlspres=geo`, the ERA5 composite geostrophic wind (~0 m/s for CASS days) plus Coriolis
drains the imposed wind regardless of the nudging timescale. `uflux` gives a clean, apple-to-apple
comparison. Only v is nudged (toward 0, τ=3 h) to suppress Coriolis rotation.

**u=0 is a new run** (not `no_aerosols_zero_wind`) — the uflux forcing must be consistent across
all points in the sweep.

```
Step 1: Set up all 5 wind values (u=0, 2.5, 5, 7.5, 10 m/s)
        python experiments/wind_u/setup_wind_u.py

Step 2: Debug one or two values first
        bash experiments/wind_u/submit_debug_wind_u.sh 0.0 5.0
        # Verify MOM is ~constant in cass.out (not decaying) — confirms uflux is working

Step 3: Production
        bash experiments/wind_u/submit_wind_u.sh
```

5 values × 8 runs = 40 runs total.

---

## sw_scale experiment

Runs the raytracer with `swscalesfc_to_2str=true`. At each radiation timestep the
surface SW downwelling flux is multiplied by `mean(sw_2stream_sfc) / mean(sw_raytracer_sfc)`,
preserving 3D heterogeneity but enforcing the same domain-mean forcing as the 1D solution.
No separate 2stream runs are needed — the 1D fluxes are computed internally every timestep.
The scale factor is output as `sw_scale_factor` in the `radiation` stats group.

```
Step 1: Set up run dirs
        python experiments/sw_scale/setup_sw_scale.py

Step 2: Debug first
        bash experiments/sw_scale/submit_debug_sw_scale.sh
        # Check sw_scale_factor in cass.default.0000000.nc — should be ~1 in clear sky,
        # < 1 when 3D RT scatters more into the domain than 1D, etc.

Step 3: Production
        bash experiments/sw_scale/submit_sw_scale.sh
```

Requires the `swscalesfc_to_2str` feature on branch `mpowell-local`.
4 raytracer reps only (no 2stream runs needed for this experiment).

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
