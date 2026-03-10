# CASS LES Project Directive

## Goal
Understand when 3D vs. 1D radiative transfer produces the largest differences in cloud LWP.
Use the CASS composite case (ARM SGP, July 24, multi-year composite) as the baseline.
Run paired 2stream (1D) and raytracer (3D) simulations across five experiment sweeps:
1. `no_aerosols` — aerosols off, standard winds (isolates aerosol direct effect vs. base)
2. `no_aerosols_zero_wind` — aerosols off, zero winds (zero-wind control; also provides nudge profiles for mean_state_nudge)
3. `cs_veg` — skin heat capacity (J&M replication)
4. `soil_moisture` — soil moisture nudging to a fixed profile
5. `mean_state_nudge` — aggressive thl/qt nudging of raytracer toward no_aerosols_zero_wind 2stream mean state, isolating the direct 3D RT effect from the mean-state-divergence pathway

---

## Repository Structure (target)

```
cases/cass/
  base/
    setup_base.py          # sets up $SCRATCH/CASS_LES/base/{2stream,raytracer}/rep_{01..04}
    sbatch_base.sh         # 4-per-node sbatch for base runs
    submit_base.sh         # submits base jobs
    submit_debug_base.sh
  experiments/
    no_aerosols/
      setup_no_aerosols.py   # sets up all no_aerosols scratch dirs
      sbatch_no_aerosols.sh  # 4-per-node sbatch
      submit_no_aerosols.sh  # submits all jobs
      submit_debug_no_aerosols.sh
    cs_veg/
      setup_cs_veg.py        # sets up all cs_veg scratch dirs
      sbatch_cs_veg.sh       # 4-per-node sbatch
      submit_cs_veg.sh       # submits all jobs
      submit_debug_cs_veg.sh
    soil_moisture/
      setup_soil_moisture.py
      sbatch_soil_moisture.sh
      submit_soil_moisture.sh
      submit_debug_soil_moisture.sh
    no_aerosols_zero_wind/
      setup_no_aerosols_zero_wind.py
      sbatch_no_aerosols_zero_wind.sh
      submit_no_aerosols_zero_wind.sh
      submit_debug_no_aerosols_zero_wind.sh
    mean_state_nudge/
      extract_nudge_profiles.py  # averages no_aerosols_zero_wind 2stream column stats → timedep nudge profiles
      setup_mean_state_nudge.py
      sbatch_mean_state_nudge.sh
      submit_mean_state_nudge.sh
      submit_debug_mean_state_nudge.sh
  shared/
    preprocessing/         # run-once ERA5/CAMS pipeline (never symlinked into run dirs)
      cass_ls2d_input.py
      compute_cams_composite.py
      download_cams.py
      download_cass_data.sh
      loop_cams_download.sh
      sbatch_era5_download.sh
      save_era5_profiles.py
      purge_rejected_pickles.py
    config/                # ini templates merged by setup scripts
      cass_base.ini
      cass_2stream.ini
      cass_raytracer.ini
    data/                  # static nc files symlinked into run dirs
      van_genuchten_parameters.nc
      cass_ls2d_input.nc     # symlink → $SCRATCH/CASS_LES/shared_data/
      cass_cams_composite.nc # symlink → $SCRATCH/CASS_LES/shared_data/
    cass_input.py          # per-run input.nc generator (symlinked into each run dir)
    cass_utils.py
    cleanup_run.py
  analysis/
    explore.ipynb
  project_directive.md
```

> Files in `shared/` are symlinked into each scratch run directory (never copied).
> Do not leave stale symlinks. Re-run setup scripts after any restructuring.

---

## Scratch Structure (target)

```
$SCRATCH/CASS_LES/
  base/
    2stream/rep_01/ ... rep_04/
    raytracer/rep_01/ ... rep_04/
  experiments/
    no_aerosols/
      2stream/rep_01/ ... rep_04/
      raytracer/rep_01/ ... rep_04/
    cs_veg/
      cs_veg_{VALUE}/      # VALUE = 0, 42000, 420000, 4200000, 42000000
        2stream/rep_01/ ... rep_04/
        raytracer/rep_01/ ... rep_04/
    soil_moisture/
      theta_{VALUE}/       # VALUE = 0.1, 0.2, 0.3, 0.4
        2stream/rep_01/ ... rep_04/
        raytracer/rep_01/ ... rep_04/
    no_aerosols_zero_wind/
      2stream/rep_01/ ... rep_04/
      raytracer/rep_01/ ... rep_04/
    mean_state_nudge/
      nudge_{TIMESCALE}s/  # e.g. nudge_3600s, nudge_1800s, nudge_900s
        raytracer/rep_01/ ... rep_04/
        # NOTE: no 2stream subdir — control is no_aerosols_zero_wind/2stream
```

- Each rep directory is self-contained: symlinks to shared data + resources, its own `cass.ini` and `cass_input.nc`.
- Old flat dirs (`CASS_LES_2stream`, `CASS_LES_raytracer`) are superseded and can be removed.

---

## Run Structure

- **4 reps per radiation type per parameter value** (rep_01–rep_04 = rndseed 1–4)
- **2 radiation types**: `2stream` (1D) and `raytracer` (3D)
- 4 sims batched per GPU node (1 node = 4 GPUs), following tijhuis/sensitivity pattern
- Always via Slurm (never interactive for production runs)
- `hbm80g` constraint required for raytracer runs; standard `gpu` for 2stream

---

## Global INI Changes (apply everywhere, including base)

| Parameter | Old value | New value | Reason |
|-----------|-----------|-----------|--------|
| `c_veg`   | 0.95      | 1.0       | Route all SEB through veg tile so cs_veg is active |
| `swaerosol` | true    | experiment-specific (see below) | |

---

## Base Case Configuration

- Uses standard CASS composite winds (u, v from ERA5 nudging targets)
- Aerosols: **ON** (`swaerosol=true`)
- `c_veg=1.0`, `cs_veg=0` (no skin heat capacity)
- No soil moisture nudging

---

## Experiment: no_aerosols

**Science**: Isolate the aerosol direct radiative effect on cloud LWP by comparing with the base case.

**Configuration** (on top of base):
- `[aerosol] swaerosol = false`
- Standard winds (same as base — no `--zero-winds`)
- No soil moisture nudging
- `cs_veg = 0` (same as base)

**Run structure**: 4 reps × 2 RT = 8 runs total (same as base)

**input.nc changes**: none — `cass_input.py` called with no flags

---

## Experiment: cs_veg Sweep

**Parameter values** (cs_veg in J m⁻² K⁻¹):

| Label         | cs_veg value |
|---------------|-------------|
| cs_veg_0      | 0           |
| cs_veg_42000  | 41,840      |
| cs_veg_420000 | 418,400     |
| cs_veg_4200000| 4,184,000   |
| cs_veg_42000000| 41,840,000 |

**INI overlays** (on top of base):
- `[land_surface] cs_veg = <VALUE>`
- `[land_surface] c_veg = 1.0`
- `[aerosol] swaerosol = false`

**Wind condition**: zero winds (see below)

**input.nc changes**: none — `cs_veg` is ini-only

---

## Experiment: Soil Moisture Sweep

**Science**: Isolate effect of soil moisture on LWP via Bowen ratio and thermal conductivity.

**Parameter values** (uniform theta through 4 soil layers):

| Label       | theta_nudge profile        |
|-------------|---------------------------|
| theta_0.1   | [0.1, 0.1, 0.1, 0.1]     |
| theta_0.2   | [0.2, 0.2, 0.2, 0.2]     |
| theta_0.3   | [0.3, 0.3, 0.3, 0.3]     |
| theta_0.4   | [0.4, 0.4, 0.4, 0.4]     |

**INI overlays** (on top of base):
- `[land_surface] swnudge_theta = true`
- `[land_surface] nudge_theta_timescale = 3600`
- `[aerosol] swaerosol = false`

**Wind condition**: zero winds (see below)

**input.nc changes**: `theta_nudge[4]` variable must be added to the `soil` group AND
`theta_soil` must be initialised to the target value.
Implementation: `--theta-nudge FLOAT` argument to `cass_input.py`. When provided,
(a) writes `theta_soil = [VALUE]*4` (uniform initial profile = nudge target), and
(b) writes `theta_nudge = [VALUE]*4` into the soil group.
The run therefore **starts at** the target moisture and is held there by the nudging.
Default (no flag): ERA5 composite theta_soil, no theta_nudge written (existing runs unaffected).

**Note**: The soil moisture nudging feature (branch `mpowell-local`) has been compiled and
**validated** — theta_soil initialises at the target and is held there by nudging. GPU path
still unverified.

---

## Experiment: no_aerosols_zero_wind

**Science**: Zero-wind control case alongside no_aerosols. Eliminates mean wind and wind shear
so that surface-cloud coupling is the dominant signal. Also serves as the source of nudge
profiles for the mean_state_nudge experiment.

**Configuration** (on top of base):
- `[aerosol] swaerosol = false`
- Zero winds (`--zero-winds` flag to `cass_input.py`)
- No soil moisture nudging, `cs_veg = 0`

**Run structure**: 4 reps × 2 RT = 8 runs (same as no_aerosols)

**Dual purpose**:
1. Direct science result: zero-wind 2stream vs. raytracer LWP differences
2. Provides nudge profiles for mean_state_nudge (extract from 2stream column output)

---

## Experiment: mean_state_nudge

**Science**: In Tijhuis et al. 2024, 3D and 1D simulations diverge in mean qt even with a 3h
nudging timescale. This experiment asks: if the raytracer is forced onto the same instantaneous
mean thermodynamic state as the 2stream, do LWP differences persist? A "yes" implicates the
direct radiative effect of 3D geometry on the cloud; a "no" implicates the mean-state-divergence
pathway.

**Design**:
1. The no_aerosols_zero_wind 2stream reps (rep_01–04) run freely.
2. Domain-mean `thl(z, t)` and `qt(z, t)` are extracted from their column output and
   averaged across all 4 reps → ensemble-mean 1D profiles, noise-suppressed.
3. Those profiles become the timedep nudge target for the raytracer runs, which are nudged
   aggressively toward the 2stream mean state.

**Sequential dependency**: no_aerosols_zero_wind 2stream must complete before raytracer setup.
The comparison is: no_aerosols_zero_wind 2stream (free) vs. mean_state_nudge raytracer (constrained).
Both use identical boundary conditions (no aerosols, zero winds); only the RT scheme and nudge differ.

**INI overlays** (raytracer only, on top of no_aerosols base):
- `[force] nudgelist = u,v,thl,qt`
- `[force] timedeplist_nudge = u,v,thl,qt`
- nudgefac set via input.nc (see below)

**Key code constraint**: `nudgefac` (1/s) is a single height-varying profile in the `init` group
of input.nc, shared across all nudged variables. No per-variable timescale. The current baseline
is uniform at 1/10800 s⁻¹ (3h). The minimum stable aggressive timescale must be found
experimentally — the code has no tendency limiter (Bart van Stratum, pers. comm.).

**input.nc changes**: A new `--nudge-thermo PATH TIMESCALE` option in `cass_input.py`. When provided:
- Reads ensemble-mean thl/qt profiles from the specified stats file (output of `extract_nudge_profiles.py`)
- Writes `thl_nudge[time, z]` and `qt_nudge[time, z]` into the `timedep` group
- Overwrites `nudgefac` in the `init` group with `1/TIMESCALE` (uniform, in s⁻¹)

**Post-processing script** (`extract_nudge_profiles.py`):
- Reads column output from all 4 no_aerosols_zero_wind 2stream reps
- Averages `thl[t, z]` and `qt[t, z]` across reps
- Writes output in the timedep-compatible format expected by `cass_input.py`
- Output: `$SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc`

**Aerosols**: off (`swaerosol = false`) — same as no_aerosols

**Wind condition**: zero winds (same as other experiments)

**Run structure**: raytracer only, 4 reps per timescale value. Start with a single debug run at
3600s to verify stability before trying shorter timescales.

**Status**: all scripts written and ready. See `README.md` for step-by-step workflow.

---

## Zero Wind Condition (applies to all experiments, NOT the base case)

**Goal**: eliminate mean wind and wind shear so that surface-cloud coupling is the dominant
signal. Later experiments will sweep mean wind and shear independently.

**Implementation**:
- In `cass_input.py`: when a `--zero-winds` flag is set, write `u=0`, `v=0` in the `init`
  group and `u_nudge=0`, `v_nudge=0` in the `timedep` group (zero arrays, same shape).
- Keep `swlspres=geo` and `fc=8.5e-5` in ini — Coriolis rotation is retained, but geo
  target wind is zero so there is no mean pressure gradient driving.
- Keep `swnudge=1` — nudging is retained, target is just zero.

---

## Shared Preprocessing (run once, output cached)

These are expensive and do not change between experiments:
- `cass_ls2d_input.py` → `cass_ls2d_input.nc` (ERA5 composite, radiation/soil profiles)
- `download_cams.py` → raw CAMS EAC4 files in `$SCRATCH/LS2D_CAMS/cass/`
- `compute_cams_composite.py` → `cass_cams_composite.nc` (CAMS aerosol composite on native levels)

**Order**: download CAMS first (`download_cams.py`), then composite (`compute_cams_composite.py`).
Both output files are symlinked into `shared/data/` and from there into every run directory.
`cass_input.py` reads the cached composites and is fast (no CAMS/ERA5 API calls).

---

## Soil Moisture Nudging Validation

**Status: VALIDATED (CPU path)**

Confirmed: theta_soil initialises at the target value and is held there by the nudging.

**Remaining known gaps**:
- No stats output — `theta_nudge_tend` diagnostic still missing
- GPU path compiled and verified
- Only works with `sw_homogeneous=true`
- `theta_nudge` is read inside `if (sw_homogeneous)` block — crash if not set

## LSM Spin-up (open question)

Whether/how to spin up the LSM prior to production runs is unresolved.
To be discussed with advisor before science runs begin.

---

## Key Code Locations

| File | Purpose |
|------|---------|
| `shared/config/cass_base.ini` | Shared base configuration |
| `shared/config/cass_2stream.ini` | 2stream radiation + LSM overlay |
| `shared/config/cass_raytracer.ini` | Raytracer radiation + LSM overlay |
| `shared/cass_input.py` | Generates `cass_input.nc` (init, timedep, radiation, soil groups) |
| `shared/preprocessing/cass_ls2d_input.py` | ERA5 preprocessing → `cass_ls2d_input.nc` |
| `shared/preprocessing/download_cams.py` | CAMS aerosol download → `$SCRATCH/LS2D_CAMS/` |
| `shared/preprocessing/compute_cams_composite.py` | CAMS composite → `cass_cams_composite.nc` |
| `shared/data/van_genuchten_parameters.nc` | LSM soil type lookup table |
| `base/setup_base.py` | Sets up base scratch dirs |
| `experiments/no_aerosols/setup_no_aerosols.py` | Sets up no_aerosols scratch dirs |
| `experiments/cs_veg/setup_cs_veg.py` | Sets up cs_veg scratch dirs |
| `experiments/soil_moisture/setup_soil_moisture.py` | Sets up soil_moisture scratch dirs |
| `experiments/mean_state_nudge/extract_nudge_profiles.py` | Post-processes no_aerosols 2stream stats → timedep nudge profiles |
| `experiments/mean_state_nudge/setup_mean_state_nudge.py` | Sets up mean_state_nudge scratch dirs (raytracer only) |
| `shared/sbatch_debug.sh` | Single-GPU debug sbatch (30 min, debug QOS) |
| MicroHH source: `src/boundary_surface_lsm.cxx` | Soil moisture nudging (CPU) |
| MicroHH source: `src/boundary_surface_lsm.cu` | Soil moisture nudging (GPU) |
| MicroHH source: `include/soil_kernels.h` | `nudge_theta()` CPU kernel |
| MicroHH source: `src/force.cxx:468` | `timedep_dim = "time_ls"` — all timedep nudge profiles share this time dimension |

---

## Status

### Done
- [x] CASS baseline runs exist at `$SCRATCH/CASS_LES_2stream` and `$SCRATCH/CASS_LES_raytracer`
- [x] Soil moisture nudging feature implemented (branch `mpowell-local`), compiled, validated (CPU + GPU)
- [x] Project structure designed and implemented (`base/`, `experiments/`, `shared/`, `analysis/`)
- [x] `shared/data/` populated: `cass_snd.txt`, `cass_sfc.txt`, `cass_lsf.txt`, `van_genuchten_parameters.nc`
- [x] `c_veg=1.0` set in `shared/config/cass_2stream.ini` and `shared/config/cass_raytracer.ini`
- [x] `shared/cass_input.py` updated with `--zero-winds` and `--theta-nudge FLOAT` flags
- [x] `base/setup_base.py`, `base/sbatch_base.sh`, `base/submit_base.sh`, `base/submit_debug_base.sh` written
- [x] `experiments/no_aerosols/` — all scripts written (setup, sbatch, submit, submit_debug)
- [x] `experiments/cs_veg/setup_cs_veg.py`, `sbatch_cs_veg.sh`, `submit_cs_veg.sh`, `submit_debug_cs_veg.sh` written
- [x] `experiments/soil_moisture/setup_soil_moisture.py`, `sbatch_soil_moisture.sh`, `submit_soil_moisture.sh`, `submit_debug_soil_moisture.sh` written
- [x] `experiments/mean_state_nudge/` — all scripts written: `extract_nudge_profiles.py`, `setup_mean_state_nudge.py`, `sbatch_mean_state_nudge.sh`, `submit_mean_state_nudge.sh`, `submit_debug_mean_state_nudge.sh`
- [x] `shared/cass_input.py` updated with `--nudge-thermo PATH TIMESCALE` flag
- [x] `shared/sbatch_debug.sh` written (single-GPU, debug QOS, 30 min)
- [x] `shared/data/README.md` written
- [x] ERA5 composite preprocessing complete: 119 composite days (1997–2009), cached in `/pscratch/sd/m/mpowell/LS2D_ERA5/cass/`
- [x] `cass_ls2d_input.nc` written to `/pscratch/sd/m/mpowell/CASS_LES/shared_data/` and symlinked into `shared/data/`
- [x] `cass_ls2d_input.py`, `cass_utils.py`, `shcu_sgp_summer_97to09.nc` moved to `shared/preprocessing/`; hardcoded path in `cass_utils.py` updated
- [x] `compute_cams_composite.py` written and run: 60 composite days (2003–2009), output at `$SCRATCH/CASS_LES/shared_data/cass_cams_composite.nc`, symlinked into `shared/data/`
- [x] `cass_input.py` refactored to read `cass_cams_composite.nc` — fast, no CAMS API calls at run time
- [x] NaN crash (previously at t=4620s with `swaerosol=true`) fixed in MicroHH source
- [x] **All debug runs successful** — base, no_aerosols, cs_veg_42000, soil_moisture/theta_0p1, soil_moisture/theta_0p4 (both 2stream and raytracer); runs timed out at debug wall limit, not crashed

### Next Steps (in order)
1. **Resolve LSM spin-up question** (discuss with advisor before science runs)
2. **Run base case** (4 × 2stream + 4 × raytracer) via `base/submit_base.sh`
3. **Run no_aerosols** (4 × 2stream + 4 × raytracer = 8 runs) via `experiments/no_aerosols/submit_no_aerosols.sh`
4. **Run cs_veg sweep** (5 values × 8 runs = 40 runs) via `experiments/cs_veg/submit_cs_veg.sh`
5. **Run soil_moisture sweep** (4 values × 8 runs = 32 runs) via `experiments/soil_moisture/submit_soil_moisture.sh`
6. **Run no_aerosols_zero_wind** (4 × 2stream + 4 × raytracer = 8 runs) via `experiments/no_aerosols_zero_wind/submit_no_aerosols_zero_wind.sh`
7. **mean_state_nudge**: extract nudge profiles from no_aerosols_zero_wind 2stream output, run debug raytracer at 3600s to verify stability, then find minimum stable timescale

### Observations
- Raytracer runs show a **larger SEB residual** than 2stream runs — cause unknown; flag when analysing results
- NaN crash at t=4620s with aerosols **resolved** — fixed in MicroHH source code; all debug runs now pass that point cleanly

### Notes / Gotchas
- Always re-run setup scripts after reorganizing — stale symlinks will silently break runs
- `cass_input.py` reads `cass.ini` from CWD — must be called from within the run dir (setup scripts handle this)
- Raytracer runs need `--constraint=gpu&hbm80g`; 2stream only needs `--constraint=gpu`
- `nudge_theta_timescale = 3600` everywhere (debug and production); theta_soil also initialised to target so convergence is immediate
- cs_veg `0` is a valid value (no skin heat capacity buffer)
- `rndseed` in `[fields]` must differ across reps: use 1, 2, 3, 4
- `surface_an_agg.nc` (present in cass root and old run dirs) — not referenced by any script, safe to ignore
- `shared/data/cass_ls2d_input.nc` must exist before running any setup script
