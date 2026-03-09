# CASS LES Project Directive

## Goal
Understand when 3D vs. 1D radiative transfer produces the largest differences in cloud LWP.
Use the CASS composite case (ARM SGP, July 24, multi-year composite) as the baseline.
Run paired 2stream (1D) and raytracer (3D) simulations across three experiment sweeps:
1. `no_aerosols` — aerosols off, standard winds (isolates aerosol direct effect vs. base)
2. `cs_veg` — skin heat capacity (J&M replication)
3. `soil_moisture` — soil moisture nudging to a fixed profile

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
  shared/
    preprocessing/         # run-once ERA5/CAMS pipeline (never symlinked into run dirs)
      cass_ls2d_input.py
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
      shcu_sgp_summer_97to09.nc
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
- `[land_surface] nudge_theta_timescale = 86400`
- `[aerosol] swaerosol = false`

**Wind condition**: zero winds (see below)

**input.nc changes**: `theta_nudge[4]` variable must be added to the `soil` group.
Implementation: add `--theta-nudge FLOAT` argument to `cass_input.py`. When provided,
writes a uniform 4-element `theta_nudge` array into the soil group. Default: not written
(existing runs unaffected).

**Note**: The soil moisture nudging feature (branch `mpowell-local`) has been compiled but
**not verified**. Validation is required before science runs. See Validation section.

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
- `download_cams.py` → CAMS aerosol pickles in `$SCRATCH/LS2D_CAMS/`

The `sbatch_cass_les.sh` job previously re-ran preprocessing inside each job. In the new
structure, preprocessing is separated: run once via `sbatch_era5_download.sh`, output
cached in the run directory or a shared location, and only `cass_input.py` is called per
run (it is fast and run-specific).

---

## Soil Moisture Nudging Validation Plan

The nudge feature (branch `mpowell-local`) must be validated before science runs:

1. Run a short debug job (debug QOS, 30 min, small grid 64×64) with
   `swnudge_theta=true`, `nudge_theta_timescale=3600` (aggressive nudge)
2. Confirm theta_soil converges to theta_nudge target within a few timescales
3. Check energy balance closure: nudging adds/removes heat — verify dH/dt is reasonable
4. Verify CPU path only first; GPU path needs separate check

**Known gaps in the implementation**:
- No stats output — add `theta_nudge_tend` diagnostic before science runs
- GPU path compiled but unexecuted
- Only works with `sw_homogeneous=true`
- `theta_nudge` is read inside `if (sw_homogeneous)` block — crash if not set

---

## Key Code Locations

| File | Purpose |
|------|---------|
| `shared/config/cass_base.ini` | Shared base configuration |
| `shared/config/cass_2stream.ini` | 2stream radiation + LSM overlay |
| `shared/config/cass_raytracer.ini` | Raytracer radiation + LSM overlay |
| `shared/cass_input.py` | Generates `cass_input.nc` (init, timedep, radiation, soil groups) |
| `shared/preprocessing/cass_ls2d_input.py` | ERA5 preprocessing → `cass_ls2d_input.nc` |
| `shared/preprocessing/download_cams.py` | CAMS aerosol download |
| `shared/data/van_genuchten_parameters.nc` | LSM soil type lookup table |
| `base/setup_base.py` | Sets up base scratch dirs |
| `experiments/no_aerosols/setup_no_aerosols.py` | Sets up no_aerosols scratch dirs |
| `experiments/cs_veg/setup_cs_veg.py` | Sets up cs_veg scratch dirs |
| `experiments/soil_moisture/setup_soil_moisture.py` | Sets up soil_moisture scratch dirs |
| `shared/sbatch_debug.sh` | Single-GPU debug sbatch (30 min, debug QOS) |
| MicroHH source: `src/boundary_surface_lsm.cxx` | Soil moisture nudging (CPU) |
| MicroHH source: `src/boundary_surface_lsm.cu` | Soil moisture nudging (GPU) |
| MicroHH source: `include/soil_kernels.h` | `nudge_theta()` CPU kernel |

---

## Status

### Done
- [x] CASS baseline runs exist at `$SCRATCH/CASS_LES_2stream` and `$SCRATCH/CASS_LES_raytracer`
- [x] Soil moisture nudging feature implemented (branch `mpowell-local`), compiled, not validated
- [x] Project structure designed and implemented (`base/`, `experiments/`, `shared/`, `analysis/`)
- [x] `shared/data/` populated: `cass_snd.txt`, `cass_sfc.txt`, `cass_lsf.txt`, `van_genuchten_parameters.nc`
- [x] `c_veg=1.0` set in `shared/config/cass_2stream.ini` and `shared/config/cass_raytracer.ini`
- [x] `shared/cass_input.py` updated with `--zero-winds` and `--theta-nudge FLOAT` flags
- [x] `base/setup_base.py`, `base/sbatch_base.sh`, `base/submit_base.sh`, `base/submit_debug_base.sh` written
- [x] `experiments/no_aerosols/` — all scripts written (setup, sbatch, submit, submit_debug)
- [x] `experiments/cs_veg/setup_cs_veg.py`, `sbatch_cs_veg.sh`, `submit_cs_veg.sh`, `submit_debug_cs_veg.sh` written
- [x] `experiments/soil_moisture/setup_soil_moisture.py`, `sbatch_soil_moisture.sh`, `submit_soil_moisture.sh`, `submit_debug_soil_moisture.sh` written
- [x] `shared/sbatch_debug.sh` written (single-GPU, debug QOS, 30 min)
- [x] `shared/data/README.md` written
- [x] ERA5 composite preprocessing complete: 119 composite days (1997–2009), cached in `/pscratch/sd/m/mpowell/LS2D_ERA5/cass/`
- [x] `cass_ls2d_input.nc` written to `/pscratch/sd/m/mpowell/CASS_LES/shared_data/` and symlinked into `shared/data/`
- [x] `cass_ls2d_input.py`, `cass_utils.py`, `shcu_sgp_summer_97to09.nc` moved to `shared/preprocessing/`; hardcoded path in `cass_utils.py` updated

### Next Steps (in order)
1. **Validate soil moisture nudging** (debug run, short timescale, verify theta_soil converges)
2. **Run base case** (4 × 2stream + 4 × raytracer) via `base/submit_base.sh`
3. **Run no_aerosols** (4 × 2stream + 4 × raytracer = 8 runs) via `experiments/no_aerosols/submit_no_aerosols.sh`
4. **Run cs_veg sweep** (5 values × 8 runs = 40 runs) via `experiments/cs_veg/submit_cs_veg.sh`
5. **Run soil_moisture sweep** (4 values × 8 runs = 32 runs) via `experiments/soil_moisture/submit_soil_moisture.sh`

### Notes / Gotchas
- Always re-run setup scripts after reorganizing — stale symlinks will silently break runs
- `cass_input.py` reads `cass.ini` from CWD — must be called from within the run dir (setup scripts handle this)
- Raytracer runs need `--constraint=gpu&hbm80g`; 2stream only needs `--constraint=gpu`
- `nudge_theta_timescale` units: seconds (86400 = 1 day relaxation)
- cs_veg `0` is a valid value (no skin heat capacity buffer)
- `rndseed` in `[fields]` must differ across reps: use 1, 2, 3, 4
- `surface_an_agg.nc` (present in cass root and old run dirs) — not referenced by any script, safe to ignore
- `shared/data/cass_ls2d_input.nc` must exist before running any setup script
