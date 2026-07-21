# Handoff: goamazon + arm97sd + goamazon_shcu production on the NVL72 SuperPOD

Written for a fresh agent with no memory of the Perlmutter session that
built this. Read this fully before touching anything.

**This file is intentionally gitignored** (see feedback convention: never
commit agentic handoff docs). `git clone` will NOT bring it over — the
user is transferring it via Globus alongside the data files listed below.
If you're reading it, you already received it that way; treat it as a
one-time briefing, not a repo artifact to maintain.

## What this is

Three MicroHH LES cases comparing 1D (two-stream RRTMGP) vs 3D (coupled
Monte Carlo ray tracer) radiative transfer, all Amazon/SGP shallow-convection
siblings of an existing project (`cases/cass/`) studying the same
1D-vs-3D question:

- **`cases/goamazon/`** — GoAmazon day 278 single pulse (2014-10-05,
  Amazon rainforest, Tian & Zhang 2025 forcing). Shallow-to-**deep**.
- **`cases/arm97sd/`** — ARM SGP 27 June 1997 shallow-to-deep (EUROCS/GCSS
  continental deep-convection case, Guichard et al. 2004). Shallow-to-**deep**.
- **`cases/goamazon_shcu/`** — composite of 6 pure, non-precipitating
  shallow-Cu GoAmazon days (Manco & Figueroa 2025, *Atmosphere* 16(7):789).
  Never transitions to deep by construction — the shallow-**only**
  counterfactual to the other two. Newest of the three (built same session
  as this handoff); much cheaper (CASS's own shallow-Cu grid, 512x512x256
  dx=50m, not the deep-conv cases' memory-constrained grid).

  Its `goamazon_shcu_input.py` had two real bugs on the first pass (wrong
  time-origin for the raw `sfc`/`snd`/`lsf` files -- day=0.000 in the file
  is 20:00 LT the evening before, not sim t=0; and per-block z-level drift
  on fixed-pressure levels, up to ~330 m at the domain top, ignored by
  reusing block 0's z-grid for every block). Both are now fixed (commit
  `976239cc3`) and **re-verified**: debug run shows cloud onset at 10.0 LT,
  maturity (peak LWP) at 13.0 LT, and critically the `validation` group's
  LE now peaks at 383 W/m2 at 14.0 LT -- an exact match to the paper's own
  reported composite value, confirming the forcing is correctly
  time-referenced now (the first "smoke test confirmed working" claim was
  based on the buggy version and should not be trusted -- the physically
  plausible-looking result then was likely riding on solar heating being
  correctly calendared while the large-scale forcing underneath was
  silently 10 h out of phase). **Good candidate to stand up first here**
  given it's cheapest and now genuinely validated, not just cheap.

Each case's own `README.md` is the detailed reference (design choices,
known biases, provenance) — **read those first**, this file only covers
what's specific to standing the pipeline up on new hardware.

## Explicit scope: do NOT touch the in-flight Perlmutter run

The GoAmazon raytracer reps (01/02) are mid-run on Perlmutter via a
restart-chain mechanism (`sbatch_restart.sh`, resubmits itself from the
last hourly checkpoint). **Do not attempt to migrate that run here** —
restart dumps are raw binary field data written by a specific build; there
is no reason to risk cross-architecture portability of that when the run
is already progressing fine where it started. Perlmutter finishes its own
work; this machine is for **new** production that Perlmutter currently
can't get to (queue-blocked by a maintenance reservation) or new grid
variants this hardware's memory enables.

What's actually needed here is the **input pipeline** — the case code and
the already-built forcing/background data — so runs can start fresh.

## Current state on Perlmutter (for context, not action)

| Case | 2stream | Raytracer |
|---|---|---|
| GoAmazon | 4/4 reps complete | rep_01/02 mid-run, restart-chained |
| ARM97SD | reps 01/02 submitted, queued behind Perlmutter maintenance | reps 01/02 submitted, queued behind Perlmutter maintenance |
| GoAmazon ShCu | debug-only so far (bug-fixed + re-verified, see above) | debug-only so far (bug-fixed + re-verified, see above) |

GoAmazon ShCu has **no production run anywhere** — explicitly the one to
run here, not started on Perlmutter at all.

ARM97SD full-scale production hasn't actually started yet — that's the
highest-value thing to pick up here. GoAmazon reps 03/04 raytracer were
never submitted at full scale either (only 2stream got the free 4-rep
ensemble via GPU backfill) — a good second target once ARM97SD is running.

## 1. Clone the code

```bash
git clone --recursive -b mpowell-local git@github.com:magpowell/microhh.git
```

`--recursive` pulls `rte-rrtmgp-cpp` and its own submodules
(`rte-rrtmgp`, `rrtmgp-data`) — all public GitHub, no NERSC-specific data
needed there. `van_genuchten_parameters.nc` (soil lookup table) is also
already tracked in `misc/` — nothing to fetch for either of those.

This branch already has the `swqsqg_to_rad` radiation feature
(`include/thermo_moist.h`, `src/thermo_moist.cxx`/`.cu`) that both cases'
`config/*_base.ini` require (`[thermo] swqsqg_to_rad=1` — adds nsw6 qs+qg
to the radiation cloud ice, SCREAM/P3 category parity). No extra patching.

## 2. Data to Globus-transfer (the actual point of this doc)

Seven small files (<8 MB total), from Perlmutter to wherever this clone
lives. Exact source paths:

```
/global/cfs/cdirs/e3sm/inputdata/atm/cam/scam/iop/GOAMAZON_singlepulse_iopfile_4scam.nc   (714 KB)
/global/cfs/cdirs/e3sm/inputdata/atm/cam/scam/iop/ARM97_iopfile_4scam.nc                  (6.3 MB)
/pscratch/sd/m/mpowell/GOAMAZON_LES/shared_data/goamazon_ls2d_input_HYBRID.nc             (83 KB)
/pscratch/sd/m/mpowell/ARM97SD_LES/shared_data/arm97sd_ls2d_input.nc                      (266 KB)
/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/shared_data/sfc                                  (~0.5 KB)
/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/shared_data/snd                                  (~24 KB)
/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/shared_data/lsf                                  (~30 KB)
```

The last three (`sfc`/`snd`/`lsf`) are the composite shallow-Cu forcing
for `goamazon_shcu` — originally from
https://ftp.cptec.inpe.br/pesquisa/bamc/MPDI/IC_Forcings_Composite6ShCuCases.zip
(Manco & Figueroa 2025's own data release), already unpacked here. That
case's radiation background is the **same** `goamazon_ls2d_input_HYBRID.nc`
file above (reused, not a separate pull) — one fewer file to worry about.

The first two are E3SM IOP forcing files (not derivable without E3SM
inputdata access — this is the one input that's genuinely hard to get any
other way). The last two are the ERA5/LS2D-derived background+soil+padding
files this session already built and validated (see each case's README
"Provenance" / background sections for how). **Transferring these skips
re-fighting CDS/MARS entirely** — the GoAmazon one in particular took
multiple attempts to get right (grid-snapping the request coordinates,
working around an LS2D bug, and ultimately hybridizing real ERA5 soil with
an RCEMIP analytic background when the MARS model-level file wouldn't
deliver in reasonable time — full story in the case's git history if
regeneration is ever needed instead).

After transfer, point each case at its files:

```bash
# GoAmazon
mkdir -p cases/goamazon/data
ln -sf <transferred>/GOAMAZON_singlepulse_iopfile_4scam.nc <wherever setup_runs.py's IOP_FILE points>
ln -sf <transferred>/goamazon_ls2d_input_HYBRID.nc cases/goamazon/data/goamazon_ls2d_input.nc

# ARM97SD
mkdir -p cases/arm97sd/data
ln -sf <transferred>/arm97sd_ls2d_input.nc cases/arm97sd/data/arm97sd_ls2d_input.nc

# GoAmazon ShCu (reuses the goamazon background file above)
mkdir -p cases/goamazon_shcu/data
ln -sf <transferred>/sfc cases/goamazon_shcu/data/sfc
ln -sf <transferred>/snd cases/goamazon_shcu/data/snd
ln -sf <transferred>/lsf cases/goamazon_shcu/data/lsf
ln -sf <transferred>/goamazon_ls2d_input_HYBRID.nc cases/goamazon_shcu/data/goamazon_ls2d_input.nc
```

Simpler alternative if Globus makes it easy: just transfer them to the
exact same relative paths the scripts expect (see each `setup_runs.py`'s
`IOP_FILE`/`CASE_DATA` constants) and skip the symlink dance.

If for any reason you'd rather regenerate the ERA5 background from
scratch instead of transferring it: `preprocessing/*_ls2d_input.py` in
each case do this via the LS2D package (`https://github.com/LS2D/LS2D`,
plain clone, needs a CDS/Copernicus API key in `~/.cdsapirc`). Expect the
`reanalysis-era5-complete` (model-level) request to queue for a long time
in MARS — this is exactly the pain the transferred files let you skip.

## 3. Hardcoded paths to edit

`goamazon`/`arm97sd` each have the same five NERSC-personal-account
absolute paths (see each README's own "Setup on a new machine" section for
the exact file:line table): LS2D clone location, LS2D's `era5_path` cache
dir, `MICROHH_DIR`, the python interpreter (`XR_PY`), and `IOP_FILE`.
`goamazon_shcu` is simpler — no LS2D pull of its own (reuses goamazon's
background), so just `MICROHH_DIR` and `XR_PY` in its `setup_runs.py`.
Update those, and `export SCRATCH=...` for wherever run output should land
(`setup_runs.py` falls back to a personal Perlmutter path if unset, in all
three cases).

## 4. Building on NVL72 — flagged unknowns

This is the part that genuinely needs on-site investigation; don't guess
blindly.

- **Grace CPU = ARM64, not x86_64.** `config/default.cmake` currently
  hardcodes `Linux_x86_64` NVHPC SDK paths (e.g.
  `/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/...`). On this cluster those need
  to become `Linux_aarch64` (or whatever the actual installed toolkit
  layout is) — **do not reuse `config/default.cmake` unmodified**, write a
  new `config/<machine>.cmake` after checking what's actually installed
  (`module avail`, or equivalent, if bare-metal modules exist here at
  all).
- **Container-first culture.** The site guide for this cluster (`beta`
  partition) frames Pyxis+Enroot containers as the default and bare-metal
  as "the exception, not the norm" — but that guide is written for
  PyTorch/NCCL-style AI training jobs, not a traditional MPI+CUDA
  Fortran/C++ HPC code like MicroHH. Figure out on-site whether host
  modules exist for NVHPC SDK / CUDA / NetCDF-C / HDF5 (bare-metal build,
  closer to how this ran on Perlmutter) or whether a container image needs
  to be built with that stack (`enroot import`/`create`, mount the repo in
  via `--container-mounts`, build inside). Don't assume either way.
- **GPU memory is dramatically bigger**: B200 HBM3e is ~186 GB/GPU
  (13.4 TB per rack / 72 GPUs) vs the A100 80 GB this was tuned for on
  Perlmutter — see the next section, this is actually a real opportunity,
  not just a build detail.
- Both cases' code otherwise doesn't care about node topology — no
  multi-node MPI needed (each MicroHH run is single-GPU or a handful of
  independent single-GPU runs sharing a node). Use `--segment=1`
  (independent work, per the site guide's own use-case matrix) — nothing
  here is a communication-heavy distributed training job, that whole
  segment-locality machinery is irrelevant to this workload.

## 5. Slurm convention translation (Perlmutter -> NVL72 `beta`)

Perlmutter scripts (`sbatch_runs.sh`, `sbatch_restart.sh`,
`sbatch_postproc.sh`, `submit_production.sh` in each case) use
Perlmutter-specific QOS/constraint names that don't exist here:

| Perlmutter | This cluster | Notes |
|---|---|---|
| `--qos=gpu_shared --gres=gpu:2 --cpus-per-task=32` | `--partition=beta --gpus-per-node=2` (or whatever fraction) | gpu_shared's "32 cores per GPU" requirement was a Perlmutter Slurm quirk, likely doesn't apply here — check |
| `--constraint="gpu&hbm80g"` | (nothing — every B200 here has the same big HBM3e) | drop entirely |
| `--constraint=gpu` (2stream, no hbm80g needed) | `--partition=beta` | |
| `--qos=debug`, 30 min cap | check this cluster's debug-equivalent QOS/partition, if any | |
| implicit 4 GPUs/node via `--gres=gpu:4` | `--gpus-per-node=4` (18 nodes/rack x 4 GPUs matches Perlmutter's packing pattern) | the "don't leave GPUs idle" job-packing lesson from this session (see below) still applies directly |
| `--account=m1266` (NERSC project) | `--account=<your allocation on this cluster>` | |

Add `--segment=1` per the site guide's own recommendation for independent,
loosely-coupled work like this.

## 6. Job-packing lesson from this session (still applies here)

Learned the hard way on Perlmutter: **never size a whole-node job around
the fastest job in the mix.** A 2stream run (fast, ~8-12h) and a raytracer
run (slow, ~35-90h+ depending on domain/convective intensity) sharing one
node both blocked on `--gres=gpu:4` idles 2 GPUs the moment 2stream
finishes. Either request only as many GPUs as a job needs (this cluster's
equivalent of `gpu_shared`, if one exists), or — the trick used here —
`srun --jobid=<id> --overlap` launches extra work onto GPUs a running job
already vacated, without waiting for a new allocation. Worth setting up
similarly if `beta` allows overlapping srun into a live allocation.

## 7. The actual opportunity: larger domain

The Perlmutter grid (512x512x384, dx=200m, dz=66.7m, 102.4km domain,
~101M cells) was **memory-constrained to A100's 80GB**, not resolution- or
science-constrained — it was chosen via an empirical fit test showing
raytracer memory scales linearly at ~0.60 GB/Mcell, and dz=50m at full
domain (134M cells, 80.5GB) came in at 98% of the card, rejected as too
risky. See `cases/goamazon/README.md` "Grid selection" section for the
full reasoning and measured numbers.

At ~186 GB/GPU here, the same 0.60 GB/Mcell scaling projects roughly
**3x the cell budget** (~310 Mcells before other overhead) — e.g. full
512x512 domain at dz~33m (768 levels, ~201M cells, ~121GB), or a bigger
horizontal domain at the existing dz=66.7m (640x640x384 ~157M cells
~94GB is comfortable; pushing toward SCREAM's full 200km domain at
dx=200m, i.e. ~1024x1024, would need ~419 Mcells and likely won't fit
even here at dz=66.7m — do the arithmetic against whatever the fit test
actually measures before committing).

**Don't just extrapolate this number and commit to a production grid.**
The 0.60 GB/Mcell scaling was measured empirically on A100/Perlmutter;
re-run the same fit-test methodology here before trusting it on B200 —
architecture, driver, and autotuner differences could shift the constant.
`setup_runs.py --fit-test` (see either case's README "Workflow" section)
already does exactly this: spins up several grid candidates at
`endtime=1800` on debug/short jobs, logs `nvidia-smi` memory every 10s to
`gpu_mem.log`, so the empirical numbers are there before any production
commitment. Cheap and already built — just needs new `GRID_PRESETS` values
sized for this hardware (edit `setup_runs.py`) and a Slurm-syntax-adapted
version of `submit_fit_test.sh`.

If going bigger, ARM97SD is the more natural candidate to test it on —
it hasn't run at full production scale on Perlmutter at all yet, so
there's no existing-grid consistency to preserve. GoAmazon changing grid
would break comparability with the reps already completed on Perlmutter
at 512x512x384 — keep that one on the existing grid unless intentionally
starting a new resolution-sensitivity variant (see each README's
resolution-adequacy discussion — a same-domain, coarser-dz sensitivity
pair was flagged as a good follow-up test, which bigger memory here would
make cheap).

## 8. Recommended order of operations

1. Confirm what's actually installed here (compilers, NVHPC SDK, MPI,
   NetCDF/HDF5 — bare metal or container) before writing a new cmake
   config.
2. Build MicroHH, verify with a trivial case (e.g. `cases/bomex` or
   similar from the existing suite) before trusting the goamazon/arm97sd/
   goamazon_shcu pipeline on this hardware.
3. Globus-transfer the 7 files above; wire up `data/` symlinks and the
   hardcoded-path edits.
4. Run `setup_runs.py --debug` as a fast end-to-end smoke test before
   spending real GPU-time. goamazon/arm97sd use 64x64 columns; goamazon_shcu
   uses 128x128 at the *same* dx as production (50m) since shallow-Cu
   develops meaningfully at reduced domain size, unlike the deep-conv
   cases — its debug run on Perlmutter (post bug-fix, see above) landed
   onset/maturity/the validation-group flux peak all where the paper says
   they should be, so that pipeline is pre-validated; still worth rerunning
   here as a build/hardware sanity check. For arm97sd, cross-check ice/rain
   onset lands near 15:30-16:30 CST (each README documents what "looks
   right" for its case).
5. Fit-test the grid for goamazon/arm97sd if pursuing a bigger domain
   (existing size otherwise needs no fit test). goamazon_shcu's grid is
   CASS's own already-validated shallow-Cu grid — no fit test needed there
   either way.
6. Submit **goamazon_shcu full production first** — cheapest of the three
   (CASS-scale grid, not the deep-conv memory-constrained one) and the
   only one with literally nothing running anywhere yet, not even on
   Perlmutter. ARM97SD full production second (queued on Perlmutter but
   blocked by its maintenance reservation). GoAmazon rep_03/04 raytracer
   third, if wanted, at the SAME grid as the Perlmutter reps for
   comparability.
