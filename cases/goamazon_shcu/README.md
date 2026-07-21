# GoAmazon composite shallow-cumulus — 1D vs 3D RT (MicroHH)

Third sibling in the GoAmazon/ARM97SD 1D-vs-3D RT family, and structurally
the odd one out: this case is a composite of **6 pure, non-precipitating
shallow-Cu days** from GoAmazon 2014/15 (Manco & Figueroa 2025, *Atmosphere*
16(7):789) that **never transition to deep convection** — the shallow-only
counterfactual to `cases/goamazon` (single-pulse) and `cases/arm97sd`
(both shallow-to-**deep** transitions). Same scientific question (does 3D
RT redistribute the cloud population — CASS's core finding) asked without
a deep tower in the mix, in the Amazon instead of SGP.

Data: https://ftp.cptec.inpe.br/pesquisa/bamc/MPDI/ (`IC_Forcings_Composite6ShCuCases.zip`),
cited in the paper's Data Availability Statement. Only the composite is
published — the six individual days (Mar10, Sep3, Sep11, Oct1, Oct5, Oct8
2014) that went into it are not.

## Case design

| Element | Choice |
|---|---|
| Site / time | SIPAM Ponta Pelada, Manaus (-3.15, -59.99) — the CAMZ octagon center, not the ARM T3 site `goamazon` uses; t=0 at 06:00 LT (10:00 UTC), 14 h to 20:00 LT, per the paper |
| Forcing | `sfc`/`snd`/`lsf` composite (SAM native ASCII format — same block-header convention as CASS's own `cass_sfc.txt`/`cass_snd.txt`/`cass_lsf.txt`, but space/tab-tokenized `day levels pres0` markers rather than CASS's comma-joined string, hence `parse_sam_blocks()` in the input generator rather than reusing `cass_input.py` directly). No-inversion variant (the paper found the artificial cloud-top inversion made minimal difference and used this as final) |
| Surface | **Interactive HTESSEL LSM** (unlike the paper's own SAM run, which prescribes H/LE/TAU from `sfc`) — the `sfc` file is stored in the input's `validation` group only, for comparing interactive-LSM fluxes against the paper's composite |
| Microphysics | `2mom_warm` (matches CASS) — pure non-precipitating ShCu by construction, no ice expected. `swqsqg_to_rad` not applicable (2mom_warm has no qs/qg fields; leave default off) |
| Radiation background | **Reused from `cases/goamazon`'s LS2D/ERA5 file** (dry-season Oct 2014 only — this composite spans wet+dry season, so it's an approximation; background/gases/soil-init matter far less than the sfc/snd/lsf forcing itself for a boundary-layer-driven case) |
| Vegetation | Same as `goamazon` (IFS evergreen broadleaf, ERA5-validated at the nearby T3 site: lai=5.35, z0m=1.3, z0h=0.13, c_veg=0.91) |
| Grid | **CASS's own validated shallow-Cu grid** (512x512x256, dx=50m, dz=25m, 6.4 km domain) — no fit test needed, already proven to run the raytracer comfortably on a single A100 |

## Workflow

```bash
# Forcing + reused ERA5 background already symlinked into data/ (no LS2D
# pull needed — see cases/goamazon/README.md if it ever needs regenerating)

# Debug/smoke test (128x128, same dx=50m as production -- unlike the
# deep-conv cases, meaningful here: shallow-Cu cells are small enough to
# develop realistically at reduced domain size, not just "does it crash")
./submit_debug.sh

# Production: 4 reps x {2stream, raytracer}
python setup_runs.py
./submit_production.sh
```

Runs land in `$SCRATCH/GOAMAZON_SHCU_LES/{debug,base}/`. Both RT modes use
`gpu_shared` 2-GPU jobs (no idle GPUs); CASS's own runs at this same grid
and similar duration (13.9 h) completed comfortably in ~22 h regular-queue,
so 24 h shared-queue should be ample — restart chain is wired up
(`sbatch_restart.sh`) but not expected to be needed.

## What to check in the smoke test

Per the paper's composite results (their Figure 5): cloud formation
10-11 LT, maturity (peak cloud fraction ~9%, peak LW ~0.02 g/kg) 13-15 LT,
dissipation by 17-18 LT. Cloud top climbing from ~1 km to ~3 km over the
same window. Interactive-LSM H/LE against the `validation` group's
prescribed values (paper's own composite: H peaks ~115 W/m2, LE ~383 W/m2,
both around 13-15 LT) is the other cross-check — same pattern used for
`goamazon`/`arm97sd`.

## Caveats

- Radiation background is Oct-2014-only (dry season); the composite itself
  spans wet+dry season equally. Soil moisture init in particular may run
  drier than a true wet-season composite would want.
- Site is SIPAM Ponta Pelada (-3.15, -59.99), ~65 km from `goamazon`'s ARM
  T3 site (-3.2, -60.6) — both within the same CAMZ region/biome, vegetation
  parameters reused across both without adjustment.
- `datetime_utc` in the ini configs is nominally 2014-10-05 (one of the six
  actual composite days, and the same date as `cases/goamazon`) — solar
  position only, not tied to any specific radiative state; low-latitude
  site means seasonal insolation variation is small regardless.
