# ARM97 shallow-to-deep (SGP, 27 June 1997) — 1D vs 3D RT (MicroHH)

Continental shallow-to-deep transition companion to `cases/goamazon`: the
canonical SGP transition day (EUROCS/GCSS continental deep-convection
diurnal cycle lineage, Guichard et al. 2004), carved from the ARM97 IOP
forcing (variational analysis, 20-min cadence). Same machinery, same grid,
same physics, same `swqsqg_to_rad` radiation patch as GoAmazon — designed
for direct cross-case comparison (wet tropical forest vs dry plains).

NOT the 21 June 1997 shallow-only benchmark (that is `cases/arm`); this is
the day that transitions shallow Cu -> congestus -> deep by mid-afternoon.

## Case design (deltas vs cases/goamazon; see that README for the shared design)

| Element | Choice |
|---|---|
| Site / time | ARM SGP CF (36.605, -97.485); t=0 at 1997-06-27 11:30 UTC (05:30 CST); 16.5 h (ends 04:00 UTC, ~2 h past sunset) |
| IOP | `ARM97_iopfile_4scam.nc`, bdate 1997-06-18; sim window at tsec 819000; 35 p-levels to ~115 hPa; q is mixing ratio |
| Forcing | divT/divq only (horizontal advection; `iop_dosubsidence=false`, vertdivT unused by EAMxx — mirrored); no Coriolis |
| Winds | Nudged to IOP `u`/`v` (file has NO u_ls/v_ls; EAMxx fallback), tau=10800 s |
| Surface | Interactive LSM; SGP grassland/cropland values from the CASS project (z0m=0.075, lai=1.5, rs_veg_min=70, alb=0.20) — cross-check ERA5 printout at assembly; ERA5 soil at 11 UTC; low-veg root profile |
| Background | LS2D/ERA5 1997-06-27 11:00 -> 06-28 04:00 at (36.5, -97.5); gases at 1997 values (co2=363.7 ppm) |
| Grid | 512x512x384, dx=200 m, dz=66.7 m — identical to GoAmazon (memory-validated, ~60 GB raytracer) |
| pbot / t_sfc | 97400 Pa / 296.0 K (IOP Ps, Tg at 11:30 UTC) |

## Workflow

```bash
XR_PY=/global/homes/m/mpowell/.conda/envs/xr_env/bin/python

# 1. ERA5 via LS2D (CDS queue; re-run until complete), then symlink
$XR_PY preprocessing/arm97sd_ls2d_input.py
ln -sf $SCRATCH/ARM97SD_LES/shared_data/arm97sd_ls2d_input.nc data/

# 2. Production setup + submit (gpu_shared 2-GPU jobs, no idle GPUs)
python setup_runs.py
./submit_production.sh
```

Runs land in `$SCRATCH/ARM97SD_LES/base/`. Raytracer pair: 48 h shared job
+ dependent restart chain (estimated 60-90 h total from GoAmazon timing).
Post-processing: `sbatch_postproc.sh` (pass `T0=0` for restarted runs).

## Notes

- `rr_bot` is in the crosslist (unlike GoAmazon) for precip-area analysis.
- No fit test needed: grid identical to GoAmazon; raytracer memory ~60 GB
  measured there (0.60 GB/Mcell scaling).
- GoAmazon known-bias analysis (H excess from aerosols-off + interactive
  land) applies in spirit; SGP June 1997 was clean-ish (no biomass burning),
  so expect a smaller clear-sky SW excess than the Amazon case.

## Setup on a new machine

Same dependencies and steps as `cases/goamazon/README.md` (MicroHH clone
+ build incl. the `swqsqg_to_rad` feature already on this branch, LS2D
clone, CDS API credentials, python env). Hardcoded paths to edit:

| File | Line | What |
|---|---|---|
| `preprocessing/arm97sd_ls2d_input.py` | 21 | LS2D clone location |
| `preprocessing/arm97sd_ls2d_input.py` | 41 | `era5_path` (LS2D's ERA5 cache dir) |
| `setup_runs.py` | 26 | `MICROHH_DIR` |
| `setup_runs.py` | 31 | `XR_PY` (python interpreter with netCDF4/xarray) |
| `setup_runs.py` | 33 | `IOP_FILE` -> `ARM97_iopfile_4scam.nc` (E3SM inputdata,
  `atm/cam/scam/iop/`, same fetch mechanism as GoAmazon's IOP file) |

`export SCRATCH=...` rather than editing `setup_runs.py`'s fallback.
`gpu_shared`/`gpu_debug`/`hbm80g` constraints in the `sbatch_*.sh` /
`submit_*.sh` / restart-chain scripts are Perlmutter-specific Slurm
QOS/constraint names — adjust for another cluster's scheduler.
