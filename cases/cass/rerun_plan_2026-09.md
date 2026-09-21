# CASS LES rerun plan (September 2026)

Status snapshot 2026-09-21, branch `mpowell-local` (HEAD after commits fb37f5639 merge of upstream main,
36e6ab0ef bug fixes, 256c7f4c5 Perlmutter cmake). Written so the work can continue on another machine.
`project_directive.md` is still the science reference but has NOT yet been updated with the findings below.

## 1. Why rerun: the forcing time-origin offset

- Every CASS ini sets `[time] datetime_utc = 2005-07-24 10:30:00` (05:30 CDT, about one hour before sunrise).
- The CASS composite tables `shared/data/cass_snd.txt`, `cass_lsf.txt`, `cass_sfc.txt` are valid from day 205.5
  = 12:00 UTC (07:00 CDT), and `shared/cass_input.py` converts them with `start_day = 205.5`.
- So in every run to date the initial sounding and the large-scale tendencies are applied 1.5 h early relative to
  the model's sun (the model's own solar geometry is correct: solar noon at clock 13:36).
- Impact: the daytime LS tendencies are weak and flat (thl_ls about -0.005 K/h; w_ls zero until 16 UTC, then
  -0.4 to -0.7 cm/s at 1.5 km), so the tendency error is below 0.05 K/h. The larger effect is the 07:00 CDT sounding
  started at 05:30 CDT with an extra hour of nocturnal cooling. All runs share the offset, so 3D-vs-1D contrasts
  are unaffected; realism against the CASS case and cloud-onset timing are.
- Bonus of the fix: with a 12:00 UTC start the same 50000 s run ends at 20:53 CDT, just after sunset, instead of
  19:23 CDT, so the full afternoon decay is captured.

Decision (user, 2026-09-21): do it right and rerun the suite with the corrected origin.

## 2. Configuration for the rerun (all experiments)

- `datetime_utc = 2005-07-24 12:00:00` in `shared/config/cass_2stream.ini` and `cass_raytracer.ini`; keep
  `endtime = 50000`. Nothing else in the time base changes (`start_day = 205.5` is then consistent).
- Couvreux tracer in every run (setup block as in `experiments/no_aerosols_zero_wind/setup_no_aerosols_zero_wind.py`):
  `slist += couvreux`, flux BC 1e-5, exponential decay 900 s, `nstd_couvreux = 1`, masks `couvreux,wplus,ql`,
  `couvreux` and `evisc` in the dump list. Tracer costs about 25 % wall.
- Aerosols off everywhere, 4 reps (rndseed 1-4), 4 sims per GPU node, same binary for the whole suite.
- Walltimes with tracer: 2stream 12 h (`--constraint=gpu`), raytracer 28 h (`--constraint="gpu&hbm80g"`);
  `gpu_regular` MaxWall on Perlmutter is 48 h, so no restarts are expected. Clone a restart script per experiment
  from `experiments/no_aerosols_zero_wind/sbatch_restart_no_aerosols_zero_wind.sh` anyway.
- Drop `t_sfc` from the ini templates (upstream removed the read; it is only an unused-item warning).

## 3. Suite and compute estimate (Perlmutter A100 numbers)

Per parameter value: one 12 h 2stream node-job plus one 28 h raytracer node-job = 40 node-hours.

| Experiment | Values | Runs | Node-h regular | Node-h with raytracer on preempt | Output |
|---|---|---|---|---|---|
| base: no_aerosols_zero_wind | 1 | 8 | 40 | 21 | 1.1 TB |
| rs_scale (0.25, 0.5, 2, 4; rs=1 is base) | 4 | 32 | 160 | 82 | 4.3 TB |
| rs_scale larger (6 values) | 6 | 48 | 240 | 123 | 6.5 TB |
| sw_scale (raytracer only, `swscalesfc_to_2str`) | 1 | 4 | 28 | 9 | 0.5 TB |
| wind_geo (2.5, 5, 7.5, 10; u=0 is base) | 4 | 32 | 160 | 82 | 4.3 TB |
| wind_sun (U = 5, anti-solar tracking) | 1 | 8 | 40 | 21 | 1.1 TB |
| wind_geo_hom (0, 2.5, 5, 7.5, 10) | 5 | 40 | 200 | 103 | 5.4 TB |
| Total, 4-value rs_scale | | 124 | 630 | 320 | ~17 TB |
| Total, 6-value rs_scale | | 140 | 710 | 360 | ~19 TB |

- Preempt QOS: 2 h minimum charge then 0.25x, so a 28 h raytracer leg bills 8.5 node-hours. MicroHH saves restart
  files hourly (`savetime = 3600`), so preemption is recoverable with the restart script.
- Calendar time is queue-bound: 31-33 single-node jobs of up to 28 h, hbm80g nodes are the bottleneck; expect
  two to three weeks end to end on Perlmutter. On another machine wall time scales with GPU speed; the raytracer at
  512 x 512 x 256 needs 80 GB GPUs.
- Storage is close to the 20 TB scratch quota: archive each experiment to HPSS as it completes, or set
  `swdump = 0` (or a trimmed dump list) on sweep values (about 60 GB per run saved). Per-run size with tracer is
  about 131 GB (113 GB hourly 3D dumps, about 20 GB of 60 s xy crosses, stats).
- Cheaper variant: 2 reps on intermediate sweep values only, roughly 400 node-hours regular.

## 4. wind_geo: make it more correct

Current `cass_input.py --geo-wind U` writes `u_geo = U`, `v_geo = 0` AND nudges u, v toward the uniform (U, 0)
profile at all levels on 10800 s (`[force] swnudge = 1, nudgelist = u,v`). That partly suppresses the Ekman turning
and the sub-geostrophic mixed-layer wind the experiment is about, so the realised shear must always be diagnosed
from the u, v stats rather than taken as u_g.

Proposed change (needs one 64 x 64 debug test before committing to it): nudge u, v only in the free troposphere
(above about 3 km) and let the boundary layer find its own Ekman balance. Caveat: starting from a uniform
geostrophic profile launches an inertial oscillation with period 2 pi / f = 20.5 h at 36.5 N, so the BL wind drifts
over the day. Options: initialise the BL wind reduced and turned (roughly 0.7 u_g with a small cross-isobar
component), or accept the drift and diagnose dU from output. Do not change anything else in wind_geo.

## 5. wind_geo_hom design (approved 2026-09-21)

Question: does the 3D-minus-1D LWP excess (46 % at u_g = 0 falling to 18 % at 10 m/s) collapse because shear
displaces the 1D self-shadow off the cloud root once L / dU < z_b / w*, so that both 1D and 3D converge to the
state with no radiatively driven surface pattern?

- MicroHH already has the switches (upstream, GPU, both solvers): `[radiation] swhomogenizesfc_sw`,
  `swhomogenizesfc_lw`, `swhomogenizehr_sw`, `swhomogenizehr_lw`. Use `swhomogenizesfc_sw = true` ONLY (exact
  Veerman et al. 2022 rt-hom / 2s-hom). Not `[land_surface] swhomogenizesfc`, which flattens the turbulent fluxes
  (a different experiment). Never combine with `swscalesfc_to_2str` (the code does not forbid it; homogenize wins).
- Sweep u_g = 0, 2.5, 5, 7.5, 10 (u = 0 hom is a real run), 4 reps, both RT, tracer, output identical to wind_geo.
  Scaffold = clone of the wind_geo four files plus `cfg.set("radiation", "swhomogenizesfc_sw", "true")` and the
  tracer/mask block; setup with `cass_input.py --geo-wind U` for all values (0.0 is equivalent to `--zero-winds`).
- Verification: homogenization is invisible in every radiation cross and stat (`sw_flux_sfc_dir_rt`,
  `sw_flux_dn.xy` are separate arrays). Check it through LSM fields: `thl_fluxbot.xy` / `qt_fluxbot.xy` must lose
  the shadow imprint (correlation of H with the SW field near zero, cloud-conditioned SEB "scissor" near zero),
  and `cass.out` must echo `[radiation][swhomogenizesfc_sw] = 1` without `(default)`. Both RT types.
- Contrasts: rt - 2s (full 3D effect), 2s - 2s-hom (1D self-shadow pattern), rt - rt-hom (3D displaced pattern),
  rt-hom - 2s-hom (mean-irradiance and heating-rate term only).
- Diagnostic R = (L / dU) / (z_b / w*): L = mean cloud effective diameter from `qlqi_path.xy`, dU = vector shear
  between z_b and a surface-layer level (definition of "U_surface" still to be chosen: lowest level 12.5 m,
  subcloud mean, or ground-relative |U(z_b)|), z_b from `compute_z_sl`, w* from surface `thv_flux`. Plot deficits
  against S = 1 / R so u = 0 stays on the axis. Figures are deferred.

## 6. Other findings (2026-09-21) still to fold into project_directive.md

- wind_sun (archived on HPSS as `wind_sun.tar`) is MISALIGNED: `solar_azimuth_deg` assumed a 12:00 UTC start
  while the ini said 10:30 UTC, so the prescribed wind was 12-65 degrees off anti-solar (56 degrees at solar noon).
  Fixed in commit 36e6ab0ef (`cass_input.py` now reads `datetime_utc`); rerun as part of the suite.
- Analysis time axis is local apparent solar time (`cass_analysis.LST_OFFSET = 3.899`); the t_scale, entrainment
  and updrafts scripts now import it (they hardcoded the 5.5 h clock offset). Lifetime and timescale caches on
  HPSS are still clock-based.
- `_gen_sweep_notebooks.py` now refuses to run without group arguments (it would overwrite the hand-edited
  `wind_geo_comparison.ipynb`).
- wind_geo runs have no conditional-sampling stats (default group only); v2 and all new runs have masks.
- `catalog.py` lacks `sw_scale`; `analysis/README.md` cites `updrafts/PLAN.md` and `updrafts/verify_and_cleanup.py`,
  which do not exist; `README.md` omits wind_sun; the directive lists rs_scale as 40 runs (32 submitted) and never
  mentions wind_sun.
- `calc_mean_2d_g` has no MPI reduction: the homogenization switches are per-rank on multi-GPU runs (fine single-GPU).
- All previous CASS output lives on HPSS under `/home/m/mpowell/CASS_LES/<experiment>/<unit>.tar` (htar extracts
  to the current directory; list members with `htar -tf`, restore subsets with `-L memberlist`). Everything there is
  on the 10:30 UTC origin.

## 7. Build notes (for a new machine)

- Perlmutter build (verified 2026-09-21): `module load cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13 cray-fftw/3.3.10.11
  cudatoolkit/13.2`, then `cmake -DUSECUDA=TRUE -DCMAKE_BUILD_TYPE=RELEASE ..` with `config/default.cmake`, which now
  takes CUDA and math libs from `NVHPC_CUDA_HOME`, FFTW from `FFTW_ROOT`, NetCDF/HDF5 from `CRAY_*_PREFIX`, host
  compiler g++-14, `CMAKE_CUDA_ARCHITECTURES 80`. Binary at `build_gpu/microhh`; the pre-merge July binary is kept
  as `build_gpu/microhh.pre_merge_026d9f8`. On another machine write a new `config/<machine>.cmake` from that pattern
  and build with `SYST=<machine>`.
- Smoke test after any rebuild: 64 x 64 debug run of a non-hom case with old and new binaries. Result on Perlmutter:
  2stream bit-identical; raytracer identical for the first hour, then chaotic divergence only (the backward Monte
  Carlo is not bit-reproducible), surface fluxes and radiation agreeing to better than 0.5 %.
- Submodule pin `rte-rrtmgp-cpp` 416a6706 matches upstream main; unused ini items only warn.

## 8. Suggested execution order

1. Set `datetime_utc = 12:00:00` in both RT templates; drop `t_sfc`.
2. wind_geo nudging debug test (section 4), decide the profile treatment.
3. Scaffold `wind_geo_hom` and the tracer-bearing variants of the other experiments; add catalog entries.
4. Debug-verify homogenization (section 5) on u = 0 and u = 10, both RT.
5. Launch base first so every sweep has its reference; then the sweeps, one submit each with wait-and-report
   watchers (no resubmit loops); archive to HPSS per experiment.
6. Update `project_directive.md`, `README.md`, `analysis/README.md` from sections 1-6.
