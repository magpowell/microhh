# CASS LES rerun plan (September 2026)

Status snapshot 2026-09-21, branch `mpowell-local` (HEAD after commits fb37f5639 merge of upstream main,
36e6ab0ef bug fixes, 256c7f4c5 Perlmutter cmake). Written so the work can continue on another machine.
`project_directive.md` (the April 2026 framing) was removed on 2026-09-29; git history keeps it (last in commit
46be6510c). Its list of gotchas is in `README.md`.

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

- DONE 2026-09-21: `datetime_utc = 2005-07-24 12:00:00` in `shared/config/cass_2stream.ini` and `cass_raytracer.ini`
  (and `t_sfc` dropped); keep `endtime = 50000`. Every run set up from this commit on is on the corrected origin. Nothing else in the time base changes (`start_day = 205.5` is then consistent).
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

## 6. Other findings (2026-09-21)

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
6. Update `README.md`, `analysis/README.md` from sections 1-6.

## 9. Empire AI Alpha status (answers from the Alpha checkout, 2026-09-21)

Target machine for the rerun. Facts as reported from Alpha on 2026-09-21:

- **Storage is the blocker.** No project directory exists for `cu_rpincus_illuminating` under `/mnt/home` or
  `/mnt/lustre`; `/mnt/lustre` is still a symlink to `/mnt/home/DDN_Copy` on the VAST NFS home filesystem, and
  `/mnt/ddn` is empty. The new Slurm account carries no filesystem allocation. A 10-19 TB rerun needs an explicit
  project storage request to Empire AI naming the new project; until it exists the rerun cannot be placed there.
  Ask for the full-output figure (about 17-19 TB, section 3) plus headroom. The CASS inputs themselves are tiny
  (`shared_data/` is 180 KB: ERA5 and CAMS composites, van Genuchten table, CASS text tables); stage them at
  `/mnt/lustre/columbia/mpowell/shared_data` (the env script's `SCRATCH`) once storage exists.
- **Slurm changed.** Partition and account `columbia` are gone (`columbia` now only carries a zero-priority
  `burst` QoS). Use account `cu_rpincus_illuminating`, partition `alpha`, and an explicit `--qos`:
  test 2 h (4-node cap), standard 48 h (30), long 7 d (15), priority 24 h (45). The Alpha submit scripts for
  arm97sd and goamazon were updated to these defaults on `mpowell-local` (production: standard; debug and fit
  test: test). The CASS port must inherit them.
- **Timing (raytracer, H200 only, first 30 simulated minutes near clear sky).** Wall-to-sim ratios 1.2-1.7 for
  goamazon 512^2 x 320-512 and 2.3-2.8 for arm97sd; scaled to 256 levels roughly 1-2x real time, so a CASS
  raytracer sim (50000 s) is about 14-28 wall hours on H200, not materially better than the 28-h A100 figure.
  One raytracer sim per 48-h standard job (or long). No 512^2 two-stream measurements exist on Alpha; the 64^2
  debug runs do not scale. Raytracer memory at 512^2 x 384 peaked at 64 GB, so H100 80 GB nodes are fine.
- **Build state intact.** Production binary `microhh-alpha-sm90-416a6706` (built 2026-07-22 from 026d9f8b on
  `mpowell-local`, submodule 416a6706, mamba env `~/miniforge3/envs/microhh`, nvcc 12.6.85, gcc 13.4.0) in
  `~/validated_builds/` with a README; do not overwrite it from the `rt-nonuniform-dz` branch. Rebuild from the
  pushed `mpowell-local` for the rerun and validate the same way (section 7). The Alpha checkout currently sits
  on `nonuniform-dz-prod` (f219ea27); its `origin` is the magpowell fork.
- **Scheduling model there**: single-GPU jobs (8 GPUs per node, no shared-node QoS), ~30-min median queue wait in
  July. With N concurrent single-GPU jobs the 124-140 sims of section 3 take roughly (124 x ~20 h) / N hours of
  run time; at N = 20 that is about a week plus queue waits.

Port list for the CASS scaffolding (mirror the arm97sd July port): derive `MICROHH_DIR` from the file location,
require `SCRATCH` and `XR_PY` from `config/empireai_alpha_env.sh`, one job per (RT, rep) via `sbatch_runs_alpha.sh`-
style scripts with chained restarts, drop the Perlmutter `-A m1266`, `hbm80g` constraint and 4-per-node packing,
QoS as above. The cross_to_nc post-processing step must run inside the job under `XR_PY`.

**Port done (2026-09-21, Alpha checkout).** The seven `setup_*.py` locate the repo from their own path, require
`SCRATCH`, and honor `MICROHH_EXEC`; `shared/submit_alpha.sh` sets up and submits any experiment as single-GPU
jobs with chained raytracer restarts; `shared/sbatch_run_alpha.sh` and `shared/sbatch_restart_alpha.sh` are the
shared job bodies (cross_to_nc under `XR_PY` inside the job); `setup_shared_data.sh`, `cass_utils.py` and the
READMEs no longer carry NERSC paths. Perlmutter sbatch/submit files, htar scripts and `analysis/` are untouched.
Not yet validated end to end: the CASS inputs are not staged on Alpha (`shared/data/` symlinks point to
`/pscratch`), so the Alpha smoke test used arm97sd. Next: a site-detecting env layer and one submit driver for
both machines (Perlmutter keeps `m1266`, `gpu&hbm80g` for raytracer legs, and its verified QoS names).

## 10. Site layer verified on Perlmutter (2026-09-21)

`config/site_env.sh` detects Perlmutter via `NERSC_HOST`; all seven `setup_*.py` dry-run cleanly with it sourced.
`sacctmgr`: `shared`/`gpu_shared` MaxWall 48 h (MaxTRESPerJob gres/gpu=2, node=1); `debug` 30 min, MaxJobsPU 2.
`sbatch --test-only` with the driver's exact flags: production raytracer (`--qos=shared --constraint=gpu&hbm80g
--gres=gpu:1 -c 32 --time=48:00:00`) and 2stream are placed in `shared_gpu_ss11`, the raytracer on an hbm80g node;
the debug set gets a whole `gpu_ss11` node. Defaults in `site_env.sh` are therefore correct; no change needed.
Caveat: `debug` allows 2 jobs per user, so `shared/submit.sh <expt> --debug` for one experiment at a time.

**Alpha GPU governance (2026-09-21, found by the smoke test).** Partition `alpha` also holds alphagpu51-54, 8x RTX
PRO 6000 Blackwell each (feature `rtx6000`, 98 GB, weak FP64). Measured on one: arm97sd 64x64 debug raytracer
reached 1200 s in 34 min against 24600 s in 30 min on H100 in July (about 20x slower); 2stream about 2x slower.
The submit plugin `rtx6000_gpu_governance` routes EVERY 1-GPU job there, ignores `--constraint=h100` and
`--exclude`, and rewrites a typed 1-GPU gres to untyped; only a typed request for >= 2 GPUs is placed on H100/H200
(verified with `sbatch --test-only`). Consequences: `submit.sh` packs 2 sims per job on Alpha
(`SITE_GPUS_PER_JOB=2`, `SITE_GPU_TYPE=nvidia_h100_80gb_hbm3`), grouped within an RT mode; the compute estimate
of section 3 is unchanged in GPU-hours; the arm97sd and goamazon `submit_*_alpha.sh` scripts still submit 1-GPU
jobs and must be changed the same way before any production use. Test-QoS queue estimate at the time was about
one day. The smoke test was resubmitted as two 2-GPU H100 jobs (old+new binary side by side per RT mode).

## 10. To do (ideas queued, not yet designed)

- **Amplitude scaling of the shadow anomaly (user, 2026-09-23).** Multiply the surface SW shadow anomaly by
  0.5 and 0.25 while keeping the domain mean: F_sfc = <F> + a (F - <F>) with a = 1 (plain), 0.5, 0.25,
  0 (= `swhomogenizesfc_sw`). Find where the 3D effect drops out. That amplitude threshold is the number to compare
  with the static-heterogeneity experiments. Needs a new `[radiation]` switch (e.g. `sfc_sw_anomaly_scale`)
  applied at the same point as `swhomogenizesfc_sw` in `radiation_rrtmgp_rt.cxx` (and the 2stream solver for
  the 2s-hom analogue); no partial-scaling option exists today, only homogenize (a = 0) and `swscalesfc_to_2str`.
  Same sweep shape as wind_geo_hom: both RT modes, 4 reps, tracer.

## 11. Base case v3 (`no_aerosols_zero_wind_v3`), settled 2026-09-29

Set up by `experiments/no_aerosols_zero_wind/setup_no_aerosols_zero_wind.py --version v3`. Three configurations
(`2stream`, `raytracer`, `raytracer_swmatch`), 4 members each, the same random seed for the same member number.
Where the CASS case description (https://portal.nersc.gov/project/capt/CASS/) specifies a value, v3 follows it.

| Item | v2 | v3 | Source |
|---|---|---|---|
| Start | 10:30 UTC | 12:00 UTC | CASS tables start at day 205.5 |
| Domain top, levels | 6.4 km, 256 | 8 km, 320; damping from 6.4 km | v2 cloud tops reached the damping layer by 16 LT in 3D |
| Temperature tendency | used as theta_l tendency | divided by the Exner function | table column is a temperature tendency |
| Subsidence below the lowest table level (205 m) | constant | linear to zero at the surface | w = 0 at the surface |
| Nudging | u, v at all heights on 3 h | theta_l, q_t above 5 km on 1 h (zero below 4.5 km) toward the CASS sounding | CASS; winds are zero in this case |
| Momentum roughness z0m | 0.075 m | 0.035 m | CASS |
| Heat roughness z0h | 0.003 m | 0.003 m | choice, see below |
| Soil moisture and temperature | ERA5 composite at 10 UTC | ERA5 composite at 12 UTC | start hour |
| Vegetation cover c_veg | 1.0 | 0.898 | ERA5 low-vegetation cover 0.998 x IFS density 0.90 |
| Leaf area index | 1.5 | 1.47 | ERA5, mean over the case days |
| Minimum canopy resistance | 70 s/m | 100 s/m | IFS table, crops and mixed farming |
| Vapour-pressure-deficit coefficient gD | 0 | 0 | IFS table: 0 for all low vegetation |
| Root distribution | short grass | crops and mixed farming | IFS table for the ERA5 vegetation type |

**Land surface.** `shared/preprocessing/cass_land_composite.py` takes ERA5 at the grid point of the site (36.5 N,
97.5 W) over the 119 case days: soil type 2 (medium), low vegetation type 1 (crops, mixed farming), low-vegetation
cover 0.998, high-vegetation cover 0.002 (ignored), leaf area index 1.47. Canopy parameters are those of that
vegetation type in Table 8.1 of the IFS documentation Cy41r2 (the ERA5 cycle), Part IV. The script reproduces the
v2 soil composite at 10 UTC exactly; moving to 12 UTC leaves soil moisture unchanged (within 0.0002) and cools the
two upper soil layers by 0.2 and 0.4 K. Initial soil moisture is about halfway between wilting point (0.151) and
field capacity (0.346). The vegetation values are not tuned: the resulting surface partition is accepted as it is.

**ERA5 data.** The per-day LS2D cache of 2026-02 (`$SCRATCH/LS2D_ERA5/cass`, model levels and surface) was purged
from scratch and was never put on tape. It is not needed: the atmospheric profiles and forcing come from the CASS
tables, and the radiation background composite in `cass_ls2d_input.nc` is kept. The ERA5 surface fields downloaded
again on 2026-09-29 (`$SCRATCH/CASS_LES/shared_data/era5_land/`, one small file per year) and
`cass_land_composite.py` are now the only source of the v3 soil and vegetation. They are on tape with the other
run inputs: `/home/m/mpowell/CASS_LES/shared_data/shared_data_v3_2026-09-29.tar`. ERA5's own surface fluxes for the
case days, used only for comparison, are in `era5_flux_context_2026-09-29.tar` in the same folder.

**Roughness lengths, two choices (settled 2026-09-29, also for the wind experiment).**
- Heat: z0h = 0.003 m, as in v2. It is within 20 % of the IFS value for crops and mixed farming (0.0025 m,
  Table 8.3), the vegetation type the canopy parameters come from. CASS does not constrain it: it gives one
  roughness length and used prescribed surface fluxes, for which the heat roughness has no effect on the fluxes.
- Momentum: z0m = 0.035 m, the CASS value, not the IFS value for crops (0.25 m). In the zero-wind base case it
  matters little. In the wind experiment it sets the friction velocity and the near-surface shear, and the choice
  there is the same: the CASS value.
- So z0m / z0h = 11.7 (IFS: 100 for low vegetation). For the methods: the heat roughness matches the IFS crop
  value and the momentum roughness follows CASS.

**Surface partition, accepted as it is (test at 64 x 64 columns, job 59083205, 2026-09-29).** Bowen ratio at
15 UTC, before any cloud: 0.40 (0.34 with the hand-set vegetation of v2; CASS table 0.72). At 18 UTC, with clouds in
a small domain and therefore only indicative: 0.35 (0.28; CASS 0.67). The sum of sensible and latent heat flux at
18 UTC is 517 W/m2 against 537 in the CASS table. ERA5's own fluxes at the site, averaged over the 119 case days,
give 0.60 at 15 UTC and 0.49 at 18 UTC: between the model and the CASS composite. The model is therefore wetter
than both. Cloud base and onset are expected to differ from the benchmark for this reason.
The full-size 1D member gives the definitive partition.

**Output.** Second dump stream (`[dump] swhf`): u, v, w, thl, qt, p as float32 below 6 km from 23700 s to 38400 s
(11:58 to 16:03 local solar time), every 60 s for member 1 of `2stream` and `raytracer`, every 300 s otherwise.
Hourly full fields include p, ql and b. Statistics add the `qlcore` mask, per-process tendencies
(`swtendency`) and warm-rain process rates.

## 12. v3 run list (user, 2026-09-29)

All runs: aerosol off, 12:00 UTC start, v3 forcing and land surface (section 11).

| Run | Radiation | Surface | Wind | Members | Binary |
|---|---|---|---|---|---|
| 1 | 3D (ray tracer) | interactive | zero | 4 | validated `2.0.2-46-g0d7fd9b22` |
| 2 | 1D (two-stream) | interactive | zero | 4 | same |
| 3 | 1D | interactive | CASS composite winds | 1 | needs per-variable nudging |
| 4 | 1D | CASS prescribed fluxes | CASS composite winds | 1 | needs per-variable nudging |

Each comparison changes one thing: run 4 to run 3 the land surface, run 3 to run 2 the wind, run 2 to run 1 the
radiation. Run 4 is the one compared with the CASS composite. Runs 1 and 2 come first. The four 3D members with
the mean shortwave matched to 1D (`raytracer_swmatch`) are postponed, not dropped.

Runs 3 and 4 follow CASS: winds nudged toward the composite winds at all heights on 1 h; theta_l and q_t nudged
above 5 km on 1 h and not below. MicroHH applies one profile `nudgefac(z)` to every nudged variable, so this needs
a per-variable profile (small change in `src/force.cxx`), a rebuild and the bit-identity test. Runs 1 and 2 nudge
only theta_l and q_t and use the validated binary.

Remaining differences of run 4 from CASS: fixed droplet number 200 cm-3 (CASS: aerosol number 600 cm-3 in its
microphysics); grid 25 m, 25.6 km wide, 8 km deep (CASS: 20 m, 29 km, 16 km); RRTMGP instead of RRTMG.

## 13. Coriolis force with a zero geostrophic wind (found 2026-09-29) - fix in the wind experiment

`cass_base.ini` has `swlspres=geo` with `fc=8.5e-5`. When `cass_input.nc` holds no `u_geo`/`v_geo`, MicroHH fills
them with zero. The Coriolis force then acts on the full wind and the nudging cannot balance it. For a target
(U, 0) the steady wind above the boundary layer is U / (1 + a^2) times (1, -a), with a = f tau:

| Nudging time scale | Speed for a 5 m/s target | Turned by |
|---|---|---|
| 3 h (old default) | 3.7 m/s | 43 degrees |
| 1 h | 4.8 m/s | 17 degrees |
| 30 min | 4.9 m/s | 9 degrees |

Which old runs are affected (from `shared/cass_input.py` and the input files):

| Runs | Wind option | Geostrophic wind written | Affected |
|---|---|---|---|
| `base`, `no_aerosols` | CASS composite winds (no flag) | no | YES |
| `wind_u` sweep (archived 2026-04) | `--wind-u` | no | YES |
| `wind_geo` sweep | `--geo-wind` | yes, equal to the target | no |
| `wind_sun` | `--sun-wind` | yes, equal to the target | no (its direction error was the solar-azimuth time origin, fixed in 36e6ab0ef) |
| zero-wind runs | `--zero-winds` | no | no (no wind) |

For the affected runs the wind speeds are lower than labelled and the direction is turned, so speeds and anything
about wind direction relative to the sun cannot be used from them. Fix for v3 runs 3 and 4 and for the wind
experiment: write the geostrophic wind equal to the nudging target (time-dependent for the CASS winds, which needs
`swtimedep_geo=true`), so that the Coriolis term vanishes at the target.

## 14. Surface fluxes: two densities in the land-surface coupling (found and fixed 2026-09-29)

### The error

With `[thermo] swupdatebasestate=1` (the MicroHH default, used in every CASS run) the model holds two densities:
the density of the dynamics, frozen at the initial profile (1.1483 kg/m3 at the surface), and the density of the
thermodynamic base state, updated every step from the mean state. At the surface the second one follows the mean
skin values and falls to 1.06 to 1.08 kg/m3 at midday.

The land-surface model used the second density, in the bulk formulas of the surface energy balance and in the
conversion of H and LE to the kinematic fluxes the atmosphere receives. The atmosphere applies the kinematic fluxes
with the first density. The atmosphere therefore received more energy and more water than the surface energy
balance handed over. Upstream MicroHH has the same code. Radiation, microphysics, advection, diffusion and the
pressure solver use the first density.

| Energy the atmosphere received over energy the land model reported | 16 UTC | 18 UTC | 20 UTC | 22 UTC |
|---|---|---|---|---|
| v3 small test, old code | 1.073 | 1.078 | 1.078 | 1.068 |
| v2 full size, old code (start 10:30 UTC, so 14:30, 16:30, 18:30, 20:30 UTC) | 1.051 | 1.070 | 1.073 | 1.071 |

Over the day of the v3 small test: 1.05 MJ/m2 of energy (7.1 %) and 0.32 kg/m2 of water (7.0 %) too much.

### The fix

Commit c40a8615c: the land-surface model takes the density of the dynamics (`fields.rhorefh`, on the GPU
`fields.rhorefh_g`) wherever it used the thermodynamic one (`src/boundary_surface_lsm.cxx`, `.cu`; three lines).
The kinematic flux stays (skin minus air) over the resistance, as the surface-layer similarity assumes, and the
energy flux is that times the density of the dynamics times cp or Lv. An independent audit of the diagnosis and of
the fix (code and data) confirmed both and found nothing that blocks production.

### Tests (64 x 64, two-stream, same ini and input as the earlier tests)

| Test | Result |
|---|---|
| Received flux equals the reported flux | largest deviation of the ratio from 1: 7e-15 (old code: 0.080) |
| Surface energy balance closes | residual, net radiation minus H, LE, G and storage: mean 0.10, largest 2.0 W/m2, the same on the land side and on the atmosphere side (old code, atmosphere side: mean -24, largest 42 W/m2) |
| Prescribed-flux path unchanged | bit-identical: 0 of 1069 statistics variables differ over the whole run, 3D fields identical |
| Run with CASS winds (run 3 settings) | received equals reported, 9e-15 |
| Run 4 settings | the atmosphere receives the CASS table fluxes, largest difference 4e-11 W/m2 |

Size of the change in the small test (new code against old code, same seed):

| | 15 UTC | 18 UTC | 20 UTC | day |
|---|---|---|---|---|
| Energy delivered to the atmosphere | -4.3 % | -5.7 % | -6.1 % | -5.1 % |
| Bowen ratio, new (old) | 0.400 (0.401) | 0.344 (0.349) | 0.315 (0.312) | 0.311 (0.312) |
| Cloud base, new (old) | | 1312 (1337) m | 1487 (1537) m | |

- The change in delivered energy is less than the 7 to 8 % of the error, because the surface energy balance
  repartitions: the skin is 0.6 to 0.8 K cooler, the ground heat flux 4 % smaller, H + LE reported by the land
  model 1.6 % larger.
- First cloud 16:35 UTC (old 16:30), one statistics sample later, in both seeds.
- Second seed with both binaries (job 59086324), means over 17 to 23 UTC, change by the fix in seed 1 and seed 2
  against the difference between the seeds: cloud base -39 and -29 m (seeds differ by 1 to 9 m); highest cloud top
  -146 and -152 m (15 to 22 m); cloud cover -0.006 and -0.005 of 0.115 (0.002 to 0.003); liquid water path -1.0 and
  -1.9 of 8 to 9 g/m2 (0.1 to 1.0). Energy delivered -5.1 % in both seeds. Small domain (6.4 km, 100 m grid).

Further tests of the fixed binary:

| Test | Result |
|---|---|
| Restart at 46800 s with the fixed `sbatch_restart.sh` | same surface density of the dynamics (1.1483064403434942); received equals reported; H, LE, thl equal to the continuous run within 5e-10 |
| `swupdatebasestate=0`, old against fixed binary | 3D fields and 1084 of 1089 statistics variables identical; `thv_diff`, `thv_flux` differ in the last bit (4e-22 on 0.15) |
| Binary with the nudging factor per variable (8dfd95410) against the fixed binary, land test | 3D fields identical at 18000 s; 1096 of 1109 statistics variables identical, `thv_diff`, `thv_flux`, `ql_diff` differ in the last bit (1e-17 on 0.13) |
| Full size, ray tracer, host memory allowance of a shared job (57 GB) | model holds 26.7 GB on the host and 51 GB on the GPU (80 GB); Slurm's MaxRSS (42 to 59 GB) includes file cache |

### What it means

- v2 and every earlier land-surface run carry the error. It is the same in 1D and 3D, so differences between the
  two are affected only to second order, but absolute budgets are not closed: the atmosphere received 5 to 8 % more
  energy and water in daytime than the surface gave.
- To get what the atmosphere received in an old run, convert the kinematic surface fluxes of the statistics with
  the density of the dynamics (`rhorefh` in group `default`), not with `rhoh` of group `thermo`.
- `cass_input.py` converted the prescribed CASS fluxes with a third density (1.139) and Lv 2.5e6: the atmosphere
  received 0.8 % more than the table. With `--flux-model-density` (used for run 4) it receives the table values.
- Not changed, and small: the surface flux has no Exner factor while radiation has one (0.9 % of H); the land
  model linearises its own longwave emission (residual below 2 W/m2).
- The fix makes the coupling conserve energy and water; it does not make the density exact. The anelastic model
  carries one reference density (1.148 kg/m3 at the surface), about 4 % above that of the air at the first level
  at midday (1.109). For the methods: one clause saying so is enough (user, 2026-09-29).

## 15. Runs 3 and 4: winds (2026-09-29)

- `cass_input.py --cass-winds`: CASS composite winds, nudged on 1 h at all heights with their own factor
  (`nudgefac_u`, `nudgefac_v`), geostrophic wind equal to the nudging target (time dependent, `swtimedep_geo=true`).
  The model change that allows a nudging factor per variable is commit 5a2770606 (`src/force.cxx`, `.cu`); without
  `nudgefac_<variable>` in the input the old behaviour is kept.
- Configurations in `setup_no_aerosols_zero_wind.py`: `2stream_wind` (run 3) and `2stream_cass` (run 4); both with
  hourly 3D output only. `2stream_prescribed` (zero winds) stays as the test of the prescribed-flux path.
- Small tests: the wind nudging tendency equals minus (wind minus target) over 3600 s to 3e-17; thl and qt nudging
  is zero below 4.5 km. The wind stays within about 1 m/s of the target; more where the CASS target itself changes
  by several m/s per hour.
- With the CASS winds the interactive land surface gives a Bowen ratio of 0.60 at 15 UTC and 0.46 at 18 UTC, against
  0.40 and 0.34 without wind (ERA5 0.60 and 0.49, CASS table 0.72 and 0.67).

## 16. Cost of the statistics (2026-09-29)

From `cass.out` of the v2 production runs, wall time per 20 iterations against the number of radiation, statistics
and dump times in the interval:

| v2, member 1 | Iterations | Radiation and 60 s output | Statistics | Total |
|---|---|---|---|---|
| Two-stream | 3.6 h | 2.0 h | 6.1 h | 11.7 h |
| Ray tracer | 6.0 h | 13.3 h | 6.7 h | 26.0 h |

One statistics sample cost 133 s in v2 and costs 163 s in v3 (320 levels, four masks, tendencies). The statistics
are computed on the host, on one CPU thread. Sampling every 300 s costs 7.6 h per v3 run.

What the 163 s consist of (job 59087932, four full-size variants on one node, three samples each, wall time of an
iteration with a statistics sample minus the 18.7 s of one without):

| Variant | Statistics sample | Share of the production cost |
|---|---|---|
| Production: four conditional masks, tendencies, microphysics budget | 163 s | |
| Without tendencies | 144 s | tendencies 19 s, 12 % |
| Without conditional masks | 60 s | four masks 103 s, 63 %, about 26 s each |
| Without both and without the microphysics budget | 50 s | domain statistics 50 s, 31 % |

- The tendencies are a small share: dropping them would save 0.9 h per run. They are kept.
- The copies from the GPU to the host are not part of this cost: they are made at every 60 s output and are in the
  18.7 s.
- The statistics kernels carry OpenMP directives over the levels, but the Perlmutter build does not enable OpenMP,
  so the 32 CPU cores of a job are idle during the statistics. A build with OpenMP is untested here; it is the
  lever for the later sweeps.
- BEFORE ANY BUILD WITH OPENMP: `Dump::do_dump` stores two flags (`do_regular`, `do_hf`, `src/dump.cxx`) that
  `needs` and `save_dump` read later. With OpenMP the statistics and dumps run as a task on a second thread while
  the main thread calls `do_dump` again, so the flags can be reset while a dump is being written and the remaining
  variables are skipped without a message, in the regular stream as well. Fix (pass the two decisions to the
  writer instead of storing them) and test bit-identity first. Not done during production: no rebuild now.

Wall time projection for v3 (v2 shares scaled by the levels, measured radiation and statistics costs): two-stream
about 16 h (limit 20 h), ray tracer about 33 h (limit 48 h). One ray tracer step costs 116 s at sunrise (v2 at the
same sun height 112 s, daily mean in v2 54 s).

Each ray tracer job has a restart job chained to it (`sbatch_restart.sh`, jobs 59088462, 59088464, 59088467,
59088468), which exits at once when the run is complete.

## 17. 60 s cross-sections at two heights (2026-09-29, runs 1 and 2)

The 60 s cross-section of w was taken at `xy = 0`, the surface, where w is zero (in every run, v2 included). Runs 1
and 2 now have `[cross] xy = 0,100` and u, v in the cross-section list; runs 3 and 4 keep `xy = 0`.

- MicroHH applies the heights to every 3D variable of the list: 13 more slices per frame (at 100 m: u, v, w, thl,
  qt, ql, b and four radiation fluxes; at the lowest level: u, v). Full-level variables are at 12.5 and 112.5 m, w
  and the radiation fluxes at 0 and 100 m. About 23 GB more per member during the run.
- The eight inis were changed while the jobs were held and before any had started; the previous ini is kept in each
  folder as `cass.ini.before_cross_change`. The setup script writes the same inis (`V3_CROSS_XY`, `V3_CROSS_ADD`).
- Check (job 59088618, 64 x 64, same binary): same sequence of time steps and byte-identical 3D fields at 3600,
  10800 and 18000 s; 1105 of 1109 statistics variables identical, `thv_diff` and `thv_flux` differ in the last bit
  (relative 7e-16), as in the comparisons of section 14. The ray tracer run writes the new fields.
- The new fields are consistent: w at 25 m equals the layer depth times the convergence of u and v at 12.5 m to
  3e-16; the w cross-section at 100 m equals the 3D file.

## 18. Changes to the model source relative to upstream MicroHH (review of 2026-09-29)

Upstream reference: `origin/main` 736a2f5ef. The branch differs in 21 source files (447 lines added, 30 removed)
and in the build files. The radiation submodule is at the commit upstream pins (416a6706), unchanged.

| Change | Files | Switch, default | Effect on an upstream configuration |
|---|---|---|---|
| Land surface uses the density of the dynamics (c40a8615c) | `boundary_surface_lsm.cxx`, `.cu` | none | CHANGES the solution of every run with the land-surface model: section 14 |
| Nudging factor per variable (5a2770606) | `force.cxx`, `.cu`, `force.h` | input variable `nudgefac_<var>`, absent | none |
| Second 3D dump stream (c58b72478) | `dump.*`, `field3d_io.*`, `fields.cxx`, `thermo_moist.cxx` | `[dump] swhf`, off | none; when on, the sample times enter the time-step limiter, which leaves the steps unchanged only if each sample time is already an output or radiation time (true in v3) |
| `rs_scale` (5cf6006a7) | land-surface kernels, `boundary_surface_lsm.*` | `[land_surface] rs_scale`, 1.0 | none |
| Soil moisture nudging (544b1d2ec) | soil kernels, `boundary_surface_lsm.*` | `[land_surface] swnudge_theta`, off | none |
| `swscalesfc_to_2str` (5cf6006a7) | `radiation_rrtmgp_rt.*` | `[radiation] swscalesfc_to_2str`, off | none |
| `swqsqg_to_rad` (9ec3989a8, GoAmazon and ARM97SD) | `thermo_moist.*` | `[thermo] swqsqg_to_rad`, off | none |
| Build | `src/CMakeLists.txt`, `config/default.cmake`, `config/empireai_alpha.cmake` | | none on the solution |

- One change alters the physics of an existing configuration: the land-surface density. It is a correction of an
  error that upstream has as well; which density belongs in the bulk formulas is a question for the MicroHH
  developers (the air at the first level has 1.109 kg/m3 at midday, the density of the dynamics 1.148, the
  thermodynamic surface value 1.065).
- The Couvreux tracer, the decay and the `qlcore` mask are upstream features. No source change was found for the
  NaN crash noted in early 2026.
- Found in the review, not yet acted on: the dump stream stores two flags in `do_dump` that a build with OpenMP could
  reset while a dump is being written (our builds have OpenMP off); `rs_scale` could be replaced by scaling
  `rs_veg_min` and `rs_soil_min` in the case input; soil moisture nudging reads its target only for homogeneous
  surfaces but applies it regardless; the edit of `src/CMakeLists.txt` (cuFFT and cuRAND removed from the link line)
  belongs in the machine file.

## 19. Decisions of 2026-09-29 on the model changes, and the production tag

- Conditional masks: all four kept; pairing of 1D and 3D matters more than 4.6 h per run.
- `thv_diff`, `thv_flux` last-bit differences: accepted (order of a floating-point sum, not the physics).
- `rs_scale`: to be moved out of the model into the case input (scale `rs_veg_min` and `rs_soil_min`), which removes
  one change from the branch and makes the Bowen ratio sweep reproducible with upstream MicroHH. Not urgent: do it
  when the land-surface setup is next touched, and test once that scaled input values give the same result as the
  switch (so far from reading the code only).
- `swqsqg_to_rad` (snow and graupel passed to the radiation as ice, GoAmazon and ARM97SD): not used in CASS, left
  alone.
- Dump stream flags: see section 16.
- Restart chain and model failures: MicroHH stops itself when the CFL number is not finite or above 10 (checked
  every `outputiter` iterations), and the job then ends as FAILED. `sbatch_restart.sh` restarts only when the
  previous job (`PARENT_JOB`) ended as TIMEOUT, NODE_FAIL or PREEMPTED, when the last lines of `cass.out` are
  finite and when the restart files are finite (`shared/check_finite.py`). A run job cancelled by hand is not
  restarted either. `test_sbatch_restart.sh`: 19 checks.
- Production tag: `v3-production` = commit adc50ae7b (set 2026-09-29 evening, pushed to the fork), whose case
  scripts write the inis and inputs of runs 1 to 4 and whose source is that of the ONE production binary:

| Binary | Source commit | sha256 | Runs |
|---|---|---|---|
| `microhh_2.0.2-73-gadc50ae7b` | adc50ae7b | 4b3c10ab8b15b04de62c91479b46d71ce9c2f83563e310aafead9a7ef36a7fad | 1 to 4 |

  Read-only in `~/validated_builds/`, on tape in `/home/m/mpowell/CASS_LES/shared_data/binaries_v3_2026-09-29_h2ofix.tar`.
  It replaced `microhh_2.0.2-51-gc40a8615c` (runs 1 and 2) and `microhh_2.0.2-53-g8dfd95410` (runs 3 and 4)
  in all ten production folders on 2026-09-29 at 19:15, while every job was pending (the old link is kept as
  `microhh.before_h2ofix` in each folder). The two-stream path of the new binary is bit-identical to both
  (section 20, last subsection); the ray tracer path has the water vapour correction of section 20.

## 20. Surface energy balance residual of ray tracer runs: a diagnostic, not a leak (2026-09-29)

The old figures showed a residual of net radiation minus (H + LE + G + S) in ray tracer runs: zero before the first
clouds, down to -11 W/m2 at midday, up to +5 W/m2 in the late afternoon; zero in two-stream runs.

- How the residual was computed: `analysis/cass_analysis.py`, `seb_residual`. H, LE, G, S are the values of the land
  surface model (statistics group `land_surface`). Net radiation came from the profiles `sw_flux_dn`, `sw_flux_up`,
  `lw_flux_dn`, `lw_flux_up` of group `radiation` at the surface level.
- What the model does in ray tracer mode (`src/radiation_rrtmgp_rt.cu`): every shortwave call computes the
  two-stream fluxes (`rte_sw_rt.rte_sw`, l.1802) and, in daytime, the ray-traced ones (`raytracer.trace_rays`,
  l.1835). The land surface model receives the ray-traced surface fluxes (`store_surface_fluxes_rt`, l.133-153 and
  l.2240-2252: downwelling = direct + diffuse, upwelling from the ray tracer; read by the land model through
  `get_surface_radiation_g`, `src/boundary_surface_lsm.cu` l.113-116). The atmosphere is heated by the ray-traced
  absorption (`calc_tendency_rt`, l.59-85, l.2208-2219). The profiles `sw_flux_up`, `sw_flux_dn`, `sw_flux_dn_dir`
  of the statistics and of the cross-sections are filled with the TWO-STREAM fluxes (l.2367-2369, l.2562-2564).
  Longwave is one solver in both modes.
- The fluxes the land model uses are written: time series `sw_flux_sfc_dir_rt`, `sw_flux_sfc_dif_rt`,
  `sw_flux_sfc_up_rt` in group `radiation` (l.2608-2610), cross-sections of the first two (in our list).
- Test on v2 (four members, 06:30 to 17:30 local solar time, `analysis/characterize/seb_residual.py`):

| Two-stream minus ray-traced surface shortwave | rms difference from the residual | correlation |
|---|---|---|
| net | 0.48 W/m2 | 0.999 |
| downwelling | 1.98 | 0.999 |
| upwelling | 3.83 | 0.999 |
| direct | 17.9 | -0.03 |
| diffuse | 16.2 | 0.53 |

  The residual of the old figure (-11.4 to +5.0 W/m2) IS the two-stream net shortwave minus the ray-traced one. With
  the net radiation the land model receives the residual is 0.44 W/m2 in the mean and at most 1.1 W/m2 in 3D
  (1D: 0.10 and 0.56).
- v3 small ray tracer test (old binary, to 20 UTC, clouds from 16:35 UTC): residual with the ray-traced net
  radiation mean 0.42, largest 0.86 W/m2; with the two-stream net radiation mean -3.0, largest 14.0. Fixed binary
  (to 15 UTC, no clouds yet): 0.49 and 0.57 on the land and on the atmosphere side, received equals reported flux
  to 8e-15. The check of section 14 (mean 0.10, largest 2.0 W/m2) was a two-stream run.
- The ray tracer conserves energy (v3 small test, 13 to 20 UTC): net shortwave at the domain top minus net at the
  surface minus the absorption in the atmosphere is -0.03 to -0.27 W/m2 of up to 934; the heating applied to the
  model equals shortwave absorption plus longwave flux convergence to 0.000 W/m2. Two-stream minus ray-traced with
  clouds: domain top net -11.6, surface net -11.7, absorption below 0.7 W/m2.
- Consequences: in ray tracer runs take the net radiation and the surface shortwave from the `_rt` time series
  or cross-sections. `sw_flux_dn`, `sw_flux_up`, `sw_flux_dn_dir` (profiles and cross-sections) of a ray tracer
  run are what a two-stream calculation on the 3D run's clouds gives. `cass_analysis.load_stats` now returns the
  net radiation the land model receives (`Rnet`) and `sw_dn_sfc`, `sw_up_sfc`; `sw_dn`, `sw_up` stay two-stream.
  `shared/check_surface_coupling.py` does the same.
- The time axis of the old figure is local clock time (the features are at 9.8, 12.8 and 15.7 h local solar time).

### Where else the two-stream outputs of ray tracer runs were read (check of 2026-09-29)

| Analysis | Reads for 3D runs | Verdict |
|---|---|---|
| Shadow maps and composites: `characterize/composite.py` l.142-147, `characterize/neighbours.py` l.58-62, `cloud_roots/sw_surface_composite.py` l.181-194, `updrafts/slice_prep/prep_sw_surface.py` l.73-79, `cass_analysis.load_sfc_sw_dn`, `load_xy_files` (`sw_flux_sfc_rt`) | ray-traced direct + diffuse cross-sections | right |
| `base_comparison.ipynb` cells 3, 11, 15, 16, 23; OLD notebooks (`load_sfc_sw_dn`) | the same | right |
| Conditioned means (`compute_seb_cache.py`, `_gen_sweep_notebooks.py`) | `sw_flux_sfc_rt`, falling back to `sw_flux_dn` only when the ray-traced files are missing | right; the fallback is silent |
| Net radiation and its residual from the statistics (`cass_analysis.load_stats`; `Rnet` in `rs_scale_comparison.ipynb`, `wind_geo_comparison.ipynb`) | two-stream | WRONG by up to 11 W/m2 of about 600; module fixed, notebooks correct themselves when rerun |
| Shortwave heating or flux profiles | not read by any script or notebook | nothing affected |

- The shortwave-matched run (`swscalesfc_to_2str`) is matched inside the model, at every radiation step, to the
  two-stream surface flux of the SAME run (`src/radiation_rrtmgp_rt.cu` l.2266-2295): the ray-traced surface fluxes
  are multiplied by `sw_scale_factor` = mean two-stream downwelling over mean ray-traced downwelling. No output
  variable enters. The mean it matches is what 1D radiation gives on the 3D run's own clouds, not the mean of the 1D
  run. In that run the `_rt` outputs are the unscaled fluxes; the land receives them times `sw_scale_factor`
  (a time series in the statistics), and the heating is not scaled.
- v3 has cross-sections at two heights: the loaders (`cass_analysis.load_sfc_xy`, `load_xy_files`,
  `characterize/les_io.lowest`) now take the lowest level.

### The shortwave-matched run against the 1D run, from its statistics (four members each, v2 era)

Statistics of `sw_scale` retrieved from tape (`sw_scale.tar`) to `$SCRATCH/CASS_LES/restore/sw_scale/`. Same grid,
time origin and land surface as v2; no tracer.

| Domain mean, 10 to 17 local solar time | 1D | 3D | 3D matched |
|---|---|---|---|
| Surface shortwave the land received [W/m2] | 774.8 | 781.6 | 775.2 |
| Difference from 1D [W/m2] | | +6.8 | +0.4 |
| Liquid water path [g/m2] | 7.9 | 11.3 | 10.9 |
| Cloud cover | 0.138 | 0.140 | 0.140 |

- The matched run received the 1D run's mean shortwave within 0.4 W/m2 over the afternoon; hour by hour within
  -4.9 to +3.6 W/m2 (3D minus 1D: up to +15.5; spread of the 1D members up to 2.8).
- It keeps 88 % of the increase of the liquid water path. The result that the domain-mean flux does not explain the
  increase stands.
- Its energy balance closes with the scaled ray-traced net radiation (mean 0.47, largest 0.92 W/m2); with the
  unscaled `_rt` outputs it would show 5.6 and 12.2 W/m2.

### The two radiation code paths differ in clear sky (found and traced 2026-09-29)

v3 small tests, same binary and input, no clouds, same water vapour path and same flux at the domain top:

| UTC | Surface downwelling, two-stream run | Ray tracer run, its two-stream | Ray tracer run, ray-traced |
|---|---|---|---|
| 13 | 237.3 | +0.8 | +0.5 |
| 14 | 441.1 | +1.4 | +1.4 |
| 15 | 632.6 | +2.1 | +2.4 |
| 16 | 793.6 | +3.0 | +3.5 |

- The difference is in the direct beam and changes sign with height (16 UTC: -2.8 W/m2 at 6 km, -0.8 at 2 km, +2.9
  at the surface): the ray tracer path absorbs more aloft and less below, 0.4 % more reaches the surface.
- It is a difference between `radiation_rrtmgp` and `radiation_rrtmgp_rt` (same settings, same coefficient
  files), not a 3D effect. About 3 of the 7 W/m2 by which the domain-mean surface shortwave of 3D exceeds 1D
  around midday may come from it. The matched run is matched to the two-stream of the ray tracer path and carries
  the same offset (+1.7 to +3.6 W/m2 against the 1D run before and around the first clouds).

### How to state the domain-mean result until the clear-sky difference is traced (user, 2026-09-29)

"3D exceeds 1D by about 7 W/m2 in the afternoon mean, of which up to about 3 W/m2 may come from a clear-sky
difference between the radiation code paths rather than from 3D effects." The matched-run conclusion is
unaffected, since the matching absorbed the offset. Runs 1 and 2 use the two code paths, so the offset is inside
every 3D minus 1D domain-mean shortwave number of the production runs.

### Trace of the clear-sky difference (2026-09-29)

- Radiation library: the binary is built from the submodule `rte-rrtmgp-cpp` inside the MicroHH repository
  (`CMakeLists.txt` l.155-167, `main/CMakeLists.txt` l.48), commit 416a6706 of 2024-02-20, the commit upstream
  MicroHH pins on `main` and `develop`. Two-stream runs use its `src_cuda` and `src_kernels_cuda`, ray tracer runs
  its `src_cuda_rt` and `src_kernels_cuda_rt`: two copies of the gas optics and of the two-stream solver. The
  library's own repository (microhh/rte-rrtmgp-cpp, `main` at b602dc26) is 223 commits ahead of the pin, with
  rewritten gas optics kernels in both copies. The separate checkout `~/repos/rte-rrtmgp-cpp` is not used by MicroHH.
- Established from the model output (v3 small tests, clear sky): the flux at the domain top is the same (0.001
  W/m2); the optical depth of the direct beam per layer in the ray tracer path is 11 to 15 % larger at 8 km, equal
  near 3.5 km and 6 to 10 % smaller at the surface than in the two-stream path, at every height of the sun; the
  column total is 1 to 1.6 % smaller; the longwave agrees to 0.04 W/m2.
- Ruled out by reading: pressure, temperature and dry air columns passed by MicroHH (`src/radiation_rrtmgp.cu`
  l.920-935, `src/radiation_rrtmgp_rt.cu` l.1624-1642; the first reading took the water vapour for the same in
  both because both set it at the same place, and missed which object is passed on), the background column and the
  flux per spectral point at the domain top (`solve_shortwave_column` is the same code in both), the radiation
  settings, and the kernels for interpolation, major gases, minor gases, Rayleigh scattering and gas columns,
  which are the same mathematics in both copies.
- Standalone programs of the pinned library on one column of the model (`$SCRATCH/CASS_LES/debug/
  rte_codepath_test/run`, job 59090947): the two copies give the same fluxes (0.0000 W/m2) and the same optical
  depth per spectral point (ratio 1.000000). The library is not the cause.

### Cause: ray tracer runs give the shortwave gas optics the water vapour of the input file (upstream MicroHH)

- `Radiation_rrtmgp_rt::exec_shortwave_rt` writes the current water vapour to the device object
  (`src/radiation_rrtmgp_rt.cu` l.1635, `gas_concs_gpu->set_vmr("h2o", h2o)`) and computes the dry air column from
  it (l.1643), but hands the gas optics `gas_concs` (l.1724). That name is the host member
  (`include/radiation_rrtmgp_rt.h` l.305), filled once from group `init` of the input file
  (`src/radiation_rrtmgp_rt.cxx` l.1218-1219) and converted to a device object without a message
  (`rte-rrtmgp-cpp/include/Gas_concs.h` l.77). The library applies a profile with one column to all columns
  (`rte-rrtmgp-cpp/src_cuda_rt/Gas_optics_rrtmgp_rt.cu` l.864-898, l.1138-1140).
- So in ray tracer runs the shortwave (its two-stream and the ray tracer, which share the optical properties) uses
  the `h2o` profile of the input file, the same in every column and at every time. The longwave of ray tracer runs
  and everything in two-stream runs use the water vapour of the model (`gas_concs_gpu`, l.1167, l.1203).
- It came with upstream commit 6599ea277 (2024-01-31, "bring radiation up-to-date with new interface of ray
  tracer"), which replaced `Gas_concs_gpu gas_concs_subset(*gas_concs_gpu, col_s, n_col_subset)` by `gas_concs`.
  Upstream `main` (736a2f5ef) and `develop` (fd3414284) have the same line (l.1710 there). Nothing was sent upstream.
- In the CASS input `h2o` is the ERA5 profile (`shared/cass_input.py` l.428-430), the model starts from the CASS
  sounding. Input `h2o` over water vapour of the model, 13 to 16 UTC: 0.83 to 0.87 at the surface, 0.90 at 1 km,
  0.98 to 1.00 at 2 to 3.5 km, 1.27 to 1.32 at 6 to 8 km; column 0.95 to 0.97.
- Check (`rte_codepath_test/run_h2o`, job 59091317): standalone library on the column of the two-stream run at
  16 UTC with the atmosphere above, once with the water vapour of the model and once with the input profile (dry
  air column of the model in both). Standalone numbers times the sun distance factor of the run (0.968):

| Height | Direct beam, ray tracer run minus two-stream run | Standalone, input minus model water vapour | Optical depth ratio per layer, runs | Standalone |
|---|---|---|---|---|
| 8 km | +0.00 | +0.00 | 1.1452 | 1.1452 |
| 6 km | -2.78 | -2.78 | 1.0885 | 1.0885 |
| 4 km | -2.01 | -2.02 | 1.0236 | 1.0236 |
| 2 km | -0.82 | -0.83 | 0.9900 | 0.9900 |
| 1 km | +0.19 | +0.19 | 0.9281 | 0.9280 |
| surface | +2.90 | +2.90 | 0.8977 | 0.8977 |

  The offset is reproduced (fluxes within 0.01 W/m2, optical depth ratios to four digits). Absorbed shortwave at 16 UTC with the input profile: -3.1 W/m2 (-8.9 %)
  below 1 km, -0.9 between 1 and 2 km, -1.1 between 2 and 4 km, +2.1 (+4.1 %) between 4 and 8 km; net at the
  surface +2.3 W/m2.
- Reach: every ray tracer run of MicroHH since the commit (CASS up to v2, the small v3 tests, GoAmazon, ARM97SD),
  with a size set by the `h2o` of each input. 3D_RT_DPSCREAM is not affected: all its legs come from one program
  and one input file (`scripts/rt/run_raytracer_all.sh` l.49, l.123-153).
- The model heats the air with the ray-traced absorption, so 3D runs had less shortwave heating below 1 km and
  more aloft than 1D runs, unrelated to 3D transport: ray-traced heating of the model at 16 UTC against the
  two-stream run -9.9 % below 1 km, -2.8 % between 1 and 4 km, +4.1 % between 4 and 8 km.
- User, 2026-09-29: correct it before the 3D jobs start. v2-era runs will not be used in a publication; the JAS
  paper is not affected (offline fluxes).

### Whole-day small ray tracer test with the corrected density (2026-09-29)

`debug/no_aerosols_zero_wind_v3_rtfix/raytracer/rep_01` (64 x 64, binary of runs 1 and 2), job 59089981 to 29294 s
(ended by the time limit as planned), restart job 59089983 to 50000 s (COMPLETED). `check_surface_coupling.py`:

| Segment | Received over reported flux, minus one | Residual of the energy balance, mean | Largest |
|---|---|---|---|
| 0, to 20:08 UTC | 7e-15 | 0.39 W/m2 | 0.77 W/m2 |
| 1, to 24:24 UTC | 7e-15 | 0.26 W/m2 | 1.9 W/m2 |
| 1, 24:30 to 25:05 UTC | 7e-15 | | 15.2 W/m2 |

- The coupling passes for the whole day, the restart included.
- Evening transition, 24:24 to 25:10 UTC (cosine of the zenith angle 0.23 to 0.10): the land surface of the whole
  domain changes between two states from one 60 s cross-section to the next (latent heat flux 65 and 15 W/m2,
  upwelling longwave 485 and 468 W/m2), so a sample of the fluxes does not balance the net radiation of the
  same time. The shortwave is smooth. It is not the ray tracer and not the density correction: the two-stream
  test with the old binary and seed 2 (`_v3_seed2old`) shows the same (64, 60, 18, 48, 42, 11 W/m2 at 300 s),
  the other two-stream tests do not, the test with CASS winds is smooth. Cause not looked at. It is after the
  window of the analysis (10 to 17 local solar time).
- User, 2026-09-29: log it and leave it. A guess, not a diagnosis: the wet-skin fraction switching on and off as
  dew forms under weak turbulence (whole domain, two latent heat states, only without wind).

### Correction of the water vapour in the ray tracer shortwave, and its tests (2026-09-29)

- Commit adc50ae7b: `*gas_concs_gpu` in place of `gas_concs` at `src/radiation_rrtmgp_rt.cu` l.1724. It was the only
  place: the other `gas_concs` in that file are function arguments of the initialisation (l.351, l.653), and the
  longwave of the ray tracer class sets and passes the device object (l.1167, l.1203, l.1207-1216).
- Binary `2.0.2-73-gadc50ae7b`, built from the clean commit; `~/validated_builds/candidate_microhh_2.0.2-73-gadc50ae7b`,
  sha256 4b3c10ab8b15b04de62c91479b46d71ce9c2f83563e310aafead9a7ef36a7fad.
- Tests (64 x 64, `debug/no_aerosols_zero_wind_v3_h2ofix`, jobs 59096375 and 59096377): (1) the two radiation
  classes agree in clear sky, (2) the longwave of the ray tracer run agrees with the two-stream run, (3) the
  two-stream runs (runs 2, 3 and 4) are bit-identical to the tests with the binaries of 2026-09-29.
- The four 3D production folders point to the candidate binary since 17:35 (jobs pending before and after), the
  old link is kept as `microhh.before_h2ofix`. Rule of the user: if test (1) fails, point them back at once,
  before the jobs can start. If the tests pass: switch the six 1D folders only if (3) holds, then tag adc50ae7b
  as `v3-production`.
- Results (jobs 59096375 and 59096377, complete runs; two-stream runs to 50000 s, ray tracer to 29100 s):
  (1) PASS: ray tracer run's two-stream minus two-stream run 0.00 W/m2 at the surface, 2 km and 6 km at 13 to 16
  UTC, optical depth ratio per layer 1.0000 at 8 km and 0.9995 to 1.0002 at 12 m; ray-traced surface flux -0.2 to
  +0.6 W/m2 from the two-stream (Monte Carlo noise, as within the ray tracer run before). Absorbed shortwave by
  layer identical from the two-stream fluxes; ray-traced heating of the model -1.1 % below 1 km, 0.0 % between 4
  and 8 km (before: -9.9 %, +4.1 %). (2) PASS: longwave within 0.03 W/m2 at the surface and the top. (3) PASS:
  runs 2, 3 and 4 configurations bit-identical to the tests with the binaries of the morning: 0 of 225, 227 and
  218 statistics variables differ, every line of `cass.out` identical, all 381, 381 and 325 3D files
  byte-identical (`$SCRATCH/CASS_LES/debug/h2ofix_checks/bitident.py`, `clearsky.py`).
- Corrected ray tracer run: coupling passes (6.7e-15), residual 0.38 mean, 0.64 max W/m2; against the
  uncorrected ray tracer run: surface fluxes 0.5 % lower, first cloud 16:30 instead of 16:35 UTC, cloud base and
  cover alike, LWP within the seed spread.
- All ten production folders were switched to `microhh_2.0.2-73-gadc50ae7b` at 19:15 with 14 jobs pending and
  none running; tag `v3-production` on adc50ae7b, pushed.
- 2026-09-30: the shared QoS accrues age priority for only two pending jobs per user (`MaxJobsAccruePU = 2`), which
  would have serialised the eight base-case members in pairs behind runs 3 and 4. The eight jobs (zero age) and their
  restarts were cancelled and resubmitted as two whole-node jobs in the regular QoS, four members each: 1D 59136701
  (20 h), 3D 59136702 (48 h) with chained restart 59136703. Same binary, same folders. Run 3 (59088071) started
  2026-09-30 15:06; run 4 (59088074) stays in the shared QoS.

## 21. Analyses to rerun when the v3 production lands (plan of 2026-10-02)

State: runs 2, 3, 4 complete and checked (coupling pass, residual below 0.6 W/m2 before 24.5 UTC). Run 1 (3D,
four members, job 59136702, 40 h limit, restart 59136703) pending.

### 0. Groundwork before any rerun (does not need the 3D runs)

- One loader for v3 snapshots: the hourly dumps lack T, qi, u, v; the 60 s fields (`*_hf`, 23700 to 38400 s, to
  6 km, float32: u, v, w, thl, qt, p) lack ql. Liquid water rebuilt from thl, qt with the base state of the nearest
  hour matches the dumped ql (358 498 of 358 501 cloudy cells at 28800 s, values to float32 precision, 2026-10-02).
  So `snapshot.load` reads the 60 s fields and derives T, ql, thv, b; u and v come with them.
- Times by SOLAR hour, not seconds: v2 snapshots 28800 to 39600 s were solar 11.9 to 14.9; in v3 (origin 12 UTC)
  the same solar times are 23364 to 34164 s. Replace every `TIMES = (28800, ...)` (composite_summary, dilution,
  cluster, open_ground, neighbours, figures, depth_w, robust, overshoot, persistence) by solar hours mapped to the
  nearest 60 s field. The 60 s window covers solar 12.0 to 16.1.
- Ray tracer runs: surface shortwave for any analysis is `sw_flux_sfc_dir_rt + sw_flux_sfc_dif_rt` (statistics
  `sw_flux_*` are two-stream). Already done in `cass_analysis`, `check_surface_coupling`, `widening.surface_sw`.
- Expt name `no_aerosols_zero_wind_v3`; result folders separate from v2.
- 3D tracking (`track3d.py`): tested on 1D member 1 (job 59209403, 18 min). Untouched clouds of at least 16 cells
  live 6.3 min (p99 12, max 16), as in 2D; every track longer than 20 min has merges and splits (the longest shed
  100 to 235 pieces). Overlap tracking in 3D is still a site tracker. The lifetime of active clouds needs a core-based
  identity that survives splits of the shell; not available yet.
- Run 1 landed 2026-10-05 (job 59136702, 31.8 h, no restart). All ten runs checked: coupling passes, residuals below
  0.6 (1D) and 1.7 W/m2 (3D) before 24.5 UTC. The clear-sky offset is gone (3D minus 1D surface shortwave
  -0.3 to +0.6 W/m2 at 13 to 16 UTC). 16 to 23 UTC means, four members each: cover 0.117 in both; LWP 6.54 (1D)
  and 8.06 g/m2 (3D), ratio 1.23; cloud base 1435 and 1445 m; highest top 3331 and 3875 m; H 107.7 and 109.4,
  LE 344.6 and 345.8 W/m2. The LWP ratio is smaller than in v2 (about 1.4); 3D LWP is equal or lower until
  19 UTC (solar 12.4) and diverges after (ratio 1.74 at 22 UTC).

### 1. Checks, as each run lands

- `check_surface_coupling.py` per member and per segment (3D: ray-traced net radiation, restart segment); `check_finite`.
- Domain means 1D against 3D (cover, LWP, base, top, surface fluxes, shortwave received), with the clear-sky offset
  now gone: the 3D minus 1D surface shortwave before the first cloud must be about zero.

### 2. Benchmark chain (needs only runs 2 to 4; can start now)

- Run 4 against the published CASS results (reference data to be located; not on disk yet).
- Run 3 against 4 (interactive land vs prescribed fluxes, CASS winds), 2 against 3 (winds), then 1 against 2.

### 3. Characterize pipeline on v3 (all need run 1)

Order chosen so the headline statements come first.

| Order | Analysis | Scripts | Figure |
|---|---|---|---|
| a | Mean state, cloud base, core, root width, entrainment, activation | snapshot, parcel, activation, collect | 1, 4, 5 |
| b | Population: number against area at fixed cover; births, location, fate; suppression on open ground; merges | width_gap (arithmetic), births, merges, suppression | 17, 19 |
| c | Width and depth: depth-width, dilution, persistence, widening regression, width gap | cloud_w, depth_w, dilution, persistence, widening, width_gap | 10, 15, 16, 18 |
| d | Composites and forces: circulation under clouds, pressure split, root work | composite, composite_summary, pressure, updraft_budget, roots | 6, 7 |
| e | Open ground: divergence, subsidence, barrier | open_ground | |
| f | Timing, tracking, overshoot, clustering, neighbours | timing, track, lifetime, overshoot, cluster, segment, robust, neighbours | 8, 9, 11 to 14 |
| g | Layer budget from tendencies | budget | |

### 4. New with v3 only (60 s fields)

- Pressure split checked against the saved model pressure `p` (v2 had only the masked statistics).
- Buoyancy-pressure force at low level composited around clouds over the surface shortwave anomaly: the catchment
  as a computed object, and the outflow from shaded open ground (Jeevanjee and Romps 2016, doi:10.1002/qj.2683).
- Low-level divergence over open ground every 60 s from u and v at 100 m (cross-sections) and the 60 s fields:
  hypothesis A of the third effect, followed in time rather than at four snapshots.
- Barrier to initiation (surface parcel against the cloud-base level) every minute against the births on open ground.
- 3D cloud lifetimes, depth histories and pulses on persistent sites (`track3d.py`): replaces the withdrawn 2D lifetimes.

### 5. Not rerun

- Shortwave-matched run and every v2 3D number affected by the water vapour error; v2 results are not for publication.
- User notebooks: not edited; their loaders already take the lowest cross-section level and the ray-traced fluxes.

### 6. After the analyses

Tape archive of v3 (statistics, 60 s fields, cross-sections, restart files at the analysis times) before the 8-week
scratch purge; then delete from scratch.
