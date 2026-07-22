# Handoff: returning to Perlmutter (written 2026-07-22 from Empire AI Alpha)

Gitignored by convention (`cases/HANDOFF_*.md`). Transfer it alongside the
data, or read it from the Alpha checkout.

## The one thing that matters: two code versions are now in play

`mpowell-local` has been merged with upstream `microhh/main`, so the code is
current and the raytracer submodule stays pinned at `416a6706` -- the same
commit upstream microhh pins, and what a fresh `clone --recursive` gives anyone.
Clean merge, no conflicts, `swqsqg_to_rad` intact.

Do NOT bump the raytracer submodule to its own main (`a969650c`, 216 commits
ahead). Upstream microhh has not adopted it, so it is not the tested
combination, and it changes the `trace_rays` API (extra args) and needs a
double-precision fix (`find_index` typed `float` not `Float`). None of that is
worth carrying in production. The nonuniform-dz feature is developed against
`a969650c` on its own branch because a PR to `microhh/rte-rrtmgp-cpp` targets
that repo's main; production and CASS have no reason to follow.

CASS is **not** finished -- more experiments expected -- and its reps were
built at `416a6706`, which is exactly where production still sits, so there is
no version split to manage. Keep a copy of each built binary outside the build
dir; on Alpha that is `~/validated_builds/` with a README recording the microhh
commit, submodule commit, toolchain and what was validated.

## Do not rebuild under a live restart chain

goamazon raytracer rep_01/02 were mid-restart-chain on Perlmutter when this was
written. Restart dumps are raw binary field data written by a specific build,
so swapping the binary under a chain that resumes from them is unsafe. Let them
finish, or accept that you are rerunning them anyway (see next section).

## Why goamazon may be worth rerunning rather than resuming

The ray tracer currently requires an equidistant vertical grid, which forces
boundary-layer resolution all the way to the 25.6 km domain top. A branch is
open to remove that:

- `rt-nonuniform-dz` in both `magpowell/microhh` and `magpowell/rte-rrtmgp-cpp`,
  based on their respective upstream mains (not on `mpowell-local`, so no
  site-specific code leaks into the PR).
- Chiel and Menno have been emailed about it. Nobody upstream is working on it
  and there is no commit history suggesting it was tried and rejected.
- Estimated saving is roughly a factor of 2.5 in cell count for the same
  near-surface spacing, in both memory and runtime.

If that lands before the goamazon raytracer reps are needed, rerunning on a
stretched grid is cheaper than finishing them on the uniform one.

Design notes for whoever picks it up: the tracer is null-collision (delta
tracking), so `k_ext` is a coefficient in 1/m and grid spacing never enters the
path integration -- the physics is already grid-independent. The background
atmosphere already traverses a nonuniform level array (`z_lev_bg`), so the
pattern exists in-tree. The work is replacing the constant `dz` multiply in the
cell index lookup. Prefer a uniform height->layer lookup table over a binary
search, because photons in a warp sit at scattered heights and a search
diverges. The null-collision grid can stay uniform in z; it only has to bound
extinction, not align with cells. The part most likely to bite is `s_min`, a
single global epsilon used to nudge photons across interfaces, which cannot
suit both thin near-surface and thick upper layers on a stretched grid.

## Behaviour change in setup_runs.py

All three cases (`goamazon`, `arm97sd`, `goamazon_shcu`) now:

- derive `MICROHH_DIR` from the script location, so no hardcoded repo path
- **require `SCRATCH` in the environment** and exit with a message if unset,
  instead of silently falling back to a personal path
- default `XR_PY` to `sys.executable`, so whichever interpreter runs the script
  is used
- take `IOP_FILE` from the environment, defaulting to `shared_data/` in-repo

Perlmutter sets `SCRATCH` itself, so nothing should need doing there. The E3SM
IOP files are public on the LCRC mirror
(`https://web.lcrc.anl.gov/public/e3sm/inputdata/atm/cam/scam/iop/`), so they no
longer need to come from `/global/cfs`.

## Alpha status, for context

Working: build (sm_90, toolchain in a mamba env under `$HOME` because Alpha's
login and compute nodes mount different `/cm/shared` trees), input pipeline,
and all three cases validated at debug scale. goamazon_shcu reproduced the
paper's composite `lhflx` peak of 382.8 W/m2 at 14.0 LT; arm97sd put ice, snow
and graupel onset at 13.75-13.83 CST peaking 16.1-16.3.

Blocked: no production runs. Robert Pincus has no project account on Alpha, so
there is no group scratch, and home is a hard 100 GB. Empire AI have
acknowledged the project was never set up. `/mnt/lustre` is a symlink into home
NFS, not the DDN Lustre the docs describe; the real filesystem is in fstab as
`/ddn2 -> /mnt/ddn` but is mounted nowhere.

Measured on H200: raytracer memory is 0.62 GB/Mcell, linear to within 1.6% over
84-134 Mcells. The 512x512x512 dz=50 m grid that was rejected on Perlmutter at
98% of an A100 sits at 59% of an H200.
