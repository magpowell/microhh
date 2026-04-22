# CASS LES Project Directive (v2 — April 2026)

## Scientific Goal

Understand how 3D radiative transfer affects shallow cumulus cloud depth and LWP
over land, and what controls the magnitude of this effect.

Central framework (Stevens 2007):

```
dh/dt ~ alpha * <Q_rho,0> * [1 - (1-k) z_a/eta] / Delta_theta_rho
```

where alpha = Q_rho,0^cld / <Q_rho,0> measures how RT redistributes surface
buoyancy flux relative to cloud locations:

```
alpha = 1 + gamma_s * (1-a) * delta_SW_dn^cld / <Q_rho,0>
gamma_s = (1 - f_G) * (Bo + eps) / ((1 + Bo) * rho * cp),   eps ~ 0.073
```

Three controls on the 3D-1D difference:
1. **Solar geometry** -- shadow displacement ~ h_cld * tan(SZA)
2. **Bowen ratio** -- dry surfaces (high Bo) amplify alpha
3. **Wind speed** -- advects clouds over shadows; U_crit ~ 2*w_star quenches 3D advantage

---

## Key Results So Far

1. **alpha framework validated**: alpha_3D/alpha_1D predicts dh/dt ratio (r=0.81, slope=0.92)
2. **gamma_s ~ 0.20**: dominated by sensible heat (beta_H ~ 0.17, beta_LE ~ 0.04)
3. **Domain-mean Q_rho profile approximately linear** (validates quasi-steady assumption)
4. **alpha_1D ~ 0.69, alpha_3D ~ 1.06** (time-mean): 1D suppresses cloud-root flux by ~31%
5. **LWP divergence**: 3D RT produces ~50% higher LWP / ~60% deeper clouds by late afternoon
6. **Mean-state nudge widens the gap** (unexpected; archived, see Observations)
7. **Full α closure from observables**: α = 1 + β(t)·[(1-a)·δSW_cr + δLW_1D_mean]/(ρ·cp·<Q>),
   with zero-fit proxies for every unobserved quantity:
   - SW_out ≈ median(SW_dn | SW_dn > SW_clr)
   - f_sh = f_sh_dom + (1 - f_sh_dom)·cf·μ₀²  (validated on CASS + Tijhuis 220 scenes)
   - δLW: single constant from 1D model, same for both schemes
   - Shadow definition: SW_dn < SW_clr (no arbitrary threshold)
   - Result: α_3D r=0.976, α_3D/α_1D − 1 r=0.97 (see `analysis/delta_sw_prediction.ipynb`)

---

## Active Experiments

| Experiment | Config basis | Sweep parameter | Values | Runs | Status |
|---|---|---|---|---|---|
| no_aerosols | base, standard winds | -- | -- | 8 | COMPLETE |
| no_aerosols_zero_wind | base, zero winds, no aero | -- | -- | 8 | COMPLETE |
| **rs_scale** | no_aero_zero_wind | `[land_surface] rs_scale` | 0.25, 0.5, 1.0, 2.0, 4.0 | 40 | scripts written |
| **sw_scale** | no_aero_zero_wind, RT only | `[radiation] swscalesfc_to_2str=true` | -- | 4 | SUBMITTED |
| **wind_azi** | no_aero_zero_wind | wind toward sun, varying speed | TBD | TBD | TODO: finalize design |
| wind_geo | base, no aero | `u_geo` via `--geo-wind` | 2.5, 5, 7.5, 10 m/s | 32 | scripts written |

**rs_scale=1.0 note**: identical to no_aerosols_zero_wind. Can reuse existing data
instead of re-running (saves 8 runs). Keep in sweep table for plotting continuity.

### rs_scale (Bowen ratio sweep)

Tests alpha ~ (Bo + eps)/(1 + Bo) by scaling surface resistance. `rs_scale`
multiplies rs before computing fLE in the SEB solver; Bo diagnosed post-hoc from
output H and LE. Zero winds, no aerosols, 4 reps x 2 RT per value.

Requires MicroHH branch `mpowell-local` with rs_scale feature (implemented in
`land_surface_kernels.h`, `land_surface_kernels_gpu.h`, `boundary_surface_lsm.*`).

Location: `$SCRATCH/CASS_LES/experiments/rs_scale/rs_<VALUE>/`

### sw_scale (mean-corrected 3D heterogeneity)

Raytracer surface SW scaled each timestep so domain mean = 2stream mean.
Preserves 3D heterogeneity, removes mean radiative bias. Answers: is the
3D effect purely spatial redistribution? 2stream fluxes computed internally
(no separate 1D run needed). Scale factor output as `sw_scale_factor` stat.

Requires `swscalesfc_to_2str` feature on branch `mpowell-local`.

Location: `$SCRATCH/CASS_LES/experiments/sw_scale/`

### wind_azi (wind toward sun)

TODO: finalize design. Key decisions:
- Fixed afternoon azimuth vs time-varying u/v tracking solar position
- Parameter values (likely 0, 1, 2.5, 5, 7.5, 10 m/s)
- Discuss with advisor

### wind_geo (geostrophic wind sweep)

Realistic Ekman shear via `swlspres=geo`. Complements wind_azi by providing
physically realistic wind profiles with vertical shear.

Location: `$SCRATCH/CASS_LES/experiments/wind_geo/u_<VALUE>/`

---

## Archived Experiments

Code removed from repo (recoverable from git history).

| Experiment | HPSS | Scratch | Note |
|---|---|---|---|
| soil_moisture | `/home/m/mpowell/CASS_LES/soil_moisture/` (7 tars, verified) | deleted | 4.2 TB archived |
| mean_state_nudge | not archived (re-derivable) | deleted | 599 GB freed |
| base | not archived | deleted 2026-04-08 | superseded by no_aerosols |
| cs_veg (5 values) | `/home/m/mpowell/CASS_LES/cs_veg/` (5 tars, verified) | deleted | 4.7 TB archived, 40 runs |
| wind_u (5 values) | `/home/m/mpowell/CASS_LES/wind_u/` (5 tars, verified) | deleted | 5.7 TB archived, production COMPLETE |

---

## Run Configuration

- 4 reps per RT per parameter value (rndseed 1--4), 4 sims/node
- Raytracer: `--constraint=gpu&hbm80g`; 2stream: `--constraint=gpu`
- Completion criterion: last binary dump is `wl_skin.0046800`

---

## Gotchas

- **MicroHH `zi` broken for shallow Cu** -- saturates at domain top. Use min(thv_flux) in [500,3500]m or ql_frac for cloud base.
- Raytracer has larger SEB residual than 2stream (cause unknown).
- `cass_input.py` reads `cass.ini` from CWD -- must be called from within run dir.
- `swlspres=geo` silently fills u_geo/v_geo with zeros if absent from input.nc.
- Re-run setup scripts after restructuring -- stale symlinks break runs silently.

---

## Observations

- **mean_state_nudge widens 3D-1D gap**: LWP integral +88 vs +25 g/m2/h. Mechanism unclear --
  Hovmoller shows positive dq_t and d_theta_l inside/above cloud (more condensate + latent heating).
  Apparent entrainment dramatically higher (72 vs ~20 mm/s), possibly diagnostic artifact from
  nudge-maintained sharp gradients. See `analysis/cloud_roots/nudge_mechanism.ipynb`.
