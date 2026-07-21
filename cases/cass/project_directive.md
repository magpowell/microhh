# CASS LES Project Directive (v3 — April 2026)

## Scientific Goal

Why does 3D radiative transfer produce **fewer, larger, longer-lived** shallow
cumulus clouds — at the same total cloud cover as 1D RT — over land?

The observation that motivates the question (afternoon time series for
`no_aerosols_zero_wind`, see `analysis/base_comparison.ipynb` §1):

- **Cloud cover** is nearly identical between 1D and 3D throughout the day.
- After ~14 LST, the two diverge in:
  - mean cloud effective diameter (3D larger),
  - cloud count (3D fewer),
  - cloud-top height (3D taller),
  - LWP (3D ~50% higher).

So the mass is being redistributed into **fewer, larger, deeper** clouds — not
into more cloud area. What boundary-layer mechanism produces this redistribution?

### Candidate mechanisms

| ID  | Mechanism                                               | Signature                                  |
|----|----------------------------------------------------------|--------------------------------------------|
| M1 | Stronger updrafts (faster `w_up`)                        | Larger `M_up` driven by `Δw`               |
| M2 | Wider updraft roots (larger `a_up` per object)           | Larger `M_up` driven by `Δa`               |
| M3 | Reduced entrainment dilution                             | Cores penetrate higher before evaporating  |
| M4 | Organisational / aggregation                             | Fewer, more clustered thermals; per-object size unchanged |

These are not mutually exclusive. The diagnostics below distinguish them.

### Predicted mechanism chain (3D)

1. 3D RT displaces the cloud-column shadow downwind/downsun (cloud no longer self-shadows).
2. Surface H, LE on the **sunlit side** of each cloud are sustained, while the **shaded side** is suppressed.
3. An **asymmetric warm pool** forms, centred a fraction of a cloud diameter to the sunlit side of the cloud root and reaching deep into the subcloud BL.
4. Inflow into the cloud root becomes **wider, stronger, and asymmetric** in the sun-parallel direction.
5. Cloud roots are wider and feed deeper updrafts → clouds grow larger and live longer.
6. Net effect: same total cloud cover, fewer cloud objects, larger mean diameter, taller cloud tops, higher LWP.

**1D counterfactual**: the parameterised radiation places the shadow directly beneath the cloud → symmetric surface-flux suppression beneath each cloud root → narrow, undercut roots → smaller, shorter-lived clouds.

The H1–H3 hypotheses below are direct tests of this chain.

---

## Key Results So Far

1. **Cloud population redistribution** (`base_comparison.ipynb` Figure 1, post ~14 LST):
   - Same cloud cover.
   - 3D mean cloud diameter ≈ 1.2× of 1D's; 3D cloud count ≈ 0.6× of 1D's.
   - 3D cloud-top height continues to climb through the afternoon while 1D's flattens.
   This is the mechanism question to be explained.
2. **LWP divergence**: 3D RT produces ~50 % higher LWP / ~60 % deeper clouds by late afternoon.
3. **Domain-mean `Q_ρ` profile approximately linear** — validates the quasi-steady assumption used in any column-budget framing.
4. **Mean-state nudge widens the gap** (unexpected; archived, see Observations).

---

## Primary Diagnostic — Couvreux Passive Tracer

The Couvreux tracer (Couvreux et al. 2010) is a surface-emitted scalar with
exponential decay (τ = 900 s) used to identify coherent rising thermals by their
boundary-layer history. Implemented in `no_aerosols_zero_wind_v2` (surface
flux F = 1×10⁻⁵, decay τ = 900 s, hourly 3D dumps; offline mask in
`analysis/updrafts/diagnostics.py`; full plan in `analysis/updrafts/PLAN.md`).

It is uniquely suited to this mechanism question because it:

1. **Extends below cloud base.** The shell decomposition's
   (`ql > 0` AND `w > 0`) core mask returns no points in the subcloud layer;
   the Couvreux mask returns a connected thermal-history mask from surface
   to cloud top.
2. **Decomposes `M_up = ρ · a_up · w_up`** — separates "each thermal is wider"
   from "each thermal is faster" without conflating with Eulerian fluid
   geometry.

### Hypotheses (sharpened)

| ID  | Statement                                                                                                   | Test                                                                                                                                                                  |
|----|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **H1** | **Asymmetric cloud-root forcing.** 3D RT produces an asymmetric warm pool around each cloud root, displaced toward the sunlit side, which drives wider and stronger inflow than the symmetric, self-shadowed cloud-root structure in 1D. The parallel/perpendicular contrast (relative to the solar-projected horizontal direction) is the signature. | Build the **tracer mass-flux composite `M_up(r, z)`** centred on cloud-root events, decomposed into sun-parallel (∥) and sun-perpendicular (⊥) components — same azimuth rotation as the existing b' / circulation composites. Predict: 3D shows wider and stronger `M_up` in **∥** vs 1D, with little change in **⊥**. Decompose `ΔM/M ≈ Δa/a + Δw/w` per (r, z). **Consistency check**: the 3D−1D `ΔM_up` composite must match the spatial structure of the 3D−1D `Δb'` composite already in `base_comparison.ipynb` §3 — same dipole, same parallel/perpendicular asymmetry. **Falsification**: if the 3D tracer composite is symmetric (no ∥/⊥ contrast), the warm-pool mechanism is wrong and the redistribution is driven by something else. |
| **H2** | The 3D excess mass flux originates **below cloud base** — in the subcloud layer.                         | Plot `M_up(z)` continuously from the surface; identify the lowest z at which 3D and 1D diverge by more than the ensemble σ. Heus & Jonker cannot do this. **Verification anchor**: the 3D−1D `Δb'` composite already shows the warm-pool anomaly extends deep into the subcloud layer (not just near surface). Tracer `ΔM_up(z)` must mirror that depth — divergence should appear *well below* z_b. If the tracer profile only diverges at cloud base, the diagnostic is missing the subcloud signal that the b' composite already shows. |
| **H3** | The redistribution is **per-object** (each thermal bigger/stronger), not aggregate (fewer thermals).     | Extract the size distribution of contiguous Couvreux-mask regions on 2D horizontal slices at cloud base. Rightward shift → per-object change (M1+M2). Count-only drop without size change → organisation (M4). |

**H3 is a scope addition** to the original PLAN, which explicitly deferred
morphology. The lightweight version requested here is *per-slice connected-
component statistics at cloud base only* (analogous to the `qlqi_path` cloud-
size analysis already in `base_comparison`), **not** full Lagrangian object
tracking.

### Cloud-root-centred composites — primary deliverable

Domain-mean `M_up(z, t)` profiles are useful but not sufficient. The H1
asymmetry signature is **only visible in cloud-root-centred composites**
because the warm-pool offset is defined relative to each cloud's local
geometry, not the domain. Composite analysis is therefore the central
methodology for H1, with domain-mean profiles as supporting context.

Pipeline:

1. **Detect cloud-root events** — reuse the event detection in
   `analysis/cloud_roots/cloud_root_composite_prep.py` (chord ≥ 1 km,
   11:30–17:00 LST), already validated for the existing b' / circulation
   composites in `base_comparison.ipynb` §3.
2. **Compute composite tracer mass flux** `M_up(r, z)` per event, where
   `r` is signed horizontal distance from the cloud-root centroid. Use
   the existing chord-normalisation (`r/L ∈ [−1, 1]`).
3. **Azimuth-rotate** into sun-parallel (∥) and sun-perpendicular (⊥)
   components — exactly the rotation already used for the shadow figures.
4. **Compare 1D vs 3D** in the same parallel/perpendicular layout used for
   `b'` and `(u', w')`.  Decompose `ΔM_up = ρ · (Δa · w + a · Δw + Δa · Δw)`
   to attribute the asymmetry to thermal width vs thermal velocity.
5. **Cross-link with b' composite** — the 3D−1D `ΔM_up` and `Δb'` composites
   must show co-located, similarly-oriented anomalies. If they don't, either
   the tracer mask is decoupled from the buoyancy field (mask issue) or the
   warm-pool hypothesis is wrong.

This subsumes the H1 test and provides the strongest visual link between
the surface-forcing chain (predicted above) and the diagnosed updraft
structure.

### Secondary diagnostic — entrainment ε(z) and detrainment δ(z)

Tracer-dilution entrainment `ε = −(1/χ) dχ/dz` (Romps 2010) and detrainment
`δ = ε − (1/M)·dM/dz` from mass continuity are reported as **3D−1D
contrasts (Δε, Δδ)**, not as absolute values. Pilot rep_01 gives
`χ(z_t) ≈ 1.05` instead of the literature 0.2–0.7; the cause is physical
(uniform surface emission keeps the environment tracer concentration
non-zero, so cores entraining environment air don't dilute toward zero).
Absolute χ / ε are therefore biased — but the bias is largely the same in
1D and 3D, so the **3D−1D difference** in ε and δ should still resolve
the mechanism's effect on plume mixing.

Recovering trustworthy *absolute* χ would need a localised source, an
anti-tracer subtraction, or a different ε estimator (Romps' bulk-MSE form).
Out of scope for this pass.

### Sanity-check baseline (mandatory)

Before any 3D-vs-1D mechanistic claim, the offline mask must reproduce
**Couvreux et al. 2010 §3.1–3.2** coverage values in the matched window
(LST 12–14):

- subcloud layer (`z < z_b`): `a_up ≈ 0.10–0.20`
- cloud layer (`z_b ≤ z ≤ z_t`): `a_up ≈ 0.03–0.05`

If these don't match within a factor of ~1.5, the mask implementation must be
revisited before mechanistic interpretation. The current pilot reports `⟨a_up⟩ ≈ 8 %`
averaged over `z ∈ [500, 2800] m` — but that's a single number across both
layers; the subcloud/cloud-layer split must be done first.

### Analysis windows (not time-means)

Macroscopic fields agree before ~14 LST and diverge after. Time-averaged
statistics over the full day blur the signal. All tracer analyses report in
two windows:

| Window     | LST    | Expected behaviour                                                                                                                                          |
|-----------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Matched   | 12–14 | 1D and 3D macroscopic fields agree → expect `ΔM_up ≈ 0`. If not, the diagnostic carries a bias and must be diagnosed before the diverged-window result is interpretable. |
| Diverged  | 14–18 | Macroscopic fields differ → mechanism signature is here.                                                                                                  |

Hovmöllers of `M_up(z, t)`, `a_up(z, t)`, `w_up(z, t)` are also required —
window means alone hide the temporal onset of the divergence.

---

## Active Experiments

| Experiment                       | Config basis                       | Sweep parameter                          | Values                       | Runs | Status                                                  |
|----------------------------------|-------------------------------------|------------------------------------------|------------------------------|-----:|---------------------------------------------------------|
| no_aerosols                      | base, standard winds                | --                                       | --                           |    8 | COMPLETE                                                |
| no_aerosols_zero_wind            | base, zero winds, no aero           | --                                       | --                           |    8 | COMPLETE                                                |
| **no_aerosols_zero_wind_v2**     | + Couvreux passive tracer           | --                                       | --                           |    8 | 2stream timed out (LST ~17.5 h) — restart queued; raytracer queued |
| **sw_scale**                     | no_aero_zero_wind, RT only          | `[radiation] swscalesfc_to_2str=true`    | --                           |    4 | SUBMITTED                                               |
| **rs_scale**                     | no_aero_zero_wind                   | `[land_surface] rs_scale`                | 0.25, 0.5, 1.0, 2.0, 4.0     |   40 | scripts written                                         |
| wind_geo                         | base, no aero                       | `u_geo` via `--geo-wind`                 | 2.5, 5, 7.5, 10 m/s          |   32 | COMPLETE                                                |

### Re-motivation under the mechanistic framing

The surface response remains central — the thermals that build cumulus are
generated by the surface heat flux, so anything that modulates that flux
(spatially via shadows; in partitioning via Bowen ratio; in horizontal
coupling via wind shear) directly tests a thermal-generation mechanism.

- **`no_aerosols_zero_wind_v2`** — primary experiment for H1–H3; tracer-bearing
  version of the baseline. The H1/H2/H3 tests live here.
- **`sw_scale`** — keep. Directly tests whether **spatial SW heterogeneity**
  (independent of any mean SW bias) is the upstream driver of the cloud
  population redistribution. If `sw_scale` reproduces the size redistribution,
  the mechanism is purely heterogeneity-driven.
- **`rs_scale`** — keep. Reframed: tests whether the **surface-flux
  partitioning** (sensible vs latent) modulates thermal vigour and width.
  Higher Bowen ratio → more sensible heat per unit Q_net → potentially
  stronger and/or wider thermals from sunlit patches (mechanism M1/M2).
  Run as planned; the original α-framework motivation no longer drives the
  interpretation but the experiment itself is still the right test of
  surface-response sensitivity.
- **`wind_geo`** — COMPLETE. Reframed: wind shear disrupts the spatial coupling
  between shadows and the thermals beneath them. Tests whether the cloud-
  population redistribution requires a coherent shadow–thermal column. The
  4-value sweep is sufficient; **`wind_azi` is dropped** (the wind_geo data
  already varies the shadow-advection timescale relative to thermal lifetime).

### sw_scale (mean-corrected 3D heterogeneity)

Raytracer surface SW scaled each timestep so domain mean = 2stream mean.
Preserves 3D heterogeneity, removes mean radiative bias. Answers: is the
3D effect purely spatial redistribution? 2stream fluxes computed internally
(no separate 1D run needed). Scale factor output as `sw_scale_factor` stat.

Requires `swscalesfc_to_2str` feature on branch `mpowell-local`.

Location: `$SCRATCH/CASS_LES/experiments/sw_scale/`

### wind_geo (geostrophic wind sweep) — COMPLETE

Realistic Ekman shear via `swlspres=geo`. Production sweep already finished;
analysis in `analysis/wind_geo_comparison.ipynb`.

Location: `$SCRATCH/CASS_LES/experiments/wind_geo/u_<VALUE>/`

### rs_scale (Bowen ratio sweep)

`rs_scale` multiplies `rs` before computing fLE in the SEB solver; Bo
diagnosed post-hoc from output H and LE. Zero winds, no aerosols, 4 reps × 2
RT per value. Requires MicroHH branch `mpowell-local` with rs_scale feature
(implemented in `land_surface_kernels.h`, `land_surface_kernels_gpu.h`,
`boundary_surface_lsm.*`).

Mechanistic interpretation: a higher Bo means more of `Q_net` becomes sensible
heat, which feeds buoyant thermals more directly. If 3D produces wider /
faster thermals only at high Bo, the cloud-population redistribution depends
on land-surface partitioning. If the redistribution survives at low Bo, the
SW-heterogeneity / thermal-generation pathway is robust to surface
properties.

Location: `$SCRATCH/CASS_LES/experiments/rs_scale/rs_<VALUE>/`

---

## Archived Results (α-framework closure)

The α-framework closure was demonstrated and **is no longer the project's
focus**. Retained here for completeness:

1. α framework validated: `α_3D / α_1D` predicts `dh/dt` ratio (r = 0.81, slope = 0.92).
2. `γ_s ≈ 0.20`, dominated by sensible heat (`β_H ≈ 0.17`, `β_LE ≈ 0.04`).
3. Time-mean `α_1D ≈ 0.69`, `α_3D ≈ 1.06` — 1D suppresses cloud-root flux by ~31 %.
4. Full α closure from observables, with zero-fit proxies for every unobserved
   quantity (`analysis/delta_sw_prediction.ipynb`; validated on CASS + Tijhuis
   Zenodo 15649286). `α_3D` r=0.976; `α_3D / α_1D − 1` r=0.97.

The closure tells us *how much* RT redistributes the surface buoyancy flux
relative to cloud locations. It does not explain *which* boundary-layer
mechanism produces the cloud-population redistribution. That is the v3
focus.

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
- **Wall time with passive tracer**: the Couvreux tracer (added in `no_aerosols_zero_wind_v2` for the H1/H2 updraft analysis) costs ~25% extra wall.
  Pre-tracer 2stream fit comfortably in 10 h; with the tracer it timed out at ~LST 17.5h.
  Bump 2stream to 12 h and raytracer to ~28 h on tracer-bearing experiments,
  or use the restart workflow below.

## Restart workflow (after a TIMEOUT)

Use `experiments/<case>/sbatch_restart_<case>.sh`.  It:

1. Auto-detects the latest savetime from `couvreux.NNNNNNN`.
2. Verifies all prognostic vars (`thl, qt, ql, w, u, v, couvreux, qr, nr`)
   exist at that timestamp.
3. Backs up `cass.ini` to `cass.ini.before_restart` and patches
   `[time] starttime` to the restart point.
4. Calls `microhh run cass` (no `init` step, no wipe of restart files).

**Post-processing after a restart**: `cass.ini` now has `starttime > 0`.
When converting binary dumps with `3d_to_nc.py` or `cross_to_nc.py`, pass
`-t0 0` to override the patched value and convert the **full** simulation
timeline, not just the restart segment.

```bash
# After restart finishes, for each rep:
python $MICROHH/python/3d_to_nc.py    -t0 0 -v thl qt ql w b u v couvreux
python $MICROHH/python/cross_to_nc.py -t0 0
```

---

## Gotchas

- **MicroHH `zi` broken for shallow Cu** -- saturates at domain top. Use min(thv_flux) in [500,3500]m or ql_frac for cloud base.
- Raytracer has larger SEB residual than 2stream (cause unknown).
- `cass_input.py` reads `cass.ini` from CWD -- must be called from within run dir.
- `swlspres=geo` silently fills u_geo/v_geo with zeros if absent from input.nc.
- Re-run setup scripts after restructuring -- stale symlinks break runs silently.
