# Radiative coupling analysis tasks

## Task 1: α and γ_s diagnostic — COMPLETE
α(t) = Q_rho,0^cr / <Q_rho,0>. γ_s = δQ_rho,0^cr / δR_n^cr with γ_s ~ 0.20
(γ_H ~ 0.17, γ_LE ~ 0.04). Implementation: `radiative_coupling.ipynb`.

## Task 2: Linear Q_rho(z) profiles — COMPLETE
Domain-mean profile is approximately linear (Stevens quasi-steady).
Cloud-conditional profile is NOT linear — the 3D vs 1D difference is a
shift in the surface anchor (α), not a change in slope.

## Task 3: f_shadow model — COMPLETE (universal form found)
Shadow definition: SW_dn < SW_clr (no arbitrary threshold).

    f_sh = f_sh_dom + (1 - f_sh_dom) · cf · μ₀²

Zero fitted parameters. Validated on CASS (cf=0.01-0.18) and Tijhuis
(cf=0.20-0.78, Zenodo 15649286). Combined r=0.92, bias=+0.03, RMSE=0.10.

## Task 4: SW_out proxy — COMPLETE
SW_out ≈ median(SW_dn | SW_dn > SW_clr). r=0.999 on CASS, r=0.984 on Tijhuis.
Zero fitted parameters.

## Task 5: Full α prediction chain — COMPLETE
α = 1 + β(t) · [(1-a)·δSW_cr + δLW_1D_mean] / (ρ·cp·<Q_rho,0>)

Chain inputs (all observable or from cheap 1D model):
- cf, μ₀, f_sh_dom, SW_clr, SW_sh, median(SW>clr) — from surface SW field + cloud mask
- <Q_rho,0>, β(t) — from LES/obs eddy covariance
- δLW_1D_mean — single constant from 1D model
- a = 0.22 (surface albedo)

Results on CASS (no_aerosols_zero_wind):
- α_3D: r=0.976, bias=-0.04
- α_3D/α_1D − 1 (fractional growth-rate change): r=0.97, RMSE=0.13

Implementation: `delta_sw_prediction.ipynb`. Caches:
- `f_shadow_cache_v2.pkl` (shadow = SW_dn < SW_clr)
- `tijhuis_cache_v2.pkl` (same definition, Tijhuis dataset)
