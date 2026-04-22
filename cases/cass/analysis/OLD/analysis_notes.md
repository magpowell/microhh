# Radiative coupling framework — reference equations

## Stevens (2007) BL growth

    Δθ_ρ · dh/dt = -<Q̃_ρ(z_a)>

Under quasi-steady linearity: Q̃_ρ(z_a) = Q̃_ρ,0 · [1 - (1-k) z_a/η], k ~ -0.2.

Q̃_ρ = w'θ_l' + a₂·Θ·w'q_t',  a₂ = R_v/R_d - 1 ≈ 0.608

## Cloud-root conditional flux

    Q̃_ρ,0^cr = <Q̃_ρ,0> + conditional average over cloudy cols

    α = Q̃_ρ,0^cr / <Q̃_ρ,0> = 1 + δQ̃_ρ,0^cr / <Q̃_ρ,0>

Growth rate ratio (shared Δθ_ρ, k, <Q̃_ρ,0>):

    (dh/dt)_3D / (dh/dt)_1D ∝ α_3D / α_1D

## Radiation-to-flux partitioning (β framework)

    δQ̃_ρ,0^cr = β · δR_n^cr / (ρ·cp)      [both in kinematic units]

where β(t) is diagnosed per-timestep from the LES. In the analytical limit:

    β ≈ f_H/(ρ·cp) + a₂·Θ·f_LE/(ρ·L_v)
      ≈ (1 - f_G) · (Bo + ε) / (1 + Bo) · 1/(ρ·cp),  ε ≈ 0.073

Use the diagnosed β(t), not the analytical form. Typical value ~0.30.

## Net radiation perturbation

    δR_n^cr = (1-a)·δSW_cr + δLW_cr

- a = 0.22 (surface albedo)
- δLW_cr: use time-mean from 1D model, ~+45 W/m² (same constant for both schemes
  in the α_3D/α_1D ratio to avoid cancellation artifacts)

## Closing the loop (fully observable)

    α = 1 + β(t) · [(1-a)·δSW_cr + δLW_1D_mean] / (ρ·cp·<Q̃_ρ,0>)

With δSW_cr = <SW>_cr − <SW>_dm and:

    <SW>_cr,3D = f_sh · SW_sh + (1 - f_sh) · SW_out

- SW_out ≈ median(SW_dn | SW_dn > SW_clr)    [zero-fit proxy]
- f_sh = f_sh_dom + (1 - f_sh_dom) · cf · μ₀²    [zero-fit universal form]

Shadow definition throughout: SW_dn < SW_clr (same reference as SW_out).

## Validation

- CASS (no_aerosols_zero_wind): α_3D r=0.976, fractional change r=0.97
- Tijhuis (Zenodo 15649286, 220 scenes): independent validation of f_sh
  and SW_out proxy, covers cf=0.20-0.78 regime

## Implementation
All in `delta_sw_prediction.ipynb`. γ_s / β and Q_rho profiles in
`radiative_coupling.ipynb`.
