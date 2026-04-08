# Task: Compute radiative coupling parameter α and partitioning coefficient β from LES output

## Background

We are investigating how 3D radiative transfer (Monte Carlo) versus 1D radiative transfer (ICA) affects shallow cumulus cloud depth and LWP through modification of surface fluxes beneath cloud roots. The theoretical framework is based on Stevens (2007, JAS, doi:10.1175/JAS3983.1), equation (26), which gives the growth rate of a nonprecipitating cumulus-topped boundary layer:

```
dh/dt = κ { Q̃_ρ,0 [1 - (1-k) h/η] / Δθ_ρ }
```

We modify this by replacing the domain-mean surface flux Q̃_ρ,0 with a cloud-root conditional flux:

```
Q̃_ρ,0^cr = ⟨Q̃_ρ,0⟩ + δQ̃_ρ,0^rad
```

where δQ̃_ρ,0^rad is the radiative perturbation to the surface buoyancy flux beneath cloud roots relative to the domain mean. This defines the radiative coupling parameter:

```
α = Q̃_ρ,0^cr / ⟨Q̃_ρ,0⟩ = 1 + δQ̃_ρ,0^rad / ⟨Q̃_ρ,0⟩
```

- α > 1 means the surface flux beneath clouds is enhanced (3D RT case: shadow displaced away from cloud root)
- α < 1 means the surface flux beneath clouds is suppressed (1D RT case: shadow directly beneath cloud)
- α = 1 means no radiative modulation

The ratio of cloud-layer growth rates between RT schemes is:

```
(dh/dt)|_3D / (dh/dt)|_1D ≈ α_3D / α_1D
```

The causal chain from radiation to cloud depth is:

```
RT geometry (3D vs 1D) → δSW↓^cr → δR_n^cr → δQ̃_ρ,0^cr → α → dh/dt → h, LWP
```

The link between the SW perturbation and the buoyancy flux perturbation goes through the interactive land surface scheme (HTESSEL-like). We define a partitioning coefficient β:

```
δQ̃_ρ,0^cr = β · δR_n^cr = β · (1 - a) · δSW↓^cr
```

where a is the surface albedo.

## Task 1: Compute β

β captures how the land surface scheme partitions a perturbation in net radiation into a perturbation in the surface buoyancy flux. It combines the Bowen ratio partitioning (how much goes to H vs LE) with the buoyancy weighting (H and LE contribute differently to Q̃_ρ).

### What to compute

The equivalent surface buoyancy flux is (following Stevens 2007, eqs. 12, 17):

```
Q̃_ρ,0 = a1 · Q_l,0 + a2 · R_0
```

where:
- Q_l,0 = w'θ_l' at the surface ≈ sensible heat flux / (ρ c_p) ≈ thl_fluxbot in MicroHH output
- R_0 = w'q' at the surface ≈ latent heat flux / (ρ L) ≈ qt_fluxbot in MicroHH output
- a1 ≈ 1
- a2 = R_v/R_d - 1 ≈ 0.608

So:

```
Q̃_ρ,0 ≈ thl_fluxbot + 0.608 · Θ · qt_fluxbot
```

where Θ is a reference potential temperature (~300 K). Check MicroHH output conventions for whether the fluxes are already kinematic or need conversion.

### Steps

1. At each timestep with cloud present, identify cloud-root columns. A "cloud-root" column is one where cloud base exists (liquid water path > 0, or ql > 0 at lowest cloud level — use whatever definition is already in the analysis pipeline).

2. For each timestep, compute:
   - ⟨Q̃_ρ,0⟩: domain-mean equivalent buoyancy flux
   - Q̃_ρ,0^cr: mean of Q̃_ρ,0 conditionally sampled over cloud-root columns only
   - δQ̃_ρ,0^cr = Q̃_ρ,0^cr - ⟨Q̃_ρ,0⟩

3. Similarly compute:
   - ⟨R_n⟩: domain-mean net radiation at the surface
   - R_n^cr: conditional mean over cloud-root columns
   - δR_n^cr = R_n^cr - ⟨R_n⟩

4. Compute β as:
   ```
   β = δQ̃_ρ,0^cr / δR_n^cr
   ```

5. Check whether β is approximately constant across timesteps or varies systematically. If it varies, characterize its dependence on soil moisture, stability, Bowen ratio, etc.

6. Also compute β separately for the sensible and latent components to understand the partitioning:
   ```
   β_H = δ(thl_fluxbot)^cr / δR_n^cr
   β_LE = δ(a2 · Θ · qt_fluxbot)^cr / δR_n^cr
   β = β_H + β_LE
   ```

7. Verify the decomposition: check that β · (1 - a) · δSW↓^cr ≈ δQ̃_ρ,0^cr. The SW term should dominate δR_n^cr but verify that LW differences are small.

## Task 2: Compute time series of α

### Steps

1. For each output timestep (or at whatever temporal resolution the surface fluxes are available), compute:
   ```
   α(t) = Q̃_ρ,0^cr(t) / ⟨Q̃_ρ,0(t)⟩
   ```
   for both the 3D RT and 1D RT simulations.

2. Plot the diurnal cycle of α for both RT schemes. We expect:
   - α_1D < 1 throughout the day, with the strongest suppression when clouds are present
   - α_3D ≥ 1, with the enhancement depending on solar zenith angle (SZA)
   - The difference α_3D - α_1D should peak when the sun is most oblique (morning/evening) because shadow displacement scales with SZA. At solar noon with the sun nearly overhead, the shadow displacement is small and the two schemes should converge.

3. Also plot:
   - Time series of δSW↓^cr for both RT schemes (this is the primary driver)
   - Time series of the cloud-root sample size (number of cloud-root columns) to assess statistical robustness — α is noisy when few clouds are present.

4. Compute the time-integrated or cloud-period-mean values of α for both schemes. These are the values to quote in the paper alongside equation (26).

### Expected output from preliminary data

From the table already computed:
- α_1D ≈ thl_fluxbot_cr / thl_fluxbot_all ≈ 0.065 / 0.099 ≈ 0.66
  (this is approximate — should use Q̃_ρ,0 not just thl_fluxbot)
- α_3D ≈ 0.105 / 0.101 ≈ 1.04

These should be recomputed using the full Q̃_ρ,0 = a1·thl_fluxbot + a2·Θ·qt_fluxbot.


## Notes

- The MicroHH LES uses an interactive land surface scheme similar to HTESSEL (Balsamo et al., 2019) for the ARM-SGP case.
- Make sure to handle the time period correctly — only include times when shallow cumulus clouds are present (avoid the morning spin-up before clouds form and any deep convective transition).
- Statistical significance: bootstrap or subsample the cloud-root conditional means to get confidence intervals on α and β.
- Units: be careful with MicroHH output units. 
