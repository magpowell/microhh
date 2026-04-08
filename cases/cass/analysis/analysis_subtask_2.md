# Task: Verify linear Q̃_ρ profiles from LES output

## Goal

Plot the equivalent "dry" virtual potential temperature flux profile Q̃_ρ(z) for both the 3D RT and 1D RT simulations. Two versions: domain-mean and conditionally sampled over cloud columns. We want to confirm that Q̃_ρ is approximately linear in z (validating Stevens 2007 quasi-steady assumption) and that the 3D vs 1D difference manifests as a shift in the surface anchor, not a change in slope.

## Definition

```
Q̃_ρ(z) = a1 * Q_l(z) + a2 * Θ * R(z)
```

where:
- Q_l(z) = w'θ_l' (resolved + subgrid flux of liquid water potential temperature)
- R(z) = w'q' (resolved + subgrid flux of total water specific humidity)
- a1 ≈ 1
- a2 = Rv/Rd - 1 ≈ 0.608
- Θ ≈ 300 K (reference potential temperature)

Check MicroHH output variable names — likely something like `thl_flux` or `w_thl` for Q_l and `qt_flux` or `w_qt` for R. Make sure to include both resolved and subgrid contributions if they are output separately.

## Plots to produce

### Figure 1: Domain-mean Q̃_ρ(z)

- Horizontal average of Q̃_ρ(z) at several representative times during the cloud period (e.g. 13, 15, 17 LST)
- Both 1D and 3D RT on the same panel
- Mark cloud base height η and cloud top z_a with horizontal dashed lines
- Overlay a straight line fit from the surface to z_a to assess linearity
- z-axis from surface to ~500 m above cloud top

### Figure 2: Cloud-conditional Q̃_ρ(z)

- Same as Figure 1 but conditionally averaged over columns where cloud is present (ql > 0 at any level, or LWP > 0)
- Same times, same axes
- Same linear fit overlay

### Figure 3: Comparison panel (for the paper)

- Two side-by-side panels: left = domain mean, right = cloud-conditional
- Single representative time (e.g. 15 LST, when α divergence is strong)
- 1D in blue, 3D in orange
- Linear fits as dashed lines
- Annotate the surface intercept values (Q̃_ρ,0) for each case
- Annotate η and z_a

## Diagnostics to compute

For each profile and each time:
1. Fit a straight line from z = 0 to z = z_a
2. Report the slope and intercept
3. Compute R² to quantify linearity
4. Compute the ratio of 3D intercept to 1D intercept — this should approximate α from the earlier analysis

## Notes

- Average over a ±15 min window around each target time to reduce noise
- For the cloud-conditional sampling, use the same cloud mask definition as in the α/β analysis
- If resolved and subgrid fluxes are separate, sum them before computing Q̃_ρ
- Normalize by the surface value if helpful for comparing slope between cases
