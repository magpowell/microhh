# Radiative coupling analysis — reference equations

## Framework (Stevens 2007)

BL growth rate for nonprecipitating cumulus:

```
dh/dt = kappa * Q_rho_0 * [1 - (1-k) h/eta] / Delta_theta_rho
```

Radiative coupling parameter:

```
alpha = Q_rho_0^cr / <Q_rho_0> = 1 + delta_Q_rho_0^cr / <Q_rho_0>
```

- alpha > 1: cloud-root flux enhanced (3D RT, shadow displaced)
- alpha < 1: cloud-root flux suppressed (1D RT, shadow beneath cloud)

Growth rate ratio: `(dh/dt)|_3D / (dh/dt)|_1D ~ alpha_3D / alpha_1D`

## Surface buoyancy flux

```
Q_rho_0 = thl_fluxbot + eps_v * Theta * qt_fluxbot
```

where eps_v = 0.608, Theta ~ 300 K.

## Surface sensitivity gamma_s (formerly beta)

```
delta_Q_rho_0^cr = gamma_s * delta_Rn^cr = gamma_s * (1-a) * delta_SW_dn^cr
```

Decomposition:

```
gamma_s = f_H / (rho * cp) + eps_v * Theta * f_LE / (rho * Lv)
```

In terms of Bowen ratio:

```
gamma_s = (1 - f_G) * (Bo + eps) / ((1 + Bo) * rho * cp),   eps ~ 0.073
```

## Q_rho profile linearity (Stevens quasi-steady test)

```
Q_rho(z) = thl_w(z) + eps_v * Theta * qt_w(z)
```

Validated: domain-mean profile is approximately linear from surface to cloud top.
Cloud-conditional profile is NOT linear (expected — different flux convergence).

## Key results (no_aerosols_zero_wind)

- alpha_1D ~ 0.69, alpha_3D ~ 1.06 (time-mean)
- gamma_s ~ 0.20 (gamma_H ~ 0.17, gamma_LE ~ 0.04)
- alpha predicts dh/dt ratio with r=0.81, slope=0.92

## Implementation

All computations in `radiative_coupling.ipynb`. Uses `load_stats`, `load_sfc_xy`,
`cloud_mask_2d`, `compute_z_sl` from `cass_analysis.py`.
