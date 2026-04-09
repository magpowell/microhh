# Radiative coupling analysis tasks (COMPLETE)

## Task 1: Compute alpha and gamma_s (formerly beta)

alpha(t) = Q_rho,0^cr(t) / <Q_rho,0(t)> measures how RT geometry modulates
surface buoyancy flux beneath cloud roots. gamma_s = delta_Q_rho,0^cr / delta_Rn^cr
captures how the LSM partitions a radiation perturbation into buoyancy flux.

Decomposition: gamma_s = gamma_H + gamma_LE, where gamma_H ~ Bo/(1+Bo) * 1/(rho*cp)
and gamma_LE ~ 1/(1+Bo) * eps_v*Theta/(rho*Lv).

Implementation: `radiative_coupling.ipynb`, cells 3-8.

## Task 2: Verify linear Q_rho(z) profiles

Confirm Q_rho(z) is approximately linear in z (Stevens 2007 quasi-steady assumption).
Domain-mean profile IS linear; cloud-conditional is NOT. The 3D vs 1D difference is a
shift in the surface anchor (alpha), not a change in slope.

Implementation: `radiative_coupling.ipynb`, cells 10-12.

## Key results

- alpha_1D ~ 0.69, alpha_3D ~ 1.06 (time-mean)
- gamma_s ~ 0.20 (gamma_H ~ 0.17, gamma_LE ~ 0.04)
- alpha_3D/alpha_1D predicts dh/dt ratio with r=0.81, slope=0.92
