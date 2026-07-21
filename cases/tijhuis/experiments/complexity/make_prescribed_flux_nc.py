#!/usr/bin/env python3
"""
Create cabauw_input_prescribed_flux.nc for the prescribed-surface-flux experiment.

Reads H(t) and LE(t) from all 6 full-complexity reference runs (paper_results),
computes the ensemble mean at each time step, interpolates onto the time_surface
grid of the input NC, converts to kinematic units, and writes updated thl_sbot
and qt_sbot into a copy of cabauw_input.nc.

These time-varying kinematic fluxes are read by MicroHH when:
  [boundary]
  swboundary   = surface
  sbcbot[thl]  = flux
  sbcbot[qt]   = flux
  swtimedep    = 1
  timedeplist  = thl,qt

Run once before setup_prescribed_flux.py:
    python make_prescribed_flux_nc.py
"""

import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d

# === Paths ===
PAPER_RESULTS = Path(
    "/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS"
    "/complexity_20140325/paper_results/20140325_t03"
)
SRC_NC = Path(
    "/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2"
    "/20140325_t03/cabauw_input.nc"
)
DST_NC = SRC_NC.parent / "cabauw_input_prescribed_flux.nc"

REFERENCE_RUNS = [
    ("rrtmgp",    "rndseed1"),
    ("rrtmgp",    "rndseed2"),
    ("rrtmgp",    "rndseed3"),
    ("rrtmgp_rt", "rndseed1"),
    ("rrtmgp_rt", "rndseed2"),
    ("rrtmgp_rt", "rndseed3"),
]

# Physical constants
CP = 1004.0    # J kg-1 K-1
LV = 2.5e6     # J kg-1


def main():
    # --- Read stats from all 6 reference runs ---
    H_runs, LE_runs = [], []
    stats_time = None

    for rad, seed in REFERENCE_RUNS:
        path = PAPER_RESULTS / rad / seed / "cabauw.default.0000000.nc"
        with nc.Dataset(path) as ds:
            H_runs.append(ds["land_surface"]["H"][:].data.copy())
            LE_runs.append(ds["land_surface"]["LE"][:].data.copy())
            if stats_time is None:
                stats_time = ds["time"][:].data.copy()

    H_mean  = np.mean(H_runs,  axis=0)   # shape (217,)
    LE_mean = np.mean(LE_runs, axis=0)

    print(f"Stats time: {stats_time[0]:.0f} – {stats_time[-1]:.0f} s  "
          f"({len(stats_time)} steps)")
    print(f"H  ensemble mean: {H_mean.mean():.2f} W/m2  "
          f"(range {H_mean.min():.1f} – {H_mean.max():.1f})")
    print(f"LE ensemble mean: {LE_mean.mean():.2f} W/m2  "
          f"(range {LE_mean.min():.1f} – {LE_mean.max():.1f})")

    # --- Get target time axis and surface density from input NC ---
    # rho_sfc = p/(R*T): use surface pressure from timedep and surface potential
    # temperature from the init profile (thl_sbot in timedep stores kinematic
    # fluxes, not temperature, so it cannot be used here).
    with nc.Dataset(SRC_NC) as ds:
        time_surface = ds["timedep"]["time_surface"][:].data.copy()
        p_sfc  = float(ds["timedep"]["p_sbot"][0])
        T_sfc  = float(ds["init"]["thl"][0])        # surface potential temp (K)
        rho_sfc = p_sfc / (287.04 * T_sfc)

    print(f"\ntime_surface: {time_surface[0]:.0f} – {time_surface[-1]:.0f} s  "
          f"({len(time_surface)} steps)")
    print(f"rho_sfc (estimated): {rho_sfc:.4f} kg/m3")

    # --- Interpolate ensemble mean onto time_surface grid ---
    interp_H  = interp1d(stats_time, H_mean,  kind="linear", fill_value="extrapolate")
    interp_LE = interp1d(stats_time, LE_mean, kind="linear", fill_value="extrapolate")

    H_interp  = interp_H(time_surface)
    LE_interp = interp_LE(time_surface)

    # --- Convert to kinematic units ---
    thl_sbot = H_interp  / (rho_sfc * CP)   # K m/s
    qt_sbot  = LE_interp / (rho_sfc * LV)   # kg/kg m/s

    print("\nKinematic fluxes at time_surface points (K m/s  |  kg/kg m/s):")
    for i, t in enumerate(time_surface):
        print(f"  t={t/3600:.1f}h  thl_sbot={thl_sbot[i]:.4e}  "
              f"qt_sbot={qt_sbot[i]:.4e}  "
              f"(H={H_interp[i]:.1f} W/m2, LE={LE_interp[i]:.1f} W/m2)")

    # --- Write output NC ---
    print(f"\nCopying {SRC_NC.name} -> {DST_NC.name}")
    shutil.copy2(SRC_NC, DST_NC)

    with nc.Dataset(DST_NC, "r+") as ds:
        ds["timedep"]["thl_sbot"][:] = thl_sbot
        ds["timedep"]["qt_sbot"][:]  = qt_sbot

    print(f"Done -> {DST_NC}")


if __name__ == "__main__":
    main()
