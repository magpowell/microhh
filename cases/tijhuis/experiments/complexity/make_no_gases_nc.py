#!/usr/bin/env python3
"""
Create cabauw_input_no_gases.nc by zeroing all gas mixing ratios in
cabauw_input.nc. Run once before setup_complexity.py.

Usage:
    python make_no_gases_nc.py [src.nc [dst.nc]]

Defaults to the 20140325_t03 source case in SCRATCH.
"""

import sys
import shutil
from pathlib import Path

import netCDF4 as nc

# NOTE on water vapor in the ray tracer:
# Within the LES domain, MicroHH feeds the radiation solver with the prognostic
# qt field (specific humidity), NOT the h2o profiles in this input file.
# Zeroing h2o / h2o_bg here therefore only removes water vapour ABOVE the domain
# top (the background column). In-domain water vapour is still active because it
# is computed self-consistently by the LES dynamics.
# This is intentional for the no_gases / dark_ocean experiments: all prescribed
# gases (CO2, CH4, O3, N2O, ...) and the background water-vapour column are
# removed, while the simulated in-domain moisture is kept. The residual in-domain
# H2O absorption is expected to be small for shortwave: H2O absorption bands lie
# in the near-IR, leaving the visible window largely transparent, so the dominant
# shortwave signal remains cloud scattering/absorption.

DEFAULT_SRC = Path(
    "/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2"
    "/20140325_t03/cabauw_input.nc"
)
DEFAULT_DST = DEFAULT_SRC.parent / "cabauw_input_no_gases.nc"

# Gas variables to zero in each group.
# Includes everything requested (n2, o2 included per user request).

INIT_GAS_VARS = [
    "o3", "h2o", "co2", "ch4",
    "n2o", "n2", "o2", "co",
    "ccl4", "cfc11", "cfc12", "hcfc22", "hfc143a", "hfc125",
    "hfc23", "hfc32", "hfc134a", "cf4", "no2",
]

# /timedep gas profiles (time-dependent)
TIMEDEP_GAS_VARS = [
    "o3", "co2", "ch4",
    "o3_bg", "h2o_bg", "co2_bg", "ch4_bg",
]

# /radiation gas profiles (static background column)
RADIATION_GAS_VARS = [
    "o3", "h2o", "co2", "ch4",
    "n2o", "n2", "o2", "co",
    "ccl4", "cfc11", "cfc12", "hcfc22", "hfc143a", "hfc125",
    "hfc23", "hfc32", "hfc134a", "cf4", "no2",
]


def zero_group_vars(group, var_names: list[str]):
    for vname in var_names:
        if vname in group.variables:
            group[vname][:] = 0.0
            print(f"  zeroed  {group.path}/{vname}")
        else:
            print(f"  (skip)  {group.path}/{vname}  -- not found")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DST

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    print(f"Copying {src.name} -> {dst.name}")
    shutil.copy2(src, dst)

    print("Zeroing gas variables...")
    with nc.Dataset(dst, "r+") as ds:
        zero_group_vars(ds["init"],      INIT_GAS_VARS)
        zero_group_vars(ds["timedep"],   TIMEDEP_GAS_VARS)
        zero_group_vars(ds["radiation"], RADIATION_GAS_VARS)

    print(f"\nDone -> {dst}")


if __name__ == "__main__":
    main()
