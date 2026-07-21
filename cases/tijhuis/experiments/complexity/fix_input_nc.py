#!/usr/bin/env python3
"""
Fix cabauw_input.nc for compatibility with the current MicroHH version.

The current MicroHH radiation module (Radiation_rrtmgp::create) calls
create_timedep_prof(..., "time_rad", ktot) for timedeplist_gas entries.
This requires two things to exist in the timedep group:

  1. A 'z' dimension local to the timedep group (gas variables like o3 use it,
     but in the raw NC the 'z' dimension lives only in the root group; calling
     nc_inq_dim with the timedep group ncid on a root-scope dimid gives
     NC_EBADDIM).

  2. A 'time_rad' dimension and variable (same size/values as 'time_ls') so
     get_dimension_size("time_rad") and get_variable("time_rad", ...) succeed.

setup_cumulus_cases.py already does this when it builds the rt/ and standard/
subdirectory NCs. This script applies the same fix in-place to the top-level
cabauw_input.nc so that downstream scripts (make_no_gases_nc.py,
make_prescribed_flux_nc.py) that copy from it will produce correct derived NCs.

Usage:
    python fix_input_nc.py                           # fix 20140325_t03 default
    python fix_input_nc.py path/to/cabauw_input.nc  # fix an arbitrary NC
"""

import sys
from pathlib import Path

import netCDF4 as nc

DEFAULT_TARGET = Path(
    "/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2"
    "/20140325_t03/cabauw_input.nc"
)


def fix_nc(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    print(f"Fixing: {path}")
    with nc.Dataset(path, "a") as ds:
        nz = ds.dimensions["z"].size
        td = ds.groups["timedep"]

        # 1. Add local z dimension so nc_inq_dim(timedep_ncid, z_dimid) works.
        if "z" not in td.dimensions:
            td.createDimension("z", nz)
            print(f"  Added timedep/z  (size={nz})")
        else:
            print(f"  timedep/z already present (size={len(td.dimensions['z'])})")

        # 2. Add time_rad dimension + variable (= copy of time_ls).
        if "time_rad" not in td.dimensions:
            n = td.dimensions["time_ls"].size
            td.createDimension("time_rad", n)
            tr = td.createVariable("time_rad", "f8", ("time_rad",))
            tr[:] = td.variables["time_ls"][:]
            print(f"  Added timedep/time_rad  (size={n})")
        else:
            print(f"  timedep/time_rad already present")

    print("  Done.")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    fix_nc(target)
