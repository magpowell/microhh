#!/usr/bin/env python3
"""
Extract ensemble-mean thl/qt nudge profiles from completed no_aerosols_zero_wind 2stream runs.

Reads the column output (cass.column.*.0000000.nc) from all 4 reps, averages
thl(z,t) and qt(z,t) across reps, then interpolates onto the time_ls grid so
the result can be used directly as timedep nudge profiles in MicroHH.

Output: nudge_profiles.nc with:
  time_ls[N]        — same time coordinate as in cass_input.nc timedep group
  z[kmax]           — LES vertical grid
  thl_nudge[N, z]   — ensemble-mean liquid-water potential temperature (K)
  qt_nudge[N, z]    — ensemble-mean total water mixing ratio (kg/kg)

Usage:
  python extract_nudge_profiles.py \\
      --run-dir $SCRATCH/CASS_LES/experiments/no_aerosols_zero_wind/2stream \\
      --output  $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc
"""

import argparse
import glob
import sys
import numpy as np
import netCDF4 as nc
from pathlib import Path


def find_column_file(rep_dir: Path) -> Path:
    """Return the column stats NC file in a run directory."""
    matches = sorted(rep_dir.glob("cass.column.*.0000000.nc"))
    if not matches:
        raise FileNotFoundError(f"No column stats file found in {rep_dir}")
    return matches[0]


def find_input_nc(rep_dir: Path) -> Path:
    p = rep_dir / "cass_input.nc"
    if not p.exists():
        raise FileNotFoundError(f"cass_input.nc not found in {rep_dir}")
    return p


def read_time_ls(input_nc_path: Path) -> np.ndarray:
    """Read the time_ls coordinate from cass_input.nc timedep group."""
    with nc.Dataset(input_nc_path) as f:
        return f.groups["timedep"].variables["time_ls"][:].data.copy()


def load_column_profiles(col_path: Path, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, profile[time, z]) from a column stats file."""
    with nc.Dataset(col_path) as f:
        t = f.variables["time"][:].data.copy()
        p = f.variables[var][:].data.copy()  # shape: (time, z)
    return t, p


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to no_aerosols_zero_wind/2stream scratch dir containing rep_01 … rep_04",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write nudge_profiles.nc",
    )
    parser.add_argument(
        "--reps", nargs="+", default=["01", "02", "03", "04"],
        help="Rep suffixes to average (default: 01 02 03 04)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect rep directories
    rep_dirs = [run_dir / f"rep_{r}" for r in args.reps]
    for d in rep_dirs:
        if not d.exists():
            print(f"ERROR: rep dir not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Read time_ls from first rep's cass_input.nc
    time_ls = read_time_ls(find_input_nc(rep_dirs[0]))
    print(f"time_ls: N={len(time_ls)}, range [{time_ls[0]:.0f}, {time_ls[-1]:.0f}] s")

    # Load thl and qt from each rep, average
    thl_all = []
    qt_all  = []
    z_ref   = None

    for rep_dir in rep_dirs:
        col_path = find_column_file(rep_dir)
        print(f"  reading {col_path.name} from {rep_dir.name}")

        t_col, thl_col = load_column_profiles(col_path, "thl")
        _,     qt_col  = load_column_profiles(col_path, "qt")

        with nc.Dataset(col_path) as f:
            z_col = f.variables["z"][:].data.copy()

        if z_ref is None:
            z_ref = z_col
        else:
            if not np.allclose(z_col, z_ref, atol=1e-3):
                print("WARNING: z grids differ across reps — using first rep's grid")

        # Interpolate each column profile onto time_ls (linearly in time)
        # thl_col: shape (N_t, kmax); interpolate along axis 0
        thl_interp = np.zeros((len(time_ls), len(z_ref)))
        qt_interp  = np.zeros((len(time_ls), len(z_ref)))
        for k in range(len(z_ref)):
            thl_interp[:, k] = np.interp(time_ls, t_col, thl_col[:, k],
                                          left=thl_col[0, k], right=thl_col[-1, k])
            qt_interp[:, k]  = np.interp(time_ls, t_col, qt_col[:, k],
                                          left=qt_col[0, k],  right=qt_col[-1, k])

        thl_all.append(thl_interp)
        qt_all.append(qt_interp)

    thl_mean = np.mean(thl_all, axis=0)  # shape: (N_time_ls, kmax)
    qt_mean  = np.mean(qt_all,  axis=0)

    print(f"Ensemble mean: thl [{thl_mean.min():.2f}, {thl_mean.max():.2f}] K, "
          f"qt [{qt_mean.min()*1e3:.3f}, {qt_mean.max()*1e3:.3f}] g/kg")

    # Write output
    with nc.Dataset(out_path, "w", datamodel="NETCDF4", clobber=True) as f:
        f.createDimension("time_ls", len(time_ls))
        f.createDimension("z", len(z_ref))

        v = f.createVariable("time_ls", np.float64, ("time_ls",))
        v[:] = time_ls
        v.units = "s"
        v.long_name = "Time since start (matches cass_input.nc timedep/time_ls)"

        v = f.createVariable("z", np.float64, ("z",))
        v[:] = z_ref
        v.units = "m"

        v = f.createVariable("thl_nudge", np.float64, ("time_ls", "z"))
        v[:] = thl_mean
        v.units = "K"
        v.long_name = "Ensemble-mean liquid-water potential temperature from no_aerosols_zero_wind 2stream"

        v = f.createVariable("qt_nudge", np.float64, ("time_ls", "z"))
        v[:] = qt_mean
        v.units = "kg kg-1"
        v.long_name = "Ensemble-mean total water mixing ratio from no_aerosols_zero_wind 2stream"

        f.n_reps = len(rep_dirs)
        f.source_dir = str(run_dir)
        f.reps = ", ".join(args.reps)

    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
