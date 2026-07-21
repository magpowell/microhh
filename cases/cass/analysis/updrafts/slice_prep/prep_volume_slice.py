#!/usr/bin/env python3
"""Extract a single-timestep 3D slice and write it as netCDF-3 (no HDF5).

Why this exists: ParaView's bundled NetCDFReader (5.13.0-gpu on Perlmutter)
fails to actually deliver data from the HDF5-backed netCDF-4 files MicroHH
writes — it reads metadata fine, then emits a stream of `NetCDF: HDF error`
messages and renders nothing. Converting just the single timestep we need
to netCDF-3 (CDF-2, 64-bit-offset) bypasses HDF5 completely and ParaView
ingests it without complaint.

Output layout:
  dims (z, y, x) — no time dim
  vars  z[z], y[y], x[x] (coords), <var>(z, y, x) [float32]
"""

import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-nc",   required=True, help="Source netCDF-4 file.")
    ap.add_argument("--var",     required=True, help="3D variable name.")
    ap.add_argument("--dump-time", type=int, required=True,
                    help="Simulation seconds (matches one of the times[] entries).")
    ap.add_argument("--out",     required=True, help="Output netCDF-3 file.")
    args = ap.parse_args()

    src = Path(args.in_nc)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(src) as ds:
        times = ds.variables["time"][:]
        idx = int(np.argmin(np.abs(times - args.dump_time)))
        t_actual = float(times[idx])
        if abs(t_actual - args.dump_time) > 1.0:
            raise SystemExit(
                f"Closest time {t_actual} differs from {args.dump_time} by "
                f"{abs(t_actual - args.dump_time):.1f}s — refusing."
            )
        z = ds.variables["z"][:].astype(np.float32)
        y = ds.variables["y"][:].astype(np.float32)
        x = ds.variables["x"][:].astype(np.float32)
        data = ds.variables[args.var][idx].astype(np.float32)
        # Force any masked values to zero (ql, couvreux are physically >=0).
        if np.ma.isMaskedArray(data):
            data = data.filled(0.0)

    with nc.Dataset(out_path, "w", format="NETCDF3_64BIT_OFFSET") as out:
        out.createDimension("z", z.size)
        out.createDimension("y", y.size)
        out.createDimension("x", x.size)
        zv = out.createVariable("z", "f4", ("z",))
        yv = out.createVariable("y", "f4", ("y",))
        xv = out.createVariable("x", "f4", ("x",))
        zv.units = "m"; yv.units = "m"; xv.units = "m"
        zv[:] = z; yv[:] = y; xv[:] = x
        v = out.createVariable(args.var, "f4", ("z", "y", "x"))
        v.units = "kg kg-1"
        v[:] = data

    print(
        f"wrote {out_path}  var={args.var}  t={t_actual:.0f}s  "
        f"shape={data.shape}  min={float(data.min()):.3e}  "
        f"max={float(data.max()):.3e}"
    )


if __name__ == "__main__":
    main()
