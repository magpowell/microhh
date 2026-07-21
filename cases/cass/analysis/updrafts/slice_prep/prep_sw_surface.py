#!/usr/bin/env python3
"""Build a clean 2D NetCDF of surface SW↓ at one dump time, for ParaView.

Reads the raw MicroHH xy binary slice(s) for the requested simulation time
and writes <out_path> with dims (y, x), var `sw_total` [W m^-2], so that
the downstream pvbatch script can load it via NetCDFReader without having
to deal with zh/at-surface dim-name quirks from cross_to_nc.py.

  2stream  : sw_flux_dn  at the surface half-level
  raytracer: sw_flux_sfc_dir_rt + sw_flux_sfc_dif_rt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import netCDF4 as nc


def _find_xy_file(run_dir, var, t_sec):
    """Return path of the xy binary file for `var` at simulation second `t_sec`.

    MicroHH writes either at-surface files (4 dot-fields) or k-indexed cross
    files (5 dot-fields). Try both glob patterns.
    """
    run_dir = Path(run_dir)
    stamp = f"{int(round(t_sec)):07d}"

    # at-surface form: <var>.xy.000.<stamp>
    cands = sorted(run_dir.glob(f"{var}.xy.000.{stamp}"))
    if cands:
        return cands[0]
    # k-indexed cross form: <var>.xy.<halflevel>.<index>.<stamp>
    cands = sorted(run_dir.glob(f"{var}.xy.*.00000.{stamp}"))
    if cands:
        return cands[0]
    raise FileNotFoundError(
        f"No xy binary for {var} at t={t_sec}s in {run_dir}"
    )


def _read_xy_slice(path, jtot, itot, dtype="<f8"):
    arr = np.fromfile(path, dtype=dtype)
    if arr.size != jtot * itot:
        raise RuntimeError(
            f"{path}: {arr.size} elements, expected {jtot * itot}"
        )
    return arr.reshape(jtot, itot).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Rep directory.")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--dump-time", type=int, required=True,
                    help="Simulation seconds (e.g. 36000).")
    ap.add_argument("--out", required=True, help="Output NetCDF path.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    # Read grid from cass_input.nc (clean and canonical) or fall back to ql.nc.
    grid_src = run_dir / "ql.nc"
    if not grid_src.exists():
        grid_src = run_dir / "cass_input.nc"
    with nc.Dataset(grid_src) as ds:
        x = ds.variables["x"][:].astype(np.float32)
        y = ds.variables["y"][:].astype(np.float32)
    itot = x.size
    jtot = y.size

    if args.rt == "2stream":
        path = _find_xy_file(run_dir, "sw_flux_dn", args.dump_time)
        sw = _read_xy_slice(path, jtot, itot)
    else:
        p_dir = _find_xy_file(run_dir, "sw_flux_sfc_dir_rt", args.dump_time)
        p_dif = _find_xy_file(run_dir, "sw_flux_sfc_dif_rt", args.dump_time)
        sw = (_read_xy_slice(p_dir, jtot, itot)
              + _read_xy_slice(p_dif, jtot, itot))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use netCDF-3 64-bit offset — ParaView's NetCDFReader handles classic
    # NC reliably; HDF5-backed NC4 files trigger "NetCDF: HDF error" and
    # silently render nothing on Perlmutter's paraview/5.13.0-gpu module.
    with nc.Dataset(out_path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("y", jtot)
        ds.createDimension("x", itot)
        yv = ds.createVariable("y", "f4", ("y",))
        xv = ds.createVariable("x", "f4", ("x",))
        yv.units = "m"
        xv.units = "m"
        yv[:] = y
        xv[:] = x
        v = ds.createVariable("sw_total", "f4", ("y", "x"))
        v.units = "W m-2"
        v.long_name = (
            "surface SW down (sw_flux_dn)" if args.rt == "2stream"
            else "surface SW down (sw_flux_sfc_dir_rt + sw_flux_sfc_dif_rt)"
        )
        v[:] = sw

    print(f"wrote {out_path}  rt={args.rt}  t={args.dump_time}s  "
          f"min={sw.min():.1f}  max={sw.max():.1f}  mean={sw.mean():.1f}")


if __name__ == "__main__":
    main()
