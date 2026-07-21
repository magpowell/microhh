#!/usr/bin/env python3
"""Single-snapshot w slice, interpolated zh → z (cell centers).

Mirrors prep_volume_slice.py / prep_couvreux_anomaly.py: reads from the
HDF5-backed netCDF-4 source, writes a netCDF-3 64-bit-offset file that
ParaView/scipy/etc. can read without HDF5 lib conflicts. The zh→z
interpolation is the standard half-step average:

    w_cc[k] = 0.5 * (w_zh[k] + w_zh[k+1])
"""

import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-nc", required=True, help="Source w.nc.")
    ap.add_argument("--dump-time", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output netCDF-3 path.")
    args = ap.parse_args()

    with nc.Dataset(args.in_nc) as ds:
        times = ds.variables["time"][:]
        idx = int(np.argmin(np.abs(times - args.dump_time)))
        t_actual = float(times[idx])
        if abs(t_actual - args.dump_time) > 1.0:
            raise SystemExit(
                f"Closest time {t_actual} differs from {args.dump_time} "
                f"by {abs(t_actual - args.dump_time):.1f}s — refusing."
            )
        zh = ds.variables["zh"][:].astype(np.float32)
        y = ds.variables["y"][:].astype(np.float32)
        x = ds.variables["x"][:].astype(np.float32)
        w_zh = ds.variables["w"][idx].astype(np.float32)            # (zh, y, x)
        if np.ma.isMaskedArray(w_zh):
            w_zh = w_zh.filled(0.0)

    # MicroHH convention: ktot cell centers z, ktot half-levels zh where
    # zh[k] is the *bottom* face of cell k (zh[0] = surface). Cell-center
    # interpolation needs the top face zh[k+1]; the file omits the very
    # top zh[ktot]. Pad the top with w=0 (no-flow upper BC, consistent
    # with MicroHH config).
    ktot, jtot, itot = w_zh.shape
    dzh = float(zh[-1] - zh[-2])
    zh_full = np.concatenate([zh, [zh[-1] + dzh]]).astype(np.float32)
    w_zh_full = np.concatenate(
        [w_zh, np.zeros((1, jtot, itot), dtype=np.float32)], axis=0
    )
    z = (0.5 * (zh_full[:-1] + zh_full[1:])).astype(np.float32)        # (ktot,)
    w_cc = (0.5 * (w_zh_full[:-1] + w_zh_full[1:])).astype(np.float32) # (ktot, j, i)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w", format="NETCDF3_64BIT_OFFSET") as out:
        out.createDimension("z", z.size)
        out.createDimension("y", y.size)
        out.createDimension("x", x.size)
        zv = out.createVariable("z", "f4", ("z",)); zv.units = "m"; zv[:] = z
        yv = out.createVariable("y", "f4", ("y",)); yv.units = "m"; yv[:] = y
        xv = out.createVariable("x", "f4", ("x",)); xv.units = "m"; xv[:] = x
        v = out.createVariable("w", "f4", ("z", "y", "x"))
        v.units = "m s-1"
        v.long_name = "vertical velocity, interpolated zh → z"
        v[:] = w_cc

    print(
        f"wrote {out_path}  t={t_actual:.0f}s  shape={w_cc.shape}  "
        f"min={float(w_cc.min()):.2f}  max={float(w_cc.max()):.2f}  "
        f"frac(w>0)={float((w_cc > 0).mean()):.3f}"
    )


if __name__ == "__main__":
    main()
