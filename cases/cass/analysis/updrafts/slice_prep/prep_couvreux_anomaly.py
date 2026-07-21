#!/usr/bin/env python3
"""Build a Couvreux-style standardized tracer anomaly slice for ParaView.

Why: rendering the raw `couvreux` field shows a high-density layer at the
surface (where the tracer is sourced) and almost nothing aloft, even
though the active updraft columns extend up through the cloud layer. The
Couvreux conditional-sampling threshold is local in z — thr(z) = ⟨C⟩_z +
m·max(σ_z(C), σ_min) — so a rendering threshold has to be local too.

We compute the per-(z) standardized anomaly

    a(z, y, x) = (C - ⟨C⟩_z) / max(σ_z(C), σ_min)

and write it as a netCDF-3 file under the variable name ``couvreux``
(intentionally — the rendering script uses that name). Active
thermals/updrafts read at ~1.5–3 σ at every height, so a single LUT/OTF
in σ-units shows them as coherent vertical columns the way Couvreux 2010
Fig. 2 does.

σ_min floor matches `diagnostics.compute_sigma_min`'s spirit: 5% of the
column max σ. Without it, the above-BL near-zero σ would inflate the
anomaly to absurd values.
"""

import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-nc", required=True, help="Source couvreux.nc.")
    ap.add_argument("--dump-time", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output netCDF-3 path.")
    ap.add_argument("--sigma-min-frac", type=float, default=0.05,
                    help="σ_min(z) = sigma_min_frac · max_z(σ_z(C)).")
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
        z = ds.variables["z"][:].astype(np.float32)
        y = ds.variables["y"][:].astype(np.float32)
        x = ds.variables["x"][:].astype(np.float32)
        # Read full column, float32 to halve RAM (still ~270 MB).
        C = ds.variables["couvreux"][idx].astype(np.float32)
        if np.ma.isMaskedArray(C):
            C = C.filled(0.0)

    C_mean = C.mean(axis=(1, 2)).astype(np.float32)         # (z,)
    sigma = C.std(axis=(1, 2)).astype(np.float32)           # (z,)
    sigma_min = float(args.sigma_min_frac * sigma.max())
    sigma_eff = np.maximum(sigma, sigma_min).astype(np.float32)

    anom = (C - C_mean[:, None, None]) / sigma_eff[:, None, None]
    anom = anom.astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w", format="NETCDF3_64BIT_OFFSET") as out:
        out.createDimension("z", z.size)
        out.createDimension("y", y.size)
        out.createDimension("x", x.size)
        zv = out.createVariable("z", "f4", ("z",)); zv.units = "m"; zv[:] = z
        yv = out.createVariable("y", "f4", ("y",)); yv.units = "m"; yv[:] = y
        xv = out.createVariable("x", "f4", ("x",)); xv.units = "m"; xv[:] = x
        # Keep the var name `couvreux` so the renderer's LUT/OTF lookups
        # under that name continue to work without modification.
        v = out.createVariable("couvreux", "f4", ("z", "y", "x"))
        v.units = "sigma"
        v.long_name = ("standardized Couvreux tracer anomaly: "
                       "(C - <C>_z) / max(sigma_z(C), sigma_min)")
        v.sigma_min_frac = float(args.sigma_min_frac)
        v.sigma_min = sigma_min
        v[:] = anom

    print(
        f"wrote {out_path}  t={t_actual:.0f}s  shape={anom.shape}  "
        f"min={float(anom.min()):.2f}  p99={float(np.percentile(anom, 99)):.2f}  "
        f"max={float(anom.max()):.2f}  sigma_min={sigma_min:.2e}"
    )


if __name__ == "__main__":
    main()
