"""Budget of total water and liquid water potential temperature in a layer, from the tendency statistics (swtendency=1).

The tendency profiles are snapshots at the sample times. The content of the layer comes from the mean profiles, so its
change is exact; the time integral of the snapshots carries a sampling error, which closure() returns as the residual.

python budget.py --run <run dir> --z0 0 --z1 1500 --t0 18000 --t1 36000
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

CP, LV = 1005., 2.501e6
SCALE = dict(qt=LV, thl=CP)          # layer integrals times this are in W m-2
TRANSPORT = ("advec", "diff", "limit")


def load(run):
    """Mean profiles, density and tendencies of all statistics segments of a run, on a rounded time axis."""
    out = []
    for f in sorted(Path(run).glob("cass.default.*.nc")):
        c = xr.open_dataset(f, decode_times=False)
        g = {k: xr.open_dataset(f, group=k, decode_times=False) for k in ("default", "thermo", "tend")}
        d = xr.Dataset({v: g["tend"][v] for v in g["tend"].data_vars if v.split("t_")[0] in SCALE})
        d["qt"], d["thl"], d["rho"] = g["thermo"]["qt"], g["thermo"]["thl"], g["default"]["rhoref"]
        out.append(d.assign_coords(time=np.round(c["time"].values), z=c["z"].values).assign(zh=("zh", c["zh"].values)))
    d = xr.concat(out, dim="time", data_vars="minimal") if len(out) > 1 else out[0]
    return d.isel(time=np.unique(d["time"].values, return_index=True)[1])


def thickness(zh, z0, z1):
    """Thickness of each level that lies inside the layer z0 to z1."""
    return np.clip(np.minimum(zh[1:], z1) - np.maximum(zh[:-1], z0), 0., None)


def integral(profile, d, z0, z1):
    """Mass-weighted integral over the layer."""
    w = xr.DataArray(thickness(d["zh"].values, z0, z1), dims="z") * d["rho"]
    return (profile * w).sum("z")


def terms(d, var, z0, z1):
    """Layer integrals of every tendency of var, and the transport (advection, diffusion, limiter)."""
    names = [v for v in d.data_vars if v.startswith(f"{var}t_")]
    out = xr.Dataset({v.split("t_")[1]: integral(d[v], d, z0, z1) for v in names})
    out["transport"] = sum(out[p] for p in TRANSPORT if p in out)
    out["content"] = integral(d[var], d, z0, z1)
    return out


def closure(d, var, z0, z1, t0, t1):
    """Change of the layer content from t0 to t1 and the time integrals of the terms (trapezoid over the snapshots)."""
    b = terms(d, var, z0, z1).sel(time=slice(t0, t1))
    out = {k: float(np.trapezoid(b[k].values, b["time"].values)) for k in b.data_vars if k != "content"}
    out["change"] = float(b["content"][-1] - b["content"][0])
    out["residual"] = out["change"] - out["total"]
    return out


def report(run, z0, z1, t0, t1):
    d = load(run)
    for var in SCALE:
        c = closure(d, var, z0, z1, t0, t1)
        s = SCALE[var] / (t1 - t0)
        print(f"\n{var}, {z0:.0f} to {z1:.0f} m, {t0:.0f} to {t1:.0f} s, mean rates in W m-2")
        for k in ("change", "total", "residual", "transport", "advec", "diff", "micro", "rad", "ls", "subs", "nudge", "damp"):
            if k in c:
                print(f"  {k:10s} {c[k] * s:10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--z0", type=float, default=0.)
    ap.add_argument("--z1", type=float, default=1500.)
    ap.add_argument("--t0", type=float, default=18000.)
    ap.add_argument("--t1", type=float, default=36000.)
    a = ap.parse_args()
    report(a.run, a.z0, a.z1, a.t0, a.t1)
