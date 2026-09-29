"""Overshoot scale dz = dw / N from cloud-base updraft speed and cloud-layer stratification.

python overshoot.py --expt no_aerosols_zero_wind_v2
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

import thermo as th
from snapshot import out_path

TIMES = (28800, 32400, 36000, 39600)
RTS = ("2stream", "raytracer")
LAYERS = ((0., 500.), (500., 1000.), (1000., 1500.), (0., 1000.))   # above cloud base
IZB = 2


def layer_mean(z, f, z0, z1):
    m = (z >= z0) & (z <= z1)
    return float(np.nanmean(f[m]))


def one(expt, rt, rep, t):
    src = out_path(expt, rt, rep, t)
    with xr.open_dataset(src) as ds, xr.open_dataset(src.with_name(f"parcel_{t:07d}.nc")) as par:
        z = ds["z"].values
        zb = float(ds["zb"].values[IZB])
        thv = ds["thv_env"].values
        n2 = th.grav / thv * np.gradient(thv, z)
        row = dict(rt=rt, rep=rep, t=t, lst=float(ds.attrs["lst_solar"]), zb=zb)
        for s in ("cu", "core"):
            w = ds[f"pt_{s}_w_{IZB}"].values
            row[f"w_{s}"] = float(w.mean())
            row[f"w_{s}_p90"] = float(np.percentile(w, 90))
        for z0, z1 in LAYERS:
            tag = f"{int(z0)}-{int(z1)}"
            row[f"N2_dry_{tag}"] = layer_mean(z, n2, zb + z0, zb + z1)
            for kind in ("undilute", "entraining"):
                B = par["B_parcel"].isel(zb_thr=IZB).sel(start="core", kind=kind).values
                row[f"N2_{kind}_{tag}"] = layer_mean(z, -np.gradient(B, z), zb + z0, zb + z1)
    return row


def summarise(d):
    rows = []
    for t, g in d.groupby("t"):
        a, b = g[g.rt == "2stream"], g[g.rt == "raytracer"]
        r = dict(t=t, lst=float(g.lst.iloc[0]))
        for s in ("w_cu", "w_core", "w_cu_p90", "w_core_p90"):
            r[f"{s}_1D"], r[f"d{s}"] = a[s].mean(), b[s].mean() - a[s].mean()
            r[f"d{s}_se"] = np.sqrt(a[s].var(ddof=1) / len(a) + b[s].var(ddof=1) / len(b))
        for c in [c for c in d.columns if c.startswith("N2_")]:
            r[f"{c}_1D"], r[f"{c}_3D"] = a[c].mean(), b[c].mean()
        for z0, z1 in LAYERS:
            tag = f"{int(z0)}-{int(z1)}"
            n2 = r[f"N2_dry_{tag}_1D"]
            for s in ("w_cu", "w_core", "w_core_p90"):
                r[f"dz_{s}_{tag}"] = r[f"d{s}"] / np.sqrt(n2) if n2 > 0. else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    d = pd.DataFrame([one(a.expt, rt, rep, t) for t in TIMES for rt in RTS for rep in range(1, 5)])
    out = out_path(a.expt, "2stream", 1, TIMES[0]).parents[2]
    d.to_csv(out / "overshoot_long.csv", index=False)
    s = summarise(d)
    s.to_csv(out / "overshoot.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(s.T.to_string(float_format=lambda v: f"{v:.4g}"))
