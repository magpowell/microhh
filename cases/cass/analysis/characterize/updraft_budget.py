"""Forces inside tracer-identified updrafts from MicroHH's own conditional statistics (5-min, couvreux mask).

python updraft_budget.py --expt no_aerosols_zero_wind_v2
"""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

import thermo as th
from les_io import Run
from snapshot import run_dir

HALF = 900.                      # averaging half-width [s]
FRACS = (0.25, 0.5, 0.75, 0.9)


def load(run, mask):
    f = run / f"cass.{mask}.0000000.nc"
    r = xr.open_dataset(f, decode_times=False)
    g = xr.open_dataset(f, group="default", decode_times=False)
    t = xr.open_dataset(f, group="thermo", decode_times=False)
    return r, g, t


def analyse(expt, rt, rep, centres):
    rd = run_dir(expt, rt, rep)
    lst = Run(rd).lst
    r, g, t = load(rd, "couvreux")
    _, g0, t0 = load(rd, "default")
    _, gq, _ = load(rd, "ql")
    time, z, zh = r["time"].values, r["z"].values, r["zh"].values
    b = th.grav * (t["thv"].values - t0["thv"].values) / t0["thv"].values          # (time, z)
    bh = np.full((time.size, zh.size), np.nan)
    bh[:, 1:-1] = 0.5 * (b[:, :-1] + b[:, 1:])
    P = -g["p_grad"].values                                                       # (time, zh)
    w = g["w"].values
    a = g["areah"].values
    cloudy = gq["area"].values
    rows = []
    for tc in centres:
        sel = np.abs(time - tc) <= HALF
        kb = int(np.flatnonzero(np.nanmean(cloudy[sel], axis=0) > 1.e-3)[0])
        zb = z[kb]
        m = lambda x: np.nanmean(x[sel], axis=0)
        wm, bm, Pm, am = m(w), m(bh), m(P), m(a)
        lay = (zh > 0.) & (zh <= zb)
        row = dict(rt=rt, rep=rep, t=tc, lst=float(lst(tc)), zb=zb,
                   w_zb=float(np.interp(zb, zh, wm)), KE_zb=0.5 * float(np.interp(zb, zh, wm)) ** 2,
                   work_b=float(np.trapz(bm[lay], zh[lay])), work_p=float(np.trapz(Pm[lay], zh[lay])))
        row["work_net"] = row["work_b"] + row["work_p"]
        for fr in FRACS:
            for name, x in (("w", wm), ("b", bm), ("P", Pm), ("area", am)):
                row[f"{name}_{fr:g}"] = float(np.interp(fr * zb, zh[1:-1], x[1:-1]))
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--t", type=int, nargs="+", default=[21600, 25200, 28800, 32400, 36000, 39600])
    a = ap.parse_args()
    import pandas as pd
    rows = [x for rt in ("2stream", "raytracer") for rep in (1, 2, 3, 4) for x in analyse(a.expt, rt, rep, a.t)]
    df = pd.DataFrame(rows)
    out = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / a.expt / "updraft_budget.csv"
    df.to_csv(out, index=False)
    print("wrote", out)
    for tc, d in df.groupby("t"):
        one, three = d[d.rt == "2stream"], d[d.rt == "raytracer"]
        print(f"\n=== solar {one.lst.mean():.2f} ===   cloud base 1D {one.zb.mean():.0f} m, 3D {three.zb.mean():.0f} m")
        for c, lab, f in (("w_zb", "updraft w at cloud base [m/s]", 1.), ("KE_zb", "kinetic energy at cloud base [J/kg]", 1.),
                          ("work_b", "work by buoyancy, surface to z_b [J/kg]", 1.),
                          ("work_p", "work by pressure force, surface to z_b [J/kg]", 1.),
                          ("work_net", "buoyancy plus pressure [J/kg]", 1.),
                          ("w_0.5", "w at 0.5 z_b [m/s]", 1.), ("b_0.25", "buoyancy at 0.25 z_b [1e-3 m/s2]", 1e3),
                          ("P_0.25", "pressure force at 0.25 z_b [1e-3 m/s2]", 1e3),
                          ("b_0.5", "buoyancy at 0.5 z_b [1e-3 m/s2]", 1e3), ("P_0.5", "pressure force at 0.5 z_b [1e-3 m/s2]", 1e3),
                          ("b_0.9", "buoyancy at 0.9 z_b [1e-3 m/s2]", 1e3), ("P_0.9", "pressure force at 0.9 z_b [1e-3 m/s2]", 1e3),
                          ("area_0.5", "updraft area fraction at 0.5 z_b", 1.)):
            m1, s1, m3, s3 = one[c].mean() * f, one[c].std(ddof=1) * f, three[c].mean() * f, three[c].std(ddof=1) * f
            se = np.sqrt(s1**2 / 4 + s3**2 / 4)
            print(f"  {lab:46s} 1D {m1:8.3f} ({s1:.3f})  3D {m3:8.3f} ({s3:.3f})  diff {m3 - m1:+8.3f}  diff/SE {(m3 - m1) / se:+5.1f}")
