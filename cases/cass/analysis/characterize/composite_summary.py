"""Ensemble summary of the per-snapshot composites and of the per-cloud tables.

python composite_summary.py --expt no_aerosols_zero_wind_v2
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from snapshot import snapshot_times

RTS = (("2stream", "1D"), ("raytracer", "3D"))
BANDS = ((0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def root(expt):
    return Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt


def load(expt, rt, t):
    return [xr.open_dataset(f).load() for f in sorted((root(expt) / rt).glob(f"rep_*/composite_{int(t):07d}.nc"))]


def box(d, v, dr, x0, x1, z0=None, z1=None):
    a = d[v].sel(dir=dr)
    a = a.sel(xl=slice(x0, x1))
    if z0 is not None:
        a = a.sel(znd=slice(z0, z1))
    return float(a.mean())


def scalars(d):
    m = dict(n_events=float(d["n_events"].sel(dir="parallel")), L=float(d["L_mean"].sel(dir="parallel")),
             zenith=d.attrs["sun_zenith_deg"])
    p = "parallel"
    for side, (x0, x1) in dict(sunward=(-0.75, -0.25), under=(-0.25, 0.25), shadow=(0.25, 0.75)).items():
        m[f"b_sfc_{side}"] = box(d, "b", p, x0, x1, 0., 0.1) * 1e3
        m[f"us_low_{side}"] = box(d, "us", p, x0, x1, 0., 0.15)
        m[f"H_{side}"] = box(d, "H", p, x0, x1)
        m[f"LE_{side}"] = box(d, "LE", p, x0, x1)
        m[f"hp_low_{side}"] = (box(d, "h_pb", p, x0, x1, 0., 0.15) + box(d, "h_pd", p, x0, x1, 0., 0.15)) * 1e3
        m[f"hpb_low_{side}"] = box(d, "h_pb", p, x0, x1, 0., 0.15) * 1e3
    m["b_sfc_dipole"] = m["b_sfc_under"] - m["b_sfc_shadow"]
    m["inflow_low"] = m["us_low_sunward"] - m["us_low_shadow"]
    m["H_domain"], m["LE_domain"] = float(d["H_domain"]), float(d["LE_domain"])
    for z0, z1 in BANDS:
        tag = f"{z0:g}-{z1:g}"
        for dr, s in (("parallel", ""), ("perpendicular", "_perp")):
            v = {k: box(d, k, dr, -0.25, 0.25, z0, z1) for k in ("w", "b", "a_pb", "a_pb_sub", "a_pb_cld", "a_pd")}
            m[f"w{s}_{tag}"] = v["w"]
            for k in ("b", "a_pb", "a_pb_sub", "a_pb_cld", "a_pd"):
                m[f"{k}{s}_{tag}"] = v[k] * 1e3
            m[f"beff{s}_{tag}"] = (v["b"] + v["a_pb"]) * 1e3
            m[f"total{s}_{tag}"] = (v["b"] + v["a_pb"] + v["a_pd"]) * 1e3
    # work along the root column, surface to cloud base [J/kg], with z_b
    zb = d.attrs["zb"]
    for k in ("b", "a_pb", "a_pd"):
        prof = d[k].sel(dir="parallel").sel(xl=slice(-0.25, 0.25)).mean("xl")
        m[f"work_{k}"] = float(np.trapz(prof.values, d["znd"].values * zb))
    m["work_total"] = m["work_b"] + m["work_a_pb"] + m["work_a_pd"]
    wz = d["w"].sel(dir="parallel").sel(xl=slice(-0.25, 0.25)).mean("xl")
    m["KE_top"] = 0.5 * float(wz.sel(znd=slice(0.95, 1.0)).mean()) ** 2
    m["w_max_root"] = float(wz.max())
    # per-cloud classification
    depth = (d["cloud_z_top"] - d["cloud_z_base"]).values + 25.
    area, core = d["cloud_area_m2"].values, d["cloud_core_cols"].values > 0
    m["n_cloud"] = float(depth.size)
    m["frac_core_number"] = float(core.mean())
    m["frac_core_area"] = float(area[core].sum() / area.sum())
    for dz in (300., 500., 1000.):
        m[f"frac_deeper_{dz:g}_number"] = float((depth > dz).mean())
        m[f"frac_deeper_{dz:g}_area"] = float(area[depth > dz].sum() / area.sum())
        m[f"n_deeper_{dz:g}"] = float((depth > dz).sum())
    m["depth_median"] = float(np.median(depth))
    m["depth_mean_core"] = float(depth[core].mean())
    m["depth_p90"] = float(np.percentile(depth, 90))
    m["top_max"] = float(d["cloud_z_top"].max())
    m["D_core_median"] = float(np.median(d["cloud_D"].values[core]))
    m["n_core"] = float(core.sum())
    m["n_forced"] = float((~core).sum())
    return m


def main(expt):
    rows = []
    for t in snapshot_times(expt, skip_first=False):
        for rt, lab in RTS:
            for d in load(expt, rt, t):
                rows += [dict(t=t, lst=d.attrs["lst_solar"], rt=lab, rep=d.attrs["rep"], metric=k, value=v)
                         for k, v in scalars(d).items()]
    df = pd.DataFrame(rows)
    df.to_csv(root(expt) / "composite_long.csv", index=False)
    g = df.groupby(["t", "metric", "rt"])["value"].agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count").unstack("rt")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["diff"] = g["mean_3D"] - g["mean_1D"]
    g["se"] = np.sqrt(g["sd_1D"]**2 / g["n_1D"] + g["sd_3D"]**2 / g["n_3D"])
    g["diff_over_se"] = g["diff"] / g["se"]
    g.to_csv(root(expt) / "composite_summary.csv")
    print("wrote", root(expt) / "composite_summary.csv")
    return g.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--show", nargs="*", default=[])
    a = ap.parse_args()
    g = main(a.expt)
    for m in a.show:
        for _, r in g[g.metric == m].sort_values("t").iterrows():
            print(f"{m:26s} {st.lt(r.lst)}  1D {r.mean_1D:9.3f} ({r.sd_1D:.3f})  3D {r.mean_3D:9.3f} ({r.sd_3D:.3f})  "
                  f"diff {r['diff']:+9.3f}  diff/SE {r.diff_over_se:+6.1f}")
