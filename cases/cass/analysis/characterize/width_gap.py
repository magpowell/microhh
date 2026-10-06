"""Two bookkeeping checks on the width gap.

python width_gap.py [--expt no_aerosols_zero_wind_v2]     -> width_gap.csv, arithmetic.csv, figure 19
1. The 3D minus 1D mean width by hour, split by the age of the site: shift of the age distribution along the 1D
   width-age relation against wider at the same age.
2. Arithmetic: hourly cloud number, mean area and cover; figure 19 shows number against mean area per minute, where
   both runs lie on the iso-cover hyperbolae.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

import widening as wd
from births import MIN_AREA, RTS
from les_io import Run
from snapshot import out_path, run_dir

HOURS = ((12., 13.), (13., 14.), (14., 15.))


def with_age(expt, d):
    """Attach the track age [min] at the sample frame to the widening rows."""
    parts = []
    for (rt, rep), g in d.groupby(["rt", "rep"]):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("features.nc")) as ds:
            f = ds[["frame", "track", "age"]].to_dataframe().set_index(["frame", "track"]).age
        g = g.copy()
        g["age"] = f.reindex(list(zip(g.frame, g.track))).values / 60.
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


AGE_BINS = np.array([0., 2., 5., 10., 20., 40., 1.e4])   # minutes


def by_age(d):
    """3D minus 1D mean width = shift of the age distribution along the 1D width-age relation + offset at the same age."""
    import depth_w as dw
    rows = []
    for h0, h1 in HOURS:
        w = d[(d.lst >= h0) & (d.lst < h1)].copy()
        w["ab"] = np.digitize(w.age, AGE_BINS) - 1
        w["depth"] = w.W        # reuse the split of depth_w on width
        a, b = w[w.rt == RTS[0]], w[w.rt == RTS[1]]
        r = dw.split(a, b, ["ab"])
        rows.append(dict(hour=f"{h0:.0f}-{h1:.0f}", gap=r["total"], older_sites=r["shift"], wider_at_same_age=r["offset"],
                         age_1D=a.age.median(), age_3D=b.age.median()))
    return pd.DataFrame(rows)


def arithmetic(expt, min_area=MIN_AREA):
    """Per minute: cover, number and mean area of clouds of at least min_area; hourly means per member."""
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        run = Run(run_dir(expt, rt, rep))
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("features.nc")) as ds:
            f = ds[["frame", "area", "lst"]].to_dataframe()
        f = f[f.area >= min_area]
        g = f.groupby("frame").agg(n=("area", "size"), total=("area", "sum"), lst=("lst", "first"))
        g["cover"] = g.total / (run.xsize * run.ysize)
        g["mean_area"] = g.total / g.n
        g["rt"], g["rep"] = rt, rep
        rows.append(g.reset_index())
    m = pd.concat(rows, ignore_index=True)
    m["hour"] = np.floor(m.lst)
    h = m[(m.hour >= 12) & (m.hour <= 15)].groupby(["hour", "rt", "rep"]).agg(cover=("cover", "mean"), n=("n", "mean"), mean_area=("mean_area", "mean")).reset_index()
    e = h.groupby(["hour", "rt"]).agg(cover=("cover", "mean"), cover_lo=("cover", "min"), cover_hi=("cover", "max"), n=("n", "mean"), mean_area=("mean_area", "mean")).unstack("rt")
    out = pd.DataFrame({"cover_1D": e["cover"][RTS[0]], "cover_1D_lo": e["cover_lo"][RTS[0]], "cover_1D_hi": e["cover_hi"][RTS[0]],
                        "cover_3D": e["cover"][RTS[1]], "cover_3D_lo": e["cover_lo"][RTS[1]], "cover_3D_hi": e["cover_hi"][RTS[1]],
                        "n_1D": e["n"][RTS[0]], "n_3D": e["n"][RTS[1]], "area_1D": e["mean_area"][RTS[0]], "area_3D": e["mean_area"][RTS[1]]})
    out["n_ratio"], out["area_ratio"], out["cover_ratio"] = out.n_3D / out.n_1D, out.area_3D / out.area_1D, out.cover_3D / out.cover_1D
    return m, out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    fmt = lambda v: f"{v:.3g}"
    res = out_path(a.expt, "2stream", 1, 0).parents[2]
    d = with_age(a.expt, wd.load(a.expt))
    ba = by_age(d)
    ba.to_csv(res / "width_gap_age.csv", index=False)
    print("--- width gap [m] by hour split by age: older sites (shift along the 1D width-age relation) against wider at the same age")
    print(ba.to_string(index=False, float_format=fmt))
    m, ar = arithmetic(a.expt)
    ar.to_csv(res / "arithmetic.csv", index=False)
    print("\n--- arithmetic by hour: cover (member mean and range), number, mean area, and their 3D over 1D ratios")
    print(ar.to_string(index=False, float_format=fmt))
