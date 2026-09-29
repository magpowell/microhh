"""Cloud depth against updraft speed at cloud base, per cloud: do 1D and 3D clouds follow the same relation?

python depth_w.py --expt no_aerosols_zero_wind_v2 [--x w_core_layer] [--min-points 4]
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

from snapshot import out_path

RTS = ("2stream", "raytracer")
TIMES = (28800, 32400, 36000, 39600)
W_BINS = np.array([0., 1., 2., 3., 4., 5., 50.])        # m/s
D_BINS = np.array([0., 400., 800., 1600., 1.e5])        # m, equivalent diameter


def load(expt, x="w_core_layer", min_points=4):
    rows = []
    for t, rt, rep in itertools.product(TIMES, RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"cloudw_{t:07d}.nc")) as ds:
            d = ds.to_dataframe()
            lst = float(ds.attrs["lst_solar"])
        d = d[d[x.replace("w_", "n_", 1)] >= min_points].copy()
        d["w"], d["t"], d["lst"], d["rt"], d["rep"] = d[x], t, lst, rt, rep
        rows.append(d[["t", "lst", "rt", "rep", "w", "depth", "D", "area", "z_base", "z_top"]])
    d = pd.concat(rows, ignore_index=True)
    d["wb"] = np.digitize(d.w, W_BINS) - 1
    d["db"] = np.digitize(d.D, D_BINS) - 1
    return d


def split(a, b, by):
    """3D (b) minus 1D (a) mean depth = shift along the 1D relation + offset at the same bin. Pooled clouds."""
    pa, pb = a.groupby(by).size() / len(a), b.groupby(by).size() / len(b)
    da, db = a.groupby(by).depth.mean(), b.groupby(by).depth.mean()
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx, fill_value=0.), pb.reindex(idx, fill_value=0.)
    da, db = da.reindex(idx), db.reindex(idx)
    da, db = da.fillna(db), db.fillna(da)
    return dict(total=b.depth.mean() - a.depth.mean(), shift=float(((pb - pa) * da).sum()), offset=float((pb * (db - da)).sum()))


def split_jackknife(d, by):
    a, b = d[d.rt == RTS[0]], d[d.rt == RTS[1]]
    full = split(a, b, by)
    var = {k: 0. for k in full}
    for grp in RTS:
        est = []
        for rep in range(1, 5):
            keep = d[~((d.rt == grp) & (d.rep == rep))]
            est.append(split(keep[keep.rt == RTS[0]], keep[keep.rt == RTS[1]], by))
        for k in full:
            e = np.array([x[k] for x in est])
            var[k] += (len(e) - 1) / len(e) * ((e - e.mean()) ** 2).sum()
    return {**full, **{f"{k}_se": np.sqrt(v) for k, v in var.items()}}


def binned(d):
    g = d.groupby(["t", "rt", "rep", "wb"]).agg(n=("depth", "size"), depth=("depth", "mean"), w=("w", "mean"), D=("D", "mean")).reset_index()
    m = g.groupby(["t", "wb", "rt"])[["n", "depth", "w", "D"]]
    mean, var, cnt = m.mean().unstack("rt"), m.var(ddof=1).unstack("rt"), m.count().unstack("rt")
    out = pd.DataFrame({"n_1D": mean["n"][RTS[0]], "n_3D": mean["n"][RTS[1]], "w_1D": mean["w"][RTS[0]], "w_3D": mean["w"][RTS[1]],
                        "depth_1D": mean["depth"][RTS[0]], "depth_3D": mean["depth"][RTS[1]],
                        "D_1D": mean["D"][RTS[0]], "D_3D": mean["D"][RTS[1]]})
    out["diff"] = out.depth_3D - out.depth_1D
    out["se"] = np.sqrt(var["depth"][RTS[0]] / cnt["depth"][RTS[0]] + var["depth"][RTS[1]] / cnt["depth"][RTS[1]])
    return g, out


def main(expt, x="w_core_layer", min_points=4):
    d = load(expt, x, min_points)
    g, b = binned(d)
    rows = []
    for t, dt in d.groupby("t"):
        n = dt.groupby(["rt", "rep"]).size().groupby("rt").mean()
        for name, by in (("same speed", ["wb"]), ("same size", ["db"]), ("same speed and size", ["wb", "db"])):
            rows.append(dict(t=t, lst=dt.lst.iloc[0], control=name, n_1D=n[RTS[0]], n_3D=n[RTS[1]],
                             w_1D=dt[dt.rt == RTS[0]].w.mean(), w_3D=dt[dt.rt == RTS[1]].w.mean(), **split_jackknife(dt, by)))
    s = pd.DataFrame(rows)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    tag = "" if x == "w_core_layer" else f"_{x}"
    d.to_csv(res / f"depth_w_clouds{tag}.csv", index=False)
    b.to_csv(res / f"depth_w_binned{tag}.csv")
    s.to_csv(res / f"depth_w_split{tag}.csv", index=False)
    return d, b, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--x", default="w_core_layer")
    ap.add_argument("--min-points", type=int, default=4)
    a = ap.parse_args()
    d, b, s = main(a.expt, a.x, a.min_points)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 60)
    f = lambda v: f"{v:.3g}"
    print(f"--- binned by {a.x} (bin edges {W_BINS[:-1]} m/s), member means")
    print(b.to_string(float_format=f))
    print("\n--- split of the 3D minus 1D mean depth [m]")
    print(s.to_string(index=False, float_format=f))


def slope(w, depth):
    return float(np.polyfit(w, depth, 1)[0])


def bootstrap_slope(d, n=2000, seed=0):
    """Slope of depth on w with a 95 % interval; clouds resampled within each ensemble member."""
    rng = np.random.default_rng(seed)
    groups = [g[["w", "depth"]].values for _, g in d.groupby(["rt", "rep"])]
    est = np.empty(n)
    for i in range(n):
        s = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        est[i] = slope(s[:, 0], s[:, 1])
    lo, hi = np.percentile(est, [2.5, 97.5])
    return dict(slope=slope(d.w.values, d.depth.values), lo=float(lo), hi=float(hi), n=len(d),
                r=float(np.corrcoef(d.w.values, d.depth.values)[0, 1]))
