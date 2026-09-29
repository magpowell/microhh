"""Robustness of the per-cloud depth results: split sub-clouds, growing clouds only, core depth, per member.

python robust.py --expt no_aerosols_zero_wind_v2
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

import depth_w as dw
from snapshot import out_path

RTS = ("2stream", "raytracer")
TIMES = (32400, 36000, 39600)
MIN_POINTS = 4
W_BINS = dw.W_BINS
D_BINS4 = dw.D_BINS
D_BINS8 = np.array([0., 300., 450., 650., 900., 1250., 1750., 2500., 1.e5])
WIDTH_CLASSES = np.array([0., 500., 1000., 1500., 2000., 1.e5])
CASES = (("objects", "obj", False, "cloud_depth"), ("sub-clouds", "sub", False, "cloud_depth"),
         ("growing objects", "obj", True, "cloud_depth"), ("growing sub-clouds", "sub", True, "cloud_depth"),
         ("objects, core depth", "obj", False, "core_depth"), ("sub-clouds, core depth", "sub", False, "core_depth"))


def load(expt, kind):
    rows = []
    for t, rt, rep in itertools.product(TIMES, RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"objects_{t:07d}.nc")) as ds:
            d = ds[[v for v in ds.data_vars if v.startswith(f"{kind}_")]].to_dataframe().rename(columns=lambda c: c[len(kind) + 1:])
            d["t"], d["lst"], d["rt"], d["rep"] = t, float(ds.attrs["lst_solar"]), rt, rep
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def population(d, growing, depth):
    x = d[(d.n_core_base >= MIN_POINTS) & np.isfinite(d[depth])].copy()
    if growing:
        x = x[x.w_top_cloud > 0.]
    x["w"], x["depth"] = x.w_core_base, x[depth]
    x["wb"] = np.digitize(x.w, W_BINS) - 1
    x["db"] = np.digitize(x.D, D_BINS4) - 1
    x["db8"] = np.digitize(x.D, D_BINS8) - 1
    return x


def common_slope_offset(d):
    """c in depth = a + b D + c [3D], least squares."""
    A = np.column_stack([np.ones(len(d)), d.D.values, (d.rt == RTS[1]).values.astype(float)])
    return float(np.linalg.lstsq(A, d.depth.values, rcond=None)[0][2])


def bootstrap(d, fun, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby(["rt", "rep"])]
    est = [fun(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(n)]
    return np.percentile(est, [2.5, 97.5])


def slope_x(d, x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g[[x, "depth"]].values for _, g in d.groupby(["rt", "rep"])]
    est = []
    for _ in range(n):
        s = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        est.append(np.polyfit(s[:, 0], s[:, 1], 1)[0])
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(np.polyfit(d[x].values, d.depth.values, 1)[0]), float(lo), float(hi)


def compare(d):
    rows = []
    for t, dt in d.groupby("t"):
        r = dict(t=t, lst=dt.lst.iloc[0])
        for rt, lab in zip(RTS, ("1D", "3D")):
            c = dt[dt.rt == rt]
            r[f"n_{lab}"] = len(c) / 4.
            r[f"depth_{lab}"], r[f"D_{lab}"], r[f"w_{lab}"] = c.depth.mean(), c.D.mean(), c.w.mean()
            for x, nm in (("w", "speed"), ("D", "width")):
                r[f"slope_{nm}_{lab}"], r[f"slope_{nm}_{lab}_lo"], r[f"slope_{nm}_{lab}_hi"] = slope_x(c, x)
            r[f"r_width_{lab}"] = float(np.corrcoef(c.D, c.depth)[0, 1])
        for nm, by in (("speed", ["wb"]), ("width4", ["db"]), ("width8", ["db8"])):
            s = dw.split_jackknife(dt, by)
            r[f"offset_{nm}"], r[f"offset_{nm}_se"] = s["offset"], s["offset_se"]
        r["total"] = s["total"]
        r["offset_fit"] = common_slope_offset(dt)
        r["offset_fit_lo"], r["offset_fit_hi"] = bootstrap(dt, common_slope_offset)
        rows.append(r)
    return pd.DataFrame(rows)


def per_member(d, t=39600):
    rows = []
    dt = d[d.t == t]
    for rep in range(1, 5):
        p = dt[dt.rep == rep]
        a, b = p[p.rt == RTS[0]], p[p.rt == RTS[1]]
        rows.append(dict(rep=rep, n_1D=len(a), n_3D=len(b), total=b.depth.mean() - a.depth.mean(),
                         offset_width4=dw.split(a, b, ["db"])["offset"], offset_width8=dw.split(a, b, ["db8"])["offset"],
                         offset_fit=common_slope_offset(p)))
    return pd.DataFrame(rows)


def cores_per_object(o):
    o = o.copy()
    o["wc"] = np.digitize(o.D, WIDTH_CLASSES) - 1
    g = o.groupby(["t", "rt", "rep", "wc"]).agg(n=("D", "size"), cores=("n_seeds", "mean"), multi=("n_seeds", lambda s: (s >= 2).mean()),
                                                none=("n_seeds", lambda s: (s == 0).mean()), cores_max=("n_seeds", "max")).reset_index()
    m = g.groupby(["t", "wc", "rt"])[["n", "cores", "multi", "none", "cores_max"]]
    mean, var, cnt = m.mean().unstack("rt"), m.var(ddof=1).unstack("rt"), m.count().unstack("rt")
    out = {}
    for c in ("n", "cores", "multi", "none", "cores_max"):
        a, b = mean[c][RTS[0]], mean[c][RTS[1]]
        se = np.sqrt(var[c][RTS[0]] / cnt[c][RTS[0]] + var[c][RTS[1]] / cnt[c][RTS[1]])
        out[c] = pd.DataFrame({"1D": a, "3D": b, "diff": b - a, "se": se})
    return pd.concat(out, axis=1)


def main(expt):
    T = {k: load(expt, k) for k in ("obj", "sub")}
    res = out_path(expt, "2stream", 1, 0).parents[2]
    C, M = [], []
    for name, kind, growing, depth in CASES:
        p = population(T[kind], growing, depth)
        c = compare(p)
        c.insert(0, "case", name)
        C.append(c)
        m = per_member(p)
        m.insert(0, "case", name)
        M.append(m)
    C, M = pd.concat(C, ignore_index=True), pd.concat(M, ignore_index=True)
    K = cores_per_object(T["obj"])
    C.to_csv(res / "robust_compare.csv", index=False)
    M.to_csv(res / "robust_per_member.csv", index=False)
    K.to_csv(res / "cores_per_object.csv")
    return C, M, K


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    C, M, K = main(a.expt)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 80)
    f = lambda v: f"{v:.3g}"
    print("--- cores per object by width class (edges", WIDTH_CLASSES[:-1], "m)")
    for c in ("n", "cores", "multi", "cores_max"):
        print(c); print(K[c].to_string(float_format=f))
    cols = ["case", "lst", "n_1D", "n_3D", "depth_1D", "depth_3D", "D_1D", "D_3D", "total"]
    print("\n--- populations"); print(C[cols].to_string(index=False, float_format=f))
    cols = ["case", "lst", "slope_speed_1D", "slope_speed_1D_lo", "slope_speed_1D_hi", "slope_speed_3D", "slope_speed_3D_lo", "slope_speed_3D_hi",
            "offset_speed", "offset_speed_se"]
    print("\n--- depth against speed"); print(C[cols].to_string(index=False, float_format=f))
    cols = ["case", "lst", "r_width_1D", "r_width_3D", "slope_width_1D", "slope_width_1D_lo", "slope_width_1D_hi", "slope_width_3D", "slope_width_3D_lo",
            "slope_width_3D_hi"]
    print("\n--- depth against width"); print(C[cols].to_string(index=False, float_format=f))
    cols = ["case", "lst", "total", "offset_width4", "offset_width4_se", "offset_width8", "offset_width8_se", "offset_fit", "offset_fit_lo", "offset_fit_hi"]
    print("\n--- offset at the same width"); print(C[cols].to_string(index=False, float_format=f))
    print("\n--- per member pair, 14:53"); print(M.to_string(index=False, float_format=f))
