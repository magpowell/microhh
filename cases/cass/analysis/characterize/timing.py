"""When the 3D minus 1D differences appear: out-of-cloud moisture, cloud fraction and cloud mass flux by height
(5-minute statistics) and cloud depth (tracked clouds).

python timing.py --expt no_aerosols_zero_wind_v2
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from les_io import Run
from snapshot import out_path, run_dir

RTS = ("2stream", "raytracer")
SEGMENTS = ("0000000", "0043200")
SMOOTH = 7                       # 5-min samples in the centred running mean
HOLD = 6                         # consecutive samples beyond K standard errors
K = 2.
LAYERS = tuple((z, z + 250.) for z in np.arange(1000., 3500., 250.))
LST_RANGE = (8., 16.)
FILL = 1.e30


def _keep(rd, mask, seg):
    """The first segment can run past the restart time; the restarted segment replaces that part."""
    with xr.open_dataset(rd / f"cass.{mask}.{seg}.nc", decode_times=False) as r:
        t = r["time"].values
    t = np.round(t)
    return t, (t <= float(SEGMENTS[1])) if seg == SEGMENTS[0] else (t > float(SEGMENTS[1]))


def _read(rd, mask, group, var):
    a = []
    for seg in SEGMENTS:
        with xr.open_dataset(rd / f"cass.{mask}.{seg}.nc", group=group, decode_times=False) as g:
            a.append(g[var].values[_keep(rd, mask, seg)[1]])
    a = np.concatenate(a)
    return np.where(np.abs(a) < FILL, a, np.nan)


def statistics(expt, rt, rep):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    time = np.concatenate([t[k] for t, k in (_keep(rd, "default", s) for s in SEGMENTS)])
    a = np.nan_to_num(_read(rd, "ql", "default", "area"))
    out = xr.Dataset(coords=dict(time=time, z=run.z))
    for v in ("qt", "thl"):
        slab, cld = _read(rd, "default", "thermo", v), np.nan_to_num(_read(rd, "ql", "thermo", v))
        out[f"{v}_out"] = (("time", "z"), (slab - a * cld) / (1. - a))
    rho = np.fromfile(rd / "rhoref.0000000", dtype="<f8")
    mh = rho[run.ktot:][None, :] * np.nan_to_num(_read(rd, "ql", "default", "areah") * _read(rd, "ql", "default", "w"))
    out["cf"] = (("time", "z"), a)
    out["M"] = (("time", "z"), 0.5 * (mh[:, :-1] + mh[:, 1:]))
    out["lwp"] = ("time", _read(rd, "default", "thermo", "ql_path"))
    out["cover"] = ("time", _read(rd, "default", "thermo", "ql_cover"))
    out["lst"] = ("time", run.lst(time))
    return out


def depth_series(expt, rt, rep, time, tag=""):
    with xr.open_dataset(out_path(expt, rt, rep, 0).with_name(f"features{tag}.nc")) as ds:
        f = ds[["time", "depth", "top", "base", "area", "n_buoy"]].to_dataframe()
    f["bin"] = time[np.abs(f["time"].values[:, None] - time[None, :]).argmin(axis=1)] if len(f) else []
    f["active"] = f["n_buoy"] > 0
    g, ga = f.groupby("bin"), f[f["active"]].groupby("bin")
    d = pd.DataFrame(dict(depth_all=g["depth"].mean(), depth_p90=g["depth"].quantile(0.9),
                          depth_active=ga["depth"].mean(), depth_active_p90=ga["depth"].quantile(0.9),
                          top_p95=g["top"].quantile(0.95), deep_share=g["depth"].apply(lambda s: (s > 1000.).mean()),
                          n_cloud=g.size() / g["time"].nunique(), n_active=ga.size() / ga["time"].nunique()))
    return d.reindex(time).to_xarray().rename(bin="time")


def smooth(x, n=SMOOTH):
    """Centred running mean along the first axis, NaN-aware."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    num = np.apply_along_axis(lambda v: np.convolve(v, np.ones(n), "same"), 0, np.where(ok, x, 0.))
    den = np.apply_along_axis(lambda v: np.convolve(v, np.ones(n), "same"), 0, ok.astype(float))
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def difference(x):
    """x: (rt, rep, time, ...) -> 3D minus 1D and its standard error."""
    n = x.shape[1]
    d = np.nanmean(x[1], axis=0) - np.nanmean(x[0], axis=0)
    se = np.sqrt(np.nanvar(x[0], axis=0, ddof=1) / n + np.nanvar(x[1], axis=0, ddof=1) / n)
    return d, se


def onset(lst, d, se, hold=HOLD, k=K):
    """First time from which the difference keeps its final sign beyond k standard errors for `hold` samples."""
    ok = np.isfinite(d) & np.isfinite(se) & (se > 0)
    if not ok.any():
        return np.nan
    final = np.sign(np.nanmean(d[ok][-hold:]))
    sig = ok & (np.abs(d) > k * se) & (np.sign(d) == final)
    for i in range(sig.size - hold + 1):
        if sig[i:i + hold].all():
            return float(lst[i])
    return np.nan


def reach(lst, d, frac):
    """First time the difference reaches a fraction of its value at the end of the window and stays above it."""
    ok = np.isfinite(d)
    if not ok.any():
        return np.nan
    ref = np.nanmean(d[ok][-HOLD:])
    i = np.flatnonzero(~(ok & (d / ref >= frac)))
    k = (i[-1] + 1) if i.size else 0
    return float(lst[k]) if k < lst.size else np.nan


def main(expt, tag=""):
    S = [[statistics(expt, rt, rep) for rep in range(1, 5)] for rt in RTS]
    time, z, lst = S[0][0]["time"].values, S[0][0]["z"].values, S[0][0]["lst"].values
    D = [[depth_series(expt, rt, rep, time, tag) for rep in range(1, 5)] for rt in RTS]
    win = (lst >= LST_RANGE[0]) & (lst <= LST_RANGE[1])
    rows, out = [], xr.Dataset(coords=dict(time=time[win], z=z, lst=("time", lst[win])))

    def add(name, x, layer=""):
        x = smooth(np.moveaxis(x, 2, 0))
        x = np.moveaxis(x, 0, 2)[:, :, win]
        d, se = difference(x)
        m1, m3 = np.nanmean(x[0], axis=0), np.nanmean(x[1], axis=0)
        rows.append(dict(metric=name, layer=layer, onset=onset(lst[win], d, se), reach25=reach(lst[win], d, 0.25),
                         reach50=reach(lst[win], d, 0.5), final_1D=np.nanmean(m1[-HOLD:]), final_diff=np.nanmean(d[-HOLD:])))
        return m1, m3, d, se

    series = [(v, D) for v in D[0][0].data_vars] + [(v, S) for v in ("lwp", "cover")]
    for v, src in series:
        x = np.array([[src[i][j][v].values for j in range(4)] for i in range(2)])
        m1, m3, d, se = add(v, x)
        for n, a in (("1D", m1), ("3D", m3), ("diff", d), ("se", se)):
            out[f"{v}_{n}"] = ("time", a)
        out[f"{v}_members"] = (("rt", "rep", "time"), np.moveaxis(smooth(np.moveaxis(x, 2, 0)), 0, 2)[:, :, win])
    for v in ("qt_out", "thl_out", "cf", "M"):
        x = np.array([[S[i][j][v].values for j in range(4)] for i in range(2)])
        xs = np.moveaxis(smooth(np.moveaxis(x, 2, 0)), 0, 2)[:, :, win]
        d, se = difference(xs)
        out[f"{v}_1D"], out[f"{v}_diff"], out[f"{v}_se"] = (("time", "z"), np.nanmean(xs[0], axis=0)), (("time", "z"), d), (("time", "z"), se)
        for z0, z1 in LAYERS:
            m = (z >= z0) & (z < z1)
            add(v, np.nanmean(x[..., m], axis=-1), f"{z0 / 1000:.2f}-{z1 / 1000:.2f}")
    res = out_path(expt, "2stream", 1, 0).parents[2]
    out.to_netcdf(res / f"timing{tag}.nc")
    t = pd.DataFrame(rows)
    t.to_csv(res / f"timing_onsets{tag}.csv", index=False)
    return out, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    _, t = main(a.expt, a.tag)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
