"""Widening of each cloud over the next minutes against the shadow it has moved off its footprint and the births nearby.

python widening.py --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1     (one row per cloud and sample minute: widening.nc)
python widening.py --expt no_aerosols_zero_wind_v3 --summary                (widening_clouds.csv, widening_fit.csv, widening_binned.csv)
For every cloud of at least 16 cells at sample minutes STEP apart between 12 and 15 LT: width W (equivalent diameter);
its change dW over the next LAG minutes, where a cloud that has vanished or been absorbed by then counts as shrunk to
zero (dW = -W) and is flagged; the shadow shift under its footprint (ray-traced minus two-stream surface shortwave:
the part of its own shadow the cloud has moved off its root; zero by construction in a two-stream run); the births
within RADIUS of its edge in the preceding LAG minutes; the clouds it absorbed; its mean water path. Fits by
statsmodels OLS with standard errors clustered by ensemble member (t intervals on members minus one).
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy import ndimage

import masks as mk
from drift import wrap
from les_io import Run, lowest
from snapshot import out_path, run_dir

RTS = ("2stream", "raytracer")
MIN_AREA = 4.e4
LAG, STEP = 5, 5                 # minutes
WINDOW = (12., 15.)              # solar hours
RADIUS = 1000.                   # m, from the cloud's edge
SW_VARS = {"2stream": ("sw_flux_dn",), "raytracer": ("sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt")}


def surface_sw(rd, rt, f):
    """Surface shortwave received [W m-2] at frame f: two-stream in a 2stream run, ray-traced in a raytracer run."""
    total = None
    for v in SW_VARS[rt]:
        with xr.open_dataset(rd / f"{v}.xy.nc", decode_times=False) as ds:
            a = lowest(ds[v].isel(time=f))
        total = a if total is None else total + a
    return total


def shadow_shift(rd, rt, f):
    """Ray-traced minus two-stream surface shortwave: the shadow moved off the ground beneath a cloud; zero in a 2stream run."""
    if rt != "raytracer":
        return None
    with xr.open_dataset(rd / "sw_flux_dn.xy.nc", decode_times=False) as ds:
        two = lowest(ds["sw_flux_dn"].isel(time=f))
    return surface_sw(rd, rt, f) - two


def nearby_births(bx, by, cx, cy, W, Lx, Ly, radius=RADIUS):
    """Number of births (bx, by) within radius of the edge of each cloud (cx, cy, W)."""
    if bx.size == 0:
        return np.zeros(cx.size, dtype=int)
    d = np.hypot(wrap(bx[None, :] - cx[:, None], Lx), wrap(by[None, :] - cy[:, None], Ly))
    return (d <= radius + 0.5 * W[:, None]).sum(axis=1)


def sample_frame(path, sw, shift, tracks_now, feats_idx, death, births, f, lag, dx, dy):
    """Rows for the clouds of frame f. A track absent at f + lag has ended: dW = -W, merged if it ended by merging."""
    lab, n = mk.label_periodic(path > 0.)
    idx = np.arange(1, n + 1)
    area = mk.object_areas(lab, n) * dx * dy
    keep = area >= MIN_AREA
    if not keep.any():
        return None
    cen = mk.periodic_centroids(lab, n, dx, dy)
    W = 2. * np.sqrt(area / np.pi)
    dsw = ndimage.mean(sw, lab, idx) - sw.mean()
    dshift = ndimage.mean(shift, lab, idx) if shift is not None else np.zeros(n)
    tr = tracks_now
    later = feats_idx.get(f + lag, {})
    W_later = np.array([2. * np.sqrt(later[t][0] / np.pi) if t in later else 0. for t in tr])
    ended = np.array([t not in later for t in tr])
    merged = ended & np.array([death.get(t) == "merge" for t in tr])
    absorbed = np.array([sum(feats_idx.get(g, {}).get(t, (0., 0))[1] for g in range(f + 1, f + lag + 1)) for t in tr])
    ny, nx = lab.shape
    bsel = births[(births.frame_first >= f - lag) & (births.frame_first < f)]
    nb = nearby_births(bsel.x.values, bsel.y.values, cen[:, 0], cen[:, 1], W, nx * dx, ny * dy)
    lwp = ndimage.mean(path, lab, idx)
    rows = pd.DataFrame(dict(frame=f, track=tr, W=W, dW=W_later - W, ended=ended, merged=merged, dSW_root=dsw, dSW_shift=dshift,
                             n_births=nb, absorbed=absorbed, lwp=lwp, area=area, x=cen[:, 0], y=cen[:, 1]))
    return rows[keep]


def analyse(expt, rt, rep, lag=LAG, step=STEP):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / "features.nc") as ds:
        feats = ds.to_dataframe()
    with xr.open_dataset(res / "tracks.nc") as ds:
        death = ds.to_dataframe().set_index("track")["death"].to_dict()
    with xr.open_dataset(res / "births.nc") as ds:
        births = ds.to_dataframe()[["frame_first", "x", "y"]]
    by_frame = {f: g.track.values for f, g in feats.groupby("frame")}
    feats_idx = {f: dict(zip(g.track.values, zip(g.area.values, np.maximum(g.n_pred.values - 1, 0))))
                 for f, g in feats.groupby("frame")}
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        time = ds["time"].values.astype(float)
        dx, dy = float(ds["x"][1] - ds["x"][0]), float(ds["y"][1] - ds["y"][0])
        dt = float(time[1] - time[0])
        lag_f, step_f = int(round(lag * 60. / dt)), int(round(step * 60. / dt))
        lst = run.lst(time)
        frames = [f for f in range(0, time.size - lag_f, step_f) if WINDOW[0] <= lst[f] < WINDOW[1]]
        out = []
        for f in frames:
            path = ds["qlqi_path"].isel(time=f).values.astype(np.float32)
            rows = sample_frame(path, surface_sw(rd, rt, f), shadow_shift(rd, rt, f), by_frame.get(f, np.zeros(0, dtype=int)),
                                feats_idx, death, births, f, lag_f, dx, dy)
            if rows is not None:
                rows["lst"] = lst[f]
                out.append(rows)
    d = pd.concat(out, ignore_index=True)
    ds = xr.Dataset.from_dataframe(d)
    ds.attrs.update(expt=expt, rt=rt, rep=rep, lag_min=lag, step_min=step, radius=RADIUS, min_area=MIN_AREA)
    ds.to_netcdf(res / "widening.nc")
    return ds


def load(expt):
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("widening.nc")) as ds:
            d = ds.to_dataframe()
        d["rt"], d["rep"] = rt, rep
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


BASE = ("n_births", "absorbed", "merged", "logW", "logLWP")
TERMS = {"forcing": ("dSW_shift",) + BASE, "pooled": BASE}


def design(d, spec, offset):
    X = pd.DataFrame({"dSW_shift": d.dSW_shift.values, "n_births": d.n_births.values.astype(float), "absorbed": d.absorbed.values.astype(float),
                      "merged": d.merged.values.astype(float), "logW": np.log(d.W.values), "logLWP": np.log(np.maximum(d.lwp.values, 1.e-6))})
    X = X[list(TERMS[spec])]
    if offset:
        X["is3D"] = (d.rt.values == RTS[1]).astype(float)
    return sm.add_constant(X)


def fit(d, spec, offset=False):
    """OLS of dW on the terms, standard errors clustered by member; coefficient, SE and 95 % interval per term."""
    X = design(d, spec, offset)
    groups = pd.factorize(d.rt.astype(str) + "_" + d.rep.astype(str))[0]
    r = sm.OLS(d.dW.values, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    ci = r.conf_int()
    out = {"n": len(d), "clusters": int(groups.max() + 1), "r2": float(r.rsquared)}
    for k in X.columns[1:]:
        out[k], out[f"{k}_se"], out[f"{k}_lo"], out[f"{k}_hi"] = float(r.params[k]), float(r.bse[k]), float(ci.loc[k, 0]), float(ci.loc[k, 1])
    return out


def summary(expt):
    d = load(expt)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    d.to_csv(res / "widening_clouds.csv", index=False)
    rows = [dict(spec="pooled", sample="both", **fit(d, "pooled", offset=True))]
    for rt, lab in zip(RTS, ("1D", "3D")):
        rows.append(dict(spec="pooled", sample=lab, **fit(d[d.rt == rt], "pooled")))
    rows.append(dict(spec="forcing", sample="3D", **fit(d[d.rt == RTS[1]], "forcing")))
    f = pd.DataFrame(rows)
    g = d.groupby(["rt", "rep"]).agg(n=("dW", "size"), ended=("ended", "mean"), merged=("merged", "mean"), dSW_shift=("dSW_shift", "mean"),
                                    dSW_shift_p90=("dSW_shift", lambda s: s.quantile(0.9)), n_births=("n_births", "mean"),
                                    absorbed=("absorbed", "mean"), dW=("dW", "mean"), W=("W", "mean"), lwp=("lwp", "mean"))
    b = g.groupby("rt").agg(["mean", "min", "max"])
    f.to_csv(res / "widening_fit.csv", index=False); b.to_csv(res / "widening_binned.csv")
    return d, f, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    fmt = lambda v: f"{v:.3g}"
    if a.summary:
        d, f, b = summary(a.expt)
        print("--- dW [m per 5 min] on the terms; coefficient, clustered SE and 95 % interval (all clouds; ended clouds count dW = -W)")
        show = ["spec", "sample", "n", "clusters", "r2"] + [c for c in f.columns if c.endswith(("_lo", "_hi")) or c in ("dSW_shift", "n_births", "absorbed", "merged", "logW", "logLWP", "is3D")]
        print(f[show].to_string(index=False, float_format=fmt))
        print("\n--- sample means per run (member mean, min, max)")
        print(b.to_string(float_format=fmt))
    else:
        ds = analyse(a.expt, a.rt, a.rep)
        d = ds.to_dataframe()
        print(f"{a.rt} rep_{a.rep:02d}: {len(d)} cloud samples, ended within 5 min {d.ended.mean():.2f} (merged {d.merged.mean():.2f}); "
              f"shadow shift mean {d.dSW_shift.mean():.1f} W m-2, births nearby mean {d.n_births.mean():.2f}, dW mean {d.dW.mean():.1f} m", flush=True)
