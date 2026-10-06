"""Persistence of the cloudy site under each cloud at the snapshot times, against cloud width and depth.

python persistence.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1        (one row per cloud: persistence_<t>.nc)
python persistence.py --summary                                                    (persistence_clouds.csv, _binned.csv, _fit.csv)
Site persistence per column = consecutive minutes the column has been cloudy up to the snapshot; per cloud its mean
and maximum over the cloud's columns. Track age and merges come from the overlap tracking (track.py).
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, stats

import masks as mk
from les_io import Run
from snapshot import file_times, frames_every, out_path, run_dir, snapshot_times

RTS = ("2stream", "raytracer")
MIN_AREA = 4.e4                                            # m2, 16 cells, as in lifetime.py
D_BINS = np.array([0., 400., 800., 1600., 1.e5])           # m
DUR_BINS = np.array([0., 5., 10., 20., 40., 1.e4])         # min
LOOKBACK = (10, 30, 60)                                    # min


def cloudy_duration(path, f):
    """Consecutive frames each column has been cloudy up to frame f (inclusive)."""
    dur = np.zeros(path.shape[1:], dtype=int)
    alive = np.ones(path.shape[1:], dtype=bool)
    for k in range(f, -1, -1):
        alive &= path[k] > 0.
        if not alive.any():
            break
        dur += alive
    return dur


def per_object(lab, n, dur, path, f, dt, lookback=LOOKBACK):
    idx = np.arange(1, n + 1)
    out = dict(dur_mean=ndimage.mean(dur, lab, idx) * dt / 60., dur_max=ndimage.maximum(dur, lab, idx) * dt / 60.)
    for m in lookback:
        k = f - int(round(m * 60. / dt))
        was = (path[k] > 0.) if k >= 0 else np.zeros(lab.shape, dtype=bool)
        out[f"cloudy_{m}min_ago"] = ndimage.mean(was.astype(float), lab, idx)
    return out


def track_columns(feats, tracks, f, dt):
    """Rows of the features table at frame f with track age, merges absorbed so far and family age."""
    now = feats[feats.frame == f].copy()
    past = feats[feats.frame <= f]
    merges = past.assign(m=np.maximum(past.n_pred - 1, 0)).groupby("track").m.sum()
    fam = tracks.set_index("track").family
    fam_first = tracks.groupby("family").frame_first.min()
    now["merges"] = merges.reindex(now.track).values
    now["family_age"] = (f - fam_first.reindex(fam.reindex(now.track).values).values) * dt / 60.
    now["age"] = now.age / 60.
    return now


def analyse(expt, rt, rep, t):
    d = run_dir(expt, rt, rep)
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(d / "qlqi_path.xy.nc", decode_times=False) as ds:
        time = ds["time"].values.astype(float)
        f = int(np.argmin(np.abs(time - t)))
        path = ds["qlqi_path"].values[:f + 1].astype(np.float32)
        dx, dy = float(ds["x"][1] - ds["x"][0]), float(ds["y"][1] - ds["y"][0])
    dt = float(time[1] - time[0])
    with xr.open_dataset(res / "features.nc") as ds:
        feats = ds.to_dataframe()
    with xr.open_dataset(res / "tracks.nc") as ds:
        tracks = ds.to_dataframe()
    lab, n = mk.label_periodic(path[f] > 0.)
    now = track_columns(feats, tracks, f, dt)
    if len(now) != n or not np.allclose(np.sort(now.area.values), np.sort(mk.object_areas(lab, n) * dx * dy)):
        raise ValueError(f"features rows at frame {f} do not match the labelled objects ({len(now)} vs {n})")
    # rows of the features table are in label order (track.py appends props in label order)
    assert np.allclose(now.area.values, mk.object_areas(lab, n) * dx * dy)
    now = now.reset_index(drop=True)
    for k, v in per_object(lab, n, cloudy_duration(path, f), path, f, dt).items():
        now[k] = v
    now["D"] = mk.equivalent_diameter(now.area.values / (dx * dy), dx, dy)
    ds = xr.Dataset.from_dataframe(now[["track", "age", "family_age", "merges", "area", "D", "depth", "top", "base", "lwp", "n_buoy",
                                        "dur_mean", "dur_max"] + [f"cloudy_{m}min_ago" for m in LOOKBACK]])
    ds.attrs.update(expt=expt, rt=rt, rep=rep, t_sec=t, lst_solar=float(Run(d).lst(t)), frame=f, dt=dt, min_area=MIN_AREA)
    ds.to_netcdf(res / f"persistence_{int(t):07d}.nc")
    return ds


def load_clouds(expt, min_area=MIN_AREA, hourly=False):
    """One row per cloud; with hourly, every persistence file is read and t holds the solar hour instead of the time."""
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
      for t in (file_times(expt, rt, rep, "persistence") if hourly else snapshot_times(expt, skip_first=True)):
        f = out_path(expt, rt, rep, 0).with_name(f"persistence_{t:07d}.nc")
        with xr.open_dataset(f) as ds:
            d = ds.to_dataframe()
            lst = float(ds.attrs["lst_solar"]) if "lst_solar" in ds.attrs else float(Run(run_dir(expt, rt, rep)).lst(t))
        if hourly:
            t = int(np.floor(lst))
        d["t"], d["lst"], d["rt"], d["rep"] = t, lst, rt, rep
        rows.append(d[d.area >= min_area])
    d = pd.concat(rows, ignore_index=True)
    d["db"] = np.digitize(d.D, D_BINS) - 1
    d["ub"] = np.digitize(d.dur_mean, DUR_BINS) - 1
    return d


def rank(d):
    """Rank correlations of depth and width with the persistence measures, and the partial ones within classes."""
    r = lambda a, b: float(stats.spearmanr(a, b)[0]) if len(a) >= 10 else np.nan
    out = dict(n=len(d), r_depth_D=r(d.depth, d.D), r_depth_dur=r(d.depth, d.dur_mean), r_depth_age=r(d.depth, d.age),
               r_depth_merges=r(d.depth, d.merges), r_D_dur=r(d.D, d.dur_mean), r_D_age=r(d.D, d.age))
    w = [r(g.depth, g.dur_mean) for _, g in d.groupby("db") if len(g) >= 20]
    u = [r(g.depth, g.D) for _, g in d.groupby("ub") if len(g) >= 20]
    out["r_depth_dur_within_D"] = float(np.mean(w)) if w else np.nan
    out["r_depth_D_within_dur"] = float(np.mean(u)) if u else np.nan
    return out


def fit(d):
    """depth on log D and log site duration with a 3D offset: coefficients [m per e-fold], offset [m]."""
    X = np.c_[np.ones(len(d)), np.log(d.D.values), np.log(np.maximum(d.dur_mean.values, 0.5)), (d.rt.values == RTS[1]).astype(float)]
    a, b, c, o = np.linalg.lstsq(X, d.depth.values, rcond=None)[0]
    Xw = X[:, :2]
    _, bw = np.linalg.lstsq(np.c_[Xw, X[:, 3]], d.depth.values, rcond=None)[0][:2]
    return dict(per_efold_D=float(b), per_efold_dur=float(c), offset_3D=float(o), per_efold_D_alone=float(bw))


def bootstrap(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby(["rt", "rep"])]
    est = [fit(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(n)]
    full = fit(d)
    for k in list(full):
        v = np.array([x[k] for x in est])
        full[f"{k}_lo"], full[f"{k}_hi"] = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    return full


def binned(d):
    """Member medians of site duration, merges and depth per width class; mean and range over members."""
    g = d.groupby(["t", "rt", "rep", "db"]).agg(n=("D", "size"), D=("D", "mean"), dur=("dur_mean", "median"), age=("age", "median"),
                                                 merges=("merges", "median"), depth=("depth", "median")).reset_index()
    m = g.groupby(["t", "db", "rt"])[["n", "D", "dur", "age", "merges", "depth"]]
    mean, lo, hi = m.mean().unstack("rt"), m.min().unstack("rt"), m.max().unstack("rt")
    cols = {}
    for c in ("n", "D", "dur", "age", "merges", "depth"):
        for rt, lab in zip(RTS, ("1D", "3D")):
            cols[f"{c}_{lab}"] = mean[c][rt]
            if c in ("dur", "depth"):
                cols[f"{c}_{lab}_lo"], cols[f"{c}_{lab}_hi"] = lo[c][rt], hi[c][rt]
    return g, pd.DataFrame(cols)


def summary(expt, hourly=False):
    d = load_clouds(expt, hourly=hourly)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    tag = "_hourly" if hourly else ""
    d.to_csv(res / f"persistence_clouds{tag}.csv", index=False)
    g, b = binned(d)
    b.reset_index().to_csv(res / f"persistence_binned{tag}.csv", index=False)
    rows = []
    for t, dt in d.groupby("t"):
        row = dict(t=t, lst=dt.lst.iloc[0], **bootstrap(dt))
        for rt, lab in zip(RTS, ("1D", "3D")):
            row.update({f"{k}_{lab}": v for k, v in rank(dt[dt.rt == rt]).items()})
        rows.append(row)
    f = pd.DataFrame(rows)
    f.to_csv(res / f"persistence_fit{tag}.csv", index=False)
    return d, b, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--hourly", action="store_true", help="summary over every file, pooled by solar hour")
    ap.add_argument("--every", type=float, default=None, help="instead of --t: every N minutes of the 60 s fields within --solar")
    ap.add_argument("--solar", type=float, nargs=2, default=(11.9, 16.1))
    a = ap.parse_args()
    if a.every and a.rt:
        a.t = frames_every(Run(run_dir(a.expt, a.rt, a.rep)), a.every, a.solar)
    a.t = a.t or (list(snapshot_times(a.expt, a.rt, a.rep, skip_first=True)) if a.rt else None)
    if a.summary:
        d, b, f = summary(a.expt, a.hourly)
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 200)
        fmt = lambda v: f"{v:.3g}"
        print(f"--- per width class (edges {D_BINS[:-1]} m): member medians of site duration [min], track age [min], merges, depth [m]")
        print(b.to_string(float_format=fmt))
        print("\n--- depth on log width and log site duration (m per e-fold), 3D offset (m); rank correlations")
        print(f.to_string(index=False, float_format=fmt))
    else:
        for t in a.t:
            ds = analyse(a.expt, a.rt, a.rep, t)
            d = ds.to_dataframe()
            d = d[d.area >= MIN_AREA]
            print(f"{a.rt} rep_{a.rep:02d} t={t} clouds={len(d)} site duration median {d.dur_mean.median():.1f} min "
                  f"(max {d.dur_max.max():.0f}), track age median {d.age.median():.1f} min, r(depth, duration)={stats.spearmanr(d.depth, d.dur_mean)[0]:.2f} "
                  f"r(D, duration)={stats.spearmanr(d.D, d.dur_mean)[0]:.2f}", flush=True)
