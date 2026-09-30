"""Widening of each cloud over the next minutes against the sunlight on its footprint and the births nearby.

python widening.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1     (one row per cloud and sample minute: widening.nc)
python widening.py --summary                                                (widening_clouds.csv, widening_fit.csv, widening_binned.csv)
For every cloud of at least 16 cells at sample minutes STEP apart between 12 and 15 LT: width W (equivalent diameter),
its change dW over the next LAG minutes (width of the surviving track, so absorbed neighbours count), the surface
shortwave anomaly under its footprint (footprint mean minus domain mean of what the surface receives in that run),
the births within RADIUS of its edge during the preceding LAG minutes, and the clouds it absorbed during the LAG.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
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
RING = 3000.                     # m, outer edge of the 1 to 3 km ring: births beyond the recruitment range
SW_VARS = {"2stream": ("sw_flux_dn",), "raytracer": ("sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt")}


def surface_sw(rd, rt, f):
    total = None
    for v in SW_VARS[rt]:
        with xr.open_dataset(rd / f"{v}.xy.nc", decode_times=False) as ds:
            a = lowest(ds[v].isel(time=f))
        total = a if total is None else total + a
    return total


def nearby_births(bx, by, cx, cy, W, Lx, Ly, radius=RADIUS):
    """Number of births (bx, by) within radius of the edge of each cloud (cx, cy, W)."""
    if bx.size == 0:
        return np.zeros(cx.size, dtype=int)
    d = np.hypot(wrap(bx[None, :] - cx[:, None], Lx), wrap(by[None, :] - cy[:, None], Ly))
    return (d <= radius + 0.5 * W[:, None]).sum(axis=1)


def sample_frame(path, sw, tracks_now, feats_idx, births, f, lag, dx, dy, dt):
    """Rows for the clouds of frame f."""
    lab, n = mk.label_periodic(path > 0.)
    idx = np.arange(1, n + 1)
    area = mk.object_areas(lab, n) * dx * dy
    keep = area >= MIN_AREA
    if not keep.any():
        return None
    cen = mk.periodic_centroids(lab, n, dx, dy)
    W = 2. * np.sqrt(area / np.pi)
    dsw = ndimage.mean(sw, lab, idx) - sw.mean()
    tr = tracks_now
    later = feats_idx.get(f + lag, {})
    W_later = np.array([2. * np.sqrt(later[t][0] / np.pi) if t in later else np.nan for t in tr])
    absorbed = np.array([sum(feats_idx.get(g, {}).get(t, (0., 0))[1] for g in range(f + 1, f + lag + 1)) for t in tr])
    ny, nx = lab.shape
    bsel = births[(births.frame_first >= f - lag) & (births.frame_first < f)]
    nb = nearby_births(bsel.x.values, bsel.y.values, cen[:, 0], cen[:, 1], W, nx * dx, ny * dy)
    ring = nearby_births(bsel.x.values, bsel.y.values, cen[:, 0], cen[:, 1], W, nx * dx, ny * dy, radius=RING) - nb
    lwp = ndimage.mean(path, lab, idx)
    rows = pd.DataFrame(dict(frame=f, track=tr, W=W, dW=W_later - W, dSW_root=dsw, n_births=nb, n_births_ring=ring, lwp=lwp,
                             absorbed=absorbed, area=area, x=cen[:, 0], y=cen[:, 1]))
    return rows[keep]


def analyse(expt, rt, rep, lag=LAG, step=STEP):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / "features.nc") as ds:
        feats = ds.to_dataframe()
    with xr.open_dataset(res / "births.nc") as ds:
        births = ds.to_dataframe()[["frame_first", "x", "y"]]
    by_frame = {f: g.track.values for f, g in feats.groupby("frame")}
    feats_idx = {f: dict(zip(g.track.values, zip(g.area.values, np.maximum(g.n_pred.values - 1, 0))))
                 for f, g in feats.groupby("frame")}
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        time = ds["time"].values.astype(float)
        dx, dy = float(ds["x"][1] - ds["x"][0]), float(ds["y"][1] - ds["y"][0])
        dt = float(time[1] - time[0])
        lst = run.lst(time)
        frames = [f for f in range(0, time.size - lag, step) if WINDOW[0] <= lst[f] < WINDOW[1]]
        out = []
        for f in frames:
            path = ds["qlqi_path"].isel(time=f).values.astype(np.float32)
            rows = sample_frame(path, surface_sw(rd, rt, f), by_frame.get(f, np.zeros(0, dtype=int)), feats_idx, births, f, lag, dx, dy, dt)
            if rows is not None:
                rows["lst"] = lst[f]
                out.append(rows)
    d = pd.concat(out, ignore_index=True)
    ds = xr.Dataset.from_dataframe(d)
    ds.attrs.update(expt=expt, rt=rt, rep=rep, lag_min=lag, step_min=step, radius=RADIUS, ring=RING, min_area=MIN_AREA)
    ds.to_netcdf(res / "widening.nc")
    return ds


def load(expt):
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("widening.nc")) as ds:
            d = ds.to_dataframe()
        d["rt"], d["rep"] = rt, rep
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    return d[np.isfinite(d.dW)]


TERMS = {"user": ("dSW_root", "n_births", "logW"), "full": ("dSW_root", "n_births", "n_births_ring", "logW", "logLWP")}


def columns(d):
    return {"dSW_root": d.dSW_root.values, "n_births": d.n_births.values, "n_births_ring": d.n_births_ring.values,
            "logW": np.log(d.W.values), "logLWP": np.log(np.maximum(d.lwp.values, 1.e-6))}


def fit(d, spec="user", offset=False):
    """dW on the terms of the specification, with a 3D offset when pooled: slopes and standardised betas."""
    c = columns(d)
    names = list(TERMS[spec])
    cols = [np.ones(len(d))] + [c[k] for k in names]
    if offset:
        cols.append((d.rt.values == RTS[1]).astype(float))
    beta = np.linalg.lstsq(np.c_[tuple(cols)], d.dW.values, rcond=None)[0]
    out = {"n": len(d)}
    sd = d.dW.std()
    for k, name in enumerate(names, start=1):
        out[name] = float(beta[k])
        out[f"beta_{name}"] = float(beta[k] * np.std(c[name]) / sd)
    if offset:
        out["offset_3D"] = float(beta[-1])
    return out


def bootstrap(d, spec="user", offset=False, n=500, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby(["rt", "rep"])]
    est = [fit(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups]), spec, offset) for _ in range(n)]
    full = fit(d, spec, offset)
    for k in list(TERMS[spec]) + ["offset_3D"]:
        if k in full:
            v = np.array([x[k] for x in est])
            full[f"{k}_lo"], full[f"{k}_hi"] = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    return full


def summary(expt):
    d = load(expt)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    d.to_csv(res / "widening_clouds.csv", index=False)
    rows = []
    for spec in TERMS:
        rows.append(dict(spec=spec, sample="pooled", **bootstrap(d, spec, offset=True)))
        for rt, lab in zip(RTS, ("1D", "3D")):
            rows.append(dict(spec=spec, sample=lab, **bootstrap(d[d.rt == rt], spec)))
    f = pd.DataFrame(rows)
    g = d.groupby(["rt", "rep"]).agg(n=("dW", "size"), dSW_root=("dSW_root", "mean"), dSW_root_p10=("dSW_root", lambda s: s.quantile(0.1)),
                                    dSW_root_p90=("dSW_root", lambda s: s.quantile(0.9)), n_births=("n_births", "mean"),
                                    n_births_ring=("n_births_ring", "mean"), absorbed=("absorbed", "mean"), dW=("dW", "mean"),
                                    W=("W", "mean"), lwp=("lwp", "mean"))
    b = g.groupby("rt").agg(["mean", "min", "max"])
    f.to_csv(res / "widening_fit.csv", index=False); b.to_csv(res / "widening_binned.csv")
    return d, f, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    fmt = lambda v: f"{v:.3g}"
    if a.summary:
        d, f, b = summary(a.expt)
        print("--- dW [m per 5 min] on the footprint shortwave anomaly [W m-2], births within 1 km and in the 1-3 km ring in the preceding 5 min, log W, log LWP; standardised betas")
        print(f.to_string(index=False, float_format=fmt))
        print("\n--- sample means per run (member mean, min, max)")
        print(b.to_string(float_format=fmt))
    else:
        ds = analyse(a.expt, a.rt, a.rep)
        d = ds.to_dataframe()
        ok = d[np.isfinite(d.dW)]
        print(f"{a.rt} rep_{a.rep:02d}: {len(d)} cloud samples, {len(ok)} with a surviving track; dSW_root mean {ok.dSW_root.mean():.1f} W m-2, "
              f"births nearby mean {ok.n_births.mean():.2f}, dW mean {ok.dW.mean():.1f} m", flush=True)
