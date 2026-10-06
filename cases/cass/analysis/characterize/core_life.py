"""Life of an active cloud: core objects tracked through the one-minute fields and the shells (cloud objects) they sit in.

python core_life.py --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1      (core_life.nc: one row per core object per frame)
python core_life.py --expt no_aerosols_zero_wind_v3 --summary                 (core_life_pulses.csv, core_life_shells.csv)
A core object is a connected set of columns holding buoyant cloud (qlqicore_max_thv_prime; 96-98 % of them hold a
rising buoyant cloudy cell in the 3D fields), tracked by overlap like the clouds (track.py --mask core). Every core
object lies inside one cloud object, its shell. A pulse is one core track; its shell may change identity by merging
or splitting while the pulse goes on. The active life of a shell is the frames in which it holds a core.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import masks as mk
import track as tr
from les_io import Run
from lifetime import ensemble
from snapshot import out_path, run_dir

RTS = (("2stream", "1D"), ("raytracer", "3D"))
HOURS = (12., 13., 14., 15., 16.)
D_BINS = (0., 500., 1000., 1.e9)                  # shell width classes [m]
D_LABELS = ("<500", "500-1000", ">=1000")


def by_frame(feats):
    """Track id of every label of every frame (labels in the order the tracker numbered them)."""
    return {f: g.track.values for f, g in feats.groupby("frame")}


def core_mask(path, core, thr=0.):
    return (path > 0.) & (core > thr) & (core < tr.FILL)


def frame_map(path_f, core_f, cloud_tid, core_tid, thr=0.):
    """Rows (core_track, cloud_track, core_cols, cloud_cols) of one frame; a core object lies in one cloud object."""
    lab_c, n_c = mk.label_periodic(path_f > 0.)
    lab_k, n_k = mk.label_periodic(core_mask(path_f, core_f, thr))
    if n_k == 0:
        return None
    idx = np.arange(1, n_k + 1)
    shell = ndimage.minimum(lab_c, lab_k, idx).astype(int)
    assert np.array_equal(shell, ndimage.maximum(lab_c, lab_k, idx)) and shell.min() > 0
    return pd.DataFrame(dict(core_track=core_tid[:n_k], cloud_track=cloud_tid[shell - 1],
                             core_cols=mk.object_areas(lab_k, n_k), cloud_cols=mk.object_areas(lab_c, n_c)[shell - 1]))


def build(path, core, feats_cloud, feats_core, thr=0., min_cells=1):
    """Core-to-shell map for all frames; core tracks of fewer than min_cells columns summed over their frames are dropped."""
    tc, tk = by_frame(feats_cloud), by_frame(feats_core)
    out = []
    for f in range(path.shape[0]):
        if f not in tk:
            continue
        m = frame_map(path[f], core[f], tc[f], tk[f], thr)
        if m is not None:
            out.append(m.assign(frame=f))
    m = pd.concat(out, ignore_index=True)
    return m[m.groupby("core_track").core_cols.transform("sum") >= min_cells].reset_index(drop=True)


def pulses(m, tracks_core, lst, dx, dy):
    """One row per core track: duration, shells it lived in, whether it outlived a shell identity, its shell's width."""
    g = m.groupby("core_track")
    p = pd.DataFrame(dict(frames=g.size(), frame_first=g.frame.min(), n_shells=g.cloud_track.nunique(),
                          cols_max=g.core_cols.max(), shell_cols_max=g.cloud_cols.max()))
    t = tracks_core.set_index("track")
    p = p.join(t[["birth", "death", "merges_in", "splits_out"]])
    p = p[(p.birth != "start") & (p.death != "end")]
    p["untouched"] = (p.merges_in == 0) & (p.splits_out == 0) & (p.birth == "new") & (p.death == "gone")
    p["shell_changed"] = p.n_shells > 1
    p["shell_D"] = mk.equivalent_diameter(p.shell_cols_max.values, dx, dy)
    p["lst"] = lst[p.frame_first.values]
    return p.reset_index()


def shells(m, tracks_cloud, lst, dx, dy):
    """One row per cloud track: pulses hosted, active frames, frames before the first and after the last core."""
    g = m.groupby("cloud_track")
    per = m.groupby(["cloud_track", "core_track"]).size()
    s = pd.DataFrame(dict(n_pulses=g.core_track.nunique(), active=g.frame.nunique(), core_first=g.frame.min(), core_last=g.frame.max(),
                          longest=per.groupby(level=0).max()))
    t = tracks_cloud.set_index("track")
    t = t[(t.birth != "start") & (t.death != "end")]
    s = t[["frame_first", "frame_last", "area_max", "merges_in", "splits_out"]].join(s)
    s[["n_pulses", "active", "longest"]] = s[["n_pulses", "active", "longest"]].fillna(0).astype(int)
    s["frames"] = s.frame_last - s.frame_first + 1
    s["lead"] = s.core_first - s.frame_first
    s["decay"] = s.frame_last - s.core_last
    s["D"] = mk.equivalent_diameter(s.area_max.values / (dx * dy), dx, dy)
    s["lst"] = lst[s.frame_first.values.astype(int)]
    return s.reset_index().rename(columns={"track": "cloud_track"})


def tag_of(thr):
    return "" if thr == 0. else f"{thr:g}"


def analyse(expt, rt, rep, thr=0., min_cells=1):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    path, time, x, y = tr.load(rd, "qlqi_path")
    core = tr.load(rd, "qlqicore_max_thv_prime")[0]
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / "features.nc") as ds:
        fc = ds.to_dataframe()
    with xr.open_dataset(res / f"features_core{tag_of(thr)}.nc") as ds:
        fk = ds.to_dataframe()
    m = build(path, core, fc, fk, thr, min_cells)
    m["time"] = time[m.frame.values]
    m["lst"] = run.lst(m.time.values)
    ds = xr.Dataset.from_dataframe(m)
    ds.attrs.update(expt=expt, rt=rt, rep=rep, dx=float(x[1] - x[0]), dy=float(y[1] - y[0]), core_thr=thr, min_cells=min_cells)
    ds.to_netcdf(res / f"core_life{tag_of(thr)}.nc")
    return ds


def tables(expt, rt, rep, thr=0.):
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / f"core_life{tag_of(thr)}.nc") as ds:
        m, dx, dy = ds.to_dataframe(), float(ds.attrs["dx"]), float(ds.attrs["dy"])
    run = Run(run_dir(expt, rt, rep))
    time = tr.load(run.dir, "qlqi_path")[1]
    lst = run.lst(time)
    with xr.open_dataset(res / f"tracks_core{tag_of(thr)}.nc") as ds:
        tk = ds.to_dataframe()
    with xr.open_dataset(res / "tracks.nc") as ds:
        tc = ds.to_dataframe()
    p, s = pulses(m, tk, lst, dx, dy), shells(m, tc, lst, dx, dy)
    p["family"] = p.core_track.map(m.groupby("core_track").cloud_track.first().map(tc.set_index("track").family))
    return p, s, tc


def systems(p, tc, dx, dy, keep_cut=False):
    """One row per cloud system (all clouds connected in space and time): lifetime, widest cloud, pulses, class."""
    g = tc.groupby("family")
    y = pd.DataFrame(dict(life=g.family_lifetime.first() / 60., D=mk.equivalent_diameter(g.area_max.max().values / (dx * dy), dx, dy), lst=g.lst_first.min(),
                          f0=g.frame_first.min(), f1=g.frame_last.max(),
                          cut=g.birth.agg(lambda b: (b == "start").any()) | g.death.agg(lambda d: (d == "end").any())))
    pg = p.groupby("family")
    y = y.join(pd.DataFrame(dict(n_pulses=pg.size(), pulse_mean=pg.frames.mean(), pulse_max=pg.frames.max())))
    y["n_pulses"] = y.n_pulses.fillna(0).astype(int)
    y["kind"] = np.select([y.n_pulses == 0, y.n_pulses == 1], ["passive", "single pulse"], "multipulse")
    return y if keep_cut else y[~y.cut]


def systems_summary(expt, thr):
    rows = []
    for (rt, lab), rep in itertools.product(RTS, range(1, 5)):
        p, s, tc = tables(expt, rt, rep, thr)
        with xr.open_dataset(out_path(expt, rt, rep, 0).parent / f"core_life{tag_of(thr)}.nc") as ds:
            dx, dy = float(ds.attrs["dx"]), float(ds.attrs["dy"])
        y = systems(p, tc, dx, dy)
        y = y[(y.lst >= HOURS[0]) & (y.lst < HOURS[-1])]
        pw = p[(p.lst >= HOURS[0]) & (p.lst < HOURS[-1])]
        for name, sel in (("all", y), ("with a cloud >= 1 km", y[y.D >= 1000.])):
            for kind, g in sel.groupby("kind"):
                rows.append(dict(rt=rt, rep=rep, systems=name, kind=kind, n=len(g), life=g.life.mean(), life_p90=g.life.quantile(.9), n_pulses=g.n_pulses.mean(),
                                 pulse_mean=g.pulse_mean.mean(), pulse_longest=g.pulse_max.mean()))
        rows.append(dict(rt=rt, rep=rep, systems="pulses", kind="all pulses", n=len(pw), life=pw.frames.mean(), life_p90=pw.frames.quantile(.9)))
    d = pd.DataFrame(rows)
    e = ensemble(d, ["systems", "kind"], ["n", "life", "life_p90", "n_pulses", "pulse_mean", "pulse_longest"])
    e.to_csv(out_path(expt, "2stream", 1, 0).parents[2] / f"core_life_systems{tag_of(thr)}.csv")
    return e


def summary(expt):
    P, S = [], []
    for (rt, lab), rep in itertools.product(RTS, range(1, 5)):
        p, s, _ = tables(expt, rt, rep)
        p["hour"], s["hour"] = np.floor(p.lst), np.floor(s.lst)
        p["cls"] = pd.cut(p.shell_D, D_BINS, labels=D_LABELS, right=False)
        s["cls"] = pd.cut(s.D, D_BINS, labels=D_LABELS, right=False)
        for (h, c), g in p[(p.hour >= HOURS[0]) & (p.hour < HOURS[-1])].groupby(["hour", "cls"], observed=True):
            u = g[g.untouched]
            P.append(dict(rt=rt, rep=rep, hour=int(h), cls=c, n=len(g), dur_mean=g.frames.mean(), dur_median=g.frames.median(), dur_p90=g.frames.quantile(.9),
                          shell_changed=g.shell_changed.mean(), untouched=g.untouched.mean(), dur_untouched=u.frames.mean(), dur_untouched_p90=u.frames.quantile(.9)))
        for (h, c), g in s[(s.hour >= HOURS[0]) & (s.hour < HOURS[-1])].groupby(["hour", "cls"], observed=True):
            a = g[g.n_pulses > 0]
            S.append(dict(rt=rt, rep=rep, hour=int(h), cls=c, n=len(g), ever_active=(g.n_pulses > 0).mean(), frames=g.frames.mean(),
                          n_pulses=a.n_pulses.mean(), multi_pulse=(a.n_pulses > 1).mean(), active=a.active.mean(), active_frac=(a.active / a.frames).mean(),
                          lead=a.lead.mean(), decay=a.decay.mean(), longest=a.longest.mean(), longest_p90=a.longest.quantile(.9),
                          rate=(10. * a.n_pulses / a.frames).mean()))
    P, S = pd.DataFrame(P), pd.DataFrame(S)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    P.to_csv(res / "core_life_pulses_long.csv", index=False); S.to_csv(res / "core_life_shells_long.csv", index=False)
    eP = ensemble(P, ["hour", "cls"], [c for c in P.columns if c not in ("rt", "rep", "hour", "cls")])
    eS = ensemble(S, ["hour", "cls"], [c for c in S.columns if c not in ("rt", "rep", "hour", "cls")])
    eP.to_csv(res / "core_life_pulses.csv"); eS.to_csv(res / "core_life_shells.csv")
    return eP, eS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    ap.add_argument("--rt", choices=[r for r, _ in RTS])
    ap.add_argument("--rep", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--systems", action="store_true", help="cloud systems and their pulses (Heus and Seifert 2013 terms)")
    ap.add_argument("--core-thr", type=float, default=0.)
    ap.add_argument("--min-cells", type=int, default=1)
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    fmt = lambda v: f"{v:.2f}"
    if a.systems:
        e = systems_summary(a.expt, a.core_thr)
        print(e.loc[:, [(c, k) for c in ("n", "life", "life_p90", "n_pulses", "pulse_mean", "pulse_longest") for k in ("1D", "3D", "d_over_se")]].to_string(float_format=fmt))
    elif a.summary:
        eP, eS = summary(a.expt)
        show = lambda e, cols: e.loc[:, [(c, k) for c in cols for k in ("1D", "3D", "d_over_se")]].to_string(float_format=fmt)
        print("--- pulses (core tracks) by hour of birth and shell width: duration [min], share that outlived a shell identity, untouched share and duration")
        print(show(eP, ["n", "dur_mean", "dur_p90", "shell_changed", "untouched", "dur_untouched"]))
        print("\n--- shells (cloud tracks) by hour of birth and width: ever active, pulses hosted, active frames and fraction, frames before the first and after the last core")
        print(show(eS, ["n", "ever_active", "frames", "n_pulses", "multi_pulse", "active", "active_frac", "lead", "decay"]))
        print("\n--- sites against pulses, shells of at least 1 km: site life [min], longest pulse the site held (mean, p90), pulses per site and per 10 min of site life, mean pulse")
        big = eS.xs(D_LABELS[-1], level="cls")
        big[("pulse", "1D")], big[("pulse", "3D")] = eP.xs(D_LABELS[-1], level="cls")[("dur_mean", "1D")], eP.xs(D_LABELS[-1], level="cls")[("dur_mean", "3D")]
        print(big.loc[:, [(c, k) for c in ("frames", "longest", "longest_p90", "n_pulses", "rate", "pulse") for k in ("1D", "3D")]].to_string(float_format=fmt))
    else:
        ds = analyse(a.expt, a.rt, a.rep, a.core_thr, a.min_cells)
        p, s, _ = tables(a.expt, a.rt, a.rep, a.core_thr)
        print(f"{a.rt} rep_{a.rep:02d}: {ds.sizes['index']} core-frames, {len(p)} pulses (median {p.frames.median():.0f} min, "
              f"{p.shell_changed.mean():.2f} outlived a shell), {len(s)} shells, {(s.n_pulses > 0).mean():.2f} ever active", flush=True)
