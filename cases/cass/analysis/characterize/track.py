"""Overlap tracking of clouds in the one-minute cloud water path fields, with depth from column base and top.

python track.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
Output: .../<rt>/rep_NN/tracks.nc (one row per track) and features.nc (one row per cloud per frame)
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import masks as mk
from les_io import Run
from snapshot import out_path, run_dir

FILL = 1.e30


def overlaps(la, lb):
    """Label pairs that share pixels in two frames: a, b, number of shared pixels."""
    m = (la > 0) & (lb > 0)
    nb = int(lb.max()) + 1
    key, n = np.unique(la[m].astype(np.int64) * nb + lb[m], return_counts=True)
    return key // nb, key % nb, n


def _best(group, other, n, size):
    o = np.lexsort((other, n, group))
    g = group[o]
    last = np.r_[g[1:] != g[:-1], True]
    out = np.zeros(size + 1, dtype=int)
    out[g[last]] = other[o][last]
    return out


def link(a, b, n, na, nb):
    """pred[b] = label in the earlier frame when a and b are each other's largest overlap, else 0."""
    pred = np.zeros(nb + 1, dtype=int)
    if a.size:
        best_b, best_a = _best(a, b, n, na), _best(b, a, n, nb)
        bb = np.arange(1, nb + 1)
        ok = (best_a[bb] > 0) & (best_b[best_a[bb]] == bb)
        pred[bb[ok]] = best_a[bb[ok]]
    return pred


def props(lab, n, path, base, top, core, dx, dy):
    idx = np.arange(1, n + 1)
    area = mk.object_areas(lab, n)
    flat = lab.ravel()
    xy = mk.periodic_centroids(lab, n, dx, dy)
    zt = ndimage.maximum(top, lab, idx)
    zb = ndimage.minimum(np.where(lab > 0, base, np.inf), lab, idx)
    return dict(area=area * dx * dy, lwp=np.bincount(flat, weights=path.ravel(), minlength=n + 1)[1:] / area,
                path_max=ndimage.maximum(path, lab, idx), top=zt, base=zb, depth=zt - zb,
                n_buoy=np.bincount(flat, weights=((core > 0.) & (core < FILL)).ravel(), minlength=n + 1)[1:],
                x=xy[:, 0], y=xy[:, 1])


class Families:
    def __init__(self):
        self.p = [0]

    def new(self):
        self.p.append(len(self.p))
        return len(self.p) - 1

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def track(path, base, top, core, time, dx, dy, thr=0.):
    """Returns features (one row per cloud per frame) and tracks (one row per track)."""
    fam = Families()
    birth, death, feats = {}, {}, []
    lab0, n0, tid0 = None, 0, np.zeros(1, dtype=int)
    nt = path.shape[0]
    for f in range(nt):
        lab, n = mk.label_periodic(path[f] > thr)
        tid = np.zeros(n + 1, dtype=int)
        npred, nsucc = np.zeros(n + 1, dtype=int), np.zeros(n0 + 1, dtype=int)
        if n and n0:
            a, b, m = overlaps(lab0, lab)
            pred = link(a, b, m, n0, n)
            np.add.at(npred, b, 1)
            np.add.at(nsucc, a, 1)
        else:
            a = b = np.zeros(0, dtype=int)
            pred = np.zeros(n + 1, dtype=int)
        tid[pred > 0] = tid0[pred[pred > 0]]
        for j in np.flatnonzero(pred[1:] == 0) + 1:
            tid[j] = fam.new()
            birth[tid[j]] = ("start" if f == 0 else "split" if npred[j] else "new", f)
        for u, v in zip(a, b):
            fam.union(tid0[u], tid[v])
        kept = np.zeros(n0 + 1, dtype=bool)
        kept[pred[pred > 0]] = True
        for i in np.flatnonzero(~kept[1:]) + 1:
            death[tid0[i]] = "merge" if nsucc[i] else "gone"
        if feats and n0:
            feats[-1]["n_succ"] = nsucc[1:]
        if n:
            p = props(lab, n, path[f], base[f], top[f], core[f], dx, dy)
            feats.append(pd.DataFrame(dict(frame=f, time=time[f], track=tid[1:], n_pred=npred[1:], n_succ=0, **p)))
        lab0, n0, tid0 = lab, n, tid
    for i in range(1, n0 + 1):
        death[tid0[i]] = "end"
    feats = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    return feats, summarise(feats, birth, death, fam, time)


def summarise(feats, birth, death, fam, time):
    if feats.empty:
        return pd.DataFrame()
    dt = float(time[1] - time[0])
    feats["age"] = (feats["frame"] - feats.groupby("track")["frame"].transform("min")) * dt
    g = feats.groupby("track")
    t = pd.DataFrame(dict(frame_first=g["frame"].min(), frame_last=g["frame"].max(), area_max=g["area"].max(),
                          area_mean=g["area"].mean(), depth_max=g["depth"].max(), depth_mean=g["depth"].mean(),
                          top_max=g["top"].max(), lwp_max=g["lwp"].max(), n_buoy_max=g["n_buoy"].max(),
                          merges_in=g["n_pred"].apply(lambda s: int(np.maximum(s - 1, 0).sum())),
                          splits_out=g["n_succ"].apply(lambda s: int(np.maximum(s - 1, 0).sum()))))
    t["age_depth_max"] = feats.loc[g["depth"].idxmax(), ["track", "age"]].set_index("track")["age"]
    t["lifetime"] = (t["frame_last"] - t["frame_first"] + 1) * dt
    t["time_first"] = time[t["frame_first"].values]
    t["birth"] = [birth[i][0] for i in t.index]
    t["death"] = [death[i] for i in t.index]
    t["family"] = [fam.find(i) for i in t.index]
    f = t.groupby("family")
    t["family_lifetime"] = ((f["frame_last"].transform("max") - f["frame_first"].transform("min") + 1) * dt)
    t["family_size"] = f["frame_first"].transform("size")
    return t.reset_index()


def load(d, var):
    with xr.open_dataset(d / f"{var}.xy.nc", decode_times=False) as ds:
        return ds[var].values.astype(np.float32), ds["time"].values.astype(float), ds["x"].values, ds["y"].values


def analyse(expt, rt, rep, thr=0., mask="cloud"):
    d = run_dir(expt, rt, rep)
    path, time, x, y = load(d, "qlqi_path")
    base, top, core = (load(d, v)[0] for v in ("qlqi_base", "qlqi_top", "qlqicore_max_thv_prime"))
    if mask == "core":
        path = np.where((core > 0.) & (core < FILL), path, 0.)
    feats, tracks = track(path, base, top, core, time, float(x[1] - x[0]), float(y[1] - y[0]), thr)
    lst = Run(d).lst(time)
    feats["lst"] = lst[feats["frame"].values]
    tracks["lst_first"] = lst[tracks["frame_first"].values]
    attrs = dict(expt=expt, rt=rt, rep=rep, path_thr=thr, mask=mask, dt=float(time[1] - time[0]))
    out = out_path(expt, rt, rep, 0).parent
    out.mkdir(parents=True, exist_ok=True)
    tag = ("" if thr == 0. else f"_thr{thr:g}") + ("" if mask == "cloud" else f"_{mask}")
    for name, df in (("features", feats), ("tracks", tracks)):
        ds = xr.Dataset.from_dataframe(df)
        ds.attrs.update(attrs)
        ds.to_netcdf(out / f"{name}{tag}.nc")
    return feats, tracks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--thr", type=float, default=0.)
    ap.add_argument("--mask", default="cloud", choices=["cloud", "core"])
    a = ap.parse_args()
    f, t = analyse(a.expt, a.rt, a.rep, a.thr, a.mask)
    c = t[(t.birth == "new") & (t.death == "gone")]
    print(f"{a.rt} rep_{a.rep:02d} features={len(f)} tracks={len(t)} simple={len(c)} "
          f"median lifetime all={t.lifetime.median() / 60:.1f} simple={c.lifetime.median() / 60:.1f} min", flush=True)
