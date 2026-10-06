"""Split projected cloud objects into sub-clouds seeded at buoyant-core regions; tabulate objects and sub-clouds.

Method: core columns (any point with ql > 0, w > 0, thv above the slab mean) are closed with a 3 x 3 element and
labelled; regions of at least CORE_MIN columns are seeds. Each object with two or more seeds is divided by a
watershed on the liquid water path grown from the seeds. Everything is doubly periodic.

python segment.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
Output: .../<rt>/rep_NN/objects_<t>.nc
"""
import argparse

import numpy as np
import xarray as xr
from scipy import ndimage
from skimage.segmentation import watershed

import masks as mk
from les_io import Run
from snapshot import load, out_path, run_dir, snapshot_times

IZB = 2
NLEV = 4            # 100 m: cloud-base layer and cloud-top layer
CORE_MIN = 4        # columns
S8 = np.ones((3, 3), dtype=bool)


def _wrap(a, p):
    return np.pad(a, p, mode="wrap")


def seeds(core_cols, core_min=CORE_MIN, close=True):
    """Labels of core regions that are large enough to seed a sub-cloud."""
    p = 2
    m = core_cols
    if close:
        m = ndimage.binary_closing(_wrap(core_cols, p), structure=S8)[p:-p, p:-p] | core_cols
    lab, n = mk.label_periodic(m)
    lab = np.where(core_cols, lab, 0)
    if n == 0:
        return lab, 0
    keep = np.flatnonzero(np.bincount(lab.ravel(), minlength=n + 1)[1:] >= core_min) + 1
    remap = np.zeros(n + 1, dtype=int)
    remap[keep] = np.arange(1, keep.size + 1)
    return remap[lab], int(keep.size)


def split(obj, nobj, seed, nseed, lwp):
    """Sub-cloud labels. Objects with fewer than two seeds are kept whole. Returns labels, n, parent, n_seeds."""
    ny, nx = obj.shape
    p = (ny // 2, nx // 2)
    nse = np.zeros(nobj + 1, dtype=int)
    if nseed:
        so = np.zeros(nseed + 1, dtype=int)
        so[seed[seed > 0]] = obj[seed > 0]
        np.add.at(nse, so[1:], 1)
    multi = nse[obj] >= 2
    sub = np.zeros_like(obj)
    if multi.any():
        mk_ = np.where(multi, seed, 0)
        ws = watershed(-_wrap(lwp, ((p[0], p[0]), (p[1], p[1]))), markers=_wrap(mk_, ((p[0], p[0]), (p[1], p[1]))),
                       mask=_wrap(multi, ((p[0], p[0]), (p[1], p[1]))), connectivity=2)
        sub = ws[p[0]:p[0] + ny, p[1]:p[1] + nx]
    whole = np.where(~multi & (obj > 0), obj + nseed, 0)
    lab = sub + whole
    ids = np.unique(lab[lab > 0])
    remap = np.zeros(lab.max() + 1, dtype=int)
    remap[ids] = np.arange(1, ids.size + 1)
    lab = remap[lab]
    parent = np.zeros(ids.size + 1, dtype=int)
    parent[lab[lab > 0]] = obj[lab > 0]
    return lab, int(ids.size), parent[1:], nse[1:]


def table(f, z, dx, dy, kb, lab, n, lwp, nlev=NLEV):
    """One row per labelled region."""
    qc, w = f["qc"], f["w"]
    cloudy = qc > 0.
    core = mk.core(qc, w, f["thv"])
    cu = mk.cloudy_updraft(qc, w)
    idx = np.arange(1, n + 1)
    area = mk.object_areas(lab, n)
    xy = mk.periodic_centroids(lab, n, dx, dy)
    out = dict(area=area * dx * dy, D=mk.equivalent_diameter(area, dx, dy), x=xy[:, 0], y=xy[:, 1],
               lwp=ndimage.sum(lwp, lab, idx) / np.maximum(area, 1), core_cols=ndimage.sum(core.any(axis=0), lab, idx))
    for name, m3 in (("cloud", cloudy), ("core", core)):
        k, j, i = np.nonzero(m3)
        l = lab[j, i]
        ok = l > 0
        k, j, i, l = k[ok], j[ok], i[ok], l[ok]
        kt, kbm = np.full(n + 1, -1), np.full(n + 1, z.size)
        np.maximum.at(kt, l, k)
        np.minimum.at(kbm, l, k)
        has = kt[1:] >= 0
        out[f"{name}_top"] = np.where(has, z[np.maximum(kt[1:], 0)], np.nan)
        out[f"{name}_base"] = np.where(has, z[np.minimum(kbm[1:], z.size - 1)], np.nan)
        out[f"{name}_depth"] = out[f"{name}_top"] - out[f"{name}_base"]
        sel = k >= kt[l] - (nlev - 1)
        cnt = np.bincount(l[sel], minlength=n + 1)[1:]
        out[f"w_top_{name}"] = np.where(cnt > 0, np.bincount(l[sel], weights=w[k[sel], j[sel], i[sel]], minlength=n + 1)[1:]
                                        / np.maximum(cnt, 1), np.nan)
    for name, m3 in (("core", core), ("cu", cu)):
        c = ndimage.sum(m3[kb:kb + nlev + 1].sum(axis=0), lab, idx)
        s = ndimage.sum(np.where(m3[kb:kb + nlev + 1], w[kb:kb + nlev + 1], 0.).sum(axis=0), lab, idx)
        out[f"n_{name}_base"], out[f"w_{name}_base"] = c, np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def analyse(expt, rt, rep, t, core_min=CORE_MIN, close=True, keep_maps=True):
    run = Run(run_dir(expt, rt, rep))
    src = out_path(expt, rt, rep, t)
    with xr.open_dataset(src) as s:
        kb = int(s["kb"].values[IZB])
        attrs = {a: s.attrs[a] for a in ("expt", "rt", "rep", "t_sec", "lst_solar")}
    f, bs = load(run, t)
    lwp = (f["qc"] * bs["rhoref"][:, None, None]).sum(axis=0) * (run.z[1] - run.z[0])
    core_cols = mk.core(f["qc"], f["w"], f["thv"]).any(axis=0)
    obj, nobj = mk.label_periodic(f["qc"].max(axis=0) > 0.)
    seed, nseed = seeds(core_cols, core_min, close)
    sub, nsub, parent, nse = split(obj, nobj, seed, nseed, lwp)
    ds = xr.Dataset()
    for tag, lab, n in (("obj", obj, nobj), ("sub", sub, nsub)):
        for v, a in table(f, run.z, run.dx, run.dy, kb, lab, n, lwp).items():
            ds[f"{tag}_{v}"] = (tag, a)
    ds["obj_n_seeds"] = ("obj", nse)
    ds["sub_parent"] = ("sub", parent)
    ds["sub_parent_n_seeds"] = ("sub", nse[parent - 1])
    if keep_maps:
        for v, a in (("map_obj", obj), ("map_sub", sub), ("map_seed", seed), ("map_lwp", lwp.astype(np.float32))):
            ds[v] = (("y", "x"), a)
    ds.attrs.update(attrs, kb=kb, zb=float(run.z[kb]), core_min=core_min, close=int(close), dx=run.dx, dy=run.dy,
                    n_seeds=nseed)
    enc = {v: dict(zlib=True, complevel=4) for v in ds.data_vars if v.startswith("map_")}
    ds.to_netcdf(src.with_name(f"objects_{int(t):07d}.nc"), encoding=enc)
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    a = ap.parse_args()
    a.t = a.t or list(snapshot_times(a.expt, a.rt, a.rep, skip_first=True))
    for t in a.t:
        d = analyse(a.expt, a.rt, a.rep, t)
        print(f"{a.rt} rep_{a.rep:02d} t={t} objects={d.sizes['obj']} seeds={d.attrs['n_seeds']} sub-clouds={d.sizes['sub']} "
              f"objects with 2+ seeds={int((d.obj_n_seeds >= 2).sum())}", flush=True)
