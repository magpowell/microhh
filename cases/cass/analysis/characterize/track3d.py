"""Three-dimensional cloud tracking on the 60 s fields of the v3 runs (thl, qt, w to 6 km, float32).

python track3d.py masks --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1 [--workers 32]
    liquid water from thl, qt by saturation adjustment with the base state of the nearest hour; per frame the cloudy
    mask (ql > 0) and the core mask (cloudy, w > 0, thv above the slab mean) as packed bits: track3d/masks_<t>.npz
python track3d.py link --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1
    3D objects = face-connected cloudy cells (periodic in x and y); frames linked by mutual largest overlap as in
    track.py; output track3d/features.nc (one row per object per frame) and track3d/tracks.nc (one row per track)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import masks as mk
import thermo as th
from les_io import Run
from track import Families, link, overlaps

Z_LO, Z_HI = 1000., 5000.                      # m, levels searched for cloud
FACE = ndimage.generate_binary_structure(3, 1)


def run_dir(expt, rt, rep):
    return Path(os.environ["SCRATCH"]) / "CASS_LES" / "experiments" / expt / rt / f"rep_{rep:02d}"


def out_dir(expt, rt, rep):
    d = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt / rt / f"rep_{rep:02d}" / "track3d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hf_times(rd):
    return sorted(int(p.name.split(".")[-1]) for p in rd.glob("thl_hf.*"))


def levels(run):
    return int(np.searchsorted(run.z, Z_LO)), int(np.searchsorted(run.z, Z_HI))


def frame_masks(rd, t, k0, k1):
    """Cloudy and core masks over levels k0..k1-1 at time t."""
    run = Run(rd)
    nz = int(os.path.getsize(rd / f"thl_hf.{t:07d}") // (4 * run.itot * run.jtot))
    rd_ = lambda v: np.memmap(rd / f"{v}_hf.{t:07d}", dtype="<f4", mode="r", shape=(nz, run.jtot, run.itot))
    hour = int(round(t / 3600.)) * 3600
    bs = run.basestate(hour)
    sl = slice(k0, k1)
    thl, qt = np.asarray(rd_("thl")[sl], dtype=float), np.asarray(rd_("qt")[sl], dtype=float)
    wh = np.asarray(rd_("w")[k0:k1 + 1], dtype=float)
    w = 0.5 * (wh[:-1] + wh[1:])
    p, exn = bs["pref"][sl, None, None], bs["exnref"][sl, None, None]
    a = th.sat_adjust(thl, qt, p, exn)
    ql, qi = a["ql"], a["qi"]
    cloudy = (ql + qi) > 0.
    thv = th.theta_v(thl, qt, ql, qi, exn)
    core = cloudy & (w > 0.) & (thv > thv.mean(axis=(1, 2), keepdims=True))
    return cloudy, core, (ql + qi).astype(np.float32)


def _mask_job(args):
    rd, t, k0, k1, out = args
    cloudy, core, qc = frame_masks(rd, t, k0, k1)
    np.savez_compressed(out / f"masks_{t:07d}.npz", cloudy=np.packbits(cloudy, axis=None), core=np.packbits(core, axis=None),
                        lwp_k=(qc.sum(axis=0)).astype(np.float32), shape=np.array(cloudy.shape))
    return t, int(cloudy.sum()), int(core.sum())


def masks(expt, rt, rep, workers):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    k0, k1 = levels(run)
    out = out_dir(expt, rt, rep)
    jobs = [(rd, t, k0, k1, out) for t in hf_times(rd)]
    with ProcessPoolExecutor(workers) as ex:
        for t, nc, ncore in ex.map(_mask_job, jobs):
            print(f"t={t} cloudy cells {nc} core cells {ncore}", flush=True)
    np.save(out / "levels.npy", np.array([k0, k1]))


def load_masks(out, t):
    f = np.load(out / f"masks_{t:07d}.npz")
    shape = tuple(f["shape"])
    n = int(np.prod(shape))
    return (np.unpackbits(f["cloudy"], count=n).reshape(shape).astype(bool),
            np.unpackbits(f["core"], count=n).reshape(shape).astype(bool))


def label_periodic_3d(mask):
    """Face-connected labels of a (z, y, x) mask, periodic in x and y. Returns labels, n."""
    lab, n = ndimage.label(mask, structure=FACE)
    if n == 0:
        return lab, 0
    fam = Families()
    for _ in range(n):
        fam.new()
    for a, b in ((lab[:, :, 0], lab[:, :, -1]), (lab[:, 0, :], lab[:, -1, :])):
        m = (a > 0) & (b > 0)
        for u, v in set(zip(a[m].tolist(), b[m].tolist())):
            fam.union(u, v)
    roots = np.array([fam.find(i) for i in range(n + 1)])
    uniq = np.unique(roots[1:])
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, uniq.size + 1)
    return remap[roots][lab], int(uniq.size)


def props(lab, n, core, z, dx, dy):
    idx = np.arange(1, n + 1)
    kk = np.arange(lab.shape[0])[:, None, None]
    vol = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    cvol = np.bincount(lab.ravel(), weights=core.ravel(), minlength=n + 1)[1:]
    top = ndimage.maximum(np.where(lab > 0, kk, -1), lab, idx).astype(int)
    base = ndimage.minimum(np.where(lab > 0, kk, 10 ** 6), lab, idx).astype(int)
    lab2 = lab.max(axis=0)          # projected area; a column holding two objects counts for the higher label
    col = np.bincount(lab2.ravel(), minlength=n + 1)[1:]
    xy = mk.periodic_centroids(lab2, n, dx, dy)
    return dict(volume=vol * dx * dy, core_volume=cvol * dx * dy, z_base=z[base], z_top=z[top], depth=z[top] - z[base],
                area=col * dx * dy, x=xy[:, 0], y=xy[:, 1])


def link_run(expt, rt, rep):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    out = out_dir(expt, rt, rep)
    k0, k1 = np.load(out / "levels.npy")
    z = run.z[k0:k1]
    dz = float(run.z[1] - run.z[0])
    times = hf_times(rd)
    fam = Families()
    birth, death, feats = {}, {}, []
    lab0, n0, tid0 = None, 0, np.zeros(1, dtype=int)
    for f, t in enumerate(times):
        cloudy, core = load_masks(out, t)
        lab, n = label_periodic_3d(cloudy)
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
            p = props(lab, n, core, z, run.dx, run.dy)
            p["volume"] *= dz; p["core_volume"] *= dz
            feats.append(pd.DataFrame(dict(frame=f, time=t, track=tid[1:], n_pred=npred[1:], n_succ=0, **p)))
        lab0, n0, tid0 = lab, n, tid
        if f % 20 == 0:
            print(f"frame {f} t={t} objects {n}", flush=True)
    for i in range(1, n0 + 1):
        death[tid0[i]] = "end"
    feats = pd.concat(feats, ignore_index=True)
    dt = float(times[1] - times[0])
    feats["age"] = (feats.frame - feats.groupby("track").frame.transform("min")) * dt
    feats["lst"] = run.lst(feats.time.values)
    g = feats.groupby("track")
    tr = pd.DataFrame(dict(frame_first=g.frame.min(), frame_last=g.frame.max(), volume_max=g.volume.max(), core_volume_max=g.core_volume.max(),
                           area_max=g.area.max(), depth_max=g.depth.max(), top_max=g.z_top.max(),
                           merges_in=g.n_pred.apply(lambda s: int(np.maximum(s - 1, 0).sum())),
                           splits_out=g.n_succ.apply(lambda s: int(np.maximum(s - 1, 0).sum())), lst_first=g.lst.min()))
    tr["lifetime"] = (tr.frame_last - tr.frame_first + 1) * dt
    tr["birth"] = [birth[i][0] for i in tr.index]
    tr["death"] = [death[i] for i in tr.index]
    tr["family"] = [fam.find(i) for i in tr.index]
    tr = tr.reset_index()
    attrs = dict(expt=expt, rt=rt, rep=rep, dt=dt, z_lo=float(z[0]), z_hi=float(z[-1]))
    for name, df in (("features", feats), ("tracks", tr)):
        ds = xr.Dataset.from_dataframe(df)
        ds.attrs.update(attrs)
        ds.to_netcdf(out / f"{name}.nc")
    return feats, tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "link"])
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    ap.add_argument("--rt", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "masks":
        masks(a.expt, a.rt, a.rep, a.workers)
    else:
        f, t = link_run(a.expt, a.rt, a.rep)
        u = t[(t.birth == "new") & (t.death == "gone") & (t.merges_in == 0) & (t.splits_out == 0)]
        print(f"objects {len(f)}, tracks {len(t)}, untouched {len(u)}: lifetime mean {u.lifetime.mean() / 60.:.1f} min, "
              f"p99 {np.percentile(u.lifetime, 99) / 60.:.1f}, max {u.lifetime.max() / 60.:.1f}", flush=True)
