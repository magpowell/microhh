"""Per-cloud depth and updraft speed at cloud base, from the 3D snapshots (one row per cloud).

python cloud_w.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
Output: .../<rt>/rep_NN/cloudw_<t>.nc
"""
import argparse

import numpy as np
import xarray as xr
from scipy import ndimage

import masks as mk
from les_io import Run
from snapshot import cloud_base_index, frames_every, load, out_path, run_dir, snapshot_times

IZB = 2
NLEV = 4            # levels above the cloud-base level included in the cloud-base layer (100 m)


def layer_mean(m3, w, lab, n, k0, k1):
    """Count and mean w of masked points in levels k0..k1, per cloud."""
    idx = np.arange(1, n + 1)
    cnt = ndimage.sum(m3[k0:k1 + 1].sum(axis=0), lab, idx)
    tot = ndimage.sum(np.where(m3[k0:k1 + 1], w[k0:k1 + 1], 0.).sum(axis=0), lab, idx)
    return cnt, np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def own_base_mean(m3, w, lab, n, nlev):
    """Same, over the lowest nlev + 1 levels in which each cloud holds masked points. Also returns that base level."""
    k, j, i = np.nonzero(m3)
    l = lab[j, i]
    ok = l > 0
    k, j, i, l = k[ok], j[ok], i[ok], l[ok]
    k0 = np.full(n + 1, m3.shape[0], dtype=int)
    np.minimum.at(k0, l, k)
    sel = k <= k0[l] + nlev
    cnt = np.bincount(l[sel], minlength=n + 1)[1:]
    tot = np.bincount(l[sel], weights=w[k[sel], j[sel], i[sel]], minlength=n + 1)[1:]
    return cnt, np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), np.where(cnt > 0, k0[1:], -1)


def table(f, z, dx, dy, kb, nlev=NLEV):
    qc, w = f["qc"], f["w"]
    cloudy = qc > 0.
    core, cu = mk.core(qc, w, f["thv"]), mk.cloudy_updraft(qc, w)
    lab, n = mk.label_periodic(cloudy.any(axis=0))
    idx = np.arange(1, n + 1)
    k = np.arange(z.size)[:, None, None]
    top = ndimage.maximum(np.where(cloudy, k, -1).max(axis=0), lab, idx).astype(int)
    base = ndimage.minimum(np.where(cloudy, k, z.size).min(axis=0), lab, idx).astype(int)
    area = mk.object_areas(lab, n)
    out = dict(area=area * dx * dy, D=mk.equivalent_diameter(area, dx, dy), z_top=z[top], z_base=z[base],
               depth=z[top] - z[base], core_cols=ndimage.sum(core.any(axis=0), lab, idx))
    for name, m3 in (("core", core), ("cu", cu)):
        out[f"n_{name}_level"], out[f"w_{name}_level"] = layer_mean(m3, w, lab, n, kb, kb)
        out[f"n_{name}_layer"], out[f"w_{name}_layer"] = layer_mean(m3, w, lab, n, kb, kb + nlev)
        out[f"n_{name}_own"], out[f"w_{name}_own"], kk = own_base_mean(m3, w, lab, n, nlev)
        out[f"z_{name}_own"] = np.where(kk >= 0, z[np.maximum(kk, 0)], np.nan)
    return out


def analyse(expt, rt, rep, t, fields=None):
    """fields: (f, run) already loaded for this t, to share one load between scripts."""
    f, run = fields if fields is not None else (None, Run(run_dir(expt, rt, rep)))
    if f is None:
        f, _ = load(run, t)
    kb = cloud_base_index(expt, rt, rep, t, IZB)
    tab = table(f, run.z, run.dx, run.dy, kb)
    ds = xr.Dataset({v: ("cloud", a) for v, a in tab.items()})
    ds.attrs.update(expt=expt, rt=rt, rep=rep, t_sec=int(t), lst_solar=float(run.lst(t)), kb=kb, zb=float(run.z[kb]), layer_levels=NLEV + 1)
    out = out_path(expt, rt, rep, 0).parent
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / f"cloudw_{int(t):07d}.nc")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    ap.add_argument("--every", type=float, default=None, help="instead: every N minutes of the 60 s fields within --solar")
    ap.add_argument("--solar", type=float, nargs=2, default=(11.9, 16.1))
    a = ap.parse_args()
    if a.every:
        a.t = frames_every(Run(run_dir(a.expt, a.rt, a.rep)), a.every, a.solar)
    a.t = a.t or list(snapshot_times(a.expt, a.rt, a.rep, skip_first=False))
    for t in a.t:
        d = analyse(a.expt, a.rt, a.rep, t)
        print(f"{a.rt} rep_{a.rep:02d} t={t} clouds={d.sizes['cloud']} with core={int((d.core_cols > 0).sum())} "
              f"core at cloud-base layer={int((d.n_core_layer > 0).sum())} at level={int((d.n_core_level > 0).sum())}", flush=True)
