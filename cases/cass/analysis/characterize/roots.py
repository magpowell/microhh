"""Root columns beneath cloud-base cloudy updrafts: buoyancy work against kinetic energy, and low-level inflow reach.

python roots.py --expt no_aerosols_zero_wind_v2
"""
import argparse
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy import ndimage

import masks as mk
from les_io import Run
from snapshot import out_path, run_dir, snapshot_times

IZB = 2                      # cloud-base threshold 1e-3
Z_LOW = 200.                 # inflow layer depth [m]
R_EDGES = np.arange(0., 3001., 100.)


def periodic_distance(mask, dx):
    """Distance [m] from every point to the nearest True point on a doubly periodic domain."""
    n = mask.shape[0]
    d = ndimage.distance_transform_edt(~np.tile(mask, (3, 3)))
    return d[n:2 * n, n:2 * n] * dx


def analyse(args):
    expt, rt, rep, t = args
    run = Run(run_dir(expt, rt, rep))
    with xr.open_dataset(out_path(expt, rt, rep, t)) as s:
        kb = int(s["kb"].values[IZB])
    z = run.z
    n = kb + 1
    wh = np.asarray(run.field("w", t)[:n + 1], dtype=float)
    w = 0.5 * (wh[:-1] + wh[1:])
    ql = np.asarray(run.field("ql", t)[kb], dtype=float) + np.asarray(run.field("qi", t)[kb], dtype=float)
    root = (ql > 0.) & (w[kb] > 0.)
    b = np.array(run.field("b", t)[:n], dtype=float)
    b -= b.mean(axis=(1, 2), keepdims=True)
    u = np.asarray(run.field("u", t)[:n], dtype=float)
    v = np.asarray(run.field("v", t)[:n], dtype=float)
    div = (np.roll(u, -1, axis=2) - u) / run.dx + (np.roll(v, -1, axis=1) - v) / run.dy
    w_root, b_root, d_root = (f[:, root].mean(axis=1) for f in (w, b, div))
    klow = int(np.searchsorted(z, Z_LOW))
    dlow = div[:klow].mean(axis=0)
    r = periodic_distance(root, run.dx)
    idx = np.digitize(r.ravel(), R_EDGES) - 1
    ok = idx < R_EDGES.size - 1
    cnt = np.bincount(idx[ok], minlength=R_EDGES.size - 1)
    d_r = np.bincount(idx[ok], weights=dlow.ravel()[ok], minlength=R_EDGES.size - 1) / np.maximum(cnt, 1)
    wl_r = np.bincount(idx[ok], weights=w[klow].ravel()[ok], minlength=R_EDGES.size - 1) / np.maximum(cnt, 1)
    zr = z[:n] / z[kb]
    grid = np.linspace(0.05, 1., 20)
    return dict(rt=rt, rep=rep, t=t, zb=z[kb], root_frac=root.mean(),
                BW=np.trapz(b_root, z[:n]), KE=0.5 * (w[kb][root] ** 2).mean(), KE_mean=0.5 * w_root[kb] ** 2,
                w_prof=np.interp(grid, zr, w_root), b_prof=np.interp(grid, zr, b_root),
                d_prof=np.interp(grid, zr, d_root), d_r=d_r, w_r=wl_r, grid=grid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    ap.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args()
    a.t = a.t or list(snapshot_times(a.expt, a.rt, a.rep, skip_first=False))
    jobs = [(a.expt, rt, rep, t) for t in a.t for rt in ("2stream", "raytracer") for rep in (1, 2, 3, 4)]
    with Pool(a.nproc) as p:
        res = p.map(analyse, jobs)
    ds = xr.Dataset(
        {k: (("case",), [r[k] for r in res]) for k in ("rt", "rep", "t", "zb", "root_frac", "BW", "KE", "KE_mean")}
        | {k: (("case", "zrel"), np.array([r[k] for r in res])) for k in ("w_prof", "b_prof", "d_prof")}
        | {k: (("case", "r"), np.array([r[k] for r in res])) for k in ("d_r", "w_r")},
        coords=dict(zrel=res[0]["grid"], r=0.5 * (R_EDGES[:-1] + R_EDGES[1:])))
    out = out_path(a.expt, "2stream", 1, 0).parents[2] / "roots.nc"
    ds.to_netcdf(out)
    print("wrote", out)
