"""Verification: offline thermodynamics vs the model's dumped T, ql, b; snapshot slab means vs 5-min statistics.

python closure_check.py --rt 2stream --rep 1 --t 36000
"""
import argparse

import numpy as np
import xarray as xr

import masks as mk
import thermo as th
from les_io import Run
from snapshot import run_dir


def thermo_closure(run, t):
    bs = run.basestate(t)
    F = {v: run.field(v, t) for v in ("thl", "qt", "T", "ql", "qi", "b")}
    acc = {k: [0., 0., 0] for k in ("T", "ql", "qi", "b", "T_sat")}
    ncold = 0
    for k in range(run.ktot):
        thl, qt = (np.asarray(F[v][k], dtype=float) for v in ("thl", "qt"))
        a = th.sat_adjust(thl, qt, bs["pref"][k], bs["exnref"][k])
        ncold += a["n_cold"]
        thv = th.theta_v(thl, qt, F["ql"][k], F["qi"][k], bs["exnref"][k])
        d = dict(T=a["T"] - F["T"][k], ql=a["ql"] - F["ql"][k], qi=a["qi"] - F["qi"][k],
                 b=th.buoyancy(thv, bs["thvref"][k]) - F["b"][k])
        sat = np.asarray(F["ql"][k]) > 0
        if sat.any():
            d["T_sat"] = d["T"][sat]
        for name, x in d.items():
            acc[name][0] = max(acc[name][0], float(np.abs(x).max()))
            acc[name][1] += float((x**2).sum())
            acc[name][2] += x.size
    print(f"thermo closure, {run.dir.parent.name}/{run.dir.name} t={t}: cold saturated points = {ncold}")
    for name, (mx, ss, n) in acc.items():
        print(f"  {name:6s} max|diff| = {mx:.3e}   rms = {np.sqrt(ss / max(n, 1)):.3e}   n = {n}")


def stats_closure(run, t):
    f = run.dir / "cass.default.0000000.nc"
    th_ = xr.open_dataset(f, group="thermo", decode_times=False)
    de = xr.open_dataset(f, group="default", decode_times=False)
    root = xr.open_dataset(f, decode_times=False)
    it = int(np.argmin(np.abs(root["time"].values - t)))
    print(f"stats closure at stats time {float(root['time'].values[it]):.0f} s (snapshot {t} s)")
    ql = np.asarray(run.field("ql", t), dtype=float)
    pairs = [("thl", th_, "thl"), ("qt", th_, "qt"), ("ql", th_, "ql")]
    for v, g, name in pairs:
        snap = ql.mean(axis=(1, 2)) if v == "ql" else np.asarray(run.field(v, t), dtype=float).mean(axis=(1, 2))
        ref = g[name].values[it]
        print(f"  slab mean {v:4s} max|snapshot - stats| = {np.abs(snap - ref).max():.3e}  (max |stats| {np.abs(ref).max():.3e})")
    if "ql_frac" in th_:
        print(f"  cloud fraction max|diff| = {np.abs((ql > 0).mean(axis=(1, 2)) - th_['ql_frac'].values[it]).max():.3e}"
              f"  (max {th_['ql_frac'].values[it].max():.3e})")
    w = np.asarray(run.field("w", t), dtype=float)
    if "w_2" in de:
        print(f"  w variance (zh) max|diff| = {np.abs((w**2).mean(axis=(1, 2)) - de['w_2'].values[it][:run.ktot]).max():.3e}"
              f"  (max {de['w_2'].values[it].max():.3e})")
    for k in ("phydro", "rho"):
        if k in th_:
            ref = run.basestate(t)["pref" if k == "phydro" else "rhoref"]
            print(f"  stats {k} vs thermo_basestate: max rel diff = {np.abs(th_[k].values[it] / ref - 1.).max():.3e}")
    fq = run.dir / "cass.ql.0000000.nc"
    if fq.exists():
        dq = xr.open_dataset(fq, group="default", decode_times=False)
        if "area" in dq:
            print(f"  ql-mask area max|diff| vs snapshot cloudy fraction = "
                  f"{np.abs((ql > 0).mean(axis=(1, 2)) - dq['area'].values[it]).max():.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, default=36000)
    a = ap.parse_args()
    r = Run(run_dir(a.expt, a.rt, a.rep))
    print(f"grid {r.shape}, dx={r.dx}, z[0]={r.z[0]}, start {r.t0} UTC, solar LST at t = {float(r.lst(a.t)):.3f} h")
    thermo_closure(r, a.t)
    stats_closure(r, a.t)
