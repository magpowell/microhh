"""Split of the dumped pressure into the part forced by buoyancy and the rest, from full or high-frequency 3D files.

The buoyancy part is solved offline from thl and qt, with buoyancy on half levels as the model computes it. The rest is
the dumped pressure minus the buoyancy part: the dynamic pressure, with the small parts from diffusion, Coriolis and
the time stepping. Pressures are divided by density and have their slab mean removed; accelerations are on half levels.

python pressure_split.py --run <run dir> --t 28800 [--full]
"""
import argparse
from pathlib import Path

import numpy as np

import pressure as pr
import thermo as th
from les_io import Run


def read(run, var, t, hf=True):
    """3D field as float64 (z, y, x); the high-frequency file holds the lowest levels in float32."""
    if not hf:
        return np.array(run.field(var, t), dtype=float)
    f = run.dir / f"{var}_hf.{int(t):07d}"
    nk, rem = divmod(f.stat().st_size, 4 * run.jtot * run.itot)
    if rem:
        raise ValueError(f"{f} does not hold whole levels")
    return np.fromfile(f, dtype="<f4").reshape(nk, run.jtot, run.itot).astype(float)


def basestate(run, t):
    """Base state of the last restart time at or before t, and the density of the dynamics."""
    times = sorted(int(f.name.split(".")[1]) for f in run.dir.glob("thermo_basestate.[0-9]*"))
    bs = run.basestate(max(s for s in times if s <= t))
    rr = np.fromfile(run.dir / "rhoref.0000000", dtype="<f8")
    bs["rho"], bs["rhoh"] = rr[:run.ktot], rr[run.ktot:]
    return bs


def buoyancy_half(thl, qt, bs):
    """Buoyancy anomaly on the nk + 1 half levels of the fields, zero on the first and the last."""
    nk = thl.shape[0]
    b = np.zeros((nk + 1,) + thl.shape[1:])
    k3 = (slice(1, nk), None, None)
    thlh, qth = 0.5 * (thl[1:] + thl[:-1]), 0.5 * (qt[1:] + qt[:-1])
    a = th.sat_adjust(thlh, qth, bs["prefh"][k3], bs["exnrefh"][k3])
    thv = th.theta_v(thlh, qth, a["ql"], a["qi"], bs["exnrefh"][k3])
    b[1:nk] = th.grav * (thv - thv.mean(axis=(1, 2), keepdims=True)) / bs["thvrefh"][k3]
    return b


def anomaly(f):
    return f - f.mean(axis=(1, 2), keepdims=True)


def solve(run, bs, b, pad=True):
    """Pressure of the buoyancy b given on the lowest half levels. With pad the column is continued to the model top
    with zero buoyancy anomaly; without it a wall closes the column on top of the data."""
    nk = b.shape[0] - 1
    n = run.ktot if pad else nk
    bf = np.zeros((n + 1,) + b.shape[1:])
    bf[:nk] = b[:nk]
    return anomaly(pr.project(None, None, bf, bs["rho"][:n], bs["rhoh"][:n + 1], run.dx, run.dy, run.z[1] - run.z[0])[:nk])


def split(run, t, hf=True, nk=None, pad=True):
    """Dumped pressure p, buoyancy part pb, rest pd, and the accelerations b, -dpb/dz, -dpd/dz on half levels.

    nk limits the data to the lowest nk levels."""
    thl, qt, p = (read(run, v, t, hf) for v in ("thl", "qt", "p"))
    nk = nk or thl.shape[0]
    thl, qt, p = thl[:nk], qt[:nk], anomaly(p[:nk])
    bs = basestate(run, t)
    dz = run.z[1] - run.z[0]
    b = buoyancy_half(thl, qt, bs)
    pb = solve(run, bs, b, pad)
    pd = p - pb
    return dict(p=p, pb=pb, pd=pd, b=b, a_pb=-pr.grad_z(pb, dz), a_pd=-pr.grad_z(pd, dz), z=run.z[:nk], zh=run.zh[:nk + 1])


def offline_total(run, t, hf=True):
    """Pressure from buoyancy and second-order advection, to compare with the dumped one."""
    thl, qt, u, v, w = (read(run, n, t, hf) for n in ("thl", "qt", "u", "v", "w"))
    nk = thl.shape[0]
    bs = basestate(run, t)
    dz = run.z[1] - run.z[0]
    wh = np.zeros((nk + 1,) + w.shape[1:])
    wh[:nk] = w
    rho, rhoh = bs["rho"][:nk], bs["rhoh"][:nk + 1]
    Tu, Tv, Tw = pr.advection(u, v, wh, rho, rhoh, run.dx, run.dy, dz)
    Tw += buoyancy_half(thl, qt, bs)
    return anomaly(pr.project(Tu, Tv, Tw, rho, rhoh, run.dx, run.dy, dz))


def agreement(a, b, z, z0, z1):
    """Correlation and ratio of root mean squares of two fields between z0 and z1."""
    k = (z >= z0) & (z <= z1)
    x, y = a[k].ravel(), b[k].ravel()
    return float(np.corrcoef(x, y)[0, 1]), float(np.sqrt((x**2).mean() / (y**2).mean()))


def report(run, t, hf):
    s = split(run, t, hf)
    z = s["z"]
    print(f"{run.dir}\nt = {t} s, {'high-frequency' if hf else 'full'} files, {z.size} levels to {z[-1] + 0.5 * (z[1] - z[0]):.0f} m")
    for name, (z0, z1) in dict(subcloud=(0., 1000.), cloud=(1000., 4000.)).items():
        k = (z >= z0) & (z <= z1)
        rms = {n: float(np.sqrt((s[n][k]**2).mean())) for n in ("p", "pb", "pd")}
        r, q = agreement(offline_total(run, t, hf), s["p"], z, z0, z1)
        print(f"  {name:8s} rms p {rms['p']:.4f}  pb {rms['pb']:.4f}  pd {rms['pd']:.4f} m2 s-2;"
              f"  offline total against dumped: r = {r:.3f}, rms ratio {q:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--t", type=int, nargs="+", required=True)
    ap.add_argument("--full", action="store_true", help="hourly full files instead of the high-frequency ones")
    a = ap.parse_args()
    for t in a.t:
        report(Run(a.run), t, not a.full)
