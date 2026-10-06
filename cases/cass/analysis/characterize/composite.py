"""Cloud-centred composites of one snapshot in sun-parallel and sun-perpendicular slices, with the forces on the air.

Slices pass through the cloud centroid; r is positive away from the sun (toward the shadow) in the parallel slice.
Horizontal distance is scaled by the chord L of the cloud along the slice, height by cloud base z_b.

python composite.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
"""
import argparse

import numpy as np
import xarray as xr
from scipy import ndimage

import masks as mk
import pressure as pr
import thermo as th
from les_io import Run, eqtime_h, lowest
from snapshot import cloud_base_index, frames_every, load, out_path, run_dir, snapshot_times

IZB = 2                                   # cloud-base threshold 1e-3
XL = np.linspace(-1., 1., 200)
ZND = np.linspace(0., 1., 100)
MIN_D, MIN_CHORD, R_MAX = 500., 1000., 4000.
DIRS = ("parallel", "perpendicular")
V3 = ("b", "w", "us", "qt", "thl", "a_pb", "a_pb_sub", "a_pb_cld", "a_pd", "h_pb", "h_pd")
V1 = ("sw", "H", "LE")


def solar_azimuth(run, t):
    """Azimuth of the sun [rad from north, clockwise] (NOAA)."""
    h0 = run.t0.hour + run.t0.minute / 60. + run.t0.second / 3600. + t / 3600.
    doy = run.t0.timetuple().tm_yday
    g = 2. * np.pi / 365. * (doy - 1. + h0 / 24.)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((h0 % 24. * 60. + eqtime_h(doy, h0) * 60. + 4. * run.lon) / 4. - 180.)
    lat = np.radians(run.lat)
    az = np.arctan2(-np.cos(decl) * np.sin(ha), np.sin(decl) * np.cos(lat) - np.cos(decl) * np.sin(lat) * np.cos(ha))
    zen = np.arccos(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha))
    return az % (2. * np.pi), zen


def line_run(on, i0):
    """Start and length of the contiguous run of True nearest index i0 on a line (not periodic)."""
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return 0, 0
    i = int(idx[np.argmin(np.abs(idx - i0))])
    a = b = i
    while a > 0 and on[a - 1]:
        a -= 1
    while b < on.size - 1 and on[b + 1]:
        b += 1
    return a, b - a + 1


def forces(run, f, bs, kb, kc):
    """Buoyancy anomaly, winds and the accelerations from the buoyancy and dynamic pressure on levels 0..kc.
    Fields that stop below the model top (the 60 s fields end at 6 km) are continued with zero forcing to the top, so
    the pressure is solved with the lid at the real top."""
    nz, nf = f["thl"].shape[0], run.ktot_full
    dz = run.z[1] - run.z[0]
    rr = np.fromfile(run.dir / "rhoref.0000000", dtype="<f8")
    rho, rhoh = rr[:nf], rr[nf:]
    full = lambda a: np.concatenate([a, np.zeros((nf - a.shape[0],) + a.shape[1:])]) if a.shape[0] < nf else np.asarray(a, dtype=float)
    u, v = full(f["u"]), full(f["v"])
    w = np.zeros((nf + 1, run.jtot, run.itot))
    w[:nz] = f["wh"]
    b = f["b"] if "b" in f else th.grav * (f["thv"] - f["thv"].mean(axis=(1, 2), keepdims=True)) / bs["thvref"][:, None, None]
    b = full(b - b.mean(axis=(1, 2), keepdims=True))
    sl = slice(0, kc + 1)
    out = dict(b=b[sl].copy(), w=pr.half_to_full(w)[sl],
               uc=0.5 * (u + np.roll(u, -1, axis=2))[sl], vc=0.5 * (v + np.roll(v, -1, axis=1))[sl])

    def vert(pi):
        return -pr.half_to_full(pr.grad_z(pi, dz))[sl]

    def horiz(pi):
        gx, gy = pr.grad_h_centre(pi[sl], run.dx, run.dy)
        return -gx, -gy

    pib = pr.project(None, None, pr.full_to_half(b), rho, rhoh, run.dx, run.dy, dz)
    out["a_pb"] = vert(pib)
    out["hbx"], out["hby"] = horiz(pib)
    bsub = b.copy()
    bsub[kb:] = 0.
    out["a_pb_sub"] = vert(pr.project(None, None, pr.full_to_half(bsub), rho, rhoh, run.dx, run.dy, dz))
    out["a_pb_cld"] = out["a_pb"] - out["a_pb_sub"]
    del bsub, b
    if "p" in f:        # the model's own pressure (per unit density): the rest is the dynamic part
        anom = lambda a: a - a.mean(axis=(1, 2), keepdims=True)
        pid = anom(full(f["p"])) - anom(pib)
        del u, v
    else:
        del pib
        Tu, Tv, Tw = pr.advection(u, v, w, rho, rhoh, run.dx, run.dy, dz)
        del u, v
        pid = pr.project(Tu, Tv, Tw, rho, rhoh, run.dx, run.dy, dz)
        del Tu, Tv, Tw
    out["a_pd"] = vert(pid)
    out["hdx"], out["hdy"] = horiz(pid)
    return out, w[:nz + 1]


def cloud_table(run, qc, w_full, thv, labels, n):
    """One row per cloud: size, base, top, whether it holds a buoyant cloudy updraft."""
    cloudy = qc > 0.
    core = mk.core(qc, w_full, thv)
    k = np.arange(run.ktot)[:, None, None]
    top = np.where(cloudy, k, -1).max(axis=0)
    base = np.where(cloudy, k, run.ktot).min(axis=0)
    idx = np.arange(1, n + 1)
    area = mk.object_areas(labels, n)
    return dict(area_m2=area * run.dx * run.dy, D=mk.equivalent_diameter(area, run.dx, run.dy),
                z_top=run.z[ndimage.maximum(top, labels, idx).astype(int)],
                z_base=run.z[ndimage.minimum(base, labels, idx).astype(int)],
                core_cols=ndimage.sum(core.any(axis=0), labels, idx),
                w_max=ndimage.maximum(np.where(cloudy, w_full, -np.inf).max(axis=0), labels, idx),
                lwp=ndimage.sum((qc * np.fromfile(run.dir / "rhoref.0000000", dtype="<f8")[:run.ktot, None, None]
                                 ).sum(axis=0) * (run.z[1] - run.z[0]), labels, idx) / np.maximum(area, 1))


def analyse(expt, rt, rep, t):
    run = Run(run_dir(expt, rt, rep))
    kb = cloud_base_index(expt, rt, rep, t)
    kc = kb + 2
    zb = run.z[kb]
    az, zen = solar_azimuth(run, t)
    e = dict(parallel=(-np.sin(az), -np.cos(az)), perpendicular=(np.cos(az), -np.sin(az)))

    f, bs = load(run, t)
    F, wh = forces(run, f, bs, kb, kc)
    qc, thl, qt, thv = f["qc"], f["thl"], f["qt"], f["thv"]
    F["qt"] = (qt - qt.mean(axis=(1, 2), keepdims=True))[:kc + 1]
    F["thl"] = (thl - thl.mean(axis=(1, 2), keepdims=True))[:kc + 1]
    labels, n = mk.label_periodic(qc.max(axis=0) > 0.)
    tab = cloud_table(run, qc, f["w"], thv, labels, n)
    del f, thl, qt, thv, qc, wh
    cen = mk.periodic_centroids(labels, n, run.dx, run.dy)

    it = int(round(t / 60.))
    S = {}
    for name, var, fac in (("H", "thl_fluxbot", bs["rhoref"][0] * th.cp), ("LE", "qt_fluxbot", bs["rhoref"][0] * th.Lv)):
        with xr.open_dataset(run.dir / f"{var}.xy.nc", decode_times=False) as d:
            S[name] = np.squeeze(d[var].isel(time=it).values).astype(float) * fac
    rtsw = [run.dir / f"sw_flux_sfc_{k}_rt.xy.nc" for k in ("dir", "dif")]
    sw_from_raytracer = all(f.exists() for f in rtsw)
    S["sw"] = 0.
    for f in (rtsw if sw_from_raytracer else [run.dir / "sw_flux_dn.xy.nc"]):   # sw_flux_dn is the two-stream field
        with xr.open_dataset(f, decode_times=False) as d:
            S["sw"] = S["sw"] + lowest(d[list(d.data_vars)[0]].isel(time=it))

    kk = np.arange(kc + 1)
    zr = run.z[:kc + 1] / zb
    j1 = np.clip(np.searchsorted(zr, ZND), 1, kc)
    wt = np.clip((ZND - zr[j1 - 1]) / (zr[j1] - zr[j1 - 1]), 0., 1.)
    ds_line = 0.5 * run.dx
    rl = np.arange(-R_MAX, R_MAX + 1., ds_line)
    i0 = rl.size // 2

    acc = {d: {v: np.zeros((ZND.size, XL.size)) for v in V3} | {v: np.zeros(XL.size) for v in V1} for d in DIRS}
    nev = dict.fromkeys(DIRS, 0)
    ev = {d: dict(L=[], D=[], zen=[]) for d in DIRS}
    for c in np.flatnonzero(tab["D"] >= MIN_D):
        for d in DIRS:
            ex, ey = e[d]
            ix = np.rint((cen[c, 0] + rl * ex) / run.dx - 0.5).astype(int) % run.itot
            iy = np.rint((cen[c, 1] + rl * ey) / run.dy - 0.5).astype(int) % run.jtot
            a, m = line_run(labels[iy, ix] == c + 1, i0)
            L = m * ds_line
            if L < MIN_CHORD or a == 0 or a + m == rl.size:
                continue
            r = rl[a] + 0.5 * (m - 1) * ds_line + XL * L
            cx = (cen[c, 0] + r * ex) / run.dx - 0.5
            cy = (cen[c, 1] + r * ey) / run.dy - 0.5
            co3 = np.array([np.repeat(kk[:, None], XL.size, 1), np.tile(cy, (kc + 1, 1)), np.tile(cx, (kc + 1, 1))])

            def samp(name):
                x = ndimage.map_coordinates(F[name], co3, order=1, mode="grid-wrap")
                return x[j1 - 1] * (1. - wt)[:, None] + x[j1] * wt[:, None]

            for vname in ("b", "w", "qt", "thl", "a_pb", "a_pb_sub", "a_pb_cld", "a_pd"):
                acc[d][vname] += samp(vname)
            acc[d]["us"] += samp("uc") * ex + samp("vc") * ey
            acc[d]["h_pb"] += samp("hbx") * ex + samp("hby") * ey
            acc[d]["h_pd"] += samp("hdx") * ex + samp("hdy") * ey
            for vname in V1:
                acc[d][vname] += ndimage.map_coordinates(S[vname], np.array([cy, cx]), order=1, mode="grid-wrap")
            nev[d] += 1
            ev[d]["L"].append(L)
            ev[d]["D"].append(tab["D"][c])

    out = xr.Dataset(coords=dict(dir=list(DIRS), znd=ZND, xl=XL))
    for vname in V3:
        out[vname] = (("dir", "znd", "xl"), np.array([acc[d][vname] / max(nev[d], 1) for d in DIRS]))
    for vname in V1:
        out[vname] = (("dir", "xl"), np.array([acc[d][vname] / max(nev[d], 1) for d in DIRS]))
        out[f"{vname}_domain"] = float(S[vname].mean())
    out["n_events"] = ("dir", [nev[d] for d in DIRS])
    out["L_mean"] = ("dir", [float(np.mean(ev[d]["L"])) if nev[d] else np.nan for d in DIRS])
    for k, x in tab.items():
        out[f"cloud_{k}"] = ("cloud", np.asarray(x, dtype=float))
    out.attrs.update(expt=expt, rt=rt, rep=rep, t_sec=int(t), lst_solar=float(run.lst(t)), zb=float(zb),
                     sun_azimuth_deg=float(np.degrees(az)), sun_zenith_deg=float(np.degrees(zen)),
                     sw_from_raytracer=int(sw_from_raytracer))
    f = out_path(expt, rt, rep, t)
    out.to_netcdf(f.with_name(f"composite_{int(t):07d}.nc"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    ap.add_argument("--every", type=float, default=None, help="instead: every N minutes of the 60 s fields within --solar")
    ap.add_argument("--solar", type=float, nargs=2, default=(12., 16.1))
    a = ap.parse_args()
    if a.every:
        a.t = frames_every(Run(run_dir(a.expt, a.rt, a.rep)), a.every, a.solar)
    a.t = a.t or list(snapshot_times(a.expt, a.rt, a.rep, skip_first=False))
    for t in a.t:
        o = analyse(a.expt, a.rt, a.rep, t)
        print(f"{a.rt} rep_{a.rep:02d} t={t} LST={o.attrs['lst_solar']:.2f} zenith={o.attrs['sun_zenith_deg']:.1f} "
              f"azimuth={o.attrs['sun_azimuth_deg']:.1f} events={o['n_events'].values} "
              f"L={np.round(o['L_mean'].values)} clouds={o.sizes['cloud']}", flush=True)
