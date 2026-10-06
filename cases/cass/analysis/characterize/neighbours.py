"""Small clouds and surface fields around large clouds, in coordinates aligned with the shadow displacement.

+x points away from the sun (the direction the shadow is displaced), y is perpendicular; distances are in units of
the large cloud's equivalent diameter D. The same orientation is used in 1D.

python neighbours.py --expt no_aerosols_zero_wind_v2            (per-snapshot files, then the summary)
Output: .../<rt>/rep_NN/neighbours_<t>.nc, neighbours_ratio.csv, neighbours_sides.csv
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import thermo as th
from composite import solar_azimuth
from les_io import Run, lowest
from snapshot import out_path, run_dir, snapshot_times

RTS = ("2stream", "raytracer")
LARGE, SMALL = 1000., 500.          # m, equivalent diameter
R_IN, R_OUT = 0.5, 3.               # in units of D
G = np.linspace(-3., 3., 61)        # composite grid in units of D
HB = np.linspace(-3., 3., 13)       # histogram bin edges for centroids
NLOW = 4                            # levels in the lowest 100 m
POPS = ("small", "small_growing")


def to_sun_frame(dx, dy, az):
    """Components along the shadow displacement (+x) and perpendicular to it (+y)."""
    ex, ey = (-np.sin(az), -np.cos(az)), (np.cos(az), -np.sin(az))
    return dx * ex[0] + dy * ex[1], dx * ey[0] + dy * ey[1]


def relative(x0, y0, x, y, lx, ly):
    dx = (x - x0 + 0.5 * lx) % lx - 0.5 * lx
    dy = (y - y0 + 0.5 * ly) % ly - 0.5 * ly
    return dx, dy


def side_counts(xs, ys):
    """Counts in the half annulus R_IN < r <= R_OUT on the sunlit (x < 0) and shadow (x > 0) sides."""
    r = np.hypot(xs, ys)
    ring = (r > R_IN) & (r <= R_OUT)
    return int((ring & (xs < 0.)).sum()), int((ring & (xs > 0.)).sum())


def fields(run, t):
    """Surface and low-level fields on the cell centres."""
    it = int(round(t / 60.))
    bs = run.basestate_at(t)
    S = {}
    with xr.open_dataset(run.dir / "thl_fluxbot.xy.nc", decode_times=False) as d:
        S["H"] = np.squeeze(d["thl_fluxbot"].isel(time=it).values).astype(float) * bs["rhoref"][0] * th.cp
    rtsw = [run.dir / f"sw_flux_sfc_{k}_rt.xy.nc" for k in ("dir", "dif")]
    S["sw"] = 0.
    for f in (rtsw if all(x.exists() for x in rtsw) else [run.dir / "sw_flux_dn.xy.nc"]):
        with xr.open_dataset(f, decode_times=False) as d:
            S["sw"] = S["sw"] + lowest(d[list(d.data_vars)[0]].isel(time=it))
    if run.has_hf(t):       # 60 s fields: buoyancy from thl and qt (no cloud water at these levels), winds from the files
        thl, qt = (np.array(run.field_hf(n, t)[:2], dtype=float) for n in ("thl", "qt"))
        thv = th.theta_v(thl, qt, 0., 0., bs["exnref"][:2, None, None])
        b = th.grav * (thv - thv.mean(axis=(1, 2), keepdims=True)) / bs["thvref"][:2, None, None]
        u, v = (np.array(run.field_hf(n, t)[:NLOW], dtype=float) for n in ("u", "v"))
    else:
        b = np.array(run.field("b", t)[:2], dtype=float)
        u, v = (np.array(run.field(n, t)[:NLOW], dtype=float) for n in ("u", "v"))
    S["b"] = (b - b.mean(axis=(1, 2), keepdims=True)).mean(axis=0)
    S["conv"] = -((np.roll(u, -1, axis=2) - u) / run.dx + (np.roll(v, -1, axis=1) - v) / run.dy).mean(axis=0)
    return S


def analyse(expt, rt, rep, t):
    run = Run(run_dir(expt, rt, rep))
    az, zen = solar_azimuth(run, t)
    src = out_path(expt, rt, rep, t)
    with xr.open_dataset(src.with_name(f"objects_{t:07d}.nc")) as ds:
        o = ds[[v for v in ds.data_vars if v.startswith("obj_")]].to_dataframe()
        zb = float(ds.attrs["zb"])
    lx, ly = run.itot * run.dx, run.jtot * run.dy
    small = (o.obj_D < SMALL).values
    grow = small & (o.obj_core_cols > 0).values & (o.obj_w_top_cloud > 0.).values
    pop = dict(small=small, small_growing=grow)
    big = np.flatnonzero((o.obj_D >= LARGE).values)
    S = fields(run, t)
    X, Y = np.meshgrid(G, G)
    ex, ey = (-np.sin(az), -np.cos(az)), (np.cos(az), -np.sin(az))
    comp = {k: np.zeros(X.shape) for k in S}
    hist = {k: np.zeros((HB.size - 1, HB.size - 1)) for k in POPS}
    rows = []
    for c in big:
        D, x0, y0 = o.obj_D.values[c], o.obj_x.values[c], o.obj_y.values[c]
        px = (x0 + D * (X * ex[0] + Y * ey[0])) / run.dx - 0.5
        py = (y0 + D * (X * ex[1] + Y * ey[1])) / run.dy - 0.5
        for k in S:
            comp[k] += ndimage.map_coordinates(S[k], np.array([py, px]), order=1, mode="grid-wrap")
        dx, dy = relative(x0, y0, o.obj_x.values, o.obj_y.values, lx, ly)
        xs, ys = to_sun_frame(dx, dy, az)
        xs, ys = xs / D, ys / D
        row = dict(cloud=int(c), D=D, depth=o.obj_cloud_depth.values[c], area_ring=0.5 * np.pi * (R_OUT**2 - R_IN**2) * D**2)
        for k, m in pop.items():
            row[f"{k}_sunlit"], row[f"{k}_shadow"] = side_counts(xs[m], ys[m])
            hist[k] += np.histogram2d(ys[m], xs[m], bins=[HB, HB])[0]
        rows.append(row)
    n = max(big.size, 1)
    out = xr.Dataset(coords=dict(yg=G, xg=G, yb=0.5 * (HB[1:] + HB[:-1]), xb=0.5 * (HB[1:] + HB[:-1])))
    for k in S:
        out[k] = (("yg", "xg"), comp[k] / n)
        out[f"{k}_domain"] = float(S[k].mean())
    for k in POPS:
        out[f"hist_{k}"] = (("yb", "xb"), hist[k])
        out[f"density_{k}"] = float(pop[k].sum() / (lx * ly))
    tab = pd.DataFrame(rows)
    for v in tab.columns:
        out[f"big_{v}"] = ("big", tab[v].values)
    out.attrs.update(expt=expt, rt=rt, rep=rep, t_sec=int(t), lst_solar=float(run.lst(t)), zb=zb, n_large=int(big.size),
                     sun_azimuth_deg=float(np.degrees(az)), sun_zenith_deg=float(np.degrees(zen)),
                     shadow_offset_m=float(zb * np.tan(zen)))
    out.to_netcdf(src.with_name(f"neighbours_{t:07d}.nc"))
    return out


def ratio(tabs, k, rng=None, n=2000):
    """Sunlit over shadow counts pooled over large clouds; interval from resampling large clouds within members."""
    tot = lambda ts: sum(t[f"{k}_sunlit"].sum() for t in ts) / max(sum(t[f"{k}_shadow"].sum() for t in ts), 1)
    out = dict(ratio=tot(tabs), sunlit=sum(t[f"{k}_sunlit"].sum() for t in tabs), shadow=sum(t[f"{k}_shadow"].sum() for t in tabs))
    if rng is not None:
        est = [tot([t.iloc[rng.integers(0, len(t), len(t))] for t in tabs if len(t)]) for _ in range(n)]
        out["lo"], out["hi"] = np.percentile(est, [2.5, 97.5])
    return out


def summary(expt, seed=0):
    rng = np.random.default_rng(seed)
    rows, sides = [], []
    for t, rt in itertools.product(snapshot_times(expt, skip_first=True), RTS):
        tabs, dens = [], {k: [] for k in POPS}
        for rep in range(1, 5):
            with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"neighbours_{t:07d}.nc")) as ds:
                tabs.append(ds[[v for v in ds.data_vars if v.startswith("big_")]].to_dataframe())
                for k in POPS:
                    dens[k].append(float(ds[f"density_{k}"]))
                    exp = float(ds[f"density_{k}"]) * tabs[-1]["big_area_ring"].sum()
                    sides.append(dict(t=t, rt=rt, rep=rep, pop=k, n_large=len(tabs[-1]),
                                      sunlit_rel=tabs[-1][f"big_{k}_sunlit"].sum() / exp if exp else np.nan,
                                      shadow_rel=tabs[-1][f"big_{k}_shadow"].sum() / exp if exp else np.nan))
                lst, off = float(ds.attrs["lst_solar"]), float(ds.attrs["shadow_offset_m"])
        tabs = [x.rename(columns=lambda c: c[4:]) for x in tabs]
        for k in POPS:
            rows.append(dict(t=t, lst=lst, rt=rt, pop=k, n_large=sum(len(x) for x in tabs),
                             D_large=np.mean(np.concatenate([x.D.values for x in tabs])), shadow_offset_m=off,
                             **ratio(tabs, k, rng)))
    r, s = pd.DataFrame(rows), pd.DataFrame(sides)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    r.to_csv(res / "neighbours_ratio.csv", index=False)
    s.to_csv(res / "neighbours_sides.csv", index=False)
    return r, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        for t, rt, rep in itertools.product(snapshot_times(a.expt, skip_first=True), RTS, range(1, 5)):
            o = analyse(a.expt, rt, rep, t)
            print(f"{rt} rep_{rep:02d} t={t} large clouds={o.attrs['n_large']} shadow offset={o.attrs['shadow_offset_m']:.0f} m", flush=True)
    r, s = summary(a.expt)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(s.groupby(["pop", "t", "rt"])[["n_large", "sunlit_rel", "shadow_rel"]].agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.3g}"))
