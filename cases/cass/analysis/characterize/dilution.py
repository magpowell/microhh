"""Per-cloud dilution of conserved variables against cloud width, from the 3D snapshots (one row per cloud).

python dilution.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
Output: .../<rt>/rep_NN/dilution_<t>.nc
python dilution.py --summary [--expt ...]
Output: .../dilution_clouds.csv, dilution_binned.csv, dilution_fit.csv
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import masks as mk
from les_io import Run
from snapshot import load, out_path, run_dir

IZB = 2
NLEV = 4            # levels above a cloud's own base forming its base layer (100 m)
MIN_PTS = 3         # masked points a cloud needs at a level for that level to count
MIN_LEVELS = 8      # levels in the fit (200 m)
Z_F = 200.          # height above the base at which the mixing fraction is read
VARS = ("qt", "thl", "h")
RTS = ("2stream", "raytracer")
TIMES = (28800, 32400, 36000, 39600)
D_BINS = np.array([0., 400., 800., 1600., 1.e5])


def cloud_level_means(m3, lab, n, fields):
    """Per cloud and level: count and mean of each field over the masked points. Arrays (n, nz)."""
    nz = m3.shape[0]
    k, j, i = np.nonzero(m3)
    l = lab[j, i]
    ok = l > 0
    k, j, i, l = k[ok], j[ok], i[ok], l[ok]
    flat = l * nz + k
    cnt = np.bincount(flat, minlength=(n + 1) * nz).reshape(n + 1, nz)
    means = {}
    for name, f in fields.items():
        s = np.bincount(flat, weights=f[k, j, i], minlength=(n + 1) * nz).reshape(n + 1, nz)
        means[name] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)[1:]
    return cnt[1:], means


def fit_run(cnt, min_pts=MIN_PTS):
    """First and last level of the contiguous run of levels with enough points, from the lowest such level."""
    ok = cnt >= min_pts
    if not ok.any():
        return -1, -1
    k0 = int(np.argmax(ok))
    bad = np.flatnonzero(~ok[k0:])
    k1 = k0 + (int(bad[0]) - 1 if bad.size else ok.size - k0 - 1)
    return k0, k1


def fit_eps(z, phi_c, phi_env, k0, k1):
    """phi_c(k) = a - eps I(k) with I the integral of (phi_c - phi_env) dz from k0: least-squares eps and its r."""
    sl = slice(k0, k1 + 1)
    d = phi_c[sl] - phi_env[sl]
    I = np.concatenate([[0.], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(z[sl]))])
    p = phi_c[sl]
    vi = ((I - I.mean()) ** 2).sum()
    if vi == 0.:
        return np.nan, np.nan
    eps = -((I - I.mean()) * (p - p.mean())).sum() / vi
    r = np.corrcoef(I, p)[0, 1]
    return float(eps), float(r)


def mixing_fraction(z, phi_c, phi_env, phi_cb, cnt, k0, k1, dz=Z_F, min_pts=MIN_PTS):
    """Fraction of environment air at the first level at least dz above the base, by two-component mixing."""
    kf = k0 + int(np.searchsorted(z[k0:k1 + 1], z[k0] + dz))
    if kf > k1 or cnt[kf] < min_pts:
        return np.nan, np.nan
    den = phi_cb - phi_env[kf]
    return (np.nan if den == 0. else float((phi_cb - phi_c[kf]) / den)), float(z[kf])


def table(f, z, dx, dy, nlev=NLEV, min_pts=MIN_PTS, min_levels=MIN_LEVELS):
    qc, w = f["qc"], f["w"]
    cloudy = qc > 0.
    lab, n = mk.label_periodic(cloudy.any(axis=0))
    idx = np.arange(1, n + 1)
    k = np.arange(z.size)[:, None, None]
    top = ndimage.maximum(np.where(cloudy, k, -1).max(axis=0), lab, idx).astype(int)
    base = ndimage.minimum(np.where(cloudy, k, z.size).min(axis=0), lab, idx).astype(int)
    area = mk.object_areas(lab, n)
    out = dict(area=area * dx * dy, D=mk.equivalent_diameter(area, dx, dy), z_top=z[top], z_base=z[base], depth=z[top] - z[base])
    clear = mk.clear_columns(qc)
    env = {v: mk.column_mean(f[v], clear) for v in VARS}
    fields = {v: f[v] for v in VARS}
    for name, m3 in (("core", mk.core(qc, w, f["thv"])), ("cu", mk.cloudy_updraft(qc, w))):
        cnt, mean = cloud_level_means(m3, lab, n, fields)
        cols = {s: np.full(n, np.nan) for s in ("z_own", "z_fit0", "z_fit1", "z_f")}
        cols["n_levels"] = np.zeros(n, dtype=int)
        for v in VARS:
            for s in ("eps", "r", "f", "excess"):
                cols[f"{s}_{v}"] = np.full(n, np.nan)
        for c in range(n):
            k0, k1 = fit_run(cnt[c], min_pts)
            if k0 < 0:
                continue
            kb1 = min(k0 + nlev, k1)
            wts = cnt[c, k0:kb1 + 1]
            cols["z_own"][c] = z[k0]
            cols["n_levels"][c] = k1 - k0 + 1
            if k1 - k0 + 1 < min_levels:
                continue
            cols["z_fit0"][c], cols["z_fit1"][c] = z[k0], z[k1]
            for v in VARS:
                phi_cb = float((mean[v][c, k0:kb1 + 1] * wts).sum() / wts.sum())
                cols[f"excess_{v}"][c] = phi_cb - float((env[v][k0:kb1 + 1] * wts).sum() / wts.sum())
                cols[f"eps_{v}"][c], cols[f"r_{v}"][c] = fit_eps(z, mean[v][c], env[v], k0, k1)
                cols[f"f_{v}"][c], cols["z_f"][c] = mixing_fraction(z, mean[v][c], env[v], phi_cb, cnt[c], k0, k1)
        for s, a in cols.items():
            out[f"{name}_{s}"] = a
    return out


def analyse(expt, rt, rep, t):
    run = Run(run_dir(expt, rt, rep))
    src = out_path(expt, rt, rep, t)
    with xr.open_dataset(src) as s:
        attrs = {a: s.attrs[a] for a in ("expt", "rt", "rep", "t_sec", "lst_solar")}
    f, _ = load(run, t)
    tab = table(f, run.z, run.dx, run.dy)
    ds = xr.Dataset({v: ("cloud", a) for v, a in tab.items()})
    ds.attrs.update(attrs, nlev=NLEV, min_pts=MIN_PTS, min_levels=MIN_LEVELS, z_f=Z_F)
    ds.to_netcdf(src.with_name(f"dilution_{int(t):07d}.nc"))
    return ds


def load_clouds(expt, mask="core"):
    rows = []
    for t, rt, rep in itertools.product(TIMES, RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"dilution_{t:07d}.nc")) as ds:
            d = ds.to_dataframe()
            lst = float(ds.attrs["lst_solar"])
        keep = ["area", "D", "z_top", "z_base", "depth"] + [c for c in d.columns if c.startswith(mask + "_")]
        d = d[keep].rename(columns={c: c[len(mask) + 1:] for c in keep if c.startswith(mask + "_")})
        d["t"], d["lst"], d["rt"], d["rep"] = t, lst, rt, rep
        rows.append(d[np.isfinite(d.eps_qt)])
    d = pd.concat(rows, ignore_index=True)
    d["db"] = np.digitize(d.D, D_BINS) - 1
    return d


def binned(d, var="qt"):
    """Median eps per width class and member, then mean and range over the members."""
    e = f"eps_{var}"
    g = d.groupby(["t", "rt", "rep", "db"]).agg(n=(e, "size"), eps=(e, "median"), D=("D", "mean"), depth=("depth", "median")).reset_index()
    m = g.groupby(["t", "db", "rt"])[["n", "eps", "D", "depth"]]
    mean, lo, hi = m.mean().unstack("rt"), m.min().unstack("rt"), m.max().unstack("rt")
    out = pd.DataFrame({"n_1D": mean["n"][RTS[0]], "n_3D": mean["n"][RTS[1]], "D_1D": mean["D"][RTS[0]], "D_3D": mean["D"][RTS[1]],
                        "eps_1D": mean["eps"][RTS[0]], "eps_1D_lo": lo["eps"][RTS[0]], "eps_1D_hi": hi["eps"][RTS[0]],
                        "eps_3D": mean["eps"][RTS[1]], "eps_3D_lo": lo["eps"][RTS[1]], "eps_3D_hi": hi["eps"][RTS[1]],
                        "depth_1D": mean["depth"][RTS[0]], "depth_3D": mean["depth"][RTS[1]]})
    return g, out


def loglog_fit(d, var="qt"):
    """log eps = a + b log D + c [3D]: slope b and 3D offset c (as a ratio)."""
    e = d[f"eps_{var}"].values
    ok = e > 0.
    X = np.c_[np.ones(ok.sum()), np.log(d.D.values[ok]), (d.rt.values[ok] == RTS[1]).astype(float)]
    a, b, c = np.linalg.lstsq(X, np.log(e[ok]), rcond=None)[0]
    return dict(slope=float(b), ratio_3D=float(np.exp(c)), n=int(ok.sum()))


def bootstrap_fit(d, var="qt", n=1000, seed=0):
    """Clouds resampled within each member."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby(["rt", "rep"])]
    est = []
    for _ in range(n):
        s = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        est.append(loglog_fit(s, var))
    full = loglog_fit(d, var)
    for k in ("slope", "ratio_3D"):
        v = np.array([x[k] for x in est])
        full[f"{k}_lo"], full[f"{k}_hi"] = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    return full


def depth_given_eps(d, var="qt"):
    """Rank correlation of depth with eps and with D, and of depth with eps within width classes."""
    from scipy import stats
    e = f"eps_{var}"
    out = dict(r_depth_D=float(stats.spearmanr(d.depth, d.D)[0]), r_depth_eps=float(stats.spearmanr(d.depth, d[e])[0]),
               r_eps_D=float(stats.spearmanr(d[e], d.D)[0]))
    within = [stats.spearmanr(g.depth, g[e])[0] for _, g in d.groupby("db") if len(g) >= 20]
    out["r_depth_eps_within_D"] = float(np.mean(within)) if within else np.nan
    return out


def summary(expt, mask="core"):
    d = load_clouds(expt, mask)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    tag = "" if mask == "core" else f"_{mask}"
    d.to_csv(res / f"dilution_clouds{tag}.csv", index=False)
    rows, fits = [], []
    for var in ("qt", "thl"):
        g, b = binned(d, var)
        b["var"] = var
        rows.append(b.reset_index())
        for t, dt in d.groupby("t"):
            f = dict(var=var, t=t, lst=dt.lst.iloc[0], **bootstrap_fit(dt, var))
            for rt, lab in zip(RTS, ("1D", "3D")):
                f.update({f"{k}_{lab}": v for k, v in depth_given_eps(dt[dt.rt == rt], var).items()})
                f[f"eps_median_{lab}"] = float(dt[dt.rt == rt][f"eps_{var}"].median())
            fits.append(f)
    b, f = pd.concat(rows, ignore_index=True), pd.DataFrame(fits)
    b.to_csv(res / f"dilution_binned{tag}.csv", index=False)
    f.to_csv(res / f"dilution_fit{tag}.csv", index=False)
    return d, b, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--t", type=int, nargs="+", default=list(TIMES))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--mask", default="core", choices=["core", "cu"])
    a = ap.parse_args()
    if a.summary:
        d, b, f = summary(a.expt, a.mask)
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
        fmt = lambda v: f"{v:.3g}"
        print(f"--- eps per width class (edges {D_BINS[:-1]} m), member medians: mean and range [1/m]")
        print(b.to_string(index=False, float_format=fmt))
        print("\n--- log eps on log D with a 3D offset; rank correlations")
        print(f.to_string(index=False, float_format=fmt))
    else:
        for t in a.t:
            ds = analyse(a.expt, a.rt, a.rep, t)
            ok = np.isfinite(ds["core_eps_qt"].values)
            print(f"{a.rt} rep_{a.rep:02d} t={t} clouds={ds.sizes['cloud']} fitted (core)={int(ok.sum())} "
                  f"median eps_qt={np.nanmedian(ds['core_eps_qt'].values):.2e} eps_thl={np.nanmedian(ds['core_eps_thl'].values):.2e} 1/m", flush=True)
