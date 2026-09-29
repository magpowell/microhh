"""Check the model input and the tendencies the model applies against the CASS tables.

python check_forcing.py --run <run dir with cass_input.nc, cass.ini and, optionally, statistics>
Output: $SCRATCH/CASS_LES/analysis/forcing_check/<name>/ (report.md and tables)
"""
import argparse
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

RD, CP, LV, GRAV, P00 = 287.04, 1005., 2.5e6, 9.81, 1.e5
DAY0 = 205.5
DATA = Path(__file__).resolve().parent / "data"


def blocks(path, ncol):
    """Time blocks of a CASS table: list of (day, array[level, column])."""
    out, cur, day = [], None, None
    for line in open(path):
        s = line.replace(",", " ").split()
        if len(s) >= 3 and "day" in line and s[0][0].isdigit():
            if cur is not None:
                out.append((day, np.array(cur)))
            day, cur = float(s[0]), []
        elif cur is not None and len(s) >= ncol:
            try:
                cur.append([float(x) for x in s[:ncol]])
            except ValueError:
                pass
    out.append((day, np.array(cur)))
    return out


def exner(p_pa):
    return (p_pa / P00)**(RD / CP)


def qsat(p, T):
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return 0.622 * es / (p - 0.378 * es)


def sounding_pressure(z, theta, q, p0):
    """Hydrostatic pressure on the sounding levels, integrated upward from the surface pressure p0."""
    p = np.empty(z.size)
    pk, zk = p0, 0.
    for k in range(z.size):
        for _ in range(3):
            pm = 0.5 * (pk + p[k]) if _ else pk
            Tv = theta[k] * exner(pm) * (1. + 0.61 * q[k])
            p[k] = pk * np.exp(-GRAV * (z[k] - zk) / (RD * Tv))
        pk, zk = p[k], z[k]
    return p


def compare_input(f, lsf, snd):
    z = f["z"][:]
    t_in = f["timedep"]["time_ls"][:]
    rows = []
    for name, col in (("thl_ls", 2), ("qt_ls", 3), ("w_ls", 6)):
        a = f["timedep"][name][:]
        d, ref = [], []
        for i, (day, b) in enumerate(lsf):
            m = (b[:, 0] >= z[0]) & (b[:, 0] <= z[-1])
            d.append(np.interp(b[m, 0], z, a[i]) - b[m, col])
            ref.append(b[m, col])
        d, ref = np.concatenate(d), np.concatenate(ref)
        rows.append(dict(field=name, table_column=("tls", "qls", "wls")[(2, 3, 6).index(col)], n=d.size,
                         max_abs_table=np.abs(ref).max(), max_abs_diff=np.abs(d).max(),
                         max_rel_diff=np.abs(d).max() / np.abs(ref).max()))
    s = snd[0][1]
    m = (s[:, 0] >= z[0]) & (s[:, 0] <= z[-1])
    for name, col, fac in (("thl", 2, 1.), ("qt", 3, 1.e-3)):
        d = np.interp(s[m, 0], z, f["init"][name][:]) - s[m, col] * fac
        rows.append(dict(field=name, table_column=("tp", "q")[(2, 3).index(col)], n=d.size, max_abs_table=np.abs(s[m, col] * fac).max(),
                         max_abs_diff=np.abs(d).max(), max_rel_diff=np.abs(d).max() / np.abs(s[m, col] * fac).max()))
    t_tab = (np.array([d for d, _ in lsf]) - DAY0) * 86400.
    times = pd.DataFrame(dict(block=np.arange(t_tab.size), table_time_s=t_tab, input_time_s=t_in,
                              nominal_s=3600. * np.arange(t_tab.size), table_minus_nominal_s=t_tab - 3600. * np.arange(t_tab.size)))
    return pd.DataFrame(rows), times


def exner_effect(lsf):
    rows = []
    for day, b in lsf:
        pi = exner(b[:, 1] * 100.)
        for k in range(b.shape[0]):
            rows.append(dict(hour_utc=(day - 205.) * 24., z=b[k, 0], p_hpa=b[k, 1], exner=pi[k], tls_K_day=b[k, 2] * 86400.,
                             theta_tendency_K_day=b[k, 2] / pi[k] * 86400.))
    d = pd.DataFrame(rows)
    d["missing_K_day"] = d.theta_tendency_K_day - d.tls_K_day
    return d


def subsidence_at_surface(f, lsf):
    z = f["z"][:]
    w = f["timedep"]["w_ls"][:]
    return pd.DataFrame(dict(hour_utc=[(d - 205.) * 24. for d, _ in lsf], lowest_table_level_m=[b[0, 0] for _, b in lsf],
                             w_table_lowest=[b[0, 6] for _, b in lsf], w_input_first_level=w[:, 0],
                             w_tapered_first_level=[b[0, 6] * z[0] / b[0, 0] for _, b in lsf]))


def sounding_saturation(snd, p0_hpa):
    s = snd[0][1]
    z, th, q = s[:, 0], s[:, 2], s[:, 3] * 1.e-3
    p = sounding_pressure(z, th, q, p0_hpa * 100.)
    T = th * exner(p)
    rh = q / qsat(p, T)
    return pd.DataFrame(dict(z=z, p_hpa=p / 100., theta=th, T=T, q_g_kg=q * 1.e3, rh=rh))


def applied(run, lsf):
    """Tendencies the model reports (statistics group tend) against the input and the CASS tables."""
    f = sorted(run.glob("cass.default.*.nc"))
    if not f:
        return None, None, None
    with nc.Dataset(f[0]) as S, nc.Dataset(run / "cass_input.nc") as I:
        if "tend" not in S.groups:
            return None, None, None
        t, z = S["time"][:], S["z"][:]
        T, TH = S["tend"], S["thermo"]
        tin = I["timedep"]["time_ls"][:]
        at = lambda name: np.array([[np.interp(tt, tin, I["timedep"][name][:, k]) for k in range(z.size)] for tt in t])
        nf = I["init"]["nudgefac"][:]
        rows, prof = [], {}
        for v, tab in (("thl", "tls"), ("qt", "qls")):
            mean = np.ma.filled(TH[v][:], np.nan)
            exp = dict(ls=at(f"{v}_ls"), subs=-at("w_ls") * np.gradient(mean, z, axis=1))
            if f"{v}_nudge" in I["timedep"].variables:
                exp["nudge"] = -nf[None, :] * (mean - at(f"{v}_nudge"))
            for proc in ("ls", "subs", "nudge", "damp", "rad", "micro"):
                name = f"{v}t_{proc}"
                if name not in T.variables:
                    continue
                m = np.ma.filled(T[name][:], np.nan)
                prof[name] = m
                r = dict(tendency=name, max_abs_model=np.nanmax(np.abs(m[1:])))
                if proc in exp:
                    e = exp[proc]
                    k = slice(2, -2)
                    r.update(max_abs_expected=np.nanmax(np.abs(e[1:, k])), max_abs_diff=np.nanmax(np.abs(m[1:, k] - e[1:, k])),
                             rms_diff=float(np.sqrt(np.nanmean((m[1:, k] - e[1:, k])**2))),
                             rms_expected=float(np.sqrt(np.nanmean(e[1:, k]**2))))
                rows.append(r)
        sfc = None
        if "land_surface" in S.groups and "H" in S["land_surface"].variables:
            tab = np.loadtxt(DATA / "cass_sfc.txt", skiprows=1)
            ts = (tab[:, 0] - DAY0) * 86400.
            H, LE = (np.ma.filled(S["land_surface"][k][:], np.nan) for k in ("H", "LE"))
            sfc = pd.DataFrame(dict(hour_utc=12. + ts / 3600., H_cass=tab[:, 2], LE_cass=tab[:, 3],
                                    H_model=np.interp(ts, t, H, right=np.nan), LE_model=np.interp(ts, t, LE, right=np.nan)))
        names = sorted(T.variables)
    return pd.DataFrame(rows), sfc, names


def main(run, out):
    out.mkdir(parents=True, exist_ok=True)
    lsf, snd = blocks(DATA / "cass_lsf.txt", 7), blocks(DATA / "cass_snd.txt", 6)
    p0 = float(open(DATA / "cass_snd.txt").readlines()[1].replace(",", " ").split()[2])
    with nc.Dataset(run / "cass_input.nc") as f:
        cmp_, times = compare_input(f, lsf, snd)
        wsfc = subsidence_at_surface(f, lsf)
        nudge = pd.DataFrame(dict(z=f["z"][:], nudgefac=f["init"]["nudgefac"][:]))
    ex = exner_effect(lsf)
    sat = sounding_saturation(snd, p0)
    same = bool(np.array_equal(snd[0][1], snd[1][1]))
    app, sfc, names = applied(run, lsf)
    if app is not None:
        app.to_csv(out / "applied_tendencies.csv", index=False)
    if sfc is not None:
        sfc.to_csv(out / "surface_fluxes.csv", index=False)
    for name, d in (("input_vs_tables", cmp_), ("time_axis", times), ("exner", ex), ("w_ls_surface", wsfc),
                    ("sounding", sat), ("nudgefac", nudge)):
        d.to_csv(out / f"{name}.csv", index=False)
    return dict(cmp=cmp_, times=times, exner=ex, wsfc=wsfc, sat=sat, nudge=nudge, sounding_blocks_identical=same,
                applied=app, sfc=sfc, tend_names=names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "forcing_check" / a.run.parents[1].name
    r = main(a.run, out)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    f = lambda v: f"{v:.4g}"
    print("--- input against tables, on the table levels inside the domain"); print(r["cmp"].to_string(index=False, float_format=f))
    print("\n--- time axis"); print(r["times"].to_string(index=False, float_format=f))
    print("\n--- sounding blocks identical:", r["sounding_blocks_identical"])
    print("--- sounding: largest relative humidity", f"{r['sat'].rh.max():.3f}", "at", f"{r['sat'].z[r['sat'].rh.idxmax()]:.0f} m")
    e = r["exner"]
    print("\n--- temperature tendency against theta tendency [K/day], by pressure level (range over all times)")
    g = e.groupby("p_hpa").agg(z=("z", "mean"), exner=("exner", "first"), tls_min=("tls_K_day", "min"), tls_max=("tls_K_day", "max"),
                               missing_min=("missing_K_day", "min"), missing_max=("missing_K_day", "max")).sort_index(ascending=False)
    print(g[g.z < 8200].to_string(float_format=f))
    print("\n--- subsidence at the lowest model level [m/s]"); print(r["wsfc"].to_string(index=False, float_format=f))
    if r["applied"] is not None:
        print("\n--- tendencies reported by the model against the input (K/s, kg/kg/s)"); print(r["applied"].to_string(index=False, float_format=f))
        print("    all tendency profiles in the statistics:", " ".join(r["tend_names"]))
    if r["sfc"] is not None:
        print("\n--- surface fluxes [W/m2]"); print(r["sfc"].to_string(index=False, float_format=f))
