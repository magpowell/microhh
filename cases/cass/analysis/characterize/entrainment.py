"""Bulk entrainment and detrainment of the cloud core from the model's conditional statistics (Siebesma and Cuijpers 1995).

python entrainment.py --expt no_aerosols_zero_wind_v3        (entrainment_layers.csv, entrainment_hourly.csv)
With a the core area fraction, M = rho a w_c the core mass flux, phi_c the core mean and phi_e the mean of the rest of
the slab, over a layer z1..z2:
    eps   = -[phi_c(z2) - phi_c(z1)] / integral (phi_c - phi_e) dz
    delta = eps - ln(M(z2) / M(z1)) / (z2 - z1)
qt is the variable of record, thl the check; the decaying surface tracer gives a third estimate, eps plus its known sink
C_c / (TAU w_c) (the sink term is as large as eps itself, so this one rests on the core-mean speed). Samples are the 300 s statistics of the model's qlcore mask (cloudy and buoyant; no ascent condition, unlike masks.core),
averaged per solar hour. The cloud layer runs from the hourly cloud base (lowest level with a cloud fraction of A_MIN) to
the top of the core; rates are given from START above cloud base, about its lowest quarter (200 to 330 m here; one
height for both runs so the profiles start together), below which air joins the core by condensing and not by
mixing (organised inflow; Drueke et al. 2020; Dawe and Austin 2013), in layers with a mean core area of A_MIN. The terms the
two formulas neglect are returned as rates in the same units: sources (microphysics, radiation) and time tendencies.
These are bulk dilution and detrainment rates; directly measured exchange rates are about twice as large (Romps 2010).
"""
import argparse
import itertools

import netCDF4
import numpy as np
import pandas as pd

from les_io import Run
from lifetime import ensemble
from snapshot import out_path, run_dir

RTS = ("2stream", "raytracer")
HOURS = tuple((float(h), float(h + 1)) for h in range(11, 17))
DZ_LAYER = 250.          # m
START = 300.             # m above cloud base, about the lowest quarter of the cloud layer: organised inflow below (Drueke et al. 2020)
A_MIN = 1.e-3
SOURCES = {"qt": ("qtt_micro",), "thl": ("thlt_micro", "thlt_rad")}
TAU = 900.               # s, decay time of the surface tracer (couvreux)


def layer_rates(z, zh, a, M, phi_c, phi_e, z1, z2):
    """eps and delta [1/m] over z1..z2. a, phi_c, phi_e on full levels z; M on half levels zh; edges on half levels."""
    k = (z > z1) & (z < z2)
    dz = np.diff(zh)[k]
    f1, f2 = np.interp([z1, z2], z, phi_c)
    eps = -(f2 - f1) / np.sum((phi_c[k] - phi_e[k]) * dz)
    m1, m2 = np.interp([z1, z2], zh, M)
    return eps, eps - np.log(m2 / m1) / (z2 - z1)


def layer_residuals(z, zh, a, rho, w_c, phi_c, phi_e, S_c, dphi_dt, da_dt, z1, z2):
    """The neglected terms as rates [1/m]: source and core tendency in eps, area tendency in delta (layer means)."""
    k = (z > z1) & (z < z2)
    dz = np.diff(zh)[k]
    wk = np.interp(z[k], zh, w_c)
    den = np.sum((phi_c[k] - phi_e[k]) * dz)
    return (np.sum(S_c[k] / wk * dz) / den, -np.sum(dphi_dt[k] / wk * dz) / den,
            -np.sum(da_dt[k] / (a[k] * wk) * dz) / (z2 - z1))


def hour_profiles(rd, h0, h1):
    """Sample-mean core and environment profiles of one member and solar hour, and the tendencies across the hour."""
    run = Run(rd)
    with netCDF4.Dataset(rd / "cass.qlcore.0000000.nc") as c, netCDF4.Dataset(rd / "cass.default.0000000.nc") as d:
        lst = run.lst(c["time"][:])
        t = np.flatnonzero((lst >= h0) & (lst < h1))
        g = lambda f, grp, v: np.ma.filled(f[grp][v][t], 0.).astype(float)
        z, zh = c["z"][:].astype(float), c["zh"][:].astype(float)
        area, areah, w = g(c, "default", "area"), g(c, "default", "areah"), g(c, "default", "w")
        rho, rhoh = g(d, "thermo", "rho"), g(d, "thermo", "rhoh")
        a = area.mean(axis=0)
        cf = g(d, "thermo", "ql_frac").mean(axis=0)
        out = dict(z=z, zh=zh, a=a, cf=cf, rho=rho.mean(axis=0), M=(rhoh * areah * w).mean(axis=0), n=t.size,
                   w_c=(areah * w).mean(axis=0) / np.maximum(areah.mean(axis=0), 1.e-12))
        cm = lambda x: (area * x).mean(axis=0) / np.maximum(a, 1.e-12)                 # core mean, area weighted
        half = t.size // 2
        tend = lambda x: (x[half:].mean(axis=0) - x[:half].mean(axis=0)) / (0.5 * (c["time"][t[-1]] - c["time"][t[0]]) + 150.)
        out["da_dt"] = tend(area)
        for v in ("qt", "thl", "couvreux"):
            grp = "default" if v == "couvreux" else "thermo"
            pc, pm = g(c, grp, v), g(d, grp, v)
            out[v + "_c"] = cm(pc)
            out[v + "_e"] = (pm.mean(axis=0) - a * out[v + "_c"]) / (1. - a)
            out[v + "_S"] = -out[v + "_c"] / TAU if v == "couvreux" else sum(cm(g(c, "tend", s)) for s in SOURCES[v])
            first, last = [(area[s] * pc[s]).mean(axis=0) / np.maximum(area[s].mean(axis=0), 1.e-12) for s in (slice(None, half), slice(half, None))]
            out[v + "_dt"] = (last - first) / (0.5 * (c["time"][t[-1]] - c["time"][t[0]]) + 150.)
    return out


def member_layers(expt, rt, rep, dz=DZ_LAYER):
    rd = run_dir(expt, rt, rep)
    rows = []
    for h0, h1 in HOURS:
        p = hour_profiles(rd, h0, h1)
        ok = np.flatnonzero(p["a"] >= A_MIN)
        if ok.size == 0:
            continue
        zb, ztop = p["zh"][np.argmax(p["cf"] >= A_MIN)], p["zh"][ok[-1] + 1]      # cloud base: cloud fraction of A_MIN
        zcore = p["zh"][ok[0]]
        for z1 in np.arange(zb, ztop - dz + 1., dz):            # mass flux, area and speed from cloud base; rates from START
            z2 = z1 + dz
            k = (p["z"] > z1) & (p["z"] < z2)
            above = z1 >= zb + START
            rates = above and p["a"][k].mean() >= A_MIN and p["a"][k].min() > 0.
            if p["a"][k].mean() <= 0. or (above and not rates):
                continue
            row = dict(rt=rt, rep=rep, hour=int(h0), zb=zb, zcore=zcore, ztop=ztop, z=0.5 * (z1 + z2), height=0.5 * (z1 + z2) - zb, a=p["a"][k].mean(),
                       M=float(np.interp(0.5 * (z1 + z2), p["zh"], p["M"])), dlnM=1.e3 * np.log(np.interp(z2, p["zh"], p["M"]) / np.interp(z1, p["zh"], p["M"])) / dz)
            if not rates:
                row.update(w_c=float(np.interp(0.5 * (z1 + z2), p["zh"], p["w_c"])))
                rows.append(row)
                continue
            for v in ("qt", "thl"):
                eps, delta = layer_rates(p["z"], p["zh"], p["a"], p["M"], p[v + "_c"], p[v + "_e"], z1, z2)
                src, tnd, atnd = layer_residuals(p["z"], p["zh"], p["a"], p["rho"], p["w_c"], p[v + "_c"], p[v + "_e"], p[v + "_S"], p[v + "_dt"], p["da_dt"], z1, z2)
                row.update({f"eps_{v}": 1.e3 * eps, f"delta_{v}": 1.e3 * delta, f"src_{v}": 1.e3 * src, f"tend_{v}": 1.e3 * tnd})
            # decaying tracer: its sink in the core is known, so it is part of the estimate and not a residual
            eps, _ = layer_rates(p["z"], p["zh"], p["a"], p["M"], p["couvreux_c"], p["couvreux_e"], z1, z2)
            src, tnd, _ = layer_residuals(p["z"], p["zh"], p["a"], p["rho"], p["w_c"], p["couvreux_c"], p["couvreux_e"], p["couvreux_S"], p["couvreux_dt"], p["da_dt"], z1, z2)
            row.update(eps_tracer=1.e3 * (eps + src), decay_tracer=1.e3 * src, tend_tracer=1.e3 * tnd, w_c=float(np.interp(0.5 * (z1 + z2), p["zh"], p["w_c"])))
            row["atend"] = 1.e3 * atnd
            rows.append(row)
    return pd.DataFrame(rows)


def summary(expt):
    d = pd.concat([member_layers(expt, rt, rep) for rt, rep in itertools.product(RTS, range(1, 5))], ignore_index=True)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    d.to_csv(res / "entrainment_layers.csv", index=False)
    d["hbin"] = (np.floor(d.height / 250.) * 250. + 125.).astype(int)
    cols = ["eps_qt", "eps_thl", "delta_qt", "dlnM", "src_qt", "tend_qt", "src_thl", "tend_thl", "atend", "a", "M"]
    m = d.groupby(["rt", "rep", "hour", "hbin"])[cols].mean().reset_index()
    full = m.groupby(["hour", "hbin"]).filter(lambda g: len(g) == 8)          # layers present in all eight members
    e = ensemble(full, ["hour", "hbin"], cols)
    e.to_csv(res / "entrainment_hourly.csv")
    return d, e


YLIM = (START, 1300.)
SPLIT = {"area": ("a", dict(color="k", ls="-")), "speed": ("w_c", dict(color="0.5", ls="--"))}      # a ratio is neither run


def layers(expt, dz=100.):
    f = out_path(expt, "2stream", 1, 0).parents[2] / f"entrainment_layers_dz{dz:.0f}.csv"
    if not f.exists():
        pd.concat([member_layers(expt, rt, rep, dz) for rt, rep in itertools.product(RTS, range(1, 5))], ignore_index=True).to_csv(f, index=False)
    return pd.read_csv(f)


def _members(d, rt, hour, v, n=3):
    """Height by member table of v; layers where at least n members have a core."""
    g = d[(d.rt == rt) & (d.hour == hour)].pivot_table(index="height", columns="rep", values=v)
    return g[g.notna().sum(axis=1) >= n]


def paired_ratio(d, hour, v, n=3):
    """3D over 1D member by member (members share seeds); layers where at least n pairs exist."""
    r = _members(d, RTS[1], hour, v, 0) / _members(d, RTS[0], hour, v, 0)
    return r[r.notna().sum(axis=1) >= n]


def _both(ax, d, hour, v, st):
    for rt, lab in zip(RTS, ("1D", "3D")):
        g = _members(d, rt, hour, v)
        ax.fill_betweenx(g.index, g.min(axis=1), g.max(axis=1), alpha=0.25, lw=0, **st.RT[lab])
        ax.plot(g.mean(axis=1), g.index, lw=1.8, label=lab, **st.RT[lab])


def _frame(axs, hours, st):
    for k, ax in enumerate(axs.ravel()):
        st.apply(ax)
        st.panel(ax, k)
    for ax in axs[:, 0]:
        ax.set_ylabel("height above cloud base [m]")
    axs[0, 0].set_ylim(*YLIM)
    for ax, hour in zip(axs[0], hours):
        ax.annotate(f"{hour}-{hour + 1} LT", xy=(0.5, 1.), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points", ha="center", va="bottom", fontsize=11)


def figure_rates(expt, hours=(12, 13, 14, 15)):
    """Bulk entrainment and detrainment rates of the core against height above its base, one column per hour;
    member mean and min-max band where at least three members have a core."""
    import style as st
    from style import plt
    d = layers(expt)
    rows = (("eps_qt", "entrainment [km$^{-1}$]", (0., 0.65)), ("delta_qt", "detrainment [km$^{-1}$]", (0., 7.)))
    fig, axs = plt.subplots(2, len(hours), figsize=(10., 5.4), sharey=True, sharex="row", layout="constrained")
    for i, (v, name, xlim) in enumerate(rows):
        for j, hour in enumerate(hours):
            _both(axs[i, j], d, hour, v, st)
            axs[i, j].set(xlabel=name, xlim=xlim)
    _frame(axs, hours, st)
    fig.legend(handles=axs[0, 0].lines[:2], ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "in_progress/fig_entrainment")


def figure_massflux(expt, hours=(12, 13, 14, 15)):
    """Core mass flux of both runs, and the 3D over 1D ratio of its two factors, core area and core speed (paired by
    member; mean and min-max band over the pairs)."""
    import style as st
    from matplotlib.lines import Line2D
    from style import plt
    d = layers(expt)
    d = d[d.eps_qt.notna()]           # from START above cloud base, as the rates
    fig, axs = plt.subplots(2, len(hours), figsize=(10., 5.6), sharey=True, sharex="row", layout="constrained")
    for j, hour in enumerate(hours):
        _both(axs[0, j], d, hour, "M", st)
        axs[0, j].set(xlabel="core mass flux [kg m$^{-2}$ s$^{-1}$]", xlim=(0., 0.06))
        ax = axs[1, j]
        ax.axvline(1., color="0.75", lw=0.8, zorder=0)
        for lab, (v, kw) in SPLIT.items():
            r = paired_ratio(d, hour, v)
            ax.fill_betweenx(r.index, r.min(axis=1), r.max(axis=1), alpha=0.25, lw=0, color=kw["color"])
            ax.plot(r.mean(axis=1), r.index, lw=1.6, **kw)
        ax.set(xlabel="3D / 1D [-]", xlim=(0.75, 3.05), xticks=[1., 1.5, 2., 2.5, 3.])
    _frame(axs, hours, st)
    h = axs[0, 0].lines[:2] + [Line2D([], [], lw=1.6, label=lab, **kw) for lab, (_, kw) in SPLIT.items()]
    fig.legend(handles=h, ncols=4, loc="outside lower center")
    return st.savefig(fig, expt, "in_progress/fig_core_massflux")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    d, e = summary(a.expt)
    show = lambda cols, keys=("1D", "3D", "d_over_se"): e.loc[:, [(c, k) for c in cols for k in keys]].to_string(float_format=lambda v: f"{v:.2f}")
    print("--- bulk rates [1/km] by solar hour and height above cloud base [m]; member mean, 3D minus 1D over its standard error")
    print(show(["eps_qt", "eps_thl", "delta_qt", "dlnM"]))
    print("\n--- neglected terms [1/km], member mean: source and tendency in eps (qt, thl), area tendency in delta")
    print(show(["src_qt", "tend_qt", "src_thl", "tend_thl", "atend"], keys=("1D", "3D")))
    print(figure_rates(a.expt)); print(figure_massflux(a.expt))
