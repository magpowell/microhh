"""Figures of cloud systems and their pulses (terms of Heus and Seifert 2013), cores above THR with a four-cell floor.

python systems_figures.py --expt no_aerosols_zero_wind_v3
"""
import argparse
import functools
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

import core_life as cl
import masks as mk
import style as st
import track as tr
from les_io import Run
from snapshot import out_path, run_dir
from style import plt

THR = 0.5
WINDOW = (12., 16.)
KINDS = ("passive", "single pulse", "multipulse")
KIND_COLORS = ("0.7", "C2", "C3")


@functools.lru_cache(maxsize=None)
def _member(expt, rt, rep):
    p, _, tc = cl.tables(expt, rt, rep, THR)
    with xr.open_dataset(out_path(expt, rt, rep, 0).parent / f"core_life{cl.tag_of(THR)}.nc") as ds:
        dx, dy = float(ds.attrs["dx"]), float(ds.attrs["dy"])
    return p, cl.systems(p, tc, dx, dy, keep_cut=True), tc


def members(expt):
    return {(lab, rep): _member(expt, rt, rep) for (rt, lab), rep in itertools.product(cl.RTS, range(1, 5))}


def figure_bars(expt):
    """Lifetimes of pulses, single-pulse clouds and multipulse systems, and pulses per multipulse system; member min-max."""
    M = members(expt)
    w = lambda d: d[(d.lst >= WINDOW[0]) & (d.lst < WINDOW[1])]
    stats = (("pulse lifetime [min]", lambda p, y: w(p).frames.mean()),
             ("single-pulse cloud lifetime [min]", lambda p, y: w(y[~y.cut & (y.kind == "single pulse")]).life.mean()),
             ("multipulse system lifetime [min]", lambda p, y: w(y[~y.cut & (y.kind == "multipulse")]).life.mean()),
             ("pulses per multipulse system [-]", lambda p, y: w(y[~y.cut & (y.kind == "multipulse")]).n_pulses.mean()))
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.), layout="constrained")
    rows = []
    for k, ((name, fn), ax) in enumerate(zip(stats, axs)):
        for i, lab in enumerate(("1D", "3D")):
            v = np.array([fn(*M[(lab, rep)][:2]) for rep in range(1, 5)])
            ax.bar(i, v.mean(), width=0.6, **st.RT[lab])
            ax.errorbar(i, v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]], color="k", capsize=3, lw=1)
            ax.annotate(f"{v.mean():.1f}", (i, v.max()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color="0.25")
            rows.append(dict(stat=name, rt=lab, mean=v.mean(), min=v.min(), max=v.max()))
        ax.set_xticks([0, 1], ["1D", "3D"])
        ax.set_ylabel(name)
        ax.margins(y=0.15)
        st.apply(ax)
        st.panel(ax, k)
    pd.DataFrame(rows).to_csv(st.outdir(expt).parent / "systems_bars.csv", index=False)
    return st.savefig(fig, expt, "fig_systems_bars")


def figure_series(expt, hours=np.arange(10., 17.01, 1.)):
    """Active systems present per area, and pulses per multipulse system by hour of its birth; member min-max."""
    M = members(expt)
    fig, axs = plt.subplots(1, 2, figsize=(8., 3.3), layout="constrained")
    h = []
    for rt, lab in cl.RTS:
        A, B, lstm = [], [], None
        for rep in range(1, 5):
            p, y, tc = M[(lab, rep)]
            run = Run(run_dir(expt, rt, rep))
            lst = run.lst(tr.load(run.dir, "qlqi_path")[1])
            act = y[y.kind != "passive"]
            n = np.zeros(lst.size + 1)
            np.add.at(n, act.f0.values.astype(int), 1); np.add.at(n, act.f1.values.astype(int) + 1, -1)
            n = np.cumsum(n)[:-1] / (run.xsize * run.ysize * 1.e-6)
            b = np.digitize(lst, hours) - 1
            A.append([n[b == i].mean() for i in range(hours.size - 1)])
            mp = y[~y.cut & (y.kind == "multipulse")]
            bb = np.digitize(mp.lst.values, hours) - 1
            B.append([mp.n_pulses.values[bb == i].mean() if (bb == i).sum() >= 5 else np.nan for i in range(hours.size - 1)])
        x = 0.5 * (hours[:-1] + hours[1:])
        for ax, v in zip(axs, (np.array(A), np.array(B))):
            ax.fill_between(x, np.nanmin(v, axis=0), np.nanmax(v, axis=0), alpha=0.25, lw=0, **st.RT[lab])
            line, = ax.plot(x, np.nanmean(v, axis=0), lw=1.8, marker="o", ms=4, label=lab, **st.RT[lab])
        h.append(line)
    axs[0].set_ylabel(r"active cloud systems present [km$^{-2}$]")
    axs[1].set_ylabel("pulses per multipulse system [-]")
    axs[1].set_yscale("log")
    for k, ax in enumerate(axs):
        ax.set_xlabel("local solar time [h]")
        st.apply(ax)
        st.panel(ax, k)
    axs[0].set_ylim(bottom=0.)
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig_systems_series")


BIG = 50      # pulses: a much re-fed system


@functools.lru_cache(maxsize=None)
def map_field(expt, rt, lab, rep, solar):
    """x, y [km] and, per column, the pulses the system of its cloud holds over its life (nan clear, 0 passive)."""
    p, y, tc = members(expt)[(lab, rep)]
    run = Run(run_dir(expt, rt, rep))
    path, time, x, yy = tr.load(run.dir, "qlqi_path")
    f = int(np.argmin(np.abs(run.lst(time) - solar)))
    lab2d, n = mk.label_periodic(path[f] > 0.)
    with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("features.nc")) as ds:
        ft = ds.to_dataframe()
    npul = pd.Series(ft[ft.frame == f].track.values).map(tc.set_index("track").family).map(y.n_pulses).values.astype(float)
    img = np.full(lab2d.shape, np.nan)
    img[lab2d > 0] = npul[lab2d[lab2d > 0] - 1]
    return x / 1000., yy / 1000., img


def big_share(img):
    return float((img >= BIG).sum() / np.isfinite(img).sum())


def figure_map(expt, rep=None, solar=14.5, vmax=200.):
    """Cloud field at one time, every cloud coloured by the pulses its system holds over its life; by default the
    member pair (same seed) with the largest difference in the cloud area share of systems of BIG pulses or more."""
    from matplotlib.colors import LogNorm
    sh = {r: [big_share(map_field(expt, rt, lab, r, solar)[2]) for rt, lab in cl.RTS] for r in range(1, 5)}
    for r, (a, b) in sh.items():
        print(f"rep {r}: cloud area share in systems of {BIG}+ pulses 1D {a:.2f}, 3D {b:.2f}, difference {b - a:+.2f}")
    rep = rep or max(sh, key=lambda r: sh[r][1] - sh[r][0])
    fig, axs = plt.subplots(1, 2, figsize=(9., 5.1), sharey=True, layout="constrained")
    for k, ((rt, lab), ax) in enumerate(zip(cl.RTS, axs)):
        x, yy, img = map_field(expt, rt, lab, rep, solar)
        ax.pcolormesh(x, yy, np.where(img == 0., 1., np.nan), cmap=ListedColormap(["0.75"]), rasterized=True)
        im = ax.pcolormesh(x, yy, np.where(img > 0., img, np.nan), cmap="viridis", norm=LogNorm(1., vmax), rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("x [km]")
        ax.tick_params(labelsize=8)
        st.panel(ax, k, lab)
    axs[0].set_ylabel("y [km]")
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.5, pad=0.02, extend="max")
    cb.set_label("pulses in the cloud system over its life [-]")
    print("plotted rep", rep)
    return st.savefig(fig, expt, "fig_systems_map")


def figure_pdfs(expt, hours=((12., 13.), (13., 14.), (14., 15.), (15., 16.)), nbins=9):
    """Distributions of multipulse system lifetime and of pulses per multipulse system by hour of the system's birth;
    density per decade on logarithmic bins, member min-max."""
    M = members(expt)
    rows = (("life", "multipulse system lifetime [min]", np.logspace(np.log10(4.), np.log10(400.), nbins + 1)),
            ("n_pulses", "pulses per multipulse system [-]", np.logspace(np.log10(2.), np.log10(300.), nbins + 1)))
    fig, axs = plt.subplots(2, len(hours), figsize=(3. * len(hours), 5.6), sharey="row", sharex="row", layout="constrained")
    for i, (v, name, edges) in enumerate(rows):
        x = np.sqrt(edges[:-1] * edges[1:])
        for j, (h0, h1) in enumerate(hours):
            ax, h = axs[i, j], []
            for lab in ("1D", "3D"):
                pm, med = [], []
                for rep in range(1, 5):
                    y = M[(lab, rep)][1]
                    d = y[~y.cut & (y.kind == "multipulse") & (y.lst >= h0) & (y.lst < h1)][v].values
                    pm.append(np.histogram(d, bins=edges)[0] / max(d.size, 1) / np.diff(np.log10(edges)))
                    med.append(np.median(d))
                pm = np.array(pm)
                ax.fill_between(x, pm.min(axis=0), pm.max(axis=0), alpha=0.25, lw=0, **st.RT[lab])
                h.append(ax.plot(x, pm.mean(axis=0), lw=1.8, label=lab, **st.RT[lab])[0])
                ax.axvline(np.mean(med), lw=1., ls="--", **st.RT[lab])
            ax.set_xscale("log")
            ax.set_xlabel(name)
            st.apply(ax)
            st.panel(ax, i * len(hours) + j, f"{h0:.0f}-{h1:.0f} LT" if i == 0 else "")
        axs[i, 0].set_ylabel("probability density per decade [-]")
        axs[i, 0].set_ylim(bottom=0.)
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig_systems_pdfs")


def figure_violins(expt):
    """Multipulse system lifetime and pulses per multipulse system, all members pooled: seaborn violins, axis cut at
    the 98th percentile."""
    import seaborn as sns
    M = members(expt)
    rows = (("life", "multipulse system lifetime [min]"), ("n_pulses", "pulses per multipulse system [-]"))
    d = []
    for lab in ("1D", "3D"):
        y = pd.concat([M[(lab, rep)][1] for rep in range(1, 5)])
        d.append(y[~y.cut & (y.kind == "multipulse") & (y.lst >= WINDOW[0]) & (y.lst < WINDOW[1])].assign(rt=lab))
    d = pd.concat(d)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.6), layout="constrained")
    for k, ((v, name), ax) in enumerate(zip(rows, axs)):
        sns.violinplot(data=d, x="rt", y=v, hue="rt", palette={l: st.RT[l]["color"] for l in ("1D", "3D")}, cut=0, legend=False, saturation=1., ax=ax)
        for body in ax.collections:
            body.set(alpha=0.5, edgecolor="none")
        ax.set(xlabel="", ylabel=name, ylim=(0., d[v].quantile(0.98)))
        st.apply(ax)
        st.panel(ax, k)
    return st.savefig(fig, expt, "fig_systems_violins")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    a = ap.parse_args()
    for fn in (figure_bars, figure_series, figure_map, figure_pdfs, figure_violins):
        print(fn(a.expt))
