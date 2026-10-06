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
    for fn in (figure_map, figure_violins):
        print(fn(a.expt))
