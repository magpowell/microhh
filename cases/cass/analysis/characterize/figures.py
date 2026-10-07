"""Figures 1 (MSE profiles and parcels) and 4 (cloud-base w PDF with w_crit) for one experiment."""
import argparse
import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import style as st
from snapshot import out_path, snapshot_times
from style import plt

RTS = (("2stream", "1D"), ("raytracer", "3D"))
WB = np.arange(0., 8.01, 0.4)


def load(expt, kind, rt, t):
    root = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt / rt
    return [xr.open_dataset(f).load() for f in sorted(root.glob(f"rep_*/{kind}_{int(t):07d}.nc"))]


CMAP_ANOM = "RdBu_r"
DIRS = ("parallel", "perpendicular")


def ens(expt, rt, t):
    """Ensemble mean of the per-member composites; members weighted equally."""
    m = load(expt, "composite", rt, t)
    keep = [v for v in m[0].data_vars if "cloud" not in m[0][v].dims]
    return xr.concat([d[keep] for d in m], dim="member").mean("member"), m


@functools.lru_cache(maxsize=None)
def composite_files(expt, rt):
    """All composite files of a configuration: (t, lst, path) sorted by time."""
    root = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt / rt
    out = []
    for f in sorted(root.glob("rep_*/composite_*.nc")):
        with xr.open_dataset(f) as d:
            out.append((int(d.attrs["t_sec"]), float(d.attrs["lst_solar"]), f))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def ens_hour(expt, rt, h0, h1):
    """Event-weighted mean of all composites (members and frames) with solar time in [h0, h1)."""
    acc, n = None, None
    for t, lst, f in composite_files(expt, rt):
        if not (h0 <= lst < h1):
            continue
        with xr.open_dataset(f) as d:
            d = d[[v for v in d.data_vars if "cloud" not in d[v].dims]].load()
        w = d["n_events"]
        part = d.drop_vars("n_events") * w
        acc = part if acc is None else acc + part
        n = w if n is None else n + w
    if acc is None:
        raise FileNotFoundError(f"no composites of {rt} between solar {h0} and {h1}")
    out = acc / n
    out["n_events"] = n
    return out


HOURS = ((12., 13.), (13., 14.), (14., 15.), (15., 16.))


def figure7(expt, vlim=3.e-3, hourly=False):
    """Forces on the air in the sun-parallel slice, 3D minus 1D, per snapshot or pooled over each hour of the 60 s fields."""
    rows = (("b", "buoyancy"), ("beff", "effective\nbuoyancy"), ("a_pd", "dynamic\npressure force"), ("tot", "sum"))
    times = list(HOURS) if hourly else snapshot_times(expt)
    fig, axs = plt.subplots(len(rows), len(times), figsize=(10.5, 8.2), sharex=True, sharey=True,
                            layout="constrained")
    for j, t in enumerate(times):
        if hourly:
            d = (ens_hour(expt, "raytracer", *t) - ens_hour(expt, "2stream", *t)).sel(dir="parallel")
            label = f"{t[0]:.0f}-{t[1]:.0f} LT"
        else:
            d = (ens(expt, "raytracer", t)[0] - ens(expt, "2stream", t)[0]).sel(dir="parallel")
            label = st.lt(ens(expt, "2stream", t)[1][0].attrs["lst_solar"])
        f = dict(b=d["b"], beff=d["b"] + d["a_pb"], a_pd=d["a_pd"], tot=d["b"] + d["a_pb"] + d["a_pd"])
        for i, (key, lab) in enumerate(rows):
            ax = axs[i, j]
            im = ax.pcolormesh(d["xl"], d["znd"], f[key], cmap=CMAP_ANOM, vmin=-vlim, vmax=vlim, rasterized=True)
            for x in (-0.5, 0.5):
                ax.axvline(x, color="0.4", lw=0.6, ls="--")
            ax.tick_params(labelsize=8)
            st.panel(ax, i * len(times) + j, label if i == 0 else "")
            if j == 0:
                ax.set_ylabel(r"$z / z_b$ [-]")
                st.row_label(ax, lab)
            if i == len(rows) - 1:
                ax.set_xlabel(r"$r_\parallel / L$ [-]")
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label(r"vertical acceleration, 3D minus 1D [m s$^{-2}$]")
    return st.savefig(fig, expt, "in_progress/fig_forces" + ("" if hourly else "_snapshots"))


def figure6_strip(expt, vlim=4.e-3, hourly=False, direction="parallel"):
    """Buoyancy anomaly and circulation in the sun-parallel slice, 1D, 3D and their difference, per snapshot or pooled
    over each hour of the 60 s fields; bottom row: surface shortwave along the slice relative to the domain mean."""
    times = list(HOURS) if hourly else snapshot_times(expt)
    rows = ("1D", "3D", "3D - 1D")
    fig = plt.figure(figsize=(2.1 * len(times) + 0.6, 7.4), layout="constrained")
    gs = fig.add_gridspec(4, len(times) + 1, height_ratios=(1, 1, 1, 0.3), width_ratios=[1] * len(times) + [0.05])
    k = 0
    SCALE, KEY = {"1D": 24., "3D": 24., "3D - 1D": 8.}, {"1D": 2., "3D - 1D": 0.5}     # the difference row has its own arrow scale
    for j, t in enumerate(times):
        if hourly:
            c = {lab: ens_hour(expt, rt, *t).sel(dir=direction) for rt, lab in RTS}
            label = f"{t[0]:.0f}-{t[1]:.0f} LT"
        else:
            c = {lab: ens(expt, rt, t)[0].sel(dir=direction) for rt, lab in RTS}
            label = st.lt(ens(expt, "2stream", t)[1][0].attrs["lst_solar"])
        c["3D - 1D"] = c["3D"] - c["1D"]
        for i, lab in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            d = c[lab]
            im = ax.pcolormesh(d["xl"], d["znd"], d["b"], cmap=CMAP_ANOM, vmin=-vlim, vmax=vlim, rasterized=True)
            q = d.isel(xl=slice(4, None, 12), znd=slice(4, None, 8))
            qv = ax.quiver(q["xl"], q["znd"], q["us"], q["w"], scale=SCALE[lab], width=0.005, color="0.15")
            if j == len(times) - 1 and lab in KEY:
                ax.quiverkey(qv, 0.93, 1.07, KEY[lab], f"{KEY[lab]:g} m s$^{{-1}}$", labelpos="W", labelsep=0.05, coordinates="axes", fontproperties=dict(size=7))
            for x in (-0.5, 0.5):
                ax.axvline(x, color="0.4", lw=0.6, ls="--")
            ax.set(xlim=(-1, 1), ylim=(0, 1))
            ax.tick_params(labelbottom=False, labelleft=(j == 0), labelsize=8)
            if j == 0:
                ax.set_ylabel(r"$z / z_b$ [-]")
                st.row_label(ax, lab)
            st.panel(ax, i * len(times) + j)
            if i == 0:
                ax.annotate(label, xy=(0.5, 1.), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points", ha="center", va="bottom", fontsize=11)
        ax = fig.add_subplot(gs[3, j])
        h = []
        for rt, lab in RTS:
            if hourly:
                e = ens_hour(expt, rt, *t).sel(dir=direction)
                pct = (e["sw"] / e["sw_domain"] - 1.) * 100.
                l, = ax.plot(e["xl"], pct, lw=1.6, label=lab, **st.RT[lab])
            else:
                e, m = ens(expt, rt, t)
                pct = np.array([(d["sw"].sel(dir=direction) / float(d["sw_domain"]) - 1.) * 100. for d in m])
                ax.fill_between(e["xl"], pct.min(axis=0), pct.max(axis=0), alpha=0.25, lw=0, **st.RT[lab])
                l, = ax.plot(e["xl"], pct.mean(axis=0), lw=1.6, label=lab, **st.RT[lab])
            h.append(l)
        st.zero_line(ax)
        for x in (-0.5, 0.5):
            ax.axvline(x, color="0.4", lw=0.6, ls="--")
        ax.set(xlim=(-1, 1), ylim=(-90, 30), yticks=(-80, -40, 0), xlabel=(r"$r_\parallel / L$ [-]" if direction == "parallel" else r"$r_\perp / L$ [-]"))
        ax.tick_params(labelleft=(j == 0), labelsize=8)
        if j == 0:
            ax.set_ylabel("surface SW,\nfrom domain\nmean [%]")
        st.apply(ax)
        st.panel(ax, 3 * len(times) + j)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:3, len(times)]), extend="both")
    cb.set_label(r"buoyancy anomaly [m s$^{-2}$]")
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig_circulation" + ("" if hourly else "_snapshots") + ("" if direction == "parallel" else "_perp"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    a = ap.parse_args()
    for fn in (lambda e: figure6_strip(e, hourly=True), lambda e: figure7(e, hourly=True)):
        print(fn(a.expt))
