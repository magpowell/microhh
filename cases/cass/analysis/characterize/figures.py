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


def row_label(ax, text):
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(-52, 0), textcoords="offset points",
                ha="center", va="center", rotation=90, fontweight="bold", fontsize=10)


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
                row_label(ax, lab)
            if i == len(rows) - 1:
                ax.set_xlabel(r"$r_\parallel / L$ [-]")
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label(r"vertical acceleration, 3D minus 1D [m s$^{-2}$]")
    return st.savefig(fig, expt, "fig7_forces_3D_minus_1D" + ("_hourly" if hourly else ""))


def figure17(expt):
    """Cloud births per area and hour, and their excess near existing clouds over random placement, by solar hour."""
    import births as bi
    d, area = bi.load_births(expt)
    r, l = bi.rates(d, area), bi.location(d)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
    h = []
    for lab in ("1D", "3D"):
        x = r.index.values + 0.5
        axs[0].fill_between(x, r[f"clouds_{lab}_lo"], r[f"clouds_{lab}_hi"], alpha=0.3, lw=0, **st.RT[lab])
        line, = axs[0].plot(x, r[f"clouds_{lab}"], lw=1.8, marker="o", ms=4, label=lab, **st.RT[lab])
        h.append(line)
        axs[1].fill_between(x, l[f"excess_500_{lab}_lo"], l[f"excess_500_{lab}_hi"], alpha=0.3, lw=0, **st.RT[lab])
        axs[1].plot(x, l[f"excess_500_{lab}"], lw=1.8, marker="o", ms=4, **st.RT[lab])
    axs[1].axhline(1., color="0.75", lw=0.8, zorder=0)
    axs[0].set_ylabel(r"cloud births [km$^{-2}$ h$^{-1}$]")
    axs[1].set_ylabel("births within 500 m of a cloud,\nobserved over random")
    for k, ax in enumerate(axs):
        ax.set_xlabel("local solar time [h]")
        ax.set_xlim(10., 17.)
        st.apply(ax)
        st.panel(ax, k)
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig17_births")


def figure6_strip(expt, vlim=4.e-3, hourly=False):
    """Buoyancy anomaly and circulation in the sun-parallel slice, 1D, 3D and their difference, per snapshot or pooled
    over each hour of the 60 s fields; bottom row: surface shortwave along the slice relative to the domain mean."""
    times = list(HOURS) if hourly else snapshot_times(expt)
    rows = ("1D", "3D", "3D - 1D")
    fig = plt.figure(figsize=(2.1 * len(times) + 0.6, 7.6), layout="constrained")
    gs = fig.add_gridspec(4, len(times) + 1, height_ratios=(1, 1, 1, 0.5), width_ratios=[1] * len(times) + [0.05])
    k = 0
    for j, t in enumerate(times):
        if hourly:
            c = {lab: ens_hour(expt, rt, *t).sel(dir="parallel") for rt, lab in RTS}
            label = f"{t[0]:.0f}-{t[1]:.0f} LT"
        else:
            c = {lab: ens(expt, rt, t)[0].sel(dir="parallel") for rt, lab in RTS}
            label = st.lt(ens(expt, "2stream", t)[1][0].attrs["lst_solar"])
        c["3D - 1D"] = c["3D"] - c["1D"]
        for i, lab in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            d = c[lab]
            im = ax.pcolormesh(d["xl"], d["znd"], d["b"], cmap=CMAP_ANOM, vmin=-vlim, vmax=vlim, rasterized=True)
            q = d.isel(xl=slice(4, None, 12), znd=slice(4, None, 8))
            ax.quiver(q["xl"], q["znd"], q["us"], q["w"], scale=24., width=0.005, color="0.15")
            for x in (-0.5, 0.5):
                ax.axvline(x, color="0.4", lw=0.6, ls="--")
            ax.set(xlim=(-1, 1), ylim=(0, 1))
            ax.tick_params(labelbottom=False, labelleft=(j == 0), labelsize=8)
            if j == 0:
                ax.set_ylabel(r"$z / z_b$ [-]")
                row_label(ax, lab)
            st.panel(ax, i * len(times) + j, label if i == 0 else "")
        ax = fig.add_subplot(gs[3, j])
        h = []
        for rt, lab in RTS:
            if hourly:
                e = ens_hour(expt, rt, *t).sel(dir="parallel")
                pct = (e["sw"] / e["sw_domain"] - 1.) * 100.
                l, = ax.plot(e["xl"], pct, lw=1.6, label=lab, **st.RT[lab])
            else:
                e, m = ens(expt, rt, t)
                pct = np.array([(d["sw"].sel(dir="parallel") / float(d["sw_domain"]) - 1.) * 100. for d in m])
                ax.fill_between(e["xl"], pct.min(axis=0), pct.max(axis=0), alpha=0.25, lw=0, **st.RT[lab])
                l, = ax.plot(e["xl"], pct.mean(axis=0), lw=1.6, label=lab, **st.RT[lab])
            h.append(l)
        st.zero_line(ax)
        for x in (-0.5, 0.5):
            ax.axvline(x, color="0.4", lw=0.6, ls="--")
        ax.set(xlim=(-1, 1), ylim=(-75, 30), xlabel=r"$r_\parallel / L$ [-]")
        ax.tick_params(labelleft=(j == 0), labelsize=8)
        if j == 0:
            ax.set_ylabel("surface SW,\nfrom domain\nmean [%]")
        st.apply(ax)
        st.panel(ax, 3 * len(times) + j)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:3, len(times)]))
    cb.set_label(r"buoyancy anomaly [m s$^{-2}$]")
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig6_circulation_strip" + ("_hourly" if hourly else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    a = ap.parse_args()
    for fn in (lambda e: figure6_strip(e, hourly=True), lambda e: figure7(e, hourly=True), figure17):
        print(fn(a.expt))
