"""Figures 1 (MSE profiles and parcels) and 4 (cloud-base w PDF with w_crit) for one experiment."""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

import style as st
from style import plt

TIMES = (32400, 36000, 39600)
RTS = (("2stream", "1D"), ("raytracer", "3D"))
WB = np.arange(0., 8.01, 0.4)


def load(expt, kind, rt, t):
    root = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt / rt
    return [xr.open_dataset(f).load() for f in sorted(root.glob(f"rep_*/{kind}_{int(t):07d}.nc"))]


def band(ax, x, z, horizontal=True, **kw):
    """Ensemble mean line with min-max fill; x has shape (member, z)."""
    m = np.nanmean(x, axis=0)
    ok = np.isfinite(m)
    ax.fill_betweenx(z[ok], np.nanmin(x, axis=0)[ok], np.nanmax(x, axis=0)[ok], color=kw.get("color"), alpha=0.25, lw=0)
    return ax.plot(m[ok], z[ok], **kw)[0]


def figure1(expt, izb=0):
    fig, axs = plt.subplots(1, 3, figsize=(9., 4.2), sharey=True, constrained_layout=True)
    for k, (ax, t) in enumerate(zip(axs, TIMES)):
        snap = load(expt, "snap", "2stream", t)
        z = snap[0]["z"].values / 1e3
        h = []
        h.append(band(ax, np.array([s["h_env"].values for s in snap]) / 1e3, z, color="k", ls="-", lw=1.4,
                      label=r"$h_{env}$"))
        h.append(band(ax, np.array([s["hsat_env"].values for s in snap]) / 1e3, z, color="k", ls="--", lw=1.4,
                      label=r"$h^*_{env}$"))
        for rt, lab in RTS:
            par = load(expt, "parcel", rt, t)
            for kind, ls in (("undilute", "-"), ("entraining", ":")):
                x = np.array([p["h_parcel"].isel(zb_thr=izb).sel(start="core", kind=kind).values for p in par]) / 1e3
                h.append(band(ax, x, z, ls=ls, lw=1.8, label=f"{lab} {kind}", **st.RT[lab]))
        zb = np.mean([float(s["zb"][izb]) for s in snap]) / 1e3
        ax.axhline(zb, color="0.5", lw=0.8, ls=(0, (1, 2)), zorder=0)
        ax.set(xlim=(321., 343.), ylim=(1.0, 3.6), xlabel=r"moist static energy [kJ kg$^{-1}$]")
        st.apply(ax)
        st.panel(ax, k, st.lt(snap[0].attrs["lst_solar"]))
    axs[0].set_ylabel("height [km]")
    fig.legend(handles=h, ncols=6, loc="outside lower center", columnspacing=1.2, handlelength=2.2)
    return st.savefig(fig, expt, f"fig1_mse_parcels_thr{izb}")


def pdf(w):
    c, _ = np.histogram(w, bins=WB)
    return c / max(c.sum(), 1) / np.diff(WB)


def figure4(expt, izb=0):
    fig, axs = plt.subplots(1, 3, figsize=(9., 3.4), sharey=True, constrained_layout=True)
    x = 0.5 * (WB[:-1] + WB[1:])
    for k, (ax, t) in enumerate(zip(axs, TIMES)):
        h = []
        for rt, lab in RTS:
            snap = load(expt, "snap", rt, t)
            for tag, ls, name in (("core", "-", "core"), ("cu", "--", "cloudy updraft")):
                p = np.array([pdf(s[f"pt_{tag}_w_{izb}"].values) for s in snap])
                ax.fill_between(x, p.min(axis=0), p.max(axis=0), alpha=0.2, lw=0, **st.RT[lab])
                h.append(ax.plot(x, pdf(np.concatenate([s[f"pt_{tag}_w_{izb}"].values for s in snap])), ls=ls, lw=1.8,
                                 label=f"{lab} {name}", **st.RT[lab])[0])
            par = load(expt, "parcel", rt, t)
            wc = np.array([float(p["w_crit"].isel(zb_thr=izb).sel(start="cu", kind="entraining")) for p in par])
            ax.axvspan(wc.min(), wc.max(), alpha=0.2, lw=0, **st.RT[lab])
            h.append(ax.axvline(wc.mean(), lw=1.2, ls="-.", label=f"{lab} " + r"$w_{crit}$", **st.RT[lab]))
        ax.set(xlim=(0., 7.), xlabel=r"vertical velocity at cloud base [m s$^{-1}$]")
        st.apply(ax)
        st.panel(ax, k, st.lt(load(expt, "snap", "2stream", t)[0].attrs["lst_solar"]))
    axs[0].set_ylabel(r"probability density [s m$^{-1}$]")
    axs[0].set_ylim(bottom=0.)
    fig.legend(handles=h, ncols=6, loc="outside lower center", columnspacing=1.2)
    return st.savefig(fig, expt, f"fig4_w_pdf_thr{izb}")


def figure5(expt, izb=2):
    """Active fraction of cloud-base cloudy-updraft parcels against a uniform boost in w and in h."""
    fig, axs = plt.subplots(2, 3, figsize=(9., 5.6), sharey=True, constrained_layout=True)
    h = []
    for k, t in enumerate(TIMES):
        for rt, lab in RTS:
            act = [a.isel(zb_thr=izb).sel(start="cu", kind="entraining") for a in load(expt, "activation", rt, t)]
            fw = np.array([a["f_active_dw"].values for a in act])
            x = act[0]["dw"].values
            axs[0, k].fill_between(x, fw.min(axis=0), fw.max(axis=0), alpha=0.25, lw=0, **st.RT[lab])
            l0, = axs[0, k].plot(x, fw.mean(axis=0), lw=1.8, label=lab, **st.RT[lab])
            x = act[0]["dh"].values / 1e3
            for b, ls in (("moisture", "-"), ("heat", "--")):
                fh = np.array([a["f_active_dh"].sel(boost=b).values for a in act])
                axs[1, k].fill_between(x, fh.min(axis=0), fh.max(axis=0), alpha=0.25, lw=0, **st.RT[lab])
                axs[1, k].plot(x, fh.mean(axis=0), lw=1.8, ls=ls, **st.RT[lab])
            if k == 0:
                h.append(l0)
        lt = st.lt(load(expt, "snap", "2stream", t)[0].attrs["lst_solar"])
        for r, xl in ((0, r"boost in vertical velocity [m s$^{-1}$]"),
                      (1, r"boost in moist static energy [kJ kg$^{-1}$]")):
            ax = axs[r, k]
            ax.axvline(0, color="0.75", lw=0.8, zorder=0)
            ax.set_xlabel(xl)
            st.apply(ax)
            st.panel(ax, 3 * r + k, lt)
    for r in (0, 1):
        axs[r, 0].set_ylabel("active fraction of cloud-base\ncloudy-updraft parcels [-]")
    axs[0, 0].set_ylim(0., 1.)
    h += [plt.Line2D([], [], color="k", lw=1.8, ls=ls, label=b) for b, ls in (("moisture", "-"), ("heat", "--"))]
    fig.legend(handles=h, ncols=4, loc="outside lower center")
    return st.savefig(fig, expt, f"fig5_activation_thr{izb}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    for izb in (0, 2):
        print(figure1(a.expt, izb))
        print(figure4(a.expt, izb))
    print(figure5(a.expt, 2))
