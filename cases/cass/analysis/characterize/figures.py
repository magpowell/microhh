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


CMAP_ANOM = "RdBu_r"
DIRS = ("parallel", "perpendicular")


def ens(expt, rt, t):
    """Ensemble mean of the per-member composites; members weighted equally."""
    m = load(expt, "composite", rt, t)
    keep = [v for v in m[0].data_vars if "cloud" not in m[0][v].dims]
    return xr.concat([d[keep] for d in m], dim="member").mean("member"), m


def row_label(ax, text):
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(-52, 0), textcoords="offset points",
                ha="center", va="center", rotation=90, fontweight="bold", fontsize=10)


def figure6(expt, t, vlim=4.e-3):
    """Buoyancy anomaly and circulation around clouds, one snapshot."""
    c = {lab: ens(expt, rt, t)[0] for rt, lab in RTS}
    c["3D - 1D"] = c["3D"] - c["1D"]
    fig = plt.figure(figsize=(9., 9.2), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=(1, 1, 1, 0.38), width_ratios=(1, 1, 0.035))
    k = 0
    for i, lab in enumerate(("1D", "3D", "3D - 1D")):
        for j, dr in enumerate(DIRS):
            ax = fig.add_subplot(gs[i, j])
            d = c[lab].sel(dir=dr)
            im = ax.pcolormesh(d["xl"], d["znd"], d["b"], cmap=CMAP_ANOM, vmin=-vlim, vmax=vlim, rasterized=True)
            q = d.isel(xl=slice(4, None, 10), znd=slice(4, None, 7))
            ax.quiver(q["xl"], q["znd"], q["us"], q["w"], scale=22., width=0.004, color="0.15")
            for x in (-0.5, 0.5):
                ax.axvline(x, color="0.4", lw=0.6, ls="--")
            ax.set(xlim=(-1, 1), ylim=(0, 1))
            ax.tick_params(labelbottom=False, labelleft=(j == 0), labelsize=8)
            if j == 0:
                ax.set_ylabel(r"$z / z_b$ [-]")
                row_label(ax, lab)
            st.panel(ax, k, dr if i == 0 else "")
            k += 1
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:3, 2]))
    cb.set_label(r"buoyancy anomaly [m s$^{-2}$]")
    h = []
    for j, dr in enumerate(DIRS):
        ax = fig.add_subplot(gs[3, j])
        for rt, lab in RTS:
            e, m = ens(expt, rt, t)
            pct = np.array([(d["sw"].sel(dir=dr) / float(d["sw_domain"]) - 1.) * 100. for d in m])
            ax.fill_between(e["xl"], pct.min(axis=0), pct.max(axis=0), alpha=0.25, lw=0, **st.RT[lab])
            l, = ax.plot(e["xl"], pct.mean(axis=0), lw=1.8, label=lab, **st.RT[lab])
            if j == 0:
                h.append(l)
        st.zero_line(ax)
        ax.set(xlim=(-1, 1), xlabel=(r"$r_\parallel / L$ [-]" if j == 0 else r"$r_\perp / L$ [-]"))
        ax.tick_params(labelleft=(j == 0))
        if j == 0:
            ax.set_ylabel("surface SW,\ndifference from\ndomain mean [%]")
        st.apply(ax)
        st.panel(ax, k)
        k += 1
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    fig.suptitle(st.lt(3.891 + t / 3600.), x=0.02, ha="left", fontsize=11)
    return st.savefig(fig, expt, f"fig6_circulation_{int(t):07d}")


def figure7(expt, vlim=3.e-3):
    """Forces on the air in the sun-parallel slice, 3D minus 1D, for each snapshot."""
    rows = (("b", "buoyancy"), ("beff", "effective\nbuoyancy"), ("a_pd", "dynamic\npressure force"), ("tot", "sum"))
    fig, axs = plt.subplots(len(rows), len(TIMES4), figsize=(10.5, 8.2), sharex=True, sharey=True,
                            layout="constrained")
    for j, t in enumerate(TIMES4):
        d = (ens(expt, "raytracer", t)[0] - ens(expt, "2stream", t)[0]).sel(dir="parallel")
        f = dict(b=d["b"], beff=d["b"] + d["a_pb"], a_pd=d["a_pd"], tot=d["b"] + d["a_pb"] + d["a_pd"])
        for i, (key, lab) in enumerate(rows):
            ax = axs[i, j]
            im = ax.pcolormesh(d["xl"], d["znd"], f[key], cmap=CMAP_ANOM, vmin=-vlim, vmax=vlim, rasterized=True)
            for x in (-0.5, 0.5):
                ax.axvline(x, color="0.4", lw=0.6, ls="--")
            ax.tick_params(labelsize=8)
            st.panel(ax, i * len(TIMES4) + j, st.lt(3.891 + t / 3600.) if i == 0 else "")
            if j == 0:
                ax.set_ylabel(r"$z / z_b$ [-]")
                row_label(ax, lab)
            if i == len(rows) - 1:
                ax.set_xlabel(r"$r_\parallel / L$ [-]")
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label(r"vertical acceleration, 3D minus 1D [m s$^{-2}$]")
    return st.savefig(fig, expt, "fig7_forces_3D_minus_1D")


TIMES4 = (28800, 32400, 36000, 39600)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    for izb in (0, 2):
        print(figure1(a.expt, izb))
        print(figure4(a.expt, izb))
    print(figure5(a.expt, 2))
    for t in TIMES4:
        print(figure6(a.expt, t))
    print(figure7(a.expt))
