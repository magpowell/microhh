"""Figures 1 (MSE profiles and parcels) and 4 (cloud-base w PDF with w_crit) for one experiment."""
import argparse
import functools
import os
from pathlib import Path

import numpy as np
import xarray as xr

import style as st
from snapshot import snapshot_times
from style import plt

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
    for k, (ax, t) in enumerate(zip(axs, snapshot_times(expt, skip_first=True))):
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
    for k, (ax, t) in enumerate(zip(axs, snapshot_times(expt, skip_first=True))):
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
    times = snapshot_times(expt, skip_first=True)
    fig, axs = plt.subplots(2, len(times), figsize=(3. * len(times), 5.6), sharey=True, constrained_layout=True)
    h = []
    for k, t in enumerate(times):
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
    fig.suptitle(st.lt(ens(expt, "2stream", t)[1][0].attrs["lst_solar"]), x=0.02, ha="left", fontsize=11)
    return st.savefig(fig, expt, f"fig6_circulation_{int(t):07d}")


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




def _members(ax, x, m, lab):
    ax.fill_between(x, np.nanmin(m, axis=0), np.nanmax(m, axis=0), alpha=0.25, lw=0, **st.RT[lab])
    return ax.plot(x, np.nanmean(m, axis=0), lw=1.8, label=lab, **st.RT[lab])[0]


def figure8(expt, tag="", zlim=(1.0, 3.5), qlim=0.5):
    """Timing: cloud depth in 1D and 3D, and 3D minus 1D moisture outside cloud with cloud fraction contours."""
    res = st.outdir(expt).parent
    with xr.open_dataset(res / f"timing{tag}.nc") as d:
        d = d.load()
    x, z = d["lst"].values, d["z"].values / 1e3
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 8.), sharex=True, constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[1., 1., 1.5]))
    h = [_members(axs[0], x, d["lwp_members"].values[i] * 1e3, lab) for i, (_, lab) in enumerate(RTS)]
    axs[0].set_ylabel(r"liquid water path [g m$^{-2}$]")
    for i, (_, lab) in enumerate(RTS):
        _members(axs[1], x, d["depth_active_members"].values[i], lab)
    axs[1].set_ylabel("depth of clouds with\na buoyant column [m]")
    pc = axs[2].pcolormesh(x, z, d["qt_out_diff"].values.T * 1e3, cmap="BrBG", vmin=-qlim, vmax=qlim, shading="nearest",
                           rasterized=True)
    cs = axs[2].contour(x, z, d["cf_diff"].values.T * 100., levels=[0.2, 0.5, 0.8], colors="k", linewidths=0.8)
    axs[2].clabel(cs, fmt="%.1f", fontsize=7)
    axs[2].set_ylim(*zlim)
    axs[2].set_ylabel("height [km]")
    axs[2].set_xlabel("local solar time [h]")
    fig.colorbar(pc, ax=axs[2], label=r"3D - 1D water vapour outside cloud [g kg$^{-1}$]", extend="both", pad=0.02)
    for k, ax in enumerate(axs):
        st.apply(ax)
        st.panel(ax, k)
    axs[0].set_xlim(9., 16.)
    h.append(plt.Line2D([], [], color="k", lw=0.8, label="3D - 1D cloud fraction [%]"))
    fig.legend(handles=h, ncols=3, loc="outside lower center")
    return st.savefig(fig, expt, f"fig8_timing{tag}")


def figure9(expt, tag="", window=(12., 15.)):
    """Lifetimes of clouds that never merge or split, and the share of such clouds against track length."""
    import lifetime as lf
    L = np.arange(1., 26.)
    fig, axs = plt.subplots(1, 2, figsize=(7., 3.4), constrained_layout=True)
    h = []
    for rt, lab in RTS:
        surv, share = [], []
        for rep in range(1, 5):
            t = lf.load(expt, rt, rep, tag)
            t = t[(t.lst_first >= window[0]) & (t.lst_first < window[1])]
            u = t[t.untouched].length.values
            surv.append([(u >= v).mean() for v in L])
            b = np.digitize(t.length.values, lf.LENGTH_BINS) - 1
            share.append([t.untouched.values[b == i].mean() if (b == i).any() else np.nan
                          for i in range(lf.LENGTH_BINS.size - 1)])
        h.append(_members(axs[0], L, np.array(surv), lab))
        share = np.array(share)
        xb = np.arange(share.shape[1])
        axs[1].fill_between(xb, np.nanmin(share, axis=0), np.nanmax(share, axis=0), alpha=0.25, lw=0, **st.RT[lab])
        axs[1].plot(xb, np.nanmean(share, axis=0), lw=1.8, marker="o", ms=4, **st.RT[lab])
    axs[0].set_yscale("log")
    axs[0].set_ylim(1.e-3, 1.)
    axs[0].set_xlabel("lifetime [min]")
    axs[0].set_ylabel("share of clouds living\nat least this long [-]")
    axs[1].set_xticks(np.arange(6), ["1-5", "5-10", "10-20", "20-40", "40-80", "80+"])
    axs[1].set_xlabel("track length [min]")
    axs[1].set_ylabel("share of tracks that\nnever merged or split [-]")
    for k, ax in enumerate(axs):
        st.apply(ax)
        st.panel(ax, k)
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, f"fig9_lifetime{tag}")


def figure10(expt, times=None, x="w_core_layer", xmax=7., hourly=False):
    """Cloud depth against core updraft speed at cloud base, one point per cloud."""
    times = times or snapshot_times(expt)[-2:]
    import depth_w as dw
    d = dw.load(expt, x, hourly=hourly)
    if hourly:
        times = sorted(d.t.unique())
    g, _ = dw.binned(d)
    fig, axs = plt.subplots(1, len(times), figsize=(4.2 * len(times), 3.8), sharey=True, constrained_layout=True)
    h = []
    for k, t in enumerate(times):
        ax = axs[k]
        for rt, lab in RTS:
            c = d[(d.t == t) & (d.rt == rt)]
            ax.scatter(c.w, c.depth, s=5, alpha=0.2, lw=0, rasterized=True, **st.RT[lab])
            b = g[(g.t == t) & (g.rt == rt)]
            m = b.pivot(index="rep", columns="wb", values="depth")
            xw = b.groupby("wb").w.mean().reindex(m.columns).values
            ax.fill_between(xw, np.nanmin(m.values, axis=0), np.nanmax(m.values, axis=0), alpha=0.3, lw=0, **st.RT[lab])
            l, = ax.plot(xw, np.nanmean(m.values, axis=0), lw=1.8, marker="o", ms=4, label=lab, **st.RT[lab])
            if k == 0:
                h.append(l)
        ax.set_xlim(0., xmax)
        ax.set_xlabel(r"core updraft speed at cloud base [m s$^{-1}$]")
        st.apply(ax)
        st.panel(ax, k, f"{t:.0f}-{t + 1:.0f} LT" if hourly else st.lt(float(d[d.t == t].lst.iloc[0])))
    axs[0].set_ylabel("cloud depth [m]")
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig10_depth_speed" + ("" if x == "w_core_layer" else f"_{x}") + ("_hourly" if hourly else ""))


def _window(a, j0, i0, h):
    return a[np.ix_(np.arange(j0 - h, j0 + h + 1) % a.shape[0], np.arange(i0 - h, i0 + h + 1) % a.shape[1])]


def figure11(expt, rt="raytracer", rep=1, t=None, n=3, half=40):
    """Examples of objects before and after splitting: water path with the object outline and the seeds, then sub-clouds."""
    t = t or snapshot_times(expt)[-2]
    res = st.outdir(expt).parent
    with xr.open_dataset(res / rt / f"rep_{rep:02d}" / f"objects_{t:07d}.nc") as ds:
        ds = ds.load()
    dx = ds.attrs["dx"]
    order = np.argsort(ds["obj_n_seeds"].values + 1.e-6 * ds["obj_D"].values)[::-1]
    pick = [c for c in order if ds["obj_D"].values[c] < 2. * half * dx * 0.8][:n]
    fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 6.2), constrained_layout=True, sharex=True, sharey=True)
    ext = np.array([-half, half, -half, half]) * dx / 1e3
    for k, c in enumerate(pick):
        j0, i0 = int(ds["obj_y"].values[c] / dx), int(ds["obj_x"].values[c] / dx)
        obj = _window(ds["map_obj"].values, j0, i0, half) == c + 1
        lwp = np.where(obj, _window(ds["map_lwp"].values, j0, i0, half) * 1e3, np.nan)
        seed = np.where(obj, _window(ds["map_seed"].values, j0, i0, half), 0)
        sub = np.where(obj, _window(ds["map_sub"].values, j0, i0, half), 0)
        pc = axs[0, k].imshow(lwp, origin="lower", extent=ext, cmap="Greys", vmin=0., vmax=np.nanpercentile(lwp, 99))
        axs[0, k].contour(np.linspace(ext[0], ext[1], obj.shape[1]), np.linspace(ext[2], ext[3], obj.shape[0]), seed > 0,
                          levels=[0.5], colors="C3", linewidths=0.9)
        ids = np.unique(sub[sub > 0])
        lab = np.full(sub.shape, np.nan)
        for m, i in enumerate(ids):
            lab[sub == i] = m
        axs[1, k].imshow(lab, origin="lower", extent=ext, cmap="tab10", vmin=-0.5, vmax=9.5, interpolation="nearest")
        fig.colorbar(pc, ax=axs[0, k], location="top", label=r"liquid water path [g m$^{-2}$]", shrink=0.9)
        for r in (0, 1):
            st.apply(axs[r, k])
            st.panel(axs[r, k], r * n + k)
            axs[r, k].set_aspect("equal")
        axs[1, k].set_xlabel("x [km]")
    axs[0, 0].set_ylabel("y [km]")
    axs[1, 0].set_ylabel("y [km]")
    fig.legend(handles=[plt.Line2D([], [], color="C3", lw=0.9, label="core regions used as seeds")], loc="outside lower center")
    return st.savefig(fig, expt, "fig11_split_examples")


def figure12(expt, times=None):
    """Share of objects holding two or more cores, against object width."""
    times = times or snapshot_times(expt)[-2:]
    import robust as rb
    o = rb.load(expt, "obj")
    o["wc"] = np.digitize(o.D, rb.WIDTH_CLASSES) - 1
    fig, axs = plt.subplots(1, len(times), figsize=(3.8 * len(times), 3.4), sharey=True, constrained_layout=True)
    h = []
    for k, t in enumerate(times):
        for rt, lab in RTS:
            g = o[(o.t == t) & (o.rt == rt)].groupby(["rep", "wc"]).n_seeds.apply(lambda v: (v >= 2).mean()).unstack("wc")
            g = g.reindex(columns=range(4))
            h.append(_members(axs[k], np.arange(4), g.values, lab))
        axs[k].set_xticks(np.arange(4), ["< 0.5", "0.5-1", "1-1.5", "1.5-2"])
        axs[k].set_xlabel("object width [km]")
        st.apply(axs[k])
        st.panel(axs[k], k, st.lt(float(o[o.t == t].lst.iloc[0])))
    axs[0].set_ylabel("share of objects with\ntwo or more cores [-]")
    fig.legend(handles=h[:2], ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig12_cores_per_object")


def figure13(expt):
    """Organization index of cloud objects and of split sub-clouds, with the two references."""
    import pandas as pd
    d = pd.read_csv(st.outdir(expt).parent / "cluster_long.csv")
    cases = [("obj", 0.), ("sub", 0.), ("obj", 500.), ("sub", 500.)]
    names = ["objects", "sub-clouds", "objects wider than 500 m", "sub-clouds wider than 500 m"]
    fig, axs = plt.subplots(1, 4, figsize=(11., 3.2), sharey=True, constrained_layout=True)
    h = []
    for k, (kind, dmin) in enumerate(cases):
        c = d[(d.kind == kind) & (d.min_D == dmin)]
        x = np.sort(c.lst.unique())
        for rt, lab in RTS:
            h.append(_members(axs[k], x, c[c.rt == rt].pivot(index="rep", columns="t", values="iorg").values, lab))
        l1, = axs[k].plot(x, c.groupby("t").iorg_disks.mean().values, color="k", lw=1., ls="--", label="randomly placed disks")
        l2 = axs[k].axhline(0.5, color="0.6", lw=1., ls=":", label="random points")
        axs[k].set_xlabel("local solar time [h]")
        st.apply(axs[k])
        st.panel(axs[k], k, names[k])
    axs[0].set_ylabel("organization index [-]")
    fig.legend(handles=h[:2] + [l1, l2], ncols=4, loc="outside lower center")
    return st.savefig(fig, expt, "fig13_organization")


def figure14(expt, times=None):
    """Around large clouds, +x away from the sun: surface shortwave, low-level convergence, small-cloud occurrence."""
    times = times or snapshot_times(expt)[-2:]
    import neighbours as nb
    res = st.outdir(expt).parent
    fig, axs = plt.subplots(3, 2 * len(times), figsize=(3.1 * 2 * len(times), 8.4), constrained_layout=True)
    h, k = [], 0
    for c, (t, (rt, lab)) in enumerate([(t, r) for t in times for r in RTS]):
        M = [xr.open_dataset(res / rt / f"rep_{rep:02d}" / f"neighbours_{t:07d}.nc").load() for rep in range(1, 5)]
        g = M[0]["xg"].values
        sw = np.mean([m["sw"].values - float(m["sw_domain"]) for m in M], axis=0)
        cv = np.mean([m["conv"].values for m in M], axis=0) * 1e3
        p0 = axs[0, c].pcolormesh(g, g, sw, cmap="RdBu_r", vmin=-400., vmax=400., shading="nearest", rasterized=True)
        p1 = axs[1, c].pcolormesh(g, g, cv, cmap="PuOr_r", vmin=-2.5, vmax=2.5, shading="nearest", rasterized=True)
        for r in (0, 1):
            axs[r, c].add_patch(plt.Circle((0., 0.), 0.5, fill=False, color="k", lw=0.8, ls="--"))
            axs[r, c].set_aspect("equal")
            axs[r, c].set_xlabel("x / D [-]")
        xb = M[0]["xb"].values
        for pop, ls in (("small", "-"),):
            rel = []
            for m in M:
                cor = np.abs(m["yb"].values) <= 0.5
                rel.append(m[f"hist_{pop}"].values[cor].sum(axis=0) / (float(m[f"density_{pop}"]) * (m["big_D"].values ** 2).sum() * 0.5))
            _members(axs[2, c], xb, np.array(rel), lab)
        axs[2, c].axhline(1., color="0.6", lw=0.8, zorder=0)
        axs[2, c].axvline(M[0].attrs["shadow_offset_m"] / np.mean([float(m["big_D"].mean()) for m in M]), color="k", lw=0.8, ls=":")
        axs[2, c].set_ylim(0., 2.)
        axs[2, c].set_xlabel("x / D [-]")
        for r in range(3):
            st.apply(axs[r, c])
            st.panel(axs[r, c], 4 * r + c, f"{lab}, {st.lt(M[0].attrs['lst_solar'])}" if r == 0 else "")
    for r, lab in ((0, "y / D [-]"), (1, "y / D [-]"), (2, "small clouds relative\nto random placement [-]")):
        axs[r, 0].set_ylabel(lab)
    fig.colorbar(p0, ax=axs[0, :], label="surface shortwave minus\n" + r"domain mean [W m$^{-2}$]", extend="both", pad=0.01)
    fig.colorbar(p1, ax=axs[1, :], label="low-level convergence\n" + r"[10$^{-3}$ s$^{-1}$]", extend="both", pad=0.01)
    hh = [plt.Line2D([], [], lw=1.8, label=l, **st.RT[l]) for l in ("1D", "3D")]
    hh += [plt.Line2D([], [], color="k", lw=0.8, ls="--", label="large cloud"), plt.Line2D([], [], color="k", lw=0.8, ls=":", label="shadow offset")]
    fig.legend(handles=hh, ncols=4, loc="outside lower center")
    return st.savefig(fig, expt, "fig14_neighbours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    for izb in (0, 2):
        print(figure1(a.expt, izb))
        print(figure4(a.expt, izb))
    print(figure5(a.expt, 2))
    for t in snapshot_times(a.expt):
        print(figure6(a.expt, t))
    print(figure7(a.expt))
    print(figure8(a.expt))
    print(figure9(a.expt))
    print(figure10(a.expt))
    for fn in (figure11, figure12, figure13, figure14):
        print(fn(a.expt))


def figure15(expt, times=None, var="qt", mask="core", hourly=False):
    """Fractional entrainment of each cloud against its width; member medians per width class with min-max bands."""
    times = times or snapshot_times(expt)[-2:]
    import dilution as dl
    d = dl.load_clouds(expt, mask, hourly=hourly)
    if hourly:
        times = sorted(d.t.unique())
    e = f"eps_{var}"
    fig, axs = plt.subplots(1, len(times), figsize=(4.2 * len(times), 3.8), sharey=True, constrained_layout=True)
    h = []
    for k, t in enumerate(times):
        ax = axs[k]
        g, _ = dl.binned(d[d.t == t], var)
        for rt, lab in RTS:
            c = d[(d.t == t) & (d.rt == rt) & (d[e] > 0.)]
            ax.scatter(c.D, 1.e3 * c[e], s=5, alpha=0.2, lw=0, rasterized=True, **st.RT[lab])
            b = g[g.rt == rt]
            m = b.pivot(index="rep", columns="db", values="eps")
            xd = b.groupby("db").D.mean().reindex(m.columns).values
            ax.fill_between(xd, 1.e3 * np.nanmin(m.values, axis=0), 1.e3 * np.nanmax(m.values, axis=0), alpha=0.3, lw=0, **st.RT[lab])
            l, = ax.plot(xd, 1.e3 * np.nanmean(m.values, axis=0), lw=1.8, marker="o", ms=4, label=lab, **st.RT[lab])
            if k == 0:
                h.append(l)
        ref = d[(d.t == t) & (d.rt == RTS[0][0]) & (d[e] > 0.)]
        x = np.array([300., 3000.])
        l, = ax.plot(x, 1.e3 * np.median(ref[e] * ref.D) / x, color="0.4", lw=1., ls="--", label="1/D")
        if k == 0:
            h.append(l)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(250., 4000.); ax.set_ylim(0.05, 5.)
        ax.set_xlabel("cloud width [m]")
        st.apply(ax)
        st.panel(ax, k, f"{t:.0f}-{t + 1:.0f} LT" if hourly else st.lt(float(d[d.t == t].lst.iloc[0])))
    axs[0].set_ylabel(r"fractional entrainment [km$^{-1}$]")
    fig.legend(handles=h, ncols=3, loc="outside lower center")
    return st.savefig(fig, expt, f"fig15_dilution_width_{var}" + ("" if mask == "core" else f"_{mask}") + ("_hourly" if hourly else ""))


def figure16(expt, times=None, hourly=False):
    """Cloud depth against the persistence of its cloudy site; member medians per duration class with min-max bands."""
    times = times or snapshot_times(expt)[-2:]
    import persistence as ps
    d = ps.load_clouds(expt, hourly=hourly)
    if hourly:
        times = sorted(d.t.unique())
    fig, axs = plt.subplots(1, len(times), figsize=(4.2 * len(times), 3.8), sharey=True, constrained_layout=True)
    h = []
    for k, t in enumerate(times):
        ax = axs[k]
        for rt, lab in RTS:
            c = d[(d.t == t) & (d.rt == rt)]
            ax.scatter(np.maximum(c.dur_mean, 0.5), c.depth, s=5, alpha=0.2, lw=0, rasterized=True, **st.RT[lab])
            g = c.groupby(["rep", "ub"]).agg(depth=("depth", "median"), dur=("dur_mean", "median")).reset_index()
            m = g.pivot(index="rep", columns="ub", values="depth")
            xd = g.groupby("ub").dur.mean().reindex(m.columns).values
            ax.fill_between(xd, np.nanmin(m.values, axis=0), np.nanmax(m.values, axis=0), alpha=0.3, lw=0, **st.RT[lab])
            l, = ax.plot(xd, np.nanmean(m.values, axis=0), lw=1.8, marker="o", ms=4, label=lab, **st.RT[lab])
            if k == 0:
                h.append(l)
        ax.set_xscale("log")
        ax.set_xlim(0.4, 100.)
        ax.set_xlabel("site persistence [min]")
        st.apply(ax)
        st.panel(ax, k, f"{t:.0f}-{t + 1:.0f} LT" if hourly else st.lt(float(d[d.t == t].lst.iloc[0])))
    axs[0].set_ylabel("cloud depth [m]")
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig16_depth_persistence" + ("_hourly" if hourly else ""))


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


def figure22(expt):
    """Low-level horizontal buoyancy-pressure and dynamic-pressure forces along the sun-parallel slice, pooled over each
    hour, over the surface shortwave along the slice relative to the domain mean."""
    import catchment as ca
    fig, axs = plt.subplots(3, len(HOURS), figsize=(2.1 * len(HOURS) + 0.6, 6.6), sharex=True, sharey="row", layout="constrained")
    h = []
    for j, t in enumerate(HOURS):
        for rt, lab in RTS:
            xl, pb, pd_, L, n = ca.low_level(expt, rt, *t)
            e = ens_hour(expt, rt, *t).sel(dir="parallel")
            sw = (e["sw"] / e["sw_domain"] - 1.) * 100.
            for i, y in enumerate((1.e3 * pb, 1.e3 * pd_, sw)):
                l, = axs[i, j].plot(xl, y, lw=1.6, label=lab, **st.RT[lab])
            if j == 0:
                h.append(l)
        for i in range(3):
            ax = axs[i, j]
            st.zero_line(ax)
            for x in (-0.5, 0.5):
                ax.axvline(x, color="0.4", lw=0.6, ls="--")
            ax.set_xlim(-1, 1)
            ax.tick_params(labelsize=8)
            st.apply(ax)
            st.panel(ax, i * len(HOURS) + j, f"{t[0]:.0f}-{t[1]:.0f} LT" if i == 0 else "")
        axs[2, j].set_xlabel(r"$r_\parallel / L$ [-]")
    axs[0, 0].set_ylabel("buoyancy-pressure force,\n%.2f-%.1f $z_b$ [10$^{-3}$ m s$^{-2}$]" % ca.LOW)
    axs[1, 0].set_ylabel("dynamic-pressure force,\n%.2f-%.1f $z_b$ [10$^{-3}$ m s$^{-2}$]" % ca.LOW)
    axs[2, 0].set_ylabel("surface SW,\nfrom domain mean [%]")
    axs[2, 0].set_ylim(-80., 30.)
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig22_catchment")


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
