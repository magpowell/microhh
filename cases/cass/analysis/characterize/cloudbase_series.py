"""Cloud-base updraft, area and mass flux, root buoyancy and the barrier to the LFC every few minutes of the 60 s fields.

python cloudbase_series.py --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1 [--every 5] [--solar 11.9 16.1]
    -> .../<rt>/rep_NN/cloudbase_series.nc (one row per frame)
python cloudbase_series.py --expt no_aerosols_zero_wind_v3 --summary
    -> cloudbase_series.csv (member mean, min, max per frame), cloudbase_hourly.csv (hourly means and 3D over 1D), figure 20
Cloud base = lowest level with core fraction above ZB_THR (core: cloudy, rising, buoyant against the slab mean).
Root = the columns beneath the core at cloud base. Barrier = CIN of the mean cloudy-updraft parcel at cloud base lifted
through the clear-column environment, undilute and with a fixed entrainment rate.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

import masks as mk
import thermo as th
from les_io import Run
from parcel import cin_lfc, lift_thl
from snapshot import load, out_path, run_dir

RTS = ("2stream", "raytracer")
ZB_THR = 1.e-3
EPS = {"undilute": 0., "entraining": 5.e-4}      # 1/m; the measured bulk rate is about 5e-4
Z_TOP = 4800.                                    # sponge starts here
ROOT_FRACS = (0.5, 0.9)
W_BINS = np.arange(-4., 12.05, 0.1)              # m/s; the core holds sinking points too
HOURS = ((12., 13.), (13., 14.), (14., 15.), (15., 16.))


def frame(run, t):
    f, bs = load(run, t)
    z, dz, rho = run.z, run.z[1] - run.z[0], bs["rhoref"]
    core, cu = mk.core(f["qc"], f["w"], f["thv"]), mk.cloudy_updraft(f["qc"], f["w"])
    kb = mk.cloud_base_index(core.mean(axis=(1, 2)), ZB_THR)
    out = dict(t=int(t), lst=float(run.lst(t)), zb=np.nan, cover=float((f["qc"] > 0.).any(axis=0).mean()),
               lwp=float((f["qc"] * rho[:, None, None]).sum(axis=0).mean() * dz))
    if kb < 0:
        return out
    out["zb"] = float(z[kb])
    slab = lambda a, k: float(a[k].mean())
    for name, m in (("core", core), ("cu", cu)):
        sel = m[kb]
        n = int(sel.sum())
        a = n / sel.size
        w = float(f["w"][kb][sel].mean()) if n else np.nan
        out[f"n_{name}"], out[f"a_{name}"], out[f"w_{name}"], out[f"M_{name}"] = n, a, w, rho[kb] * a * w
        out[f"thv_{name}"] = float(f["thv"][kb][sel].mean()) - slab(f["thv"], kb) if n else np.nan
        out[f"qt_{name}"] = float(f["qt"][kb][sel].mean()) - slab(f["qt"], kb) if n else np.nan
        out[f"hist_{name}"] = np.histogram(f["w"][kb][sel], bins=W_BINS)[0]
    for fr in ROOT_FRACS:
        k = int(np.argmin(np.abs(z - fr * z[kb])))
        for v in ("thv", "w", "qt"):
            out[f"{v}_root_{int(fr * 100)}"] = float(f[v][k][core[kb]].mean()) - slab(f[v], k) if out["n_core"] else np.nan
    clear = mk.clear_columns(f["qc"])
    env = xr.Dataset({v: ("z", a) for v, a in (("pref", bs["pref"]), ("exnref", bs["exnref"]),
                                                ("thl_env", mk.column_mean(f["thl"], clear)), ("qt_env", mk.column_mean(f["qt"], clear)),
                                                ("T_env", mk.column_mean(f["T"], clear)), ("qv_env", mk.column_mean(f["qv"], clear)))},
                     coords=dict(z=z))
    ktop = int(np.searchsorted(z, Z_TOP)) - 1
    sel = cu[kb]
    if sel.sum() == 0:
        return out
    thl0, qt0, w = float(f["thl"][kb][sel].mean()), float(f["qt"][kb][sel].mean()), f["w"][kb][sel]
    for tag, eps in EPS.items():
        cin, zlfc, found = cin_lfc(z, lift_thl(env, kb, ktop, thl0, qt0, eps), kb, ktop)
        if not found:       # no level of free convection below Z_TOP: the barrier is not defined
            cin = zlfc = np.nan
        out[f"cin_{tag}"], out[f"zlfc_{tag}"], out[f"lfc_{tag}"] = cin, zlfc, int(found)
        out[f"wcrit_{tag}"] = np.sqrt(2. * cin)
        out[f"frac_above_{tag}"] = float((w > np.sqrt(2. * cin)).mean()) if found else np.nan
    return out


def analyse(expt, rt, rep, every=5., solar=(11.9, 16.1)):
    run = Run(run_dir(expt, rt, rep))
    ts = np.array(run.hf_times())
    step = int(round(every * 60. / (ts[1] - ts[0])))
    rows = []
    for t in ts[::step]:
        if solar[0] <= run.lst(t) < solar[1]:
            run = Run(run_dir(expt, rt, rep))      # load() restricts the grid; start each frame from the full run
            rows.append(frame(run, int(t)))
            print(f"{rt} rep_{rep:02d} t={int(t)} zb={rows[-1]['zb']:.0f} w_core={rows[-1].get('w_core', np.nan):.2f} M_core={rows[-1].get('M_core', np.nan):.4f}", flush=True)
    nb = W_BINS.size - 1
    hist = {k: np.array([r.pop(k, np.zeros(nb, dtype=int)) for r in rows]) for k in ("hist_core", "hist_cu")}
    d = pd.DataFrame(rows)
    ds = xr.Dataset.from_dataframe(d.set_index("t"))
    ds = ds.assign_coords(w_bin=0.5 * (W_BINS[:-1] + W_BINS[1:]))
    for k, v in hist.items():
        ds[k] = (("t", "w_bin"), v)
    ds.attrs.update(expt=expt, rt=rt, rep=rep, every_min=every, zb_thr=ZB_THR)
    out = out_path(expt, rt, rep, 0).parent
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "cloudbase_series.nc")
    return ds


def mask_no_lfc(ds):
    """Barrier quantities are undefined where no level of free convection was found (files written before the lfc flag)."""
    for tag in EPS:
        found = ds[f"zlfc_{tag}"].notnull()
        for v in ("cin", "wcrit", "frac_above"):
            ds[f"{v}_{tag}"] = ds[f"{v}_{tag}"].where(found)
        if f"lfc_{tag}" not in ds:
            ds[f"lfc_{tag}"] = found.astype(int)
    return ds


def load_all(expt):
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("cloudbase_series.nc")) as ds:
            d = mask_no_lfc(ds.drop_dims("w_bin").load()).to_dataframe().reset_index()
        d["rt"], d["rep"] = rt, rep
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


VARS = ("zb", "cover", "lwp", "a_core", "w_core", "M_core", "a_cu", "w_cu", "M_cu", "thv_core", "qt_core", "thv_root_50", "w_root_50",
        "thv_root_90", "cin_undilute", "cin_entraining", "wcrit_entraining", "frac_above_entraining", "lfc_entraining")


def summary(expt):
    d = load_all(expt)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    g = d.groupby(["rt", "t"])[list(VARS) + ["lst"]]
    s = pd.concat({"mean": g.mean(), "min": g.min(), "max": g.max()}, axis=1)
    s.to_csv(res / "cloudbase_series.csv")
    rows = []
    for h0, h1 in HOURS:
        w = d[(d.lst >= h0) & (d.lst < h1)]
        m = w.groupby("rt")[list(VARS)].mean()
        row = dict(hour=f"{h0:.0f}-{h1:.0f}")
        for v in VARS:
            row[f"{v}_1D"], row[f"{v}_3D"] = m.loc[RTS[0], v], m.loc[RTS[1], v]
            row[f"{v}_ratio"] = m.loc[RTS[1], v] / m.loc[RTS[0], v] if m.loc[RTS[0], v] else np.nan
        rows.append(row)
    h = pd.DataFrame(rows)
    h.to_csv(res / "cloudbase_hourly.csv", index=False)
    return d, s, h


def load_hist(expt):
    out = {}
    for rt, rep in itertools.product(RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("cloudbase_series.nc")) as ds:
            out[(rt, rep)] = mask_no_lfc(ds[["hist_core", "hist_cu", "lst", "wcrit_entraining", "zlfc_entraining", "cin_entraining", "frac_above_entraining",
                                             "wcrit_undilute", "zlfc_undilute", "cin_undilute", "frac_above_undilute"]].load())
    return out


def figure4r(expt, xmin=-1., xmax=7., step=0.28):
    """The same distributions as a ridgeline: one row per hour on a shared axis, kernel density estimate (scipy
    gaussian_kde, Scott's rule) of the core speeds of all members, from the 0.1 m/s counts."""
    import style as st
    from scipy.stats import gaussian_kde
    from style import plt
    H = load_hist(expt)
    x = 0.5 * (W_BINS[:-1] + W_BINS[1:])
    xs = np.linspace(xmin, xmax, 400)
    fig, ax = plt.subplots(figsize=(4.6, 4.6), layout="constrained")
    h, ticks = [], []
    for k, (h0, h1) in enumerate(HOURS):
        base, front = k * step, 2 * (len(HOURS) - k)      # earliest hour at the bottom, in front
        for rt, lab in zip(RTS, ("1D", "3D")):
            c = sum(H[(rt, rep)]["hist_core"].values[((H[(rt, rep)].lst >= h0) & (H[(rt, rep)].lst < h1)).values].sum(axis=0)
                    for rep in range(1, 5))
            kde = gaussian_kde(x[c > 0], weights=c[c > 0], bw_method=c.sum() ** -0.2)      # Scott's rule on the sample count
            y, mean = kde(xs), float((x * c).sum() / c.sum())
            ax.fill_between(xs, base, base + y, color="w", lw=0, zorder=front)     # hides the row behind
            line, = ax.plot(xs, base + y, lw=1.6, zorder=front + 1, label=lab, **st.RT[lab])
            ax.plot([mean, mean], [base, base + float(kde(mean)[0])], lw=1., ls="--", zorder=front + 1, **st.RT[lab])      # the mean
            if k == 0:
                h.append(line)
        ax.axhline(base, color="0.6", lw=0.6, zorder=front)
        ticks.append((base, f"{h0:.0f}-{h1:.0f} LT"))
    ax.set(xlim=(xmin, xmax), xlabel=r"core vertical velocity at cloud base [m s$^{-1}$]")
    ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0., (len(HOURS) - 1) * step + 0.5)
    st.apply(ax)
    ax.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D
    first = ax.legend(handles=h, ncols=2, loc="upper right")
    ax.add_artist(first)
    ax.legend(handles=[Line2D([], [], color="k", lw=1., ls="--", label="mean")], loc="upper right", bbox_to_anchor=(1., 0.94))
    return st.savefig(fig, expt, "fig_w_ridge")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--every", type=float, default=5.)
    ap.add_argument("--solar", type=float, nargs=2, default=(11.9, 16.1))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    if a.summary:
        d, s, h = summary(a.expt)
        show = ["hour"] + [f"{v}_{k}" for v in ("zb", "w_core", "a_core", "M_core", "thv_root_50", "cin_entraining") for k in ("1D", "3D", "ratio")]
        print("--- hourly means, 1D, 3D and 3D over 1D")
        print(h[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print(figure4r(a.expt))
    else:
        analyse(a.expt, a.rt, a.rep, a.every, a.solar)
