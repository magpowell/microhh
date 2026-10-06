"""Why lit open ground makes fewer clouds in 3D: the flow and the thermodynamic state over ground classed by its
distance from clouds and by the sunlight it received, from the hourly 3D snapshots.

python open_ground.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1     (open_ground_<t>.nc per snapshot)
python open_ground.py --summary                                                (open_ground_classes.csv, open_ground_diff.csv)
Classes: under a cloud (>= 16 cells), within FAR of one, open and lit, open and shaded (shortwave anomaly of the
preceding LOOK minutes below LIT). Flow: mass divergence in the lowest DIV_TOP metres and w at 0.5 and 0.9 z_b.
State: thl and qt anomalies from the domain mean in the surface layer and the upper subcloud layer, and the surface
parcel (lowest 100 m) lifted to cloud base: its LCL and its virtual potential temperature excess at z_b.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

import masks as mk
import thermo as th
from les_io import Run
from composite import cloud_base_index
from snapshot import load as load_fields, out_path, run_dir, snapshot_times
from suppression import far_from_cloud, cloud_mask
from widening import surface_sw

RTS = ("2stream", "raytracer")
IZB = 2
MIN_AREA = 4.e4
FAR = 500.
LOOK, LIT = 10, -20.
DIV_TOP = 200.                     # m
SFC_TOP = 100.                     # m, surface parcel and surface layer
UPPER = (0.7, 1.0)                 # of z_b, upper subcloud layer
CLASSES = ("cloud", "near", "open_lit", "open_shaded")


def classes(qc, dx, dy, anomaly, far=FAR, lit=LIT, min_area=MIN_AREA):
    cloudy = cloud_mask(qc.max(axis=0), dx, dy, min_area)
    open_cols = far_from_cloud(cloudy, dx, dy, far)
    c = np.full(cloudy.shape, 1, dtype=int)            # near
    c[cloudy] = 0
    c[open_cols & (anomaly >= lit)] = 2
    c[open_cols & (anomaly < lit)] = 3
    return c


def mass_divergence(u, v, rho, dx, dy):
    """Horizontal anelastic mass divergence at cell centres, (z, y, x)."""
    return rho[:, None, None] * ((np.roll(u, -1, axis=2) - u) / dx + (np.roll(v, -1, axis=1) - v) / dy)


def layer_mean(f, z, z0, z1):
    k = (z >= z0) & (z < z1)
    return f[k].mean(axis=0)


def lifted_parcel(thl_p, qt_p, p, exn, kb, ktop=None):
    """Surface parcel lifted through levels 0..ktop (default kb): LCL level index (or -1) and its ql at kb."""
    ktop = kb if ktop is None else ktop
    lcl = np.full(thl_p.shape, -1, dtype=int)
    ql_kb = np.zeros(thl_p.shape)
    for k in range(ktop + 1):
        a = th.sat_adjust(thl_p, qt_p, np.full(thl_p.shape, p[k]), np.full(thl_p.shape, exn[k]))
        ql = a["ql"] + a["qi"]
        lcl = np.where((lcl < 0) & (ql > 0.), k, lcl)
        if k == kb:
            ql_kb = ql
    return lcl, ql_kb


def analyse(expt, rt, rep, t):
    run = Run(run_dir(expt, rt, rep))
    rd = run.dir
    kb = cloud_base_index(expt, rt, rep, t)
    f, bs = load_fields(run, t)
    z, zh, dx, dy = run.z, run.zh, run.dx, run.dy
    qc = f["qc"]
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        fr = int(np.argmin(np.abs(ds["time"].values.astype(float) - t)))
    sw = [surface_sw(rd, rt, g) for g in range(fr - LOOK, fr)]
    anomaly = np.mean([a - a.mean() for a in sw], axis=0)
    cl = classes(qc, dx, dy, anomaly)
    zb = z[kb]
    div = layer_mean(mass_divergence(f["u"], f["v"], bs["rhoref"], dx, dy), z, 0., DIV_TOP)
    wf = f["w"]
    w05 = wf[int(np.argmin(np.abs(z - 0.5 * zb)))]
    w09 = wf[int(np.argmin(np.abs(z - 0.9 * zb)))]
    thl_m, qt_m = f["thl"].mean(axis=(1, 2)), f["qt"].mean(axis=(1, 2))
    thl_a, qt_a = f["thl"] - thl_m[:, None, None], f["qt"] - qt_m[:, None, None]
    thl_sfc, qt_sfc = layer_mean(thl_a, z, 0., SFC_TOP), layer_mean(qt_a, z, 0., SFC_TOP)
    thl_up, qt_up = layer_mean(thl_a, z, UPPER[0] * zb, UPPER[1] * zb), layer_mean(qt_a, z, UPPER[0] * zb, UPPER[1] * zb)
    thl_p, qt_p = layer_mean(f["thl"], z, 0., SFC_TOP), layer_mean(f["qt"], z, 0., SFC_TOP)
    ktop = int(np.searchsorted(z, zb + 600.))
    lcl, ql_kb = lifted_parcel(thl_p.astype(float), qt_p.astype(float), bs["pref"], bs["exnref"], kb, ktop)
    thv_p = th.theta_v(thl_p, qt_p, ql_kb, 0., bs["exnref"][kb])
    thv_env = th.theta_v(f["thl"][kb], f["qt"][kb], f["ql"][kb], f["qi"][kb], bs["exnref"][kb])
    b_kb = thv_p - thv_env
    kl = np.maximum(lcl, 0)
    jj, ii = np.indices(kl.shape)
    thv_env_lcl = th.theta_v(f["thl"][kl, jj, ii], f["qt"][kl, jj, ii], f["ql"][kl, jj, ii], f["qi"][kl, jj, ii], bs["exnref"][kl])
    b_lcl = np.where(lcl >= 0, th.theta_v(thl_p, qt_p, 0., 0., bs["exnref"][kl]) - thv_env_lcl, np.nan)
    z_lcl = np.where(lcl >= 0, z[kl], np.nan)
    rows = []
    for c, name in enumerate(CLASSES):
        m = cl == c
        if m.sum() == 0:
            continue
        rows.append(dict(cls=name, n=int(m.sum()), frac=float(m.mean()), div_h=float(div[m].mean()), w05=float(w05[m].mean()), w09=float(w09[m].mean()),
                         thl_sfc=float(thl_sfc[m].mean()), qt_sfc=float(1.e3 * qt_sfc[m].mean()), thl_up=float(thl_up[m].mean()), qt_up=float(1.e3 * qt_up[m].mean()),
                         lcl=float(np.nanmean(z_lcl[m])), lcl_minus_zb=float(np.nanmean(z_lcl[m]) - zb), lcl_found=float((lcl[m] >= 0).mean()),
                         b_lcl=float(np.nanmean(b_lcl[m])), b_kb=float(b_kb[m].mean()), b_kb_pos=float((b_kb[m] > 0.).mean()),
                         anomaly=float(anomaly[m].mean())))
    d = pd.DataFrame(rows)
    ds = xr.Dataset.from_dataframe(d.set_index("cls"))
    ds.attrs.update(expt=expt, rt=rt, rep=rep, t_sec=int(t), lst_solar=float(run.lst(t)), kb=kb, zb=float(zb), far=FAR, lit=LIT, div_top=DIV_TOP,
                    sfc_top=SFC_TOP, upper_lo=UPPER[0], upper_hi=UPPER[1])
    out = out_path(expt, rt, rep, t)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out.with_name(f"open_ground_{int(t):07d}.nc"))
    return ds


VARS = ("frac", "div_h", "w05", "w09", "thl_sfc", "qt_sfc", "thl_up", "qt_up", "lcl", "lcl_minus_zb", "b_lcl", "b_kb", "b_kb_pos", "anomaly")


def load(expt):
    rows = []
    for t, rt, rep in itertools.product(snapshot_times(expt, skip_first=True), RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"open_ground_{t:07d}.nc")) as ds:
            d = ds.to_dataframe().reset_index()
            d["zb"], d["lst"] = float(ds.attrs["zb"]), float(ds.attrs["lst_solar"])
        d["t"], d["rt"], d["rep"] = t, rt, rep
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_series(expt):
    """Every open_ground file of the experiment (snapshot times and every-N-minute frames alike)."""
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        for f in sorted(out_path(expt, rt, rep, 0).parent.glob("open_ground_*.nc")):
            with xr.open_dataset(f) as ds:
                d = ds.to_dataframe().reset_index()
                d["t"], d["lst"], d["zb"] = int(ds.attrs["t_sec"]), float(ds.attrs["lst_solar"]), float(ds.attrs["zb"])
            d["rt"], d["rep"] = rt, rep
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def series(expt):
    """Hourly means over lit open ground of the barrier and the low-level divergence, next to the lit-ground birth rate."""
    d = load_series(expt)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    lit = d[d.cls == "open_lit"].copy()
    lit["hour"] = np.floor(lit.lst)
    cols = ["frac", "div_h", "w05", "b_lcl", "b_kb", "b_kb_pos", "lcl_minus_zb", "qt_up", "thl_up"]
    g = lit.groupby(["hour", "rt"])[cols].mean().unstack("rt")
    out = pd.DataFrame({f"{c}_{lab}": g[c][rt] for c in cols for rt, lab in zip(RTS, ("1D", "3D"))})
    try:
        s = pd.read_csv(res / "suppression_decomp_far500_min40000.csv").set_index("hour")
        out["births_lit_1D"], out["births_lit_3D"] = s["rate_lit_1D"], s["rate_lit_3D"]
    except FileNotFoundError:
        pass
    out.to_csv(res / "open_ground_series.csv")
    return d, out


def figure21(expt):
    import style as st
    from style import plt
    d = load_series(expt)
    lit = d[d.cls == "open_lit"]
    panels = (("b_lcl", r"parcel $\theta_v$ deficit at its LCL [K]"), ("div_h", r"mass divergence below 200 m [kg m$^{-3}$ s$^{-1}$]"),
              ("w05", r"w at 0.5 $z_b$ [m s$^{-1}$]"), ("qt_up", r"$q_t$ anomaly, upper subcloud [g kg$^{-1}$]"))
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.8), sharex=True, layout="constrained")
    h = []
    for k, ((v, lab), ax) in enumerate(zip(panels, axs.ravel())):
        for rt, name in zip(RTS, ("1D", "3D")):
            c = lit[lit.rt == rt].pivot_table(index="t", columns="rep", values=v)
            x = lit[lit.rt == rt].groupby("t").lst.first().reindex(c.index).values
            ax.fill_between(x, c.min(axis=1), c.max(axis=1), alpha=0.25, lw=0, **st.RT[name])
            l, = ax.plot(x, c.mean(axis=1), lw=1.6, label=name, **st.RT[name])
            if k == 0:
                h.append(l)
        ax.set_ylabel(lab)
        st.zero_line(ax)
        st.apply(ax)
        st.panel(ax, k)
        if k >= 2:
            ax.set_xlabel("local solar time [h]")
    fig.legend(handles=h, ncols=2, loc="outside lower center")
    return st.savefig(fig, expt, "fig21_open_ground_series")


def summary(expt):
    d = load(expt)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    g = d.groupby(["t", "cls", "rt"])[list(VARS)]
    mean, lo, hi = g.mean().unstack("rt"), g.min().unstack("rt"), g.max().unstack("rt")
    cols = {}
    for v in VARS:
        for rt, lab in zip(RTS, ("1D", "3D")):
            cols[f"{v}_{lab}"] = mean[v][rt]
            cols[f"{v}_{lab}_lo"], cols[f"{v}_{lab}_hi"] = lo[v][rt], hi[v][rt]
    c = pd.DataFrame(cols).reset_index()
    c.to_csv(res / "open_ground_classes.csv", index=False)
    diff = pd.DataFrame({f"{v}": mean[v][RTS[1]] - mean[v][RTS[0]] for v in VARS}).reset_index()
    diff.to_csv(res / "open_ground_diff.csv", index=False)
    return d, c, diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--series", action="store_true", help="hourly means over lit open ground from every file, and figure 21")
    ap.add_argument("--every", type=float, default=None, help="instead of --t: every N minutes of the 60 s fields within --solar")
    ap.add_argument("--solar", type=float, nargs=2, default=(11.9, 16.1))
    a = ap.parse_args()
    if a.every and a.rt:
        run = Run(run_dir(a.expt, a.rt, a.rep))
        ts = np.array(run.hf_times())
        step = int(round(a.every * 60. / (ts[1] - ts[0])))
        a.t = [int(t) for t in ts[::step] if a.solar[0] <= run.lst(t) < a.solar[1]]
    a.t = a.t or (list(snapshot_times(a.expt, a.rt, a.rep, skip_first=True)) if a.rt else None)
    if a.series:
        d, o = series(a.expt)
        print(o.to_string(float_format=lambda v: f"{v:.3g}"))
        print(figure21(a.expt))
        raise SystemExit
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 200)
    fmt = lambda v: f"{v:.3g}"
    if a.summary:
        d, c, diff = summary(a.expt)
        show = [x for x in c.columns if not x.endswith(("_lo", "_hi"))]
        print("--- by class (member means): fraction of the domain, mass divergence below 200 m [kg m-3 s-1], w at 0.5 and 0.9 z_b [m/s], "
              "surface-layer and upper-subcloud thl [K] and qt [g/kg] anomalies, LCL of the surface parcel [m], its thv excess at z_b [K] and the share positive")
        print(c[show].to_string(index=False, float_format=fmt))
        print("\n--- 3D minus 1D")
        print(diff.to_string(index=False, float_format=fmt))
    else:
        for t in a.t:
            ds = analyse(a.expt, a.rt, a.rep, t)
            d = ds.to_dataframe()
            print(f"{a.rt} rep_{a.rep:02d} t={t}: " + "; ".join(f"{k}: frac {r.frac:.2f} div {r.div_h:+.2e} w09 {r.w09:+.3f} qt_up {r.qt_up:+.2f} b_kb {r.b_kb:+.2f}" for k, r in d.iterrows()), flush=True)
