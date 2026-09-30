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
from snapshot import out_path, run_dir
from suppression import far_from_cloud, cloud_mask
from widening import surface_sw

RTS = ("2stream", "raytracer")
TIMES = (32400, 36000, 39600)
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
    rd = run_dir(expt, rt, rep)
    src = out_path(expt, rt, rep, t)
    with xr.open_dataset(src) as s:
        kb = int(s["kb"].values[IZB])
        attrs = {a: s.attrs[a] for a in ("expt", "rt", "rep", "t_sec", "lst_solar")}
    bs = run.basestate(t)
    z, zh, dx, dy = run.z, run.zh, run.dx, run.dy
    f = {v: np.asarray(run.field(v, t), dtype=np.float32) for v in ("u", "v", "w", "thl", "qt", "ql", "qi")}
    qc = f["ql"] + f["qi"]
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        fr = int(np.argmin(np.abs(ds["time"].values.astype(float) - t)))
    sw = [surface_sw(rd, rt, g) for g in range(fr - LOOK, fr)]
    anomaly = np.mean([a - a.mean() for a in sw], axis=0)
    cl = classes(qc, dx, dy, anomaly)
    zb = z[kb]
    # flow
    div = layer_mean(mass_divergence(f["u"], f["v"], bs["rhoref"], dx, dy), z, 0., DIV_TOP)
    wf = mk.w_to_full(f["w"])
    w05 = wf[int(np.argmin(np.abs(z - 0.5 * zb)))]
    w09 = wf[int(np.argmin(np.abs(z - 0.9 * zb)))]
    # state
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
    # parcel buoyancy where it first condenses, against the column at that level (dry parcel there)
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
    ds.attrs.update(attrs, kb=kb, zb=float(zb), far=FAR, lit=LIT, div_top=DIV_TOP, sfc_top=SFC_TOP, upper_lo=UPPER[0], upper_hi=UPPER[1])
    ds.to_netcdf(src.with_name(f"open_ground_{int(t):07d}.nc"))
    return ds


VARS = ("frac", "div_h", "w05", "w09", "thl_sfc", "qt_sfc", "thl_up", "qt_up", "lcl", "lcl_minus_zb", "b_lcl", "b_kb", "b_kb_pos", "anomaly")


def load(expt):
    rows = []
    for t, rt, rep in itertools.product(TIMES, RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"open_ground_{t:07d}.nc")) as ds:
            d = ds.to_dataframe().reset_index()
            d["zb"], d["lst"] = float(ds.attrs["zb"]), float(ds.attrs["lst_solar"])
        d["t"], d["rt"], d["rep"] = t, rt, rep
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


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
    ap.add_argument("--t", type=int, nargs="+", default=list(TIMES))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
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
