"""Gather snapshot and parcel files into tables with ensemble statistics (spread across members, ddof = 1).

python collect.py --expt no_aerosols_zero_wind_v2
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RTS = {"2stream": "1D", "raytracer": "3D"}


def scalars(snap, par, act=None):
    """List of (metric, zb_thr, value); zb_thr is NaN for threshold-independent metrics."""
    out = [(k, np.nan, float(snap.attrs[k])) for k in ("lst_solar", "clear_column_frac", "n_qi_points", "min_T_cloudy",
                                                        "zt_core")]
    thrs = snap["zb_thr"].values
    if int(snap["kb"].values.min()) < 0:
        return out
    for i, thr in enumerate(thrs):
        kb = int(snap["kb"][i])
        m = dict(zb=float(snap["zb"][i]), n_core_zb=float(snap["n_core_zb"][i]), n_cu_zb=float(snap["n_cu_zb"][i]),
                 frac_core_zb=float(snap["frac_core"][kb]), frac_cu_zb=float(snap["frac_cu"][kb]),
                 frac_cloudy_zb=float(snap["frac_cloudy"][kb]),
                 h_env_zb=float(snap["h_env"][kb]), hsat_env_zb=float(snap["hsat_env"][kb]),
                 thv_slab_zb=float(snap["thv_slab"][kb]))
        for v in ("h", "w", "qt", "thl", "thv", "T", "qc"):
            for tag in ("core", "cu"):
                m[f"{v}_{tag}_zb"] = float(snap[f"{v}_{tag}_zb"][i])
        for tag in ("core", "cu"):
            w = snap[f"pt_{tag}_w_{i}"].values
            for q in (10, 50, 90):
                m[f"w_{tag}_zb_p{q}"] = float(np.percentile(w, q))
        z = snap["z"].values
        for tag in ("root", "cu_root"):
            for v in ("thv", "h", "w", "thl", "qt"):
                for j, fr in enumerate(snap["root_frac"].values):
                    a = float(snap[f"{v}_{tag}_anom"][i, j])
                    k = int(np.argmin(np.abs(z - fr * z[kb])))
                    m[f"{v}_{tag}_anom_{fr:g}zb"] = a
                    m[f"{v}_{tag}_abs_{fr:g}zb"] = a + float(snap[f"{v}_slab"][k])
                    m[f"{v}_slab_{fr:g}zb"] = float(snap[f"{v}_slab"][k])
        for tag in ("core", "cu"):
            m[f"h_{tag}_minus_env_zb"] = m[f"h_{tag}_zb"] - m["h_env_zb"]
            m[f"h_{tag}_minus_hsat_env_zb"] = m[f"h_{tag}_zb"] - m["hsat_env_zb"]
            m[f"thv_{tag}_minus_slab_zb"] = m[f"thv_{tag}_zb"] - m["thv_slab_zb"]
        if par is not None and "CIN" in par:
            pi = par.isel(zb_thr=i)
            for k in ("eps_mean_qt", "eps_median_qt", "eps_int_qt", "eps_mean_thl", "eps_median_thl", "eps_int_thl",
                      "eps_parcel", "eps_z0", "eps_z1", "zt"):
                m[k] = float(pi[k])
            for v in ("G_max", "G_mean", "CIN", "z_LFC", "LFC_found", "w_crit", "CIN_thl_parcel",
                      "frac_w_above_wcrit"):
                for st in pi["start"].values:
                    for kd in pi["kind"].values:
                        m[f"{v}_{st}_{kd}"] = float(pi[v].sel(start=st, kind=kd))
        if act is not None:
            ai = act.isel(zb_thr=i)
            for st_ in ai["start"].values:
                for kd in ai["kind"].values:
                    a = ai.sel(start=st_, kind=kd)
                    m[f"f_active_{st_}_{kd}"] = float(a["f_active"])
                    m[f"f_buoyant_at_base_{st_}_{kd}"] = float(a["f_buoyant_at_base"])
                    m[f"CIN_median_{st_}_{kd}"] = float(a["CIN_median"])
                    m[f"S_w_{st_}_{kd}"] = float(a["S_w"])
                    for b in a["boost"].values:
                        m[f"S_h_{b}_{st_}_{kd}"] = float(a["S_h"].sel(boost=b))
        out += [(k, float(thr), v) for k, v in m.items()]
    for tag in ("w0", "wsig"):
        out.append((f"largest_updraft_frac_{tag}", np.nan, float(snap.attrs[f"largest_frac_{tag}"])))
        out.append((f"updraft_area_frac_{tag}", np.nan, float(snap.attrs[f"area_frac_{tag}"])))
        D = snap[f"root_D_{tag}"].values
        act = snap["cloud_has_core"].values > 0
        for sel, name in ((np.isfinite(D), "all"), (np.isfinite(D) & act, "active")):
            out.append((f"root_D_{tag}_{name}_median", np.nan, float(np.median(D[sel])) if sel.any() else np.nan))
            out.append((f"root_D_{tag}_{name}_mean", np.nan, float(D[sel].mean()) if sel.any() else np.nan))
    cd = snap["cloud_D"].values
    act = snap["cloud_has_core"].values > 0
    out += [("cloud_D_all_median", np.nan, float(np.median(cd))), ("n_cloud", np.nan, float(cd.size)),
            ("n_cloud_active", np.nan, float(act.sum())),
            ("cloud_D_active_median", np.nan, float(np.median(cd[act])) if act.any() else np.nan)]
    return out


def main(expt):
    root = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt
    rows = []
    for f in sorted(root.glob("*/rep_*/snap_*.nc")):
        with xr.open_dataset(f) as s:
            s = s.load()
        p = f.with_name(f.name.replace("snap_", "parcel_"))
        par = xr.open_dataset(p).load() if p.exists() else None
        a = f.with_name(f.name.replace("snap_", "activation_"))
        act = xr.open_dataset(a).load() if a.exists() else None
        for k, thr, v in scalars(s, par, act):
            rows.append(dict(t_sec=s.attrs["t_sec"], rt=RTS[s.attrs["rt"]], rep=s.attrs["rep"], metric=k,
                             zb_thr=thr, value=v))
    long = pd.DataFrame(rows)
    long.to_csv(root / "table_long.csv", index=False)
    g = long.fillna({"zb_thr": -1.}).groupby(["t_sec", "metric", "zb_thr", "rt"])["value"]
    summ = g.agg(mean="mean", std=lambda x: x.std(ddof=1), min="min", max="max", n="count").unstack("rt")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["diff_3D_minus_1D"] = summ["mean_3D"] - summ["mean_1D"]
    summ["se_diff"] = np.sqrt(summ["std_1D"]**2 / summ["n_1D"] + summ["std_3D"]**2 / summ["n_3D"])
    summ["diff_over_se"] = summ["diff_3D_minus_1D"] / summ["se_diff"]
    summ.to_csv(root / "table_summary.csv")
    print(f"wrote {root / 'table_long.csv'} ({len(long)} rows) and table_summary.csv ({len(summ)} rows)")
    key_table(summ.reset_index(), root / "table_key.md")
    return long, summ


KEY = [("zb", 1e-4, "cloud base z_b [m]", 1.), ("zb", 1e-3, "cloud base z_b [m]", 1.),
       ("h_core_zb", 1e-4, "core h at z_b [kJ/kg]", 1e-3), ("h_core_zb", 1e-3, "core h at z_b [kJ/kg]", 1e-3),
       ("w_core_zb", 1e-4, "core w at z_b [m/s]", 1.), ("w_core_zb", 1e-3, "core w at z_b [m/s]", 1.),
       ("h_cu_zb", 1e-3, "cloudy-updraft h at z_b [kJ/kg]", 1e-3), ("w_cu_zb", 1e-3, "cloudy-updraft w at z_b [m/s]", 1.),
       ("h_env_zb", 1e-3, "h_env at z_b [kJ/kg]", 1e-3), ("hsat_env_zb", 1e-3, "h*_env at z_b [kJ/kg]", 1e-3),
       ("G_max_core_undilute", 1e-4, "G max, undilute core [kJ/kg]", 1e-3),
       ("G_mean_core_undilute", 1e-4, "G layer mean, undilute core [kJ/kg]", 1e-3),
       ("G_mean_core_entraining", 1e-4, "G layer mean, entraining core [kJ/kg]", 1e-3),
       ("CIN_core_entraining", 1e-4, "CIN, entraining core [J/kg]", 1.),
       ("CIN_cu_entraining", 1e-3, "CIN, entraining cloudy updraft [J/kg]", 1.),
       ("w_crit_cu_entraining", 1e-3, "w_crit, cloudy updraft [m/s]", 1.),
       ("frac_w_above_wcrit_cu_entraining", 1e-3, "fraction of cloudy updraft w above w_crit", 1.),
       ("eps_int_qt", 1e-4, "eps from q_t, ratio of integrals [1/km]", 1e3),
       ("eps_int_thl", 1e-4, "eps from theta_l, ratio of integrals [1/km]", 1e3),
       ("eps_mean_qt", 1e-4, "eps from q_t, layer mean of local ratio [1/km]", 1e3),
       ("eps_mean_thl", 1e-4, "eps from theta_l, layer mean of local ratio [1/km]", 1e3),
       ("root_D_w0_active_median", -1., "root width, w > 0, median [m]", 1.),
       ("root_D_wsig_active_median", -1., "root width, w > sigma_w, median [m]", 1.),
       ("cloud_D_active_median", -1., "active cloud diameter, median [m]", 1.),
       ("n_cloud", -1., "number of clouds", 1.), ("largest_updraft_frac_w0", -1., "largest w > 0 region / domain", 1.),
       ("thv_cu_root_anom_0.5zb", 1e-3, "root theta_v anomaly at 0.5 z_b [K]", 1.),
       ("thv_cu_root_abs_0.5zb", 1e-3, "root theta_v at 0.5 z_b [K]", 1.),
       ("h_cu_root_anom_0.9zb", 1e-3, "root h anomaly at 0.9 z_b [kJ/kg]", 1e-3),
       ("h_cu_root_abs_0.9zb", 1e-3, "root h at 0.9 z_b [kJ/kg]", 1e-3),
       ("h_slab_0.5zb", 1e-3, "slab-mean h at 0.5 z_b [kJ/kg]", 1e-3),
       ("w_cu_root_anom_0.9zb", 1e-3, "root w anomaly at 0.9 z_b [m/s]", 1.),
       ("f_active_cu_entraining", 1e-3, "active fraction, cloudy updraft", 1.),
       ("f_active_cu_undilute", 1e-3, "active fraction, cloudy updraft, undilute", 1.),
       ("f_buoyant_at_base_cu_entraining", 1e-3, "buoyant at cloud base, cloudy updraft", 1.),
       ("f_active_core_entraining", 1e-3, "active fraction, core", 1.),
       ("S_w_cu_entraining", 1e-3, "sensitivity to w [per m/s]", 1.),
       ("S_h_moisture_cu_entraining", 1e-3, "sensitivity to moisture boost [per kJ/kg]", 1e3),
       ("S_h_heat_cu_entraining", 1e-3, "sensitivity to heat boost [per kJ/kg]", 1e3)]


def key_table(summ, path):
    """Markdown table of headline metrics; spread is the sample standard deviation over members."""
    L = ["| metric | z_b threshold | solar LT | 1D mean | 1D sd | 3D mean | 3D sd | 3D - 1D | SE | within 2 SE |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, thr, label, f in KEY:
        r = summ[(summ["metric"] == m) & np.isclose(summ["zb_thr"], thr)].sort_values("t_sec")
        for _, x in r.iterrows():
            lt = 3.891 + x["t_sec"] / 3600.
            d, se = x["diff_3D_minus_1D"] * f, x["se_diff"] * f
            inside = "yes" if (se > 0 and abs(d) < 2. * se) or d == 0 else "no"
            L.append(f"| {label} | {'-' if thr < 0 else f'{thr:g}'} | {lt:.2f} | {x['mean_1D']*f:.4g} | {x['std_1D']*f:.2g} | "
                     f"{x['mean_3D']*f:.4g} | {x['std_3D']*f:.2g} | {d:+.3g} | {se:.2g} | {inside} |")
    Path(path).write_text("\n".join(L) + "\n")
    print(f"wrote {path} ({len(L) - 2} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    main(ap.parse_args().expt)
