"""Per-snapshot diagnostics for one run: environment, core, cloud base, root width (tasks 1, 2, 4 inputs).

python snapshot.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1 --t 32400
Output: $SCRATCH/CASS_LES/analysis/characterize/<expt>/<rt>/rep_NN/snap_<t>.nc
"""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

import masks as mk
import thermo as th
from les_io import Run

ZB_THRS = (1.e-4, 5.e-4, 1.e-3)
W_BINS = np.arange(0., 12.05, 0.1)
ROOT_FRACS = (0.5, 0.9)


def run_dir(expt, rt, rep):
    return Path(os.environ["SCRATCH"]) / "CASS_LES" / "experiments" / expt / rt / f"rep_{rep:02d}"


def out_path(expt, rt, rep, t):
    return (Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt / rt
            / f"rep_{rep:02d}" / f"snap_{int(t):07d}.nc")


def load(run, t):
    f = {v: np.asarray(run.field(v, t), dtype=float) for v in ("T", "ql", "qi", "qt", "thl", "w")}
    bs = run.basestate(t)
    k3 = (slice(None), None, None)
    f["w"] = mk.w_to_full(f["w"])
    f["qv"] = f["qt"] - f["ql"] - f["qi"]
    f["qc"] = f["ql"] + f["qi"]
    f["thv"] = th.theta_v(f["thl"], f["qt"], f["ql"], f["qi"], bs["exnref"][k3])
    f["h"] = th.mse(f["T"], run.z[k3], f["qv"])
    f["hsat"] = th.mse_sat(f["T"], run.z[k3], bs["pref"][k3])
    return f, bs


def profiles(f, m3, col, names):
    """Level means over a 3D mask (m3) or a 2D column mask (col)."""
    out = {}
    for v in names:
        out[v] = mk.masked_mean(f[v], m3)[0] if m3 is not None else mk.column_mean(f[v], col)
    return out


def root_width(f, run, kb, ql_thr):
    """Task 4a: per-cloud equivalent diameter of the updraft region beneath it at 0.9 z_b."""
    k09 = int(np.argmin(np.abs(run.z - 0.9 * run.z[kb])))
    clab, nc = mk.label_periodic(f["qc"].max(axis=0) > ql_thr)
    w09 = f["w"][k09]
    sig = w09.std()
    res = dict(k09=k09, z09=run.z[k09], sigma_w09=sig, n_cloud=nc)
    carea = mk.object_areas(clab, nc)
    res["cloud_D"] = mk.equivalent_diameter(carea, run.dx, run.dy)
    res["cloud_has_core"] = np.bincount(clab.ravel(), weights=f["core"].any(axis=0).ravel(), minlength=nc + 1)[1:] > 0
    for tag, thr in (("w0", 0.), ("wsig", sig)):
        rlab, nr = mk.label_periodic(w09 > thr)
        rarea = mk.object_areas(rlab, nr)
        res[f"largest_frac_{tag}"] = rarea.max() / w09.size if nr else 0.
        res[f"area_frac_{tag}"] = (w09 > thr).mean()
        D = np.full(nc, np.nan)
        ov = np.zeros(nc)
        sel = (clab > 0) & (rlab > 0)
        if sel.any():
            pair, cnt = np.unique(np.stack([clab[sel], rlab[sel]]), axis=1, return_counts=True)
            for c in np.unique(pair[0]):
                j = pair[0] == c
                best = pair[1][j][np.argmax(cnt[j])]
                D[c - 1] = mk.equivalent_diameter(rarea[best - 1], run.dx, run.dy)
                ov[c - 1] = cnt[j].max() / carea[c - 1]
        res[f"root_D_{tag}"] = D
        res[f"root_overlap_{tag}"] = ov
    return res


def analyse(expt, rt, rep, t, ql_thr=0.):
    run = Run(run_dir(expt, rt, rep))
    f, bs = load(run, t)
    z = run.z
    cloudy = f["qc"] > ql_thr
    f["core"] = mk.core(f["qc"], f["w"], f["thv"], ql_thr)
    cu = mk.cloudy_updraft(f["qc"], f["w"], ql_thr)
    col = mk.clear_columns(f["qc"], ql_thr)
    names = ("h", "hsat", "T", "qv", "qt", "thl", "thv", "w", "qc")

    ds = xr.Dataset(coords=dict(z=z, zb_thr=list(ZB_THRS), w_bin=0.5 * (W_BINS[:-1] + W_BINS[1:]),
                                root_frac=list(ROOT_FRACS)))
    ds.attrs.update(expt=expt, rt=rt, rep=rep, t_sec=int(t), lst_solar=float(run.lst(t)), ql_thr=ql_thr,
                    clear_column_frac=float(col.mean()), n_qi_points=int((f["qi"] > 0).sum()),
                    min_T_cloudy=float(f["T"][cloudy].min()) if cloudy.any() else np.nan,
                    datetime_utc=str(run.t0), dx=run.dx, dz=float(z[1] - z[0]))
    for k in ("pref", "exnref", "thvref", "rhoref"):
        ds[k] = ("z", bs[k])
    for tag, m3, c2 in (("env", None, col), ("slab", None, np.ones_like(col)), ("core", f["core"], None),
                        ("cu", cu, None), ("cloudy", cloudy, None)):
        for v, p in profiles(f, m3, c2, names).items():
            ds[f"{v}_{tag}"] = ("z", p)
    ds["hsat_of_Tenv"] = ("z", th.mse_sat(ds["T_env"].values, z, bs["pref"]))
    for tag, m3 in (("core", f["core"]), ("cu", cu), ("cloudy", cloudy)):
        ds[f"frac_{tag}"] = ("z", m3.mean(axis=(1, 2)))
    ds["w2_slab"] = ("z", (f["w"] ** 2).mean(axis=(1, 2)))

    kbs = [mk.cloud_base_index(ds["frac_core"].values, thr) for thr in ZB_THRS]
    ds["kb"] = ("zb_thr", kbs)
    ds["zb"] = ("zb_thr", [z[k] if k >= 0 else np.nan for k in kbs])
    kt = np.flatnonzero(ds["frac_core"].values > ZB_THRS[0])
    ds.attrs["zt_core"] = float(z[kt[-1]]) if kt.size else np.nan
    for v in names:
        for tag in ("core", "cu"):
            ds[f"{v}_{tag}_zb"] = ("zb_thr", [float(ds[f"{v}_{tag}"].values[k]) if k >= 0 else np.nan for k in kbs])
    ds["n_core_zb"] = ("zb_thr", [int(f["core"][k].sum()) if k >= 0 else 0 for k in kbs])
    ds["n_cu_zb"] = ("zb_thr", [int(cu[k].sum()) if k >= 0 else 0 for k in kbs])
    for tag, m3 in (("core", f["core"]), ("cu", cu)):
        ds[f"w_{tag}_pdf"] = (("zb_thr", "w_bin"), np.array(
            [np.histogram(f["w"][k][m3[k]], bins=W_BINS)[0] if k >= 0 else np.zeros(W_BINS.size - 1) for k in kbs]))

    if min(kbs) >= 0:
        for i, k0 in enumerate(kbs):
            for tag, m2 in (("core", f["core"][k0]), ("cu", cu[k0])):
                for v in ("h", "w", "thl", "qt", "thv"):
                    ds[f"pt_{tag}_{v}_{i}"] = (f"pt_{tag}_{i}", f[v][k0][m2])
        for tag, m3 in (("root", f["core"]), ("cu_root", cu)):
            for v in ("thv", "h", "w", "thl", "qt"):
                anom = np.empty((len(kbs), len(ROOT_FRACS)))
                for i, k0 in enumerate(kbs):
                    for j, fr in enumerate(ROOT_FRACS):
                        k = int(np.argmin(np.abs(z - fr * z[k0])))
                        anom[i, j] = f[v][k][m3[k0]].mean() - f[v][k].mean()
                ds[f"{v}_{tag}_anom"] = (("zb_thr", "root_frac"), anom)
        kb = kbs[0]
        rw = root_width(f, run, kb, ql_thr)
        for k, v in rw.items():
            if np.ndim(v) == 0:
                ds.attrs[k] = float(v)
            else:
                ds[k] = ("cloud", np.asarray(v, dtype=float))
    out = out_path(expt, rt, rep, t)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, nargs="+", default=[28800, 32400, 36000, 39600])
    ap.add_argument("--ql-thr", type=float, default=0.)
    a = ap.parse_args()
    for t in a.t:
        d = analyse(a.expt, a.rt, a.rep, t, a.ql_thr)
        print(f"{a.rt} rep_{a.rep:02d} t={t} LST={d.attrs['lst_solar']:.2f} zb={d['zb'].values} "
              f"n_core_zb={d['n_core_zb'].values} h_core_zb={d['h_core_zb'].values[0]:.1f} "
              f"w_core_zb={d['w_core_zb'].values[0]:.3f}", flush=True)
