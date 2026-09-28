"""Per-parcel activation at cloud base: fraction of parcels whose kinetic energy exceeds their own CIN,
and its response to uniform boosts in moisture, heat and vertical velocity.

python activation.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
"""
import argparse

import numpy as np
import xarray as xr

import thermo as th
from parcel import Z_MAX, buoyancy_profile
from snapshot import out_path

DH = np.round(np.arange(-1000., 1000.1, 100.), 6)   # J/kg
DW = np.round(np.arange(-1., 1.01, 0.1), 6)         # m/s


def cin_lfc_many(z, B):
    """Vectorised cin_lfc for B of shape (parcel, level), starting at level 0. Returns CIN, found."""
    neg = np.minimum(B, 0.)
    C = np.concatenate([np.zeros((B.shape[0], 1)),
                        np.cumsum(0.5 * (neg[:, 1:] + neg[:, :-1]) * np.diff(z)[None, :], axis=1)], axis=1)
    pos = B > 0.
    found = pos.any(axis=1)
    k1 = np.where(found, pos.argmax(axis=1), B.shape[1] - 1)
    cin = -C[np.arange(B.shape[0]), k1]
    return np.where(B[:, 0] > 0., 0., cin), found


def lift_many(ds, kb, ktop, h0, qt0, eps):
    """Lift parcels (h0, qt0 arrays) from level kb; returns B of shape (parcel, level kb..ktop)."""
    z, p = ds["z"].values[kb:ktop + 1], ds["pref"].values[kb:ktop + 1]
    he, qe = ds["h_env"].values[kb:ktop + 1], ds["qt_env"].values[kb:ktop + 1]
    n = z.size
    h = np.empty((h0.size, n))
    q = np.empty((h0.size, n))
    h[:, 0], q[:, 0] = h0, qt0
    for k in range(n - 1):
        a = np.exp(-eps * (z[k + 1] - z[k]))
        h[:, k + 1] = 0.5 * (he[k] + he[k + 1]) * (1. - a) + h[:, k] * a
        q[:, k + 1] = 0.5 * (qe[k] + qe[k + 1]) * (1. - a) + q[:, k] * a
    T, ql = th.T_from_mse(h, q, z[None, :], p[None, :])
    return z, buoyancy_profile(z, p, T, q - ql, ql, ds["T_env"].values[kb:ktop + 1][None, :],
                               ds["qv_env"].values[kb:ktop + 1][None, :])


def active_fraction(w, cin, found):
    return float(((0.5 * w**2 > cin) & found).mean())


def analyse_one(ds, par, izb, tag):
    kb = int(ds["kb"].values[izb])
    ktop = int(np.searchsorted(ds["z"].values, Z_MAX) - 1)
    h0, q0, w = (ds[f"pt_{tag}_{v}_{izb}"].values for v in ("h", "qt", "w"))
    eps = float(par["eps_parcel"].isel(zb_thr=izb))
    out = xr.Dataset(coords=dict(kind=["undilute", "entraining"], dh=DH, dw=DW, boost=["moisture", "heat"]))
    f0, fb, cm, fw, fh = np.empty(2), np.empty(2), np.empty(2), np.empty((2, DW.size)), np.empty((2, 2, DH.size))
    for j, e in enumerate((0., eps)):
        z, B = lift_many(ds, kb, ktop, h0, q0, e)
        cin, found = cin_lfc_many(z, B)
        f0[j], fb[j], cm[j] = active_fraction(w, cin, found), float((B[:, 0] > 0.).mean()), float(np.median(cin))
        fw[j] = [active_fraction(np.maximum(w + d, 0.), cin, found) for d in DW]
        for b, moist in enumerate((True, False)):
            for i, d in enumerate(DH):
                zz, Bd = lift_many(ds, kb, ktop, h0 + d, q0 + (d / th.Lv if moist else 0.), e)
                c, fd = cin_lfc_many(zz, Bd)
                fh[j, b, i] = active_fraction(w, c, fd)
    i0, j0 = int(np.argmin(np.abs(DH))), int(np.argmin(np.abs(DW)))
    out["f_active"] = ("kind", f0)
    out["f_buoyant_at_base"] = ("kind", fb)
    out["CIN_median"] = ("kind", cm)
    out["f_active_dw"] = (("kind", "dw"), fw)
    out["f_active_dh"] = (("kind", "boost", "dh"), fh)
    out["S_w"] = ("kind", (fw[:, j0 + 1] - fw[:, j0 - 1]) / (DW[j0 + 1] - DW[j0 - 1]))
    out["S_h"] = (("kind", "boost"), (fh[:, :, i0 + 1] - fh[:, :, i0 - 1]) / (DH[i0 + 1] - DH[i0 - 1]))
    out["n_parcel"] = h0.size
    return out


def analyse(ds, par):
    thr = xr.DataArray(ds["zb_thr"].values, dims="zb_thr", name="zb_thr")
    pop = xr.DataArray(["cu", "core"], dims="start", name="start")
    out = xr.concat([xr.concat([analyse_one(ds, par, i, t) for i in range(ds.sizes["zb_thr"])], dim=thr)
                     for t in ("cu", "core")], dim=pop)
    out.attrs.update({k: ds.attrs[k] for k in ("expt", "rt", "rep", "t_sec", "lst_solar")})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, nargs="+", default=[28800, 32400, 36000, 39600])
    a = ap.parse_args()
    for t in a.t:
        src = out_path(a.expt, a.rt, a.rep, t)
        with xr.open_dataset(src) as ds, xr.open_dataset(src.with_name(f"parcel_{int(t):07d}.nc")) as par:
            out = analyse(ds.load(), par.load())
        out.to_netcdf(src.with_name(f"activation_{int(t):07d}.nc"))
        c = out.sel(start="cu", kind="entraining").isel(zb_thr=2)
        print(f"{a.rt} rep_{a.rep:02d} t={t} n={int(c['n_parcel'])} f_active={float(c['f_active']):.3f} "
              f"f_buoyant0={float(c['f_buoyant_at_base']):.3f} S_w={float(c['S_w']):.3f} per m/s "
              f"S_h(moist)={float(c['S_h'].sel(boost='moisture'))*1e3:.3f} per kJ/kg", flush=True)
