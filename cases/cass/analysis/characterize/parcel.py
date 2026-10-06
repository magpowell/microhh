"""Tasks 3 and 4b from snapshot files: bulk entrainment, undilute and entraining parcels, CIN, w_crit.

python parcel.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1
"""
import argparse

import numpy as np
import xarray as xr

import thermo as th
from snapshot import out_path, snapshot_times

DZ_OFF = 100.       # layer for eps: z_b + DZ_OFF .. z_t - DZ_OFF
FRAC_TOP = 1.e-3    # z_t = highest level with core fraction above this
G_DEPTH = 500.
Z_MAX = 4800.       # sponge starts here


def bulk_entrainment(z, phi_c, phi_env, k0, k1):
    """eps = -(d phi_c/dz)/(phi_c - phi_env): local profile, layer mean and median, ratio of layer integrals."""
    d = phi_c - phi_env
    loc = -np.gradient(phi_c, z) / d
    sl = slice(k0, k1 + 1)
    integ = -(phi_c[k1] - phi_c[k0]) / np.trapz(d[sl], z[sl])
    return loc, float(np.nanmean(loc[sl])), float(np.nanmedian(loc[sl])), float(integ)


def relax(z, env, x0, kb, ktop, eps):
    """Integrate dx/dz = -eps (x - env) upward from level kb."""
    x = np.full(z.size, np.nan)
    x[kb] = x0
    for k in range(kb, ktop):
        e = 0.5 * (env[k] + env[k + 1])
        x[k + 1] = e + (x[k] - e) * np.exp(-eps * (z[k + 1] - z[k]))
    return x


def buoyancy_profile(z, p, T_p, qv_p, qc_p, T_env, qv_env):
    tvp = th.virtual_temperature(T_p, qv_p, qc_p)
    tve = th.virtual_temperature(T_env, qv_env, 0.)
    return th.grav * (tvp - tve) / tve


def cin_lfc(z, B, kb, ktop):
    """Negative buoyancy area between z_b and the LFC. Returns CIN [J/kg], z_LFC, found."""
    if B[kb] > 0.:
        return 0., float(z[kb]), True
    pos = np.flatnonzero(B[kb:ktop + 1] > 0.)
    k1 = kb + pos[0] if pos.size else ktop
    cin = -np.trapz(np.minimum(B[kb:k1 + 1], 0.), z[kb:k1 + 1])
    return float(cin), float(z[k1]), bool(pos.size)


def lift_mse(ds, kb, ktop, h0, qt0, eps):
    z, p = ds["z"].values, ds["pref"].values
    hp = relax(z, ds["h_env"].values, h0, kb, ktop, eps)
    qp = relax(z, ds["qt_env"].values, qt0, kb, ktop, eps)
    sl = slice(kb, ktop + 1)
    T = np.full(z.size, np.nan)
    ql = np.full(z.size, np.nan)
    T[sl], ql[sl] = th.T_from_mse(hp[sl], qp[sl], z[sl], p[sl])
    B = buoyancy_profile(z, p, T, qp - ql, ql, ds["T_env"].values, ds["qv_env"].values)
    return hp, B


def lift_thl(ds, kb, ktop, thl0, qt0, eps):
    z, p, exn = ds["z"].values, ds["pref"].values, ds["exnref"].values
    tp = relax(z, ds["thl_env"].values, thl0, kb, ktop, eps)
    qp = relax(z, ds["qt_env"].values, qt0, kb, ktop, eps)
    sl = slice(kb, ktop + 1)
    a = th.sat_adjust(tp[sl], qp[sl], p[sl], exn[sl], tol=1.e-12, nitermax=50)
    T = np.full(z.size, np.nan); ql = np.full(z.size, np.nan)
    T[sl], ql[sl] = a["T"], a["ql"] + a["qi"]
    return buoyancy_profile(z, p, T, qp - ql, ql, ds["T_env"].values, ds["qv_env"].values)


def analyse_one(ds, izb):
    z = ds["z"].values
    kb = int(ds["kb"].values[izb])
    out = xr.Dataset(coords=dict(z=z, start=["core", "cu"], kind=["undilute", "entraining"]))
    top = np.flatnonzero(ds["frac_core"].values > FRAC_TOP)
    kt = int(top[-1])
    k0 = int(np.searchsorted(z, z[kb] + DZ_OFF))
    k1 = int(np.searchsorted(z, z[kt] - DZ_OFF, side="right") - 1)
    ktop = int(np.searchsorted(z, Z_MAX) - 1)
    for k, v in dict(zb=z[kb], zt=z[kt], eps_z0=z[k0], eps_z1=z[k1]).items():
        out[k] = float(v)

    eps = {}
    for v in ("qt", "thl"):
        loc, mean, med, integ = bulk_entrainment(z, ds[f"{v}_core"].values, ds[f"{v}_env"].values, k0, k1)
        out[f"eps_local_{v}"] = ("z", loc)
        out[f"eps_mean_{v}"], out[f"eps_median_{v}"], out[f"eps_int_{v}"] = mean, med, integ
        eps[v] = integ
    e = 0.5 * (eps["qt"] + eps["thl"])
    out["eps_parcel"] = e

    kg = int(np.searchsorted(z, z[kb] + G_DEPTH, side="right") - 1)
    hsat = ds["hsat_env"].values
    shape = (2, 2)
    G_max, G_mean, cin, lfc, found, wcrit, cin_thl, frac_above = (np.full(shape, np.nan) for _ in range(8))
    hp_all = np.full((2, 2, z.size), np.nan)
    B_all = np.full((2, 2, z.size), np.nan)
    for i, s in enumerate(("core", "cu")):
        h0, q0, t0 = (float(ds[f"{v}_{s}_zb"].values[izb]) for v in ("h", "qt", "thl"))
        w = ds[f"pt_{s}_w_{izb}"].values
        for j, ee in enumerate((0., e)):
            hp, B = lift_mse(ds, kb, ktop, h0, q0, ee)
            hp_all[i, j], B_all[i, j] = hp, B
            G = hsat[kb:kg + 1] - hp[kb:kg + 1]
            G_max[i, j], G_mean[i, j] = G.max(), G.mean()
            cin[i, j], lfc[i, j], found[i, j] = cin_lfc(z, B, kb, ktop)
            wcrit[i, j] = np.sqrt(2. * cin[i, j])
            cin_thl[i, j] = cin_lfc(z, lift_thl(ds, kb, ktop, t0, q0, ee), kb, ktop)[0]
            frac_above[i, j] = float((w > wcrit[i, j]).mean())
    d2 = ("start", "kind")
    for name, arr in (("G_max", G_max), ("G_mean", G_mean), ("CIN", cin), ("z_LFC", lfc), ("LFC_found", found),
                      ("w_crit", wcrit), ("CIN_thl_parcel", cin_thl), ("frac_w_above_wcrit", frac_above)):
        out[name] = (d2, arr)
    out["h_parcel"] = (d2 + ("z",), hp_all)
    out["B_parcel"] = (d2 + ("z",), B_all)
    return out


def analyse(ds):
    if int(ds["kb"].values.min()) < 0:
        out = xr.Dataset()
    else:
        out = xr.concat([analyse_one(ds, i) for i in range(ds.sizes["zb_thr"])],
                        dim=xr.DataArray(ds["zb_thr"].values, dims="zb_thr", name="zb_thr"))
    out.attrs.update({k: ds.attrs[k] for k in ("expt", "rt", "rep", "t_sec", "lst_solar")})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--t", type=int, nargs="+", default=None, help="default: the snapshot times of the run")
    a = ap.parse_args()
    a.t = a.t or list(snapshot_times(a.expt, a.rt, a.rep, skip_first=False))
    for t in a.t:
        src = out_path(a.expt, a.rt, a.rep, t)
        with xr.open_dataset(src) as ds:
            out = analyse(ds.load())
        out.to_netcdf(src.with_name(f"parcel_{int(t):07d}.nc"))
        if "CIN" in out:
            print(f"{a.rt} rep_{a.rep:02d} t={t} eps_int qt={out['eps_int_qt'].values} thl={out['eps_int_thl'].values} "
                  f"CIN(cu, entraining)={out['CIN'].sel(start='cu', kind='entraining').values}", flush=True)
