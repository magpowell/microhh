"""Check that the atmosphere receives the fluxes of the land surface model and that the surface energy balance closes.

Net radiation is the one the land surface model receives. In a ray tracer run that is the ray-traced surface
shortwave (time series sw_flux_sfc_dir_rt, sw_flux_sfc_dif_rt, sw_flux_sfc_up_rt); the profiles sw_flux_dn and
sw_flux_up of such a run are the two-stream fluxes the model computes alongside and are not used by the land model.

python check_surface_coupling.py --run <run dir> [--ref <run dir to compare with>] [--segment N]
python check_surface_coupling.py --run <run dir with prescribed fluxes> --prescribed
Exit status 1 when the atmosphere does not receive the fluxes of the land surface model (tolerance TOL).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CP, LV = 1005., 2.501e6    # include/constants.h
UTC0 = 12.
FRAC_MIN = 1.e-3           # cloud base: lowest level with cloud fraction above this
TOL = 1.e-12               # received over reported flux, minus one (rounding gives about 1e-15)
TOL_ABS = 1.e-9            # received minus reported flux [W m-2]


def load(run, segment=0):
    """Statistics of one segment of a run (0 is the start, later ones follow restarts)."""
    f = sorted(Path(run).glob("cass.default.*.nc"))[segment]
    import netCDF4
    with netCDF4.Dataset(f) as nc:
        groups = [k for k in ("default", "thermo", "land_surface", "radiation") if k in nc.groups]
    g = {k: xr.open_dataset(f, group=k, decode_times=False) for k in groups}
    c = xr.open_dataset(f, decode_times=False)
    c = c.assign_coords(time=np.round(c["time"].values)).coords
    return {k: v.assign_coords({n: c[n] for n in v.dims if n in c}) for k, v in g.items()}


def coupling(run, segment=0):
    """Fluxes of the land model, fluxes the atmosphere receives, and the energy balance at the surface [W m-2]."""
    g = load(run, segment)
    th, ls, rad = g["thermo"], g["land_surface"], g["radiation"].isel(zh=0)
    rho_dyn = float(g["default"]["rhorefh"][0])
    d = xr.Dataset(dict(H=ls["H"], LE=ls["LE"], G=ls["G"], S=ls["S"]))
    d["rho_thermo"] = th["rhoh"].isel(zh=0)
    d["H_atm"] = rho_dyn * CP * th["thl_flux"].isel(zh=0)
    d["LE_atm"] = rho_dyn * LV * th["qt_flux"].isel(zh=0)
    lw = rad["lw_flux_dn"] - rad["lw_flux_up"]
    d["Qnet_2s"] = rad["sw_flux_dn"] - rad["sw_flux_up"] + lw
    if "sw_flux_sfc_dir_rt" in rad:
        d["Qnet"] = rad["sw_flux_sfc_dir_rt"] + rad["sw_flux_sfc_dif_rt"] - rad["sw_flux_sfc_up_rt"] + lw
        d.attrs["shortwave"] = "ray-traced"
    else:
        d["Qnet"] = d["Qnet_2s"]
        d.attrs["shortwave"] = "two-stream"
    d["res_2s"] = d["Qnet_2s"] - (d["H"] + d["LE"] + d["G"] + d["S"])
    d["res_land"] = d["Qnet"] - (d["H"] + d["LE"] + d["G"] + d["S"])
    d["res_atm"] = d["Qnet"] - d["G"] - d["S"] - (d["H_atm"] + d["LE_atm"])
    d["bowen"] = d["H"] / d["LE"]
    d["bowen_atm"] = d["H_atm"] / d["LE_atm"]
    d.attrs["rho_dyn"] = rho_dyn
    return d.drop_vars("zh", errors="ignore")


def prescribed(run, segment=0):
    """Fluxes the atmosphere receives in a run with prescribed fluxes, against the CASS table of the run folder."""
    g = load(run, segment)
    rho_dyn = float(g["default"]["rhorefh"][0])
    tab = np.loadtxt(Path(run) / "cass_sfc.txt", skiprows=1)
    t = g["thermo"]["time"].values
    ts = (tab[:, 0] - (205. + UTC0 / 24.)) * 86400.
    d = xr.Dataset(dict(H_atm=rho_dyn * CP * g["thermo"]["thl_flux"].isel(zh=0),
                        LE_atm=rho_dyn * LV * g["thermo"]["qt_flux"].isel(zh=0)))
    d["H"], d["LE"] = ("time", np.interp(t, ts, tab[:, 2])), ("time", np.interp(t, ts, tab[:, 3]))
    return d.drop_vars("zh", errors="ignore")


def report_prescribed(run, segment=0):
    d = prescribed(run, segment)
    hours = [h for h in range(13, 26) if (h - UTC0) * 3600. in d["time"].values]
    t = hourly(d, hours)
    t["H_atm/H"], t["LE_atm/LE"] = t["H_atm"] / t["H"], t["LE_atm"] / t["LE"]
    print(f"\n{run}\nprescribed fluxes: received by the atmosphere against the CASS table")
    print(t[["H", "H_atm", "LE", "LE_atm", "H_atm/H", "LE_atm/LE"]].to_string(float_format=lambda x: f"{x:.4f}"))
    err = max(float(abs(d[f"{v}_atm"] - d[v]).max()) for v in ("H", "LE"))
    print(f"largest difference from the table: {err:.3e} W m-2")
    return err


def clouds(run):
    th = load(run)["thermo"]
    frac = th["ql_frac"]
    base = frac["z"].where(frac > FRAC_MIN).min("z")
    top = frac["z"].where(frac > FRAC_MIN).max("z")
    return xr.Dataset(dict(cover=th["ql_cover"], lwp=th["ql_path"], base=base, top=top))


def onset(c):
    """First sample time with cloud fraction above FRAC_MIN at any level [UTC hours]."""
    t = c["time"].where(c["base"].notnull(), drop=True)
    return UTC0 + float(t[0]) / 3600. if t.size else np.nan


def hourly(d, hours):
    return d.sel(time=[(h - UTC0) * 3600. for h in hours]).to_dataframe().set_axis(hours).rename_axis("UTC")


def report(run, ref=None, segment=0):
    d = coupling(run, segment)
    day = d.isel(time=np.flatnonzero((d["H"] + d["LE"] > 50.).values))
    hours = [h for h in range(13, 26) if (h - UTC0) * 3600. in d["time"].values]
    pd.set_option("display.width", 200)
    print(f"\n{run}\ndensity of the dynamics at the surface: {d.attrs['rho_dyn']:.4f} kg m-3; "
          f"net radiation of the land model: {d.attrs['shortwave']} shortwave")
    if hours:
        t = hourly(d, hours)
        t["H_atm/H"], t["LE_atm/LE"] = t["H_atm"] / t["H"], t["LE_atm"] / t["LE"]
        print(t[["rho_thermo", "H", "H_atm", "LE", "LE_atm", "H_atm/H", "LE_atm/LE", "bowen", "Qnet", "G", "S",
                 "res_land", "res_atm"]].to_string(float_format=lambda x: f"{x:.4f}"))
    if day.sizes["time"]:
        for v, w in (("H", "H_atm"), ("LE", "LE_atm")):
            r = day[w] / day[v] - 1.
            print(f"{w}/{v} - 1 while H + LE > 50 W m-2: max abs {float(abs(r).max()):.3e}  (n = {r.size})")
        for v in ("res_land", "res_atm") + (("res_2s",) if d.attrs["shortwave"] == "ray-traced" else ()):
            print(f"{v} while H + LE > 50 W m-2: mean {float(day[v].mean()):.3f}, max abs {float(abs(day[v]).max()):.3f} W m-2")
    ex = (d["H_atm"] + d["LE_atm"] - d["H"] - d["LE"]).integrate("time")
    ew = ((d["LE_atm"] - d["LE"]) / LV).integrate("time")
    print(f"received minus reported over the segment: {float(ex) / 1.e6:.4f} MJ m-2 of energy, {float(ew):.5f} kg m-2 of water")
    big = abs(d["H"]) > 5.
    err = (max(float(abs(d["H_atm"] / d["H"] - 1.).where(big).max()), float(abs(d["LE_atm"] / d["LE"] - 1.).where(big).max()))
           if bool(big.any()) else 0.)
    err_abs = max(float(abs(d["H_atm"] - d["H"]).max()), float(abs(d["LE_atm"] - d["LE"]).max()))
    ok = err < TOL and err_abs < TOL_ABS
    print(f"{'PASS' if ok else 'FAIL'}: largest |received / reported - 1| while |H| > 5 W m-2 is {err:.3e} (tolerance {TOL:.0e}); "
          f"largest |received - reported| is {err_abs:.3e} W m-2 (tolerance {TOL_ABS:.0e})")
    if ref is None:
        return ok
    e, c, cr = coupling(ref), clouds(run), clouds(ref)
    print(f"\nagainst {ref}")
    print(f"first cloud [UTC]: {onset(c):.3f} (this run), {onset(cr):.3f} (reference)")
    a, b, ca, cb = hourly(d, hours), hourly(e, hours), hourly(c, hours), hourly(cr, hours)
    t = pd.DataFrame({"H+LE atm": a["H_atm"] + a["LE_atm"], "ref": b["H_atm"] + b["LE_atm"],
                      "bowen": a["bowen"], "bowen ref": b["bowen"],
                      "base": ca["base"], "base ref": cb["base"], "top": ca["top"], "top ref": cb["top"],
                      "cover": ca["cover"], "cover ref": cb["cover"],
                      "lwp": ca["lwp"] * 1.e3, "lwp ref": cb["lwp"] * 1.e3})
    t.insert(2, "ratio", t["H+LE atm"] / t["ref"])
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--prescribed", action="store_true")
    a = ap.parse_args()
    if a.prescribed:
        report_prescribed(a.run, a.segment)
        raise SystemExit(0)
    raise SystemExit(0 if report(a.run, a.ref, a.segment) else 1)
