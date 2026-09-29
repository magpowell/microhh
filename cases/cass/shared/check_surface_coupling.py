"""Check that the atmosphere receives the fluxes of the land surface model and that the surface energy balance closes.

python check_surface_coupling.py --run <run dir> [--ref <run dir to compare with>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CP, LV = 1005., 2.501e6    # include/constants.h
UTC0 = 12.
FRAC_MIN = 1.e-3           # cloud base: lowest level with cloud fraction above this


def load(run):
    f = sorted(Path(run).glob("cass.default.*.nc"))[0]
    g = {k: xr.open_dataset(f, group=k, decode_times=False) for k in ("default", "thermo", "land_surface", "radiation")}
    c = xr.open_dataset(f, decode_times=False)
    c = c.assign_coords(time=np.round(c["time"].values)).coords
    return {k: v.assign_coords({n: c[n] for n in v.dims if n in c}) for k, v in g.items()}


def coupling(run):
    """Fluxes of the land model, fluxes the atmosphere receives, and the energy balance at the surface [W m-2]."""
    g = load(run)
    th, ls, rad = g["thermo"], g["land_surface"], g["radiation"].isel(zh=0)
    rho_dyn = float(g["default"]["rhorefh"][0])
    d = xr.Dataset(dict(H=ls["H"], LE=ls["LE"], G=ls["G"], S=ls["S"]))
    d["rho_thermo"] = th["rhoh"].isel(zh=0)
    d["H_atm"] = rho_dyn * CP * th["thl_flux"].isel(zh=0)
    d["LE_atm"] = rho_dyn * LV * th["qt_flux"].isel(zh=0)
    d["Qnet"] = rad["sw_flux_dn"] - rad["sw_flux_up"] + rad["lw_flux_dn"] - rad["lw_flux_up"]
    d["res_land"] = d["Qnet"] - (d["H"] + d["LE"] + d["G"] + d["S"])
    d["res_atm"] = d["Qnet"] - d["G"] - d["S"] - (d["H_atm"] + d["LE_atm"])
    d["bowen"] = d["H"] / d["LE"]
    d["bowen_atm"] = d["H_atm"] / d["LE_atm"]
    d.attrs["rho_dyn"] = rho_dyn
    return d.drop_vars("zh", errors="ignore")


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


def report(run, ref=None):
    d = coupling(run)
    day = d.where(d["H"] + d["LE"] > 50., drop=True)
    hours = [h for h in range(13, 26) if (h - UTC0) * 3600. in d["time"].values]
    pd.set_option("display.width", 200)
    print(f"\n{run}\ndensity of the dynamics at the surface: {d.attrs['rho_dyn']:.4f} kg m-3")
    t = hourly(d, hours)
    t["H_atm/H"], t["LE_atm/LE"] = t["H_atm"] / t["H"], t["LE_atm"] / t["LE"]
    print(t[["rho_thermo", "H", "H_atm", "LE", "LE_atm", "H_atm/H", "LE_atm/LE", "bowen", "Qnet", "G", "S",
             "res_land", "res_atm"]].to_string(float_format=lambda x: f"{x:.4f}"))
    for v, w in (("H", "H_atm"), ("LE", "LE_atm")):
        r = day[w] / day[v] - 1.
        print(f"{w}/{v} - 1 while H + LE > 50 W m-2: max abs {float(abs(r).max()):.3e}  (n = {r.size})")
    for v in ("res_land", "res_atm"):
        print(f"{v} while H + LE > 50 W m-2: mean {float(day[v].mean()):.3f}, max abs {float(abs(day[v]).max()):.3f} W m-2")
    if ref is None:
        return
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ref", type=Path, default=None)
    a = ap.parse_args()
    report(a.run, a.ref)
