"""Surface energy balance residual of ray tracer runs against the surface shortwave outputs of the same run.

python seb_residual.py --expt no_aerosols_zero_wind_v2 [--reps 1 2 3 4]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

import style as st
from les_io import Run
from style import plt


def load(run_dir):
    """Land surface fluxes and surface radiation of all statistics segments of a run [W m-2]."""
    out = []
    for f in sorted(Path(run_dir).glob("cass.default.*.nc")):
        time = np.round(xr.open_dataset(f, decode_times=False)["time"].values)
        rad, ls = (xr.open_dataset(f, group=g, decode_times=False) for g in ("radiation", "land_surface"))
        d = xr.Dataset({v: ls[v] for v in ("H", "LE", "G", "S")})
        for v in ("sw_flux_dn", "sw_flux_up", "sw_flux_dn_dir", "lw_flux_dn", "lw_flux_up"):
            d[v] = rad[v].isel(zh=0)
        for v in ("sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt", "sw_flux_sfc_up_rt"):
            if v in rad:
                d[v] = rad[v]
        out.append(d.drop_vars("zh", errors="ignore").assign_coords(time=time))
    d = xr.concat(out, "time") if len(out) > 1 else out[0]
    d = d.isel(time=np.unique(d["time"].values, return_index=True)[1])
    return d.assign_coords(lst=("time", Run(run_dir).lst(d["time"].values)))


def terms(d):
    """Residuals of the energy balance and, for a ray tracer run, two-stream minus ray-traced surface shortwave."""
    flux = d["H"] + d["LE"] + d["G"] + d["S"]
    lw = d["lw_flux_dn"] - d["lw_flux_up"]
    o = xr.Dataset(dict(residual_2s=d["sw_flux_dn"] - d["sw_flux_up"] + lw - flux))
    if "sw_flux_sfc_dir_rt" not in d:
        return o.assign(residual=o["residual_2s"])
    dn = d["sw_flux_sfc_dir_rt"] + d["sw_flux_sfc_dif_rt"]
    o["residual"] = dn - d["sw_flux_sfc_up_rt"] + lw - flux
    o["downwelling"] = d["sw_flux_dn"] - dn
    o["upwelling"] = d["sw_flux_up"] - d["sw_flux_sfc_up_rt"]
    o["net"] = o["downwelling"] - o["upwelling"]
    o["direct"] = d["sw_flux_dn_dir"] - d["sw_flux_sfc_dir_rt"]
    o["diffuse"] = d["sw_flux_dn"] - d["sw_flux_dn_dir"] - d["sw_flux_sfc_dif_rt"]
    return o


def members(expt, rt, reps):
    root = Path(os.environ["SCRATCH"]) / "CASS_LES" / "experiments" / expt / rt
    m = [terms(load(root / f"rep_{r:02d}")) for r in reps]
    n = min(x.sizes["time"] for x in m)
    return xr.concat([x.isel(time=slice(n)) for x in m], "member")


def line(ax, m, v, label, **kw):
    x = m["lst"]
    ax.fill_between(x, m[v].min("member"), m[v].max("member"), color=kw["color"], alpha=0.25, lw=0)
    ax.plot(x, m[v].mean("member"), label=label, **kw)


def figure(expt, reps):
    m3, m1 = members(expt, "raytracer", reps), members(expt, "2stream", reps)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    line(axs[0], m3, "residual_2s", "residual", **st.RESIDUAL)
    for v, kw in st.SW_PARTS.items():
        line(axs[0], m3, v, v, lw=1.8, **kw)
    axs[0].set_ylabel("domain-mean surface flux [W m$^{-2}$]")
    axs[0].legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=1.0, handlelength=1.8)
    line(axs[1], m1, "residual", "1D", lw=1.8, **st.RT["1D"])
    line(axs[1], m3, "residual", "3D", lw=1.8, **st.RT["3D"])
    axs[1].set_ylabel("domain-mean residual [W m$^{-2}$]")
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    for k, ax in enumerate(axs):
        st.apply(ax)
        st.zero_line(ax)
        st.panel(ax, k)
        ax.set_xlabel("local time [h]")
        ax.set_xlim(5., 18.)
        ax.set_xticks(np.arange(6, 19, 2))
    return st.savefig(fig, expt, "seb_residual_raytracer"), m3, m1


def summary(m3, m1):
    day = (m3["lst"] > 6.5) & (m3["lst"] < 17.5)
    r = m3["residual_2s"].mean("member")[day]
    print(f"3D, residual with the two-stream net radiation: {float(r.min()):.2f} to {float(r.max()):.2f} W m-2")
    for v in st.SW_PARTS:
        c = m3[v].mean("member")[day]
        print(f"  two-stream minus ray-traced {v:12s}: rms difference from that residual {float(np.sqrt(((r - c)**2).mean())):6.2f},"
              f" correlation {float(np.corrcoef(r, c)[0, 1]):6.3f}")
    for name, m in (("3D", m3), ("1D", m1)):
        x = m["residual"].mean("member")[day]
        print(f"{name}, residual with the net radiation of the land model: mean {float(x.mean()):.2f}, "
              f"largest {float(abs(x).max()):.2f} W m-2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()
    f, m3, m1 = figure(a.expt, a.reps)
    summary(m3, m1)
    print(f)
