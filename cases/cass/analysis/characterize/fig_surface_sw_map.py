"""Figure: surface shortwave minus its clear-sky value at one time, 1D and 3D, with the cloud outlines.

python fig_surface_sw_map.py [--solar 14.5] [--rep 2]
The clear-sky value is the surface downward clear-sky flux of each run's own statistics at that time.
"""
import argparse

import netCDF4
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

import style as st
from les_io import Run
from snapshot import run_dir
from style import plt
from widening import surface_sw

EXPT = "no_aerosols_zero_wind_v3"
RTS = (("2stream", "1D"), ("raytracer", "3D"))
VLIM = 400.  # W m-2


def fields(rt, rep, solar):
    """x, y [km], surface shortwave minus clear sky, cloud water path, clear-sky value and solar time of the frame."""
    rd = run_dir(EXPT, rt, rep)
    run = Run(rd)
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        time = ds["time"].values.astype(float)
        f = int(np.argmin(np.abs(run.lst(time) - solar)))
        path, x, y = ds["qlqi_path"].isel(time=f).values, ds["x"].values / 1000., ds["y"].values / 1000.
    with netCDF4.Dataset(rd / "cass.default.0000000.nc") as s:
        k = int(np.argmin(np.abs(s["time"][:] - time[f])))
        clear = float(s["radiation"]["sw_flux_dn_clear"][k, 0])
    return x, y, surface_sw(rd, rt, f) - clear, path, clear, float(run.lst(time[f]))


def figure(solar=14.5, rep=2):
    fig, axs = plt.subplots(1, 2, figsize=(9., 4.9), sharey=True, layout="constrained")
    for k, ((rt, lab), ax) in enumerate(zip(RTS, axs)):
        x, y, dsw, path, clear, lst = fields(rt, rep, solar)
        im = ax.pcolormesh(x, y, dsw, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, rasterized=True)
        ax.contour(x, y, (path > 0.).astype(float), levels=[0.5], colors="k", linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [km]")
        ax.tick_params(labelsize=8)
        st.panel(ax, k, lab)
        print(f"{lab}: {st.lt(lst)}, clear-sky surface shortwave {clear:.0f} W m-2, domain mean minus clear sky {dsw.mean():.1f}, "
              f"range {dsw.min():.0f} to {dsw.max():.0f}; under cloud {dsw[path > 0.].mean():.0f}, cloud-free ground {dsw[path <= 0.].mean():.1f}")
    axs[0].set_ylabel("y [km]")
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.6, pad=0.02, extend="both")
    cb.set_label("surface shortwave $-$ clear sky [W m$^{-2}$]")
    fig.legend(handles=[Line2D([], [], color="k", lw=0.8, label="cloud")], loc="outside lower left")
    fig.suptitle(st.lt(lst), x=0.02, ha="left", fontsize=11)
    return st.savefig(fig, EXPT, "fig_surface_sw_map")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solar", type=float, default=14.5)
    ap.add_argument("--rep", type=int, default=2)
    a = ap.parse_args()
    print("wrote", figure(a.solar, a.rep))
