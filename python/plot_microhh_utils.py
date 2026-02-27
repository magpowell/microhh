"""
Utilities for loading and plotting MicroHH LES output.

Works with any output directory (2-stream, raytracer, etc.) —
just pass the path containing the .nc files.

Usage:
    from microhh_utils import load_xy, load_3d, plot_cross_sections, plot_flux_sections
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configparser
from datetime import datetime, timezone
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from pathlib import Path

Lv = 2.5e6
Ls = 2.834e6
cp = 1005.0
FILL_VALUE = -1.0e9

# 3D variables used for cross-section / flux analysis
DEFAULT_3D_VARS = ["thl", "T", "qt", "ql", "qi", "b", "w"]


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def get_local_start(data_dir):
    """Parse the .ini file in *data_dir* and return start time as local pd.Timestamp."""
    ini_files = list(Path(data_dir).glob("*.ini"))
    if not ini_files:
        return None
    config = configparser.ConfigParser()
    config.read(ini_files[0])

    if "time" not in config or "datetime_utc" not in config["time"]:
        return None

    start_utc = datetime.strptime(
        config["time"]["datetime_utc"], "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)

    lat = float(config["grid"]["lat"])
    lon = float(config["grid"]["lon"])

    try:
        from timezonefinder import TimezoneFinder
        tz_name = TimezoneFinder().timezone_at(lng=lon, lat=lat)
        local_tz = ZoneInfo(tz_name)
    except ImportError:
        offset_hours = round(lon / 15)
        local_tz = timezone(pd.Timedelta(hours=offset_hours))

    return pd.Timestamp(start_utc.astimezone(local_tz))


def _apply_local_time(ds, data_dir):
    """Replace the time coordinate (seconds from start) with local datetime."""
    start = get_local_start(data_dir)
    if start is not None:
        ds = ds.assign_coords(time=start + pd.to_timedelta(ds.time.values, unit="s"))
    return ds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xy(data_dir):
    """Load all xy cross-section files from *data_dir* into one Dataset.

    Masks the MicroHH fill-value (-1e9) with NaN.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.xy.nc"))
    if not files:
        raise FileNotFoundError(f"No *.xy.nc files found in {data_dir}")
    ds = xr.open_mfdataset(files, decode_times=False)
    ds = ds.where(ds != FILL_VALUE)
    ds = _apply_local_time(ds, data_dir)
    return ds


def load_3d(data_dir, variables=None):
    """Load 3D field files and compute derived quantities.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``{var}.nc`` files.
    variables : list of str, optional
        Variable names to load (default: thl, T, qt, ql, qi, b, w).

    Returns a Dataset with added fields:
        th, qv, w_prime, thl_prime, th_prime, qv_prime, w_prime_interp
    """
    data_dir = Path(data_dir)
    if variables is None:
        variables = DEFAULT_3D_VARS

    paths = [data_dir / f"{v}.nc" for v in variables]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    ds = xr.open_mfdataset(paths, decode_times=False)

    # derived quantities
    ds["th"] = ds.thl * (1 + (ds.ql * Lv + ds.qi * Ls) / (cp * ds.T))
    ds["qv"] = ds.qt - ds.ql - ds.qi

    # perturbations
    ds["w_prime"] = ds.w - ds.w.mean(["x", "y"])
    ds["thl_prime"] = ds.thl - ds.thl.mean(["x", "y"])
    ds["th_prime"] = ds.th - ds.th.mean(["x", "y"])
    ds["qv_prime"] = ds.qv - ds.qv.mean(["x", "y"])
    ds["w_prime_interp"] = ds["w_prime"].interp(zh=ds["z"])

    ds = _apply_local_time(ds, data_dir)
    return ds


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_domain_mean(ds_xy, var="qlqi_path", ax=None, ds_xy2=None,
                     label=None, label2=None, **kwargs):
    """Plot the domain-mean timeseries of *var* from an xy Dataset.

    Parameters
    ----------
    ds_xy2 : xr.Dataset, optional
        Second xy Dataset to overlay for comparison.
    label, label2 : str, optional
        Legend labels for the first and second dataset.
    """
    if ax is None:
        _, ax = plt.subplots()

    ds_xy[var].mean(["x", "y"]).plot(ax=ax, label=label, **kwargs)

    if ds_xy2 is not None:
        ds_xy2[var].mean(["x", "y"]).plot(ax=ax, label=label2, **kwargs)
        ax.legend()

    ax.figure.autofmt_xdate()
    return ax


def plot_cross_sections(ds, ds_xy, time, yidx=100, xidx=20, zmax=3000,
                        th_vmax=310, w_vmin=-1.5, qv_vmin=0.014, flux_var_name = 'sw_flux_dn'):
    """2x4 cross-section plot (y-slice on top, x-slice on bottom).

    Panels: theta, w, qv, cloud mask (ql > 0)
    Overlays: cloud-base height, sw_flux_dn, cloud shading on last column.
    """
    cld_base = ds_xy.qlqi_base.sel(time=time).quantile(0.5).values

    fig, axs = plt.subplots(2, 4, figsize=(18, 8))

    # --- helpers for a single row ---
    def _row(row_ax, slice_dim, idx, ds_xy_ref):
        sel3d = ds.sel(time=time).isel(**{slice_dim: idx})
        sel3d.th.plot(y="z", vmax=th_vmax, ax=row_ax[0], cmap="plasma")
        sel3d.w.plot(y="zh", vmin=w_vmin, ax=row_ax[1])
        sel3d.qv.plot(y="z", vmin=qv_vmin, ax=row_ax[2])
        (sel3d.ql > 0).plot(y="z", ax=row_ax[3], cmap="Grays")

        cloud_mask = (ds_xy_ref.qlqi_path.sel(time=time) > 0).isel(**{slice_dim: idx})
        other_dim = "x" if slice_dim == "y" else "y"
        coords = ds_xy_ref[other_dim].values

        for i, ax in enumerate(row_ax):
            ax.set_ylim(0, zmax)
            ax.hlines(y=cld_base, xmin=0, xmax=coords[-1],
                      linewidth=2, color="black", linestyle="dashed")
            if flux_var_name in ds_xy_ref:
                ds_xy_ref[flux_var_name].sel(time=time).isel(**{slice_dim: idx}).plot(
                    ax=ax, color="black")
            if i == 3:
                ax.fill_between(coords, zmax, 0, where=cloud_mask.values,
                                alpha=0.2, color="gray", zorder=50,
                                edgecolor="blue", linewidth=0.5)
            ax.set_title("")
            ax.set_ylabel("")

    _row(axs[0], "y", yidx, ds_xy)
    _row(axs[1], "x", xidx, ds_xy)

    plt.tight_layout()
    return fig, axs


def plot_flux_sections(ds, ds_xy, time, yidx=100, xidx=20, zmax=None,
                       th_flux_vmax=0.5, qv_flux_vmax=1e-3):
    """2x2 turbulent-flux cross-sections (w'th' and w'qv').

    Cloud columns shown as hatched regions.
    """
    cld_base = ds_xy.qlqi_base.sel(time=time).quantile(0.5).values
    if zmax is None:
        zmax = float(cld_base)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    def _row(row_ax, slice_dim, idx):
        sel = ds.sel(time=time).isel(**{slice_dim: idx})
        (sel.th_prime * sel.w_prime_interp).plot(ax=row_ax[0], vmax=th_flux_vmax)
        row_ax[0].set_title("w'theta' Vertical Heat Flux")
        (sel.qv_prime * sel.w_prime_interp).plot(ax=row_ax[1], vmax=qv_flux_vmax)
        row_ax[1].set_title("w'qv' Vertical Moisture Flux")

        cloud_mask = (ds_xy.qlqi_path.sel(time=time) > 0).isel(**{slice_dim: idx})
        other_dim = "x" if slice_dim == "y" else "y"
        coords = ds_xy[other_dim].values

        for ax in row_ax:
            ax.fill_between(coords, zmax, 0, where=cloud_mask.values,
                            facecolor="none", hatch="/", edgecolor="black",
                            zorder=50, linewidth=1)
            ax.set_ylim(0, zmax)
            ax.set_ylabel("")

    _row(axs[0], "y", yidx)
    _row(axs[1], "x", xidx)

    plt.tight_layout()
    return fig, axs
