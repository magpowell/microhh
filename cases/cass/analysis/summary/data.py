"""Loaders for the CASS LES v3 summary figures: 1D (two-stream) vs 3D (ray tracer) radiation."""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append("/global/homes/m/mpowell/repos/microhh/cases/cass/analysis/characterize")
from les_io import Run, lowest  # noqa: E402

EXPT = "no_aerosols_zero_wind_v3"
EXPT_ROOT = Path("/pscratch/sd/m/mpowell/CASS_LES/experiments") / EXPT
FEATURES_ROOT = Path("/pscratch/sd/m/mpowell/CASS_LES/analysis/characterize") / EXPT
OUT_ROOT = Path("/pscratch/sd/m/mpowell/CASS_LES/analysis/summary") / EXPT
CACHE_DIR = OUT_ROOT / "cache"
FIG_ROOT = Path("/pscratch/sd/m/mpowell/CASS_LES/analysis/figures") / EXPT  # shared with characterize/

RT = {"1D": "2stream", "3D": "raytracer"}
REPS = (1, 2, 3, 4)
EXTRA = {"1D wind": ("2stream_wind", 1), "1D cass": ("2stream_cass", 1)}

WINDOW = (10., 17.)  # solar hours of the window means
XLIM = (9., 17.)
RHO, CP, LV = 1.1483, 1005., 2.501e6
MIN_CLOUD_AREA = 40000.  # m2, 16 columns

STATS_VARS = {
    "thermo": ("ql_path", "ql_cover", "ql_frac", "ql", "thl_flux", "qt_flux"),
    "land_surface": ("H", "LE"),
    "radiation": ("sw_flux_dn", "sw_flux_up", "sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt", "sw_flux_sfc_up_rt"),
}


def run_dir(rt, rep):
    return EXPT_ROOT / rt / f"rep_{rep:02d}"


def members(label):
    """(rt, rep) pairs of the ensemble called label ("1D" or "3D")."""
    return [(RT[label], rep) for rep in REPS]


def _stats_file(path):
    root = xr.open_dataset(path, decode_times=False)
    t = np.rint(root["time"].values).astype(int)
    parts = []
    for group, names in STATS_VARS.items():
        g = xr.open_dataset(path, group=group, decode_times=False)
        parts.append(g[[n for n in names if n in g]])
    return xr.merge(parts).assign_coords(time=t, z=root["z"].values, zh=root["zh"].values)


def stats(run_dir):
    """Domain statistics (300 s) of a run, restart segments concatenated; adds lst and sw_sfc."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("cass.default.*.nc"), key=lambda p: int(p.stem.split(".")[-1]))
    ds = xr.concat([_stats_file(f) for f in files], dim="time")
    ds = ds.isel(time=np.unique(ds["time"].values, return_index=True)[1])
    if "sw_flux_sfc_dir_rt" in ds:
        sw = ds["sw_flux_sfc_dir_rt"] + ds["sw_flux_sfc_dif_rt"]
    else:
        sw = ds["sw_flux_dn"].isel(zh=0, drop=True)
    ds["sw_sfc"] = sw.assign_attrs(units="W m-2", long_name="surface shortwave received")
    return ds.assign_coords(lst=("time", Run(run_dir).lst(ds["time"].values)))


def xy_times(run_dir):
    """Integer times [s] of the 60 s cross-sections."""
    with xr.open_dataset(Path(run_dir) / "qlqi_path.xy.nc", decode_times=False) as ds:
        return np.rint(ds["time"].values).astype(int)


def xy_frames(run_dir, var, frame):
    """Cross-section var at the lowest level for one frame index or a slice of frames."""
    with xr.open_dataset(Path(run_dir) / f"{var}.xy.nc", decode_times=False) as ds:
        return lowest(ds[var].isel(time=frame))


def surface_sw(run_dir, frame):
    """Surface shortwave received per column: two-stream sw_flux_dn, or the ray-traced dir + dif."""
    run_dir = Path(run_dir)
    if (run_dir / "sw_flux_sfc_dir_rt.xy.nc").exists():
        return xy_frames(run_dir, "sw_flux_sfc_dir_rt", frame) + xy_frames(run_dir, "sw_flux_sfc_dif_rt", frame)
    return xy_frames(run_dir, "sw_flux_dn", frame)


def surface_fluxes(run_dir, frame):
    """Per-column sensible and latent heat flux [W m-2] from the kinematic surface fluxes."""
    return RHO * CP * xy_frames(run_dir, "thl_fluxbot", frame), RHO * LV * xy_frames(run_dir, "qt_fluxbot", frame)


def cloud_mask(run_dir, frame):
    return xy_frames(run_dir, "qlqi_path", frame) > 0.


def cloud_population(rt, rep):
    """Per-minute number of clouds per km2 and mean effective diameter [m] from the tracker features."""
    run = Run(run_dir(rt, rep))
    times = xy_times(run_dir(rt, rep))  # features hold only minutes with objects; zero clouds elsewhere
    fe = xr.open_dataset(FEATURES_ROOT / rt / f"rep_{rep:02d}" / "features.nc")
    df = fe[["time", "area"]].to_dataframe()
    df = df[df["area"] >= MIN_CLOUD_AREA].assign(diameter=lambda d: 2. * np.sqrt(d["area"] / np.pi))
    g = df.groupby("time")
    number = g.size().reindex(times, fill_value=0) / (run.xsize * run.ysize / 1e6)
    diameter = g["diameter"].mean().reindex(times)
    return xr.Dataset({"number": ("time", number.values), "diameter": ("time", diameter.values)},
                      coords={"time": times, "lst": ("time", run.lst(times))})


def cache_file(rt, rep):
    return CACHE_DIR / f"surface_{rt}_rep_{rep:02d}.nc"


def load_cache(rt, rep):
    return xr.open_dataset(cache_file(rt, rep))


def ensemble(label, loader):
    """Stack loader(rt, rep) over the members of label along a new member dimension."""
    ds = xr.concat([loader(rt, rep) for rt, rep in members(label)], dim="member",
                   coords="minimal", compat="override")
    return ds.assign_coords(member=list(REPS))


def stats_ensemble(label):
    return ensemble(label, lambda rt, rep: stats(run_dir(rt, rep)))


def population_ensemble(label):
    return ensemble(label, cloud_population)


def cache_ensemble(label):
    return ensemble(label, load_cache)


def window_mean(da, window=WINDOW):
    """Time mean over the solar-hour window."""
    sel = (da["lst"] >= window[0]) & (da["lst"] <= window[1])
    return da.where(sel).mean("time")


def report(name, da1, da3, unit="", fmt="{:.3g}"):
    """Print the window means per member and per ensemble, and the 3D/1D ratio of the ensemble means."""
    m1, m3 = window_mean(da1), window_mean(da3)
    each = lambda m: " ".join(fmt.format(v) for v in np.atleast_1d(m.values))
    ratio = float(m3.mean()) / float(m1.mean())
    print(f"{name:<36s} {unit:<10s} 1D {fmt.format(float(m1.mean())):>8s} [{each(m1)}]"
          f"   3D {fmt.format(float(m3.mean())):>8s} [{each(m3)}]   3D/1D {ratio:.3f}")
