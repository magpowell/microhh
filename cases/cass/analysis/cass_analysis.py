"""
CASS LES analysis utilities.

Provides data loading, analysis functions, and plotting helpers for comparing
2stream (1D) and raytracer (3D) radiation runs across all CASS experiments.

Typical usage
-------------
    from cass_analysis import RunSet, load_stats_ensemble

    rs = RunSet("base_debug", "/pscratch/sd/m/mpowell/CASS_LES/debug/base", n_reps=1)

    # Stats (domain means, profiles)
    stats_2s, stats_2s_std = load_stats_ensemble(rs.dirs["2stream"])
    stats_rt, stats_rt_std = load_stats_ensemble(rs.dirs["raytracer"])

    # XY cross-sections (spatially resolved, surface level)
    ds_2s, ds_2s_std = load_xy_ensemble(rs.dirs["2stream"])
    ds_rt, ds_rt_std = load_xy_ensemble(rs.dirs["raytracer"])

Notes
-----
- 3D dump analysis (Section 3.2) requires running 3d_to_nc.py first; see load_3d_nc().
- XY files are loaded lazily via dask; call .compute() or .load() when needed.
"""

import sys
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

# ── Upstream utility import ───────────────────────────────────────────────────
_PYTHON_DIR = Path(__file__).parents[3] / "python"
sys.path.insert(0, str(_PYTHON_DIR))
try:
    from plot_microhh_utils import get_local_start
except ImportError:
    def get_local_start(data_dir):  # fallback: no timezone conversion
        warnings.warn("plot_microhh_utils not found; time axis will be in UTC")
        return None

# ── Physical constants ────────────────────────────────────────────────────────
Lv  = 2.5e6    # J kg⁻¹  latent heat of vaporisation
cp  = 1005.0   # J kg⁻¹ K⁻¹
rho = 1.2      # kg m⁻³  reference surface air density
eps_v = 0.608  # R_v/R_d - 1  (virtual-temperature coefficient)

# ── CASS site constants ─────────────────────────────────────────────────────
CASS_LAT = 36.5    # ARM SGP latitude [deg N]
CASS_DOY = 205     # July 24

# ── Simulation time helpers ──────────────────────────────────────────────────
LST_OFFSET = 5.5   # simulation t=0 → 05:30 LST
THETA_REF  = 300.0 # reference potential temperature [K]


def sim_time_to_lst(t_sec):
    """Convert simulation seconds to LST hours."""
    return np.asarray(t_sec) / 3600.0 + LST_OFFSET


def zenith_angle(lst_h, lat=CASS_LAT, doy=CASS_DOY):
    """Solar zenith angle [deg] for given LST hour(s)."""
    decl  = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + doy))))
    lat_r = np.radians(lat)
    ha    = np.radians(15.0 * (np.asarray(lst_h) - 12.0))
    cos_z = (np.sin(lat_r) * np.sin(decl)
             + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    return np.degrees(np.arccos(np.clip(cos_z, -1.0, 1.0)))


def lst_window_mask(t_sec, lo=11.5, hi=17.0):
    """Boolean mask for timesteps within an LST window."""
    lst = sim_time_to_lst(t_sec)
    return (lst >= lo) & (lst <= hi)


def load_sfc_xy(run_dir, varname):
    """Load a single surface xy field, squeezing z/zh singletons.

    Returns (data, t_sec) where data is (nt, ny, nx) float32 and t_sec is
    the simulation-seconds time axis.
    """
    run_dir = Path(run_dir)
    ds = xr.open_dataset(run_dir / f"{varname}.xy.nc", decode_times=False)
    arr = ds[varname].values
    t_sec = ds["time"].values
    ds.close()
    # Squeeze singleton z / zh
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    return arr, t_sec


def load_sfc_sw_dn(run_dir):
    """Load surface SW↓, handling 2stream vs raytracer file conventions.

    Returns (sw_dn, t_sec) where sw_dn is (nt, ny, nx).
    """
    run_dir = Path(run_dir)
    rt_file = run_dir / "sw_flux_sfc_dir_rt.xy.nc"
    if rt_file.exists():
        ds_dir = xr.open_dataset(rt_file, decode_times=False)
        ds_dif = xr.open_dataset(
            run_dir / "sw_flux_sfc_dif_rt.xy.nc", decode_times=False)
        sw = ds_dir["sw_flux_sfc_dir_rt"].values + ds_dif["sw_flux_sfc_dif_rt"].values
        t_sec = ds_dir["time"].values
        ds_dir.close(); ds_dif.close()
        return sw, t_sec
    # 2stream fallback
    ds = xr.open_dataset(run_dir / "sw_flux_dn.xy.nc", decode_times=False)
    sw = ds["sw_flux_dn"].values
    t_sec = ds["time"].values
    ds.close()
    if sw.ndim == 4:
        sw = sw[:, 0, :, :]
    return sw, t_sec

# ── Style maps (keep consistent across all notebooks) ────────────────────────
RT_STYLE = {
    "2stream":   dict(ls="-",  color="C0"),
    "raytracer": dict(ls="--", color="C1"),
}
RT_LABEL = {
    "2stream":   "1D (2stream)",
    "raytracer": "3D (raytracer)",
}
FLUX_COLORS = {"Rnet": "k", "H": "C1", "LE": "C0", "G": "C2", "S": "C3"}


# ══════════════════════════════════════════════════════════════════════════════
# RunSet  —  describes one experimental condition
# ══════════════════════════════════════════════════════════════════════════════

class RunSet:
    """One experimental condition: N reps × ≤2 RT types.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "base" or "soil_moisture/theta_0.3".
    root : str or Path
        Directory containing ``{rt_type}/rep_{01..N}/`` subdirectories.
    n_reps : int
        Maximum number of reps to search for. Reps that don't exist are skipped.
    rt_types : sequence of str
        Radiation type subdirs to include.

    Attributes
    ----------
    dirs : dict[str, list[Path]]
        Maps each rt_type to the list of existing rep directories.
    """

    def __init__(self, name, root, n_reps=4, rt_types=("2stream", "raytracer")):
        self.name = name
        self.root = Path(root)
        self.rt_types = list(rt_types)
        self.dirs: dict[str, list[Path]] = {}
        for rt in self.rt_types:
            found = []
            for i in range(1, n_reps + 1):
                p = self.root / rt / f"rep_{i:02d}"
                if p.exists():
                    found.append(p)
            self.dirs[rt] = found
            if not found:
                warnings.warn(f"[RunSet '{name}'] No reps found for {rt} under {self.root}")

    def __repr__(self):
        lines = [f"RunSet('{self.name}', root={self.root})"]
        for rt, dirs in self.dirs.items():
            lines.append(f"  {rt}: {len(dirs)} rep(s)")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Stats loading  (xarray, one open per group → merge)
# ══════════════════════════════════════════════════════════════════════════════

def _open_stats_group(path, group):
    """Open a single netCDF4 group as an xr.Dataset (decode_times=False)."""
    return xr.open_dataset(str(path), group=group, decode_times=False)


def load_stats(run_dir) -> xr.Dataset:
    """Load ``cass.default.0000000.nc`` from *run_dir*.

    Returns an xr.Dataset with coordinates (time, t_sec, t_local, z, zh)
    and data variables from all groups.

    Access pattern is dict-like: ``ds["H"]``, ``ds["qlqi_path"]``, etc.

    Key variables
    -------------
    H, LE, G, S, Rnet                 SEB scalars (W m-2)
    theta                             soil moisture (time, z_soil)
    qlqi_path, qlqi_cover, ql_cover   cloud scalars
    zi                                boundary-layer height (m)
    thl_w, qt_w, thv_w               resolved turbulent flux profiles (time, zh)
    thl, qt, ql, ql_frac             mean profiles (time, z)
    sza                               solar zenith angle
    """
    run_dir = Path(run_dir)
    stats_path = run_dir / "cass.default.0000000.nc"

    # Read each group as xr.Dataset
    ds_root = xr.open_dataset(str(stats_path), decode_times=False)
    ds_lsm  = _open_stats_group(stats_path, "land_surface")
    ds_rad  = _open_stats_group(stats_path, "radiation")
    ds_thm  = _open_stats_group(stats_path, "thermo")
    ds_dyn  = _open_stats_group(stats_path, "default")

    t_sec = ds_root["time"].values
    z  = ds_root["z"].values
    zh = ds_root["zh"].values

    # Build local-time coordinate
    start   = get_local_start(run_dir)
    t_local = pd.to_datetime(
        [start + pd.Timedelta(seconds=float(s)) for s in t_sec]
        if start is not None else t_sec
    )
    if hasattr(t_local, "tz") and t_local.tz is not None:
        t_local = t_local.tz_localize(None)

    # Radiation scalars: extract surface level
    sw_dn = ds_rad["sw_flux_dn"].isel(zh=0).values
    sw_up = ds_rad["sw_flux_up"].isel(zh=0).values
    lw_dn = ds_rad["lw_flux_dn"].isel(zh=0).values
    lw_up = ds_rad["lw_flux_up"].isel(zh=0).values

    # Assemble into a single Dataset
    out = xr.Dataset(
        data_vars={
            # Land surface scalars (time,)
            "H":       ("time", ds_lsm["H"].values),
            "LE":      ("time", ds_lsm["LE"].values),
            "G":       ("time", ds_lsm["G"].values),
            "S":       ("time", ds_lsm["S"].values),
            "theta":   (("time", "z_soil"), ds_lsm["theta"].values),
            "ustar":   ("time", ds_lsm["ustar"].values),
            # Radiation scalars (time,)
            "Rnet":      ("time", (sw_dn - sw_up) + (lw_dn - lw_up)),
            "sw_dn":     ("time", sw_dn),
            "sw_up":     ("time", sw_up),
            "lw_dn":     ("time", lw_dn),
            "lw_up":     ("time", lw_up),
            "sza":       ("time", ds_rad["sza"].values),
            "sw_dn_toa": ("time", ds_rad["sw_flux_dn_toa"].values),
            # Thermo scalars (time,)
            "qlqi_path":  ("time", ds_thm["qlqi_path"].values),
            "ql_cover":   ("time", ds_thm["ql_cover"].values),
            "qlqi_cover": ("time", ds_thm["qlqi_cover"].values),
            "zi":         ("time", ds_thm["zi"].values),
            "thl_bot":    ("time", ds_thm["thl_bot"].values),
            "qt_bot":     ("time", ds_thm["qt_bot"].values),
            # Thermo profiles (time, zh)
            "thl_w": (("time", "zh"), ds_thm["thl_w"].values),
            "qt_w":  (("time", "zh"), ds_thm["qt_w"].values),
            "thv_w": (("time", "zh"), ds_thm["thv_w"].values),
            # Thermo profiles (time, z)
            "thl":     (("time", "z"), ds_thm["thl"].values),
            "qt":      (("time", "z"), ds_thm["qt"].values),
            "ql":      (("time", "z"), ds_thm["ql"].values),
            "ql_frac": (("time", "z"), ds_thm["ql_frac"].values),
            # Dynamics profiles (time, z)
            "u":   (("time", "z"), ds_dyn["u"].values),
            "v":   (("time", "z"), ds_dyn["v"].values),
            "tke": (("time", "z"), ds_dyn["tke"].values),
        },
        coords={
            "time":    t_local,
            "t_sec":   ("time", t_sec),
            "z":       z,
            "zh":      zh,
        },
    )
    # Alias: callers that access ds["t_local"] get the time coordinate values
    out["t_local"] = out["time"]

    for d in (ds_root, ds_lsm, ds_rad, ds_thm, ds_dyn):
        d.close()
    return out


def load_stats_ensemble(rep_dirs: list) -> tuple[xr.Dataset, xr.Dataset]:
    """Load stats from multiple reps; return (ensemble_mean, ensemble_std).

    Time axes are truncated to the shortest run before stacking.
    """
    all_ds = [load_stats(d) for d in rep_dirs]
    if not all_ds:
        raise ValueError("No rep dirs provided")

    nt_min = min(ds.sizes["time"] for ds in all_ds)
    all_ds = [ds.isel(time=slice(None, nt_min)) for ds in all_ds]
    stacked = xr.concat(all_ds, dim="rep")
    return stacked.mean("rep"), stacked.std("rep", ddof=0)


# ══════════════════════════════════════════════════════════════════════════════
# XY loading  (xarray + dask for lazy evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def load_xy_files(run_dir, variables=None, chunks=None) -> xr.Dataset:
    """Load xy cross-section netCDF files from *run_dir*.

    Parameters
    ----------
    variables : list of str, optional
        Variable names to load (default: standard set including RT-specific ones).
    chunks : dict, optional
        Dask chunk sizes, e.g. ``{'time': 200}``. Defaults to ``{'time': 200}``.

    Returns
    -------
    xr.Dataset with local-time coordinate.  Singleton z/zh dims are squeezed.
    Missing files are skipped without error.

    Notes
    -----
    For raytracer runs, ``sw_flux_sfc_rt`` = dir + dif is added automatically.
    """
    run_dir = Path(run_dir)
    if variables is None:
        variables = [
            "qlqi_path", "thl_fluxbot", "qt_fluxbot",
            "sw_flux_dn", "sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt",
        ]
    if chunks is None:
        chunks = {"time": 200}

    datasets = []
    for var in variables:
        f = run_dir / f"{var}.xy.nc"
        if not f.exists():
            continue
        ds_v = xr.open_dataset(str(f), decode_times=False, chunks=chunks)
        for dim in ("z", "zh"):
            if dim in ds_v.dims and ds_v.sizes[dim] == 1:
                ds_v = ds_v.squeeze(dim, drop=True)
        ds_v = ds_v.where(ds_v != -1.0e9)
        datasets.append(ds_v)

    if not datasets:
        raise FileNotFoundError(f"No xy files found in {run_dir}")

    ds = xr.merge(datasets, join="inner")

    # Convenience: total 3D-RT surface SW
    if "sw_flux_sfc_dir_rt" in ds and "sw_flux_sfc_dif_rt" in ds:
        ds["sw_flux_sfc_rt"] = ds["sw_flux_sfc_dir_rt"] + ds["sw_flux_sfc_dif_rt"]

    # Local-time coordinate
    start = get_local_start(run_dir)
    if start is not None:
        t_local = pd.to_datetime(
            [start + pd.Timedelta(seconds=float(s)) for s in ds.time.values]
        )
        if t_local.tz is not None:
            t_local = t_local.tz_localize(None)
        ds = ds.assign_coords(time=("time", t_local))

    return ds



def conditioned_means_ensemble(rep_dirs: list, variables=None) -> tuple[dict, dict]:
    """Compute cloud-root-conditioned surface flux means per rep, return ensemble stats.

    For each rep:
      1. Load xy files
      2. At each timestep, partition ``thl_fluxbot`` and ``qt_fluxbot`` into
         cloud-root (qlqi_path > 0) and non-cloud-root columns.
      3. Average over x, y for each partition.

    Returns
    -------
    (mean_dict, std_dict)
        Each is {'shaded': {var: DataArray(time)}, 'unshaded': {var: DataArray(time)}}
    """
    if variables is None:
        variables = ["thl_fluxbot", "qt_fluxbot", "sw_flux_dn", "sw_flux_sfc_rt"]

    # sw_flux_sfc_rt is derived (dir+dif) inside load_xy_files, not a real file;
    # request the components explicitly so the derivation fires.
    vars_to_load = [v for v in variables if v != "sw_flux_sfc_rt"]
    if "sw_flux_sfc_rt" in variables:
        vars_to_load += ["sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt"]

    per_rep = []
    for d in rep_dirs:
        ds = load_xy_files(d, vars_to_load + ["qlqi_path"], chunks={"time": 200})
        mask = ds["qlqi_path"] > 0
        shaded_means   = {v: ds[v].where(mask).mean(["x", "y"])  for v in variables if v in ds}
        unshaded_means = {v: ds[v].where(~mask).mean(["x", "y"]) for v in variables if v in ds}
        domain_means   = {v: ds[v].mean(["x", "y"])              for v in variables if v in ds}
        per_rep.append({"shaded": shaded_means, "unshaded": unshaded_means, "domain": domain_means})

    first_var = list(per_rep[0]["shaded"])[0]
    nt_min = min(
        r["shaded"][first_var].sizes["time"] for r in per_rep
    )

    out_mean = {"shaded": {}, "unshaded": {}, "domain": {}}
    out_std  = {"shaded": {}, "unshaded": {}, "domain": {}}
    for kind in ("shaded", "unshaded", "domain"):
        for var in per_rep[0][kind]:
            stack = xr.concat(
                [r[kind][var].isel(time=slice(None, nt_min)) for r in per_rep],
                dim="rep",
            )
            out_mean[kind][var] = stack.mean("rep")
            out_std[kind][var]  = stack.std("rep")
    return out_mean, out_std


# ══════════════════════════════════════════════════════════════════════════════
# 3D dump loading  (requires 3d_to_nc.py to have been run first)
# ══════════════════════════════════════════════════════════════════════════════

_3D_VARS_DEFAULT = ["thl", "qt", "ql", "w", "b"]
_3D_VARS_CIRC    = ["thl", "qt", "ql", "w", "b", "u", "v"]


def load_3d_nc(run_dir, variables=None, chunks=None) -> xr.Dataset:
    """Load 3D netCDF dump files produced by ``python/3d_to_nc.py``.

    Expects files named ``{var}.nc`` in *run_dir*.  Adds perturbation fields
    (w_prime, thl_prime, qt_prime, b_prime, u_prime, v_prime) and interpolates
    staggered fields (w, u, v) to cell-centre levels.

    Prerequisites
    -------------
    From the run directory, run::

        python $MICROHH/python/3d_to_nc.py -v thl qt ql w b u v

    Parameters
    ----------
    variables : list of str, optional
        Variable names (default: thl, qt, ql, w, b).
    chunks : dict, optional
        Dask chunk sizes (default: ``{'time': 5, 'z': -1}``).
    """
    run_dir = Path(run_dir)
    if variables is None:
        variables = _3D_VARS_DEFAULT
    if chunks is None:
        chunks = {"time": 5, "z": -1}

    paths   = [run_dir / f"{v}.nc" for v in variables]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing 3D nc files (run 3d_to_nc.py first): "
            f"{[p.name for p in missing]}"
        )

    ds = xr.open_mfdataset([str(p) for p in paths], decode_times=False, chunks=chunks)

    # Perturbations from instantaneous horizontal mean
    for var, prime in [("thl", "thl_prime"), ("qt", "qt_prime"), ("b", "b_prime")]:
        if var in ds:
            ds[prime] = ds[var] - ds[var].mean(["x", "y"])

    # qv = qt - ql  (vapour only)
    if "qt" in ds and "ql" in ds:
        qv = ds["qt"] - ds["ql"]
        ds["qv_prime"] = qv - qv.mean(["x", "y"])

    # Interpolate w (on zh) to cell-centre z for flux computations
    if "w" in ds and "z" in ds.coords:
        ds["w_cc"]    = ds["w"].interp(zh=ds["z"])
        ds["w_prime"] = ds["w_cc"] - ds["w_cc"].mean(["x", "y"])

    # Interpolate u (on xh) and v (on yh) to cell-centre grids and compute primes
    if "u" in ds and "x" in ds.coords:
        ds["u_cc"]    = ds["u"].interp(xh=ds["x"])
        ds["u_prime"] = ds["u_cc"] - ds["u_cc"].mean(["x", "y"])
    if "v" in ds and "y" in ds.coords:
        ds["v_cc"]    = ds["v"].interp(yh=ds["y"])
        ds["v_prime"] = ds["v_cc"] - ds["v_cc"].mean(["x", "y"])

    # Cloud mask at each level
    if "ql" in ds:
        ds["cloud_3d"] = ds["ql"] > 0

    # Local-time coordinate
    start = get_local_start(run_dir)
    if start is not None:
        t_local = pd.to_datetime(
            [start + pd.Timedelta(seconds=float(s)) for s in ds.time.values]
        )
        if t_local.tz is not None:
            t_local = t_local.tz_localize(None)
        ds = ds.assign_coords(time=("time", t_local))

    return ds


# ══════════════════════════════════════════════════════════════════════════════
# SEB helpers
# ══════════════════════════════════════════════════════════════════════════════

def buoyancy_flux_equiv(thl_flux, qt_flux, theta_ref=THETA_REF):
    """Equivalent surface buoyancy flux Q_rho = thl_flux + eps_v * Theta * qt_flux.

    Works on scalars, numpy arrays, or xarray DataArrays.
    """
    return thl_flux + eps_v * theta_ref * qt_flux


def cloud_top_za(z, ql_frac, frac_of_peak=0.10):
    """Cloud-layer top height from a domain-mean ql_frac profile.

    Returns the highest z where ql_frac > frac_of_peak * max(ql_frac).
    Returns NaN if no cloud is present.
    """
    z = np.asarray(z)
    ql_frac = np.asarray(ql_frac)
    peak = ql_frac.max()
    if peak <= 0:
        return np.nan
    above = np.where(ql_frac > frac_of_peak * peak)[0]
    return float(z[above[-1]])



def seb_residual(stats) -> np.ndarray:
    """Rnet − (H + LE + G + S)  [W m⁻²]."""
    return stats["Rnet"] - (stats["H"] + stats["LE"] + stats["G"] + stats["S"])


def bowen_ratio(stats, le_min: float = 1.0) -> np.ndarray:
    """H / LE, masked where LE < *le_min*  (avoids near-zero division)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(stats["LE"] > le_min, stats["H"] / stats["LE"], np.nan)


def evaporative_fraction(stats, le_min: float = 1.0) -> np.ndarray:
    """LE / (H + LE), masked where (H + LE) is small."""
    denom = stats["H"] + stats["LE"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denom) > le_min, stats["LE"] / denom, np.nan)


def compute_z_sl(stats, time_idx: int = -1) -> float:
    """Subcloud-layer height: lowest level where domain-mean ql_frac > 0 (cloud base).

    MicroHH's zi diagnostic uses the max thl gradient, which in shallow cumulus
    is dominated by the sponge layer near the domain top — not physically meaningful.
    Cloud base (first ql_frac > 0) correctly identifies the subcloud layer top.
    """
    ql_frac = stats["ql_frac"][time_idx]   # (nz,)
    z       = stats["z"]                   # (nz,)
    idx = np.argmax(ql_frac > 0)
    if ql_frac[idx] == 0:
        # No cloud present — fall back to domain-mean thv_w minimum below 4 km
        thv_w = stats["thv_w"][time_idx]
        zh    = stats["zh"]
        mask  = zh < 4000.0
        idx   = np.argmin(thv_w[mask])
        return float(zh[idx])
    return float(z[idx])


# ══════════════════════════════════════════════════════════════════════════════
# Cloud root identification and cross-section analysis
# ══════════════════════════════════════════════════════════════════════════════

def cloud_mask_2d(qlqi_path_2d: np.ndarray) -> np.ndarray:
    """Boolean array: True where qlqi_path > 0 (cloud-root column)."""
    return qlqi_path_2d > 0


def find_cloud_objects(mask_2d: np.ndarray, dx: float, dy: float,
                       min_L: float = 500.0) -> tuple[np.ndarray, list]:
    """Label connected cloud-root objects in a 2D bool mask.

    Parameters
    ----------
    mask_2d : (ny, nx) bool
    dx, dy : grid spacing in metres
    min_L : minimum effective diameter [m] to retain (default 500 m)

    Returns
    -------
    labeled : (ny, nx) int array (0 = background, ≥1 = object label)
    props : list of dicts with keys:
        label, area_m2, L (effective diameter), cy (centroid y-idx), cx (centroid x-idx)
    """
    labeled, n_obj = ndimage.label(mask_2d)
    props = []
    for lbl in range(1, n_obj + 1):
        obj = labeled == lbl
        area_pix = obj.sum()
        area_m2  = float(area_pix) * dx * dy
        L = 2.0 * np.sqrt(area_m2 / np.pi)
        if L < min_L:
            labeled[obj] = 0      # remove small objects
            continue
        ys, xs = np.where(obj)
        props.append(dict(
            label   = lbl,
            area_m2 = area_m2,
            L       = L,
            cy      = int(round(ys.mean())),
            cx      = int(round(xs.mean())),
        ))
    # Sort by size descending
    props.sort(key=lambda p: p["area_m2"], reverse=True)
    return labeled, props


def compute_normalized_cloud_root_profiles(ds_3d: xr.Dataset,
                                            ds_xy: xr.Dataset,
                                            stats_mean: dict) -> dict:
    """Cloud-root flux profiles normalised by the sub-cloud domain-mean integral.

    At each 3D dump time:
      1. Cloud-root mask   : LWP > 0 from nearest XY output
      2. Conditioned mean  : w'θ_l' and w'q_v' averaged over cloud-root columns
      3. Domain mean       : same fields averaged over all columns
      4. Norm scalar       : trapz(F_domain(z), z)  for z ≤ z_sl(t)
      5. ζ = z / z_sl(t)  : non-dimensionalise using that time step's z_sl
      6. Interpolate F_cloud / norm onto a common ζ grid (0–1.5, 200 pts)

    Profiles are aligned in physical structure before averaging, so a feature
    always at 0.3 z_sl will not be smeared by z_sl variability across dump times.

    Parameters
    ----------
    ds_3d      : Dataset with all dump times (dims: time, z, y, x);
                 must contain w_prime, thl_prime, qv_prime
    ds_xy      : Dataset with qlqi_path (dims: time, y, x)
    stats_mean : stats dict used to compute z_sl at each dump time

    Returns
    -------
    dict with keys:
      ``zeta``         (n_zeta,)      – common z/z_sl coordinate
      ``z``            (nz,)          – original height coordinate (m)
      ``z_sl``         (n_times,)     – sub-cloud layer height per dump (m)
      ``norm_thl``     (n_times,)     – sub-cloud integral of domain-mean heat flux
      ``norm_qv``      (n_times,)     – sub-cloud integral of domain-mean moisture flux
      ``thl_cloud_nd`` (n_times, n_zeta) – cloud-root/domain-mean heat flux ratio on zeta grid
      ``qv_cloud_nd``  (n_times, n_zeta) – cloud-root/domain-mean moisture flux ratio on zeta grid
    """
    for req in ("w_prime", "thl_prime", "qv_prime"):
        if req not in ds_3d:
            raise ValueError(f"Dataset missing '{req}'; ensure load_3d_nc() loaded qt and ql.")

    zdim = "z" if "z" in ds_3d.dims else "zh"
    z    = ds_3d[zdim].values

    zeta_grid = np.linspace(0, 1.5, 200)

    thl_cloud_nd_list, qv_cloud_nd_list   = [], []
    f_cloud_thl_list,  f_cloud_qv_list   = [], []
    f_free_thl_list,   f_free_qv_list    = [], []
    f_domain_thl_list, f_domain_qv_list  = [], []
    cloud_frac_list                      = []
    norm_thl_list, norm_qv_list          = [], []
    z_sl_list                            = []
    t_hours_list                         = []

    # Process all dump times; LST filtering is applied at load time using cached t_hours
    t_vals  = ds_3d.time.values   # local datetime64 (assigned by load_3d_nc)
    t_hours = (pd.DatetimeIndex(t_vals).hour
               + pd.DatetimeIndex(t_vals).minute / 60.0)

    xy_times_f = ds_xy.time.values.astype("datetime64[ns]").astype(float)
    st_times_f = stats_mean["t_local"].values.astype("datetime64[ns]").astype(float)

    for tidx_3d in range(len(t_vals)):
        t3d   = ds_3d.time.values[tidx_3d]
        t3d_f = np.datetime64(t3d, "ns").astype(float)
        ds_t  = ds_3d.isel(time=int(tidx_3d)).load()

        # Cloud mask from nearest XY time
        mask_2d = cloud_mask_2d(
            ds_xy["qlqi_path"].isel(time=int(np.argmin(np.abs(xy_times_f - t3d_f)))).values
        )

        # z_sl from stats at nearest time; skip timesteps with no cloud
        z_sl = compute_z_sl(stats_mean, time_idx=int(np.argmin(np.abs(st_times_f - t3d_f))))
        if z_sl <= 0:
            continue

        # Flux fields
        flux_thl = (ds_t["w_prime"] * ds_t["thl_prime"]).squeeze()
        flux_qv  = (ds_t["w_prime"] * ds_t["qv_prime"]).squeeze() * 1e3
        mask_da  = xr.DataArray(mask_2d, dims=["y", "x"])

        f_cloud_thl  = flux_thl.where(mask_da).mean(["x", "y"]).values
        f_cloud_qv   = flux_qv.where(mask_da).mean(["x", "y"]).values
        f_free_thl   = flux_thl.where(~mask_da).mean(["x", "y"]).values
        f_free_qv    = flux_qv.where(~mask_da).mean(["x", "y"]).values
        f_domain_thl = flux_thl.mean(["x", "y"]).values
        f_domain_qv  = flux_qv.mean(["x", "y"]).values
        cf           = float(mask_da.mean().values)

        # Non-dimensionalise z → ζ = z/z_sl(t), interpolate raw profiles onto common grid
        zeta_t = z / z_sl
        _interp = lambda arr: np.interp(zeta_grid, zeta_t, arr, left=np.nan, right=np.nan)

        f_cloud_thl_list.append(_interp(f_cloud_thl))
        f_cloud_qv_list.append( _interp(f_cloud_qv))
        f_free_thl_list.append( _interp(f_free_thl))
        f_free_qv_list.append(  _interp(f_free_qv))
        f_domain_thl_list.append(_interp(f_domain_thl))
        f_domain_qv_list.append( _interp(f_domain_qv))
        cloud_frac_list.append(cf)

        # Sub-cloud peak normalisation (scalar per timestep, for thl_cloud_nd backward compat)
        scl      = z <= z_sl
        norm_thl = float(np.nanmax(np.abs(f_domain_thl[scl]))) if scl.any() else np.nan
        norm_qv  = float(np.nanmax(np.abs(f_domain_qv[scl])))  if scl.any() else np.nan
        norm_thl_list.append(norm_thl)
        norm_qv_list.append(norm_qv)

        if norm_thl > 1e-6 and norm_qv > 1e-6:
            thl_cloud_nd_list.append(_interp(f_cloud_thl) / norm_thl)
            qv_cloud_nd_list.append( _interp(f_cloud_qv)  / norm_qv)
        else:
            thl_cloud_nd_list.append(np.full(len(zeta_grid), np.nan))
            qv_cloud_nd_list.append( np.full(len(zeta_grid), np.nan))

        z_sl_list.append(z_sl)
        t_hours_list.append(t_hours[tidx_3d])

    return {
        "zeta":          zeta_grid,
        "z":             z,
        "z_sl":          np.array(z_sl_list),
        "norm_thl":      np.array(norm_thl_list),
        "norm_qv":       np.array(norm_qv_list),
        "thl_cloud_nd":  np.array(thl_cloud_nd_list),    # (n_times, n_zeta) normalised
        "qv_cloud_nd":   np.array(qv_cloud_nd_list),
        # Raw flux profiles on zeta grid — cache these to avoid recomputing if
        # the normalisation changes
        "f_cloud_thl":   np.array(f_cloud_thl_list),     # (n_times, n_zeta) K m/s
        "f_cloud_qv":    np.array(f_cloud_qv_list),      # (n_times, n_zeta) g/kg m/s
        "f_free_thl":    np.array(f_free_thl_list),      # cloud-free columns
        "f_free_qv":     np.array(f_free_qv_list),
        "f_domain_thl":  np.array(f_domain_thl_list),
        "f_domain_qv":   np.array(f_domain_qv_list),
        "cloud_frac":    np.array(cloud_frac_list),       # (n_times,) scalar
        "t_hours":       np.array(t_hours_list),           # (n_times,) LST hours of each dump
    }


def lwp_integral(diff, dt_s: float, cloudy_mask=None) -> float:
    """Time-integral of LWP difference over the cloudy period.

    Parameters
    ----------
    diff : array-like (g m⁻²)
    dt_s : timestep in seconds
    cloudy_mask : bool array, same length as diff.
        If None, integrates over all times.

    Returns
    -------
    float  [g m⁻² h]
    """
    diff = np.asarray(diff)
    if cloudy_mask is not None:
        diff = diff[cloudy_mask]
    return float(diff.sum() * dt_s / 3600.0)


def lwp_normalized_diff(diff_gm2, lwp_1d_gm2, min_lwp: float = 0.1) -> np.ndarray:
    """(3D − 1D) / 1D LWP  as a relative change (masked where 1D LWP is small)."""
    denom = np.asarray(lwp_1d_gm2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > min_lwp, np.asarray(diff_gm2) / denom, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_plottime(t):
    """Strip tz-info so matplotlib fill_between handles the time axis."""
    import matplotlib.dates as mdates
    ti = pd.DatetimeIndex(t)
    if ti.tz is not None:
        ti = ti.tz_localize(None)
    return mdates.date2num(ti)


def _shade_ensemble(ax, t, mean_arr, std_arr, color, alpha=0.15):
    """Plot mean ± std shading on *ax*.  Handles tz-aware time arrays."""
    try:
        t_num = _to_plottime(t)
    except Exception:
        t_num = t
    ax.fill_between(t_num, mean_arr - std_arr, mean_arr + std_arr,
                    color=color, alpha=alpha)


def plot_seb_timeseries(ax, stats_mean: dict, stats_std: dict = None,
                        rt_type: str = "2stream") -> plt.Axes:
    """Panel: SEB component timeseries (H, LE, G, S, Rnet).

    Parameters
    ----------
    ax : matplotlib Axes
    stats_mean, stats_std : dicts from load_stats_ensemble
    rt_type : '2stream' or 'raytracer' (controls linestyle)
    """
    t     = stats_mean["t_local"]
    t_num = _to_plottime(t)     
    sty   = RT_STYLE[rt_type]

    for flux, color in FLUX_COLORS.items():
        y = stats_mean[flux]
        ax.plot(t_num, y, color=color, label=flux, ls=sty["ls"])
        if stats_std is not None and stats_std.get(flux) is not None:
            _shade_ensemble(ax, t, y, stats_std[flux], color)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Flux (W m⁻²)")     
    ax.xaxis_date()   
    ax.figure.autofmt_xdate()
    return ax


def plot_seb_residual(ax, stats_mean: dict, rt_type: str = "2stream") -> plt.Axes:
    """Panel: SEB residual  Rnet − (H + LE + G + S)."""
    t     = stats_mean["t_local"]
    t_num = _to_plottime(t)          # ← match panel 1
    ax.plot(t_num, seb_residual(stats_mean),
            label=RT_LABEL[rt_type], **RT_STYLE[rt_type])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Rnet − (H+LE+G+S)  (W m⁻²)")
    ax.xaxis_date()
    ax.figure.autofmt_xdate()
    return ax



def plot_flux_conditioned(ax, t_local, shaded_mean, unshaded_mean,
                          shaded_std=None, unshaded_std=None,
                          rt_type: str = "2stream",
                          ylabel: str = "", scale: float = 1.0) -> plt.Axes:
    """Panel: flux variable conditioned on cloud mask (shaded vs. unshaded).

    Parameters
    ----------
    t_local : time coordinate
    shaded_mean, unshaded_mean : DataArray or array (time,)
    shaded_std, unshaded_std : optional spread arrays
    scale : multiply flux by this value (e.g. rho*cp to convert thl_fluxbot → H)
    """
    ls    = RT_STYLE[rt_type]["ls"]
    sm    = np.asarray(shaded_mean) * scale
    um    = np.asarray(unshaded_mean) * scale
    t_num = _to_plottime(t_local)
    ax.plot(t_num, sm, color="steelblue",  ls=ls, label="cloud-root")
    ax.plot(t_num, um, color="darkorange", ls=ls, label="cloud-free")
    if shaded_std is not None:
        _shade_ensemble(ax, t_local, sm, np.asarray(shaded_std) * scale, "steelblue")
    if unshaded_std is not None:
        _shade_ensemble(ax, t_local, um, np.asarray(unshaded_std) * scale, "darkorange")
    ax.axhline(0, color="gray", lw=0.5)
    ax.xaxis_date()
    ax.set_ylabel(ylabel)
    ax.figure.autofmt_xdate()
    return ax



# ══════════════════════════════════════════════════════════════════════════════
# L&P composite  —  standard grid + helpers shared by prep script and notebook
# ══════════════════════════════════════════════════════════════════════════════

# Standard non-dimensional composite grid (Lohou & Patton 2014)
XL_GRID  = np.linspace(-1.0, 1.0, 200)   # x/L (or y/L) axis
ZND_GRID = np.linspace( 0.0, 1.0, 100)   # z/z_sl axis

# Variables saved per event (keys used in events_xz.nc / events_yz.nc)
COMPOSITE_VARS = (
    "w_thl", "w_qv", "w_prime", "ql", "thl_prime", "qt_prime",
    # circulation composite extras (added when u, v, b are included in prep)
    "b_prime", "horiz_wind_prime",
)


def chord_length_1d(labeled: np.ndarray, label: int,
                    cy: int, cx: int,
                    dx: float, dy: float,
                    orientation: str) -> float:
    """Chord length of a cloud object through the centroid row or column.

    Parameters
    ----------
    labeled : (ny, nx) int label array from find_cloud_objects
    label : object label integer
    cy, cx : centroid row / column indices
    dx, dy : grid spacing in metres
    orientation : 'y' → xz-slice (chord along x through row cy)
                  'x' → yz-slice (chord along y through column cx)
    """
    if orientation == "y":
        return float((labeled[cy, :] == label).sum()) * dx
    else:
        return float((labeled[:, cx] == label).sum()) * dy


def interp_event_to_std_grid(field_2d: np.ndarray,
                              x_nd: np.ndarray,
                              z_nd: np.ndarray,
                              xl_grid: np.ndarray = XL_GRID,
                              znd_grid: np.ndarray = ZND_GRID) -> np.ndarray:
    """Bilinear interpolation from a non-dimensional (z/z_sl, x/L) grid
    to the standard composite grid (ZND_GRID × XL_GRID).

    Parameters
    ----------
    field_2d : (nz, nhoriz) array on the original non-dim coordinates
    x_nd     : (nhoriz,) x/L coordinates of the original grid, monotone increasing
    z_nd     : (nz,)     z/z_sl coordinates of the original grid, monotone increasing
    xl_grid, znd_grid : output grid arrays (default: module-level constants)

    Returns
    -------
    (n_znd, n_xL) array on the standard grid; NaN outside the original extent
    """
    from scipy.interpolate import RegularGridInterpolator
    fn = RegularGridInterpolator(
        (z_nd, x_nd), field_2d,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    ZZ, XX = np.meshgrid(znd_grid, xl_grid, indexing="ij")  # (n_znd, n_xL)
    return fn(np.stack([ZZ.ravel(), XX.ravel()], axis=1)).reshape(
        len(znd_grid), len(xl_grid)
    )


def load_composite_events(path) -> xr.Dataset:
    """Load per-event composite file written by cloud_root_composite_prep.py."""
    return xr.open_dataset(str(path))


def composite_mean_std(events_ds: xr.Dataset,
                       vnames: tuple = COMPOSITE_VARS) -> xr.Dataset:
    """Mean and std of per-event fields in a composite events Dataset.

    Returns Dataset with ``<var>_mean``, ``<var>_std`` (dims: z_nd, xL)
    and scalar ``n_events``.
    """
    out = {}
    n = events_ds.sizes["event"]
    for vn in vnames:
        if vn not in events_ds:
            continue
        arr = events_ds[vn].values   # (n_events, n_znd, n_xL)
        out[f"{vn}_mean"] = xr.DataArray(np.nanmean(arr, axis=0), dims=["z_nd", "xL"])
        out[f"{vn}_std"]  = xr.DataArray(np.nanstd(arr,  axis=0), dims=["z_nd", "xL"])
    out["n_events"] = xr.DataArray(np.int32(n))
    return xr.Dataset(
        out,
        coords={"xL": events_ds.xL, "z_nd": events_ds.z_nd},
        attrs=events_ds.attrs,
    )


def plot_cloud_root_composite(ax_thl, ax_qt,
                               comp_ds: xr.Dataset,
                               var_thl: str = "w_thl_mean",
                               var_qt:  str = "w_qv_mean",
                               vlim_thl: tuple = None,
                               vlim_qv:  tuple = None,
                               label: str = "",
                               show_xlabel: bool = True,
                               orient: str = "xz") -> None:
    """Colour-mesh plot of a cloud-root composite on (x/L, z/z_sl) axes.

    Parameters
    ----------
    ax_thl, ax_qt : matplotlib Axes (left: heat flux, right: moisture flux)
    comp_ds       : Dataset from composite_mean_std or average_cloud_root_composite.py
    var_thl, var_qt : variable names in comp_ds
    vlim_thl, vlim_qv : (vmin, vmax) colour limits; if None, 95th-percentile symmetric
    label         : string appended to subplot titles
    show_xlabel   : if False, suppress the x-axis label (use for non-bottom rows)
    orient        : 'xz' or 'yz' — controls the x-axis label
    """
    xL   = comp_ds.xL.values
    z_nd = comp_ds.z_nd.values

    def _sym_lim(arr):
        v = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 95))
        return (-v, v)

    thl_arr = comp_ds[var_thl].values
    qv_arr  = comp_ds[var_qt].values

    if vlim_thl is None:
        vlim_thl = _sym_lim(thl_arr)
    if vlim_qv is None:
        vlim_qv = _sym_lim(qv_arr)

    pcm_thl = ax_thl.pcolormesh(xL, z_nd, thl_arr, cmap="RdBu_r", shading="auto",
                                 vmin=vlim_thl[0], vmax=vlim_thl[1])
    pcm_qv  = ax_qt.pcolormesh( xL, z_nd, qv_arr,  cmap="BrBG",   shading="auto",
                                 vmin=vlim_qv[0],  vmax=vlim_qv[1])
    plt.colorbar(pcm_thl, ax=ax_thl, label="w'θ_l'  (K m s⁻¹)")
    plt.colorbar(pcm_qv,  ax=ax_qt,  label="w'q_v'  (g kg⁻¹ m s⁻¹)")

    if "n_events_per_rep" in comp_ds:
        n_ev = int(comp_ds["n_events_per_rep"].values.sum())
    elif "n_events" in comp_ds:
        n_ev = int(comp_ds["n_events"])
    else:
        n_ev = "?"

    xlabel = "x/L" if orient == "xz" else "y/L"
    for ax in (ax_thl, ax_qt):
        ax.axvline(-0.5, color="0.4", lw=0.8, ls="--")
        ax.axvline( 0.5, color="0.4", lw=0.8, ls="--", label=f"cloud edge ({xlabel}=±½)")
        ax.axhline( 1.0, color="0.4", lw=0.8, ls=":",  label="z_sl")
        if show_xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("z / z_sl")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.0)

    suffix = f"  ({n_ev} events)"
    if label:
        suffix = f"  {label}" + suffix
    ax_thl.set_title(f"Composite w'θ_l'{suffix}")
    ax_qt.set_title( f"Composite w'q_v'{suffix}")


def plot_circulation_composite(ax, comp_ds: xr.Dataset,
                                vlim_b: tuple = None,
                                label: str = "",
                                show_xlabel: bool = True,
                                orient: str = "xz",
                                quiver_stride: tuple = (10, 5),
                                quiver_scale: float = None,
                                show_colorbar: bool = True):
    """Colour-mesh of b' with (horiz_wind', w') circulation vectors overlaid.

    Parameters
    ----------
    ax            : matplotlib Axes
    comp_ds       : Dataset from composite_mean_std / average_reps; must contain
                    b_prime_mean, horiz_wind_prime_mean, w_prime_mean.
    vlim_b        : (vmin, vmax) colour limits for b'; if None, 98th-percentile symmetric.
    label         : string appended to subplot title.
    show_xlabel   : if False, suppress the x-axis label (use for non-bottom rows).
    orient        : 'xz' or 'yz' — controls the x-axis label.
    quiver_stride : (stride_x, stride_z) subsampling of the standard grid for quivers.
    quiver_scale  : passed to ax.quiver ``scale``; if None, matplotlib auto-scales.
    show_colorbar : if False, suppress the per-axes colorbar (for shared-cbar layouts).

    Returns
    -------
    pcm : QuadMesh returned by pcolormesh (for creating a shared colorbar).
    """
    xL   = comp_ds.xL.values
    z_nd = comp_ds.z_nd.values

    b_arr = comp_ds["b_prime_mean"].values
    u_arr = comp_ds["horiz_wind_prime_mean"].values   # u' (xz) or v' (yz)
    w_arr = comp_ds["w_prime_mean"].values

    if vlim_b is None:
        v = float(np.nanpercentile(np.abs(b_arr[np.isfinite(b_arr)]), 98))
        vlim_b = (-v, v)

    pcm = ax.pcolormesh(xL, z_nd, b_arr, cmap="RdBu_r", shading="auto",
                        vmin=vlim_b[0], vmax=vlim_b[1])
    if show_colorbar:
        plt.colorbar(pcm, ax=ax, label="b'  (m s⁻²)")

    # Quiver on subsampled grid
    sx, sz = quiver_stride
    xi = np.arange(0, len(xL),   sx)
    zi = np.arange(0, len(z_nd), sz)
    Xq, Zq = np.meshgrid(xL[xi], z_nd[zi])
    Uq = u_arr[np.ix_(zi, xi)]
    Wq = w_arr[np.ix_(zi, xi)]

    qkw = dict(pivot="mid", color="k", linewidth=0.4, alpha=0.75)
    if quiver_scale is not None:
        qkw["scale"] = quiver_scale
    ax.quiver(Xq, Zq, Uq, Wq, **qkw)

    if "n_events_per_rep" in comp_ds:
        n_ev = int(comp_ds["n_events_per_rep"].values.sum())
    elif "n_events" in comp_ds:
        n_ev = int(comp_ds["n_events"])
    else:
        n_ev = "?"

    xlabel = "x/L" if orient == "xz" else "y/L"
    wind_label = "u'" if orient == "xz" else "v'"
    ax.axvline(-0.5, color="0.4", lw=0.8, ls="--")
    ax.axvline( 0.5, color="0.4", lw=0.8, ls="--")
    ax.axhline( 1.0, color="0.4", lw=0.8, ls=":")
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("z / z_sl")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.0)

    suffix = f"  ({n_ev} events)"
    if label:
        suffix = f"  {label}" + suffix
    ax.set_title(f"Composite b' + ({wind_label}, w') circulation{suffix}")
    return pcm
