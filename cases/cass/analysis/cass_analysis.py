"""
CASS LES analysis utilities.

Provides data loading, analysis functions, and plotting helpers for comparing
2stream (1D) and raytracer (3D) radiation runs across all CASS experiments.

Typical usage
-------------
    from cass_analysis import RunSet, load_stats_ensemble, load_xy_ensemble

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
import netCDF4 as nc
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
Ls  = 2.834e6  # J kg⁻¹  latent heat of sublimation
cp  = 1005.0   # J kg⁻¹ K⁻¹
g   = 9.81     # m s⁻²
rho = 1.2      # kg m⁻³  reference surface air density

FILL_VALUE = -1.0e9   # MicroHH output fill value

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
# Stats loading  (netCDF4 — handles groups cleanly)
# ══════════════════════════════════════════════════════════════════════════════

def load_stats(run_dir) -> dict:
    """Load ``cass.default.0000000.nc`` from *run_dir*.

    Returns a flat dict with arrays from all groups.

    Key entries
    -----------
    t_sec, t_local, z, zh             coordinates
    H, LE, G, S, Rnet                 SEB scalars (W m⁻²)
    theta                             soil moisture (time, 4 layers)
    qlqi_path, qlqi_cover, ql_cover   cloud scalars
    zi                                boundary-layer height (m)
    thl_w, qt_w, thv_w               resolved turbulent flux profiles at zh
    thl, qt, ql                       mean profiles at z
    sza                               solar zenith angle
    """
    run_dir = Path(run_dir)
    ds = nc.Dataset(str(run_dir / "cass.default.0000000.nc"))

    t_sec = ds.variables["time"][:]
    z  = ds.variables["z"][:]
    zh = ds.variables["zh"][:]

    start   = get_local_start(run_dir)
    t_local = pd.to_datetime(
        [start + pd.Timedelta(seconds=float(s)) for s in t_sec]
        if start is not None else t_sec
    )
    if hasattr(t_local, "tz") and t_local.tz is not None:
        t_local = t_local.tz_localize(None)

    lsm  = ds.groups["land_surface"]
    rad  = ds.groups["radiation"]
    thm  = ds.groups["thermo"]

    sw_dn = rad.variables["sw_flux_dn"][:, 0]
    sw_up = rad.variables["sw_flux_up"][:, 0]
    lw_dn = rad.variables["lw_flux_dn"][:, 0]
    lw_up = rad.variables["lw_flux_up"][:, 0]

    out = dict(
        # Coordinates
        t_sec   = np.asarray(t_sec),
        t_local = t_local,
        z       = np.asarray(z),
        zh      = np.asarray(zh),
        # Land surface scalars
        H       = np.asarray(lsm.variables["H"][:]),
        LE      = np.asarray(lsm.variables["LE"][:]),
        G       = np.asarray(lsm.variables["G"][:]),
        S       = np.asarray(lsm.variables["S"][:]),
        theta   = np.asarray(lsm.variables["theta"][:]),   # (time, 4)
        ustar   = np.asarray(lsm.variables["ustar"][:]),
        # Radiation scalars
        Rnet    = np.asarray((sw_dn - sw_up) + (lw_dn - lw_up)),
        sw_dn   = np.asarray(sw_dn),
        sw_up   = np.asarray(sw_up),
        lw_dn   = np.asarray(lw_dn),
        lw_up   = np.asarray(lw_up),
        sza     = np.asarray(rad.variables["sza"][:]),
        sw_dn_toa = np.asarray(rad.variables["sw_flux_dn_toa"][:]),
        # Thermo scalars
        qlqi_path  = np.asarray(thm.variables["qlqi_path"][:]),
        ql_cover   = np.asarray(thm.variables["ql_cover"][:]),
        qlqi_cover = np.asarray(thm.variables["qlqi_cover"][:]),
        zi         = np.asarray(thm.variables["zi"][:]),
        thl_bot    = np.asarray(thm.variables["thl_bot"][:]),
        qt_bot     = np.asarray(thm.variables["qt_bot"][:]),
        # Thermo profiles at zh — resolved turbulent fluxes
        thl_w   = np.asarray(thm.variables["thl_w"][:]),   # (time, nzh)
        qt_w    = np.asarray(thm.variables["qt_w"][:]),
        thv_w   = np.asarray(thm.variables["thv_w"][:]),
        # Thermo profiles at z — mean fields
        thl     = np.asarray(thm.variables["thl"][:]),
        qt      = np.asarray(thm.variables["qt"][:]),
        ql      = np.asarray(thm.variables["ql"][:]),
        ql_frac = np.asarray(thm.variables["ql_frac"][:]),   # (time, nz)
    )
    ds.close()
    return out


def load_stats_ensemble(rep_dirs: list) -> tuple[dict, dict]:
    """Load stats from multiple reps; return (ensemble_mean, ensemble_std).

    Time axes are truncated to the shortest run before stacking.
    """
    all_stats = [load_stats(d) for d in rep_dirs]
    if not all_stats:
        raise ValueError("No rep dirs provided")

    nt_min = min(s["t_sec"].shape[0] for s in all_stats)

    # Spatial coords: copy as-is (never slice along time axis)
    SPATIAL = {"z", "zh"}
    # t_local: slice to nt_min but don't stack across reps
    TIMECOPY = {"t_local"}
    mean_s, std_s = {}, {}

    for key in all_stats[0]:
        if key in SPATIAL:
            mean_s[key] = all_stats[0][key]
            std_s[key]  = None
            continue
        if key in TIMECOPY:
            mean_s[key] = all_stats[0][key][:nt_min]
            std_s[key]  = None
            continue
        try:
            arr = np.stack([s[key][:nt_min] for s in all_stats], axis=0)
            mean_s[key] = arr.mean(axis=0)
            std_s[key]  = arr.std(axis=0, ddof=0)
        except Exception:
            mean_s[key] = all_stats[0][key]
            std_s[key]  = None

    # Align t_local to nt_min
    mean_s["t_local"] = all_stats[0]["t_local"][:nt_min]
    return mean_s, std_s


# ══════════════════════════════════════════════════════════════════════════════
# XY loading  (xarray + dask for lazy evaluation)
# ══════════════════════════════════════════════════════════════════════════════

_XY_VARS_DEFAULT = [
    "qlqi_path", "qlqi_base", "qlqi_top", "ql",
    "thl_fluxbot", "qt_fluxbot",
    "sw_flux_dn", "sw_flux_dn_dir",
    "lw_flux_dn", "lw_flux_up",
    "sw_flux_sfc_dir_rt", "sw_flux_sfc_dif_rt",   # raytracer only
    "b", "thl", "w",
]


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
        variables = _XY_VARS_DEFAULT
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
        ds_v = ds_v.where(ds_v != FILL_VALUE)
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


def load_xy_ensemble(rep_dirs: list, variables=None,
                     chunks=None) -> tuple[xr.Dataset, xr.Dataset]:
    """Load xy files from multiple reps; return (ensemble_mean, ensemble_std).

    Time axes are truncated to the shortest run before computing statistics.
    """
    all_ds = [load_xy_files(d, variables, chunks) for d in rep_dirs]
    nt_min = min(ds.sizes["time"] for ds in all_ds)
    all_ds = [ds.isel(time=slice(None, nt_min)) for ds in all_ds]
    stack  = xr.concat(all_ds, dim="rep")
    return stack.mean("rep"), stack.std("rep")


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

    per_rep = []
    for d in rep_dirs:
        ds = load_xy_files(d, variables + ["qlqi_path"], chunks={"time": 200})
        mask = ds["qlqi_path"] > 0
        shaded_means   = {v: ds[v].where(mask).mean(["x", "y"])  for v in variables if v in ds}
        unshaded_means = {v: ds[v].where(~mask).mean(["x", "y"]) for v in variables if v in ds}
        per_rep.append({"shaded": shaded_means, "unshaded": unshaded_means})

    nt_min = min(
        per_rep[0]["shaded"][list(per_rep[0]["shaded"])[0]].sizes["time"]
        for _ in per_rep
    )

    out_mean = {"shaded": {}, "unshaded": {}}
    out_std  = {"shaded": {}, "unshaded": {}}
    for kind in ("shaded", "unshaded"):
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


def load_3d_nc(run_dir, variables=None, chunks=None) -> xr.Dataset:
    """Load 3D netCDF dump files produced by ``python/3d_to_nc.py``.

    Expects files named ``{var}.nc`` in *run_dir*.  Adds perturbation fields
    (w_prime, thl_prime, qt_prime) and interpolates w to cell-centre levels.

    Prerequisites
    -------------
    From the run directory, run::

        python $MICROHH/python/3d_to_nc.py -v thl qt ql w b

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
    for var, prime in [("thl", "thl_prime"), ("qt", "qt_prime")]:
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

def seb_residual(stats: dict) -> np.ndarray:
    """Rnet − (H + LE + G + S)  [W m⁻²]."""
    return stats["Rnet"] - (stats["H"] + stats["LE"] + stats["G"] + stats["S"])


def bowen_ratio(stats: dict, le_min: float = 1.0) -> np.ndarray:
    """H / LE, masked where LE < *le_min*  (avoids near-zero division)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(stats["LE"] > le_min, stats["H"] / stats["LE"], np.nan)


def evaporative_fraction(stats: dict, le_min: float = 1.0) -> np.ndarray:
    """LE / (H + LE), masked where (H + LE) is small."""
    denom = stats["H"] + stats["LE"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denom) > le_min, stats["LE"] / denom, np.nan)


def compute_z_sl(stats: dict, time_idx: int = -1) -> float:
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


def extract_root_slice(ds_3d_t: xr.Dataset, cy: int, cx: int,
                       orientation: str = "y") -> xr.Dataset:
    """Extract a vertical slice through a cloud root centroid.

    Parameters
    ----------
    ds_3d_t : Dataset at a single time  (dims: z or zh, y, x)
    cy, cx : centroid grid indices
    orientation : 'y' → xz-slice at y=cy; 'x' → yz-slice at x=cx

    Returns
    -------
    xr.Dataset with dims (z, x) or (z, y)
    """
    if orientation == "y":
        return ds_3d_t.isel(y=cy)
    return ds_3d_t.isel(x=cx)


def nondim_slice(sl: xr.Dataset, cloud_mask_1d: np.ndarray,
                 horiz_coords: np.ndarray, z_sl: float) -> xr.Dataset:
    """Non-dimensionalise a vertical slice by cloud-root width and subcloud-layer height.

    Horizontal axis: x → (x − centroid) / L, where L is the cloud-root width
    along the slice (number of LWP > 0 columns × dx).
    Vertical axis:   z → z / z_sl.

    Parameters
    ----------
    sl : Dataset from extract_root_slice
    cloud_mask_1d : bool array (nx or ny) — True where LWP > 0 (cloud-root columns)
    horiz_coords : coordinate values [m] corresponding to cloud_mask_1d
    z_sl : subcloud-layer height [m]
    """
    sl = sl.copy(deep=False)

    # L = width of the cloud-root region along the slice
    cloud_indices = np.where(cloud_mask_1d)[0]
    if len(cloud_indices) == 0:
        raise ValueError("No cloud-root pixels in this slice")
    L = float(len(cloud_indices)) * float(horiz_coords[1] - horiz_coords[0])
    centroid_m = float(horiz_coords[cloud_indices].mean())

    # Identify horizontal dim
    horiz_dim = "x" if "x" in sl.dims else "y"
    coords_m = sl[horiz_dim].values.astype(float)
    sl = sl.assign_coords({horiz_dim: (coords_m - centroid_m) / L})
    sl[horiz_dim].attrs["long_name"] = "x/L"

    # Rescale vertical
    for zdim in ("z", "zh"):
        if zdim in sl.coords:
            sl = sl.assign_coords({zdim: sl[zdim].values / z_sl})
            sl[zdim].attrs["long_name"] = f"{zdim}/z_sl"

    sl.attrs["L_m"] = L
    sl.attrs["z_sl_m"] = z_sl
    return sl


def extract_cloud_root_profiles_1d(ds_3d_t: xr.Dataset,
                                    mask_2d: np.ndarray) -> dict:
    """Vertical profiles of turbulent fluxes conditioned on cloud-root columns (LWP > 0).

    Computes instantaneous w'θ_l' and w'q_v' and averages over three partitions:
    cloud-root columns (LWP > 0), environment columns (LWP = 0), and the full domain.

    Parameters
    ----------
    ds_3d_t : Dataset at a single time, must contain w_prime, thl_prime, qv_prime
              (dims: z, y, x).
    mask_2d : (ny, nx) bool — True where LWP > 0.

    Returns
    -------
    dict with keys ``'cloud_root'``, ``'env'``, ``'domain'``, each a dict with:
        ``flux_thl`` (K m s⁻¹), ``flux_qv`` (g kg⁻¹ m s⁻¹), ``z`` (m).
    """
    for req in ("w_prime", "thl_prime", "qv_prime"):
        if req not in ds_3d_t:
            raise ValueError(
                f"Dataset missing '{req}'; ensure load_3d_nc() loaded qt and ql."
            )

    flux_thl = (ds_3d_t["w_prime"] * ds_3d_t["thl_prime"]).squeeze()
    flux_qv  = (ds_3d_t["w_prime"] * ds_3d_t["qv_prime"]).squeeze() * 1e3

    zdim = "z" if "z" in flux_thl.dims else "zh"
    z    = flux_thl[zdim].values

    mask_da = xr.DataArray(mask_2d, dims=["y", "x"])

    def _cmean(field, m):
        return field.where(m).mean(["x", "y"]).values

    return {
        "cloud_root": {"flux_thl": _cmean(flux_thl,  mask_da),
                       "flux_qv":  _cmean(flux_qv,   mask_da), "z": z},
        "env":        {"flux_thl": _cmean(flux_thl, ~mask_da),
                       "flux_qv":  _cmean(flux_qv,  ~mask_da), "z": z},
        "domain":     {"flux_thl": flux_thl.mean(["x", "y"]).values,
                       "flux_qv":  flux_qv.mean(["x", "y"]).values,  "z": z},
    }


def plot_cloud_root_profiles_comparison(ax_thl, ax_qt,
                                         prof_2s: dict, prof_rt: dict,
                                         z_sl_2s: float = None,
                                         z_sl_rt: float = None,
                                         condition: str = "cloud_root") -> None:
    """Compare 1D vertical flux profiles conditioned on cloud root for both RT types.

    Parameters
    ----------
    ax_thl, ax_qt : matplotlib Axes (left: heat flux, right: moisture flux)
    prof_2s, prof_rt : dicts from extract_cloud_root_profiles_1d
    z_sl_2s, z_sl_rt : subcloud-layer heights [m] for reference lines
    condition : ``'cloud_root'``, ``'env'``, or ``'domain'``
    """
    _COND_LABEL = {
        "cloud_root": "cloud-root columns (LWP > 0)",
        "env":        "environment (LWP = 0)",
        "domain":     "domain mean",
    }
    for rt, prof, z_sl in [("2stream", prof_2s, z_sl_2s),
                            ("raytracer", prof_rt, z_sl_rt)]:
        p   = prof[condition]
        sty = RT_STYLE[rt]
        ax_thl.plot(p["flux_thl"], p["z"], label=RT_LABEL[rt], **sty)
        ax_qt.plot( p["flux_qv"],  p["z"], label=RT_LABEL[rt], **sty)
        if z_sl is not None:
            for ax in (ax_thl, ax_qt):
                ax.axhline(z_sl, color=sty["color"], ls=":", lw=1,
                           label=f"z_sl {RT_LABEL[rt]}")

    for ax in (ax_thl, ax_qt):
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_ylabel("z  (m)")

    ax_thl.set_xlabel("w'θ_l'  (K m s⁻¹)")
    ax_qt.set_xlabel("w'q_v'  (g kg⁻¹ m s⁻¹)")
    ax_thl.set_title(f"Turbulent fluxes  —  {_COND_LABEL.get(condition, condition)}")
    ax_thl.legend(fontsize=8)


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
      ``thl_cloud_nd`` (n_times, n_zeta) – normalised cloud-root heat flux on zeta grid
      ``qv_cloud_nd``  (n_times, n_zeta) – normalised cloud-root moisture flux on zeta grid
    """
    for req in ("w_prime", "thl_prime", "qv_prime"):
        if req not in ds_3d:
            raise ValueError(f"Dataset missing '{req}'; ensure load_3d_nc() loaded qt and ql.")

    zdim = "z" if "z" in ds_3d.dims else "zh"
    z    = ds_3d[zdim].values

    zeta_grid = np.linspace(0, 1.5, 200)

    thl_cloud_nd_list, qv_cloud_nd_list = [], []
    norm_thl_list, norm_qv_list         = [], []
    z_sl_list                           = []

    xy_times_f = ds_xy.time.values.astype("datetime64[ns]").astype(float)
    st_times_f = stats_mean["t_local"].values.astype("datetime64[ns]").astype(float)

    for tidx_3d in range(ds_3d.sizes["time"]):
        t3d   = ds_3d.time.values[tidx_3d]
        t3d_f = np.datetime64(t3d, "ns").astype(float)
        ds_t  = ds_3d.isel(time=tidx_3d).load()

        # Cloud mask from nearest XY time
        mask_2d = cloud_mask_2d(
            ds_xy["qlqi_path"].isel(time=int(np.argmin(np.abs(xy_times_f - t3d_f)))).values
        )

        # z_sl from stats at nearest time
        z_sl = compute_z_sl(stats_mean, time_idx=int(np.argmin(np.abs(st_times_f - t3d_f))))

        # Flux fields
        flux_thl = (ds_t["w_prime"] * ds_t["thl_prime"]).squeeze()
        flux_qv  = (ds_t["w_prime"] * ds_t["qv_prime"]).squeeze() * 1e3
        mask_da  = xr.DataArray(mask_2d, dims=["y", "x"])

        f_cloud_thl  = flux_thl.where(mask_da).mean(["x", "y"]).values
        f_cloud_qv   = flux_qv.where(mask_da).mean(["x", "y"]).values
        f_domain_thl = flux_thl.mean(["x", "y"]).values
        f_domain_qv  = flux_qv.mean(["x", "y"]).values

        # Sub-cloud integral of domain mean (normalization scalar)
        scl      = z <= z_sl
        norm_thl = float(np.trapz(f_domain_thl[scl], z[scl]))
        norm_qv  = float(np.trapz(f_domain_qv[scl],  z[scl]))

        # Non-dimensionalise z → ζ = z/z_sl(t), interpolate onto common grid
        zeta_t = z / z_sl
        thl_cloud_nd_list.append(np.interp(zeta_grid, zeta_t, f_cloud_thl / norm_thl))
        qv_cloud_nd_list.append( np.interp(zeta_grid, zeta_t, f_cloud_qv  / norm_qv))

        norm_thl_list.append(norm_thl)
        norm_qv_list.append(norm_qv)
        z_sl_list.append(z_sl)

    return {
        "zeta":         zeta_grid,
        "z":            z,
        "z_sl":         np.array(z_sl_list),
        "norm_thl":     np.array(norm_thl_list),
        "norm_qv":      np.array(norm_qv_list),
        "thl_cloud_nd": np.array(thl_cloud_nd_list),   # (n_times, n_zeta)
        "qv_cloud_nd":  np.array(qv_cloud_nd_list),
    }


def plot_normalized_cloud_root_profiles(ax_thl, ax_qt,
                                         result_2s: dict, result_rt: dict) -> None:
    """Time-mean normalised cloud-root flux profiles ± 1 std, 2stream vs raytracer.

    Parameters
    ----------
    ax_thl, ax_qt  : matplotlib Axes (left: heat flux, right: moisture flux)
    result_2s, result_rt : dicts from compute_normalized_cloud_root_profiles
    """
    for rt, res in [("2stream", result_2s), ("raytracer", result_rt)]:
        sty  = RT_STYLE[rt]
        zeta = res["zeta"]

        mu_thl = np.nanmean(res["thl_cloud_nd"], axis=0)
        sd_thl = np.nanstd( res["thl_cloud_nd"], axis=0)
        mu_qv  = np.nanmean(res["qv_cloud_nd"],  axis=0)
        sd_qv  = np.nanstd( res["qv_cloud_nd"],  axis=0)

        ax_thl.plot(mu_thl, zeta, label=RT_LABEL[rt], **sty)
        ax_thl.fill_betweenx(zeta, mu_thl - sd_thl, mu_thl + sd_thl,
                              alpha=0.15, color=sty["color"])
        ax_qt.plot(mu_qv, zeta, label=RT_LABEL[rt], **sty)
        ax_qt.fill_betweenx(zeta, mu_qv - sd_qv, mu_qv + sd_qv,
                             alpha=0.15, color=sty["color"])

    for ax in (ax_thl, ax_qt):
        ax.axhline(1.0, color="gray", ls="--", lw=1, label="z_sl")
        ax.axvline(0,   color="gray", lw=0.5)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("z / z_sl")

    ax_thl.set_xlabel("w'θ_l' / ∫w'θ_l' dz  (m⁻¹)")
    ax_qt.set_xlabel("w'q_v' / ∫w'q_v' dz  (m⁻¹)")
    ax_thl.set_title("Normalised cloud-root heat flux")
    ax_qt.set_title("Normalised cloud-root moisture flux")
    ax_thl.legend(fontsize=8)


# ══════════════════════════════════════════════════════════════════════════════
# LWP analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_lwp_diff(ds_2s: xr.Dataset, ds_rt: xr.Dataset) -> xr.DataArray:
    """Domain-mean LWP difference  (3D − 1D)  in g m⁻².

    Time axes are aligned to the shorter run.
    """
    lwp_1d = ds_2s["qlqi_path"].mean(["x", "y"]) * 1e3
    lwp_3d = ds_rt["qlqi_path"].mean(["x", "y"]) * 1e3
    nt = min(lwp_1d.sizes["time"], lwp_3d.sizes["time"])
    diff = lwp_3d.isel(time=slice(None, nt)) - lwp_1d.isel(time=slice(None, nt))
    diff.attrs["units"] = "g m⁻²"
    diff.attrs["long_name"] = "LWP difference (3D − 1D)"
    return diff


def compute_lwp_diff_from_stats(stats_2s: dict, stats_rt: dict) -> tuple[np.ndarray, np.ndarray]:
    """Same as compute_lwp_diff but from stats dicts.

    Returns (diff_gm2, t_local) — useful for single-rep debug runs.
    """
    nt = min(len(stats_2s["qlqi_path"]), len(stats_rt["qlqi_path"]))
    diff = (stats_rt["qlqi_path"][:nt] - stats_2s["qlqi_path"][:nt]) * 1e3
    t    = stats_2s["t_local"][:nt]
    return diff, t


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


def plot_cloud_cover(ax, stats_mean: dict, rt_type: str = "2stream") -> plt.Axes:
    """Panel: cloud fraction and LWP."""
    t   = stats_mean["t_local"]
    sty = RT_STYLE[rt_type]
    ax.plot(t, stats_mean["qlqi_cover"] * 100,
            label=f"{RT_LABEL[rt_type]} liq+ice", **sty)
    ax.plot(t, stats_mean["ql_cover"] * 100, color=sty["color"],
            ls=sty["ls"], alpha=0.5, label=f"{RT_LABEL[rt_type]} liquid")
    ax.set_ylabel("Cloud fraction (%)")
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
    ax.plot(t_num, um, color="darkorange", ls=ls, label="unshaded")
    if shaded_std is not None:
        _shade_ensemble(ax, t_local, sm, np.asarray(shaded_std) * scale, "steelblue")
    if unshaded_std is not None:
        _shade_ensemble(ax, t_local, um, np.asarray(unshaded_std) * scale, "darkorange")
    ax.axhline(0, color="gray", lw=0.5)
    ax.xaxis_date()
    ax.set_ylabel(ylabel)
    ax.figure.autofmt_xdate()
    return ax


def plot_lwp_comparison(ax,
                        ds_2s_mean: xr.Dataset, ds_2s_std: xr.Dataset,
                        ds_rt_mean: xr.Dataset, ds_rt_std: xr.Dataset) -> plt.Axes:
    """Panel: LWP timeseries for both RT types with ensemble spread."""
    for rt, ds_m, ds_s in [
        ("2stream",   ds_2s_mean, ds_2s_std),
        ("raytracer", ds_rt_mean, ds_rt_std),
    ]:
        lwp = ds_m["qlqi_path"].mean(["x", "y"]) * 1e3
        t   = ds_m.time
        t_num = _to_plottime(t.values)
        sty = RT_STYLE[rt]
        ax.plot(t_num, lwp.values, label=RT_LABEL[rt], **sty)
        if ds_s is not None and "qlqi_path" in ds_s:
            lwp_s = ds_s["qlqi_path"].mean(["x", "y"]) * 1e3
            _shade_ensemble(ax, t.values, lwp.values, lwp_s.values, sty["color"])
    ax.set_ylabel("LWP  (g m⁻²)")
    ax.xaxis_date()
    ax.legend(fontsize=8)
    ax.figure.autofmt_xdate()
    return ax


def plot_lwp_diff_timeseries(ax, diff_da: xr.DataArray,
                             integral_val: float = None,
                             smooth_n: int = 12) -> plt.Axes:
    """Panel: LWP difference (3D − 1D) with running-mean smoothing."""
    t     = diff_da.time.values
    t_num = _to_plottime(t)
    y     = diff_da.values
    y_s   = pd.Series(y).rolling(smooth_n, center=True, min_periods=1).mean().values
    ax.plot(t_num, y,   color="gray", alpha=0.4, lw=0.8, label="instantaneous")
    ax.plot(t_num, y_s, color="k",    lw=1.5,            label=f"{smooth_n}-pt mean")
    ax.axhline(0, color="gray", lw=0.5)
    title = "3D − 1D LWP"
    if integral_val is not None:
        title += f"   |   ∫dt = {integral_val:+.1f} g m⁻² h  (cloudy period)"
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("ΔLWP  (g m⁻²)")
    ax.xaxis_date()
    ax.legend(fontsize=8)
    ax.figure.autofmt_xdate()
    return ax


def plot_flux_profiles(ax_thl, ax_qt, stats_mean: dict,
                       rt_type: str = "2stream",
                       time_idx: int = -1, z_sl: float = None) -> None:
    """Panels: vertical profiles of resolved w'θ_l' and w'q_t' at one time."""
    zh  = stats_mean["zh"]
    sty = RT_STYLE[rt_type]
    ax_thl.plot(stats_mean["thl_w"][time_idx], zh,
                label=RT_LABEL[rt_type], **sty)
    ax_qt.plot(stats_mean["qt_w"][time_idx] * 1e3, zh,
               label=RT_LABEL[rt_type], **sty)
    for ax in (ax_thl, ax_qt):
        ax.axvline(0, color="gray", lw=0.5)
        if z_sl is not None:
            ax.axhline(z_sl, color="gray", ls=":", lw=1, label="z_sl")
    ax_thl.set_xlabel("w'θ_l'  (K m s⁻¹)")
    ax_qt.set_xlabel("w'q_t'  (g kg⁻¹ m s⁻¹)")
    ax_thl.set_ylabel("z  (m)")


def plot_cloud_root_slice(ax_thl, ax_qt, sl_nd: xr.Dataset,
                          cloud_mask_1d: np.ndarray = None,
                          zmax_sl: float = 1.0,
                          vlim_thl: tuple = None,
                          vlim_qv: tuple  = None) -> None:
    """Panels: w'θ_l' and w'q_v' in a non-dimensionalised cloud-root slice.

    Parameters
    ----------
    sl_nd : non-dimensionalised Dataset from nondim_slice
    cloud_mask_1d : bool array marking the cloud-root region (plotted as span)
    zmax_sl : upper z/z_sl limit for display (default 1.0 = cloud base)
    vlim_thl : (vmin, vmax) for w'θ_l' colorbar; if None, computed from data (95th pct)
    vlim_qv  : (vmin, vmax) for w'q_v' colorbar; if None, computed from data (95th pct)
    """
    horiz_dim = "x" if "x" in sl_nd.dims else "y"

    if "thl_prime" not in sl_nd or "w_prime" not in sl_nd:
        raise ValueError("Dataset missing thl_prime or w_prime; check load_3d_nc()")
    if "qv_prime" not in sl_nd:
        raise ValueError("Dataset missing qv_prime; ensure qt and ql were loaded in load_3d_nc()")

    flux_thl = (sl_nd["thl_prime"] * sl_nd["w_prime"]).squeeze()
    flux_qv  = (sl_nd["qv_prime"]  * sl_nd["w_prime"]).squeeze() * 1e3

    zdim = "z" if "z" in flux_thl.dims else "zh"
    h    = flux_thl[zdim].values
    x    = flux_thl[horiz_dim].values

    # Clip to subcloud layer for both display and color scaling
    z_mask = h <= zmax_sl
    flux_thl = flux_thl.isel({zdim: z_mask})
    flux_qv  = flux_qv.isel( {zdim: z_mask})
    h = h[z_mask]

    def _sym_lim(arr):
        v = float(np.nanpercentile(np.abs(arr), 95))
        return -v, v

    if vlim_thl is None:
        vlim_thl = _sym_lim(flux_thl.values)
    if vlim_qv is None:
        vlim_qv  = _sym_lim(flux_qv.values)

    pcm_thl = ax_thl.pcolormesh(x, h, flux_thl.values,
                                 cmap="RdBu_r", shading="auto",
                                 vmin=vlim_thl[0], vmax=vlim_thl[1])
    pcm_qv  = ax_qt.pcolormesh( x, h, flux_qv.values,
                                 cmap="BrBG",   shading="auto",
                                 vmin=vlim_qv[0],  vmax=vlim_qv[1])
    plt.colorbar(pcm_thl, ax=ax_thl, label="w'θ_l'  (K m s⁻¹)")
    plt.colorbar(pcm_qv,  ax=ax_qt,  label="w'q_v'  (g kg⁻¹ m s⁻¹)")

    for ax in (ax_thl, ax_qt):
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, zmax_sl)
        ax.set_xlabel(f"{horiz_dim}/L")
        ax.set_ylabel("z / z_sl")
        if cloud_mask_1d is not None:
            pass  # shadow region already centred at 0; shaded area is |x| ≤ 0.5
        ax.axvspan(-0.5, 0.5, color="gray", alpha=0.08, label="cloud root")

    L_m   = sl_nd.attrs.get("L_m", "?")
    z_sl_m = sl_nd.attrs.get("z_sl_m", "?")
    ax_thl.set_title(f"Cloud-root cross-section  (L={L_m:.0f} m, z_sl={z_sl_m:.0f} m)")
