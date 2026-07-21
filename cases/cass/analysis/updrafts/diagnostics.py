"""
H1/H2 updraft diagnostics for the Couvreux-tracer-based pipeline.

Functions map 1-to-1 onto PLAN.md §§1–4.  All public functions take and return
xr.DataArray / xr.Dataset for composability with the rest of the analysis.

The physics
-----------
H1. Updraft mass flux               M_up(z, t) = ρ · a_up · w_up
H2. Tracer-dilution entrainment     ε(z, t)    = -(1/χ) · dχ/dz,
                                    χ = C_cloud / C_undil,
                                    C_undil = C_cloud(z_b) · exp(-Δt/τ)

References
----------
Couvreux et al. (2010), QJRMS — offline mask and σ_min floor.
Romps (2010), JAS — tracer-dilution entrainment definition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid

# Resolve upstream helpers from the parent analysis package
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from cass_analysis import (                                  # noqa: E402
    compute_z_sl, cloud_top_za, load_3d_nc, load_stats, rho as RHO_REF,
)


# ══════════════════════════════════════════════════════════════════════════════
# §1 — Paper-strict Couvreux mask
# ══════════════════════════════════════════════════════════════════════════════

def compute_sigma_min(sigma_C: xr.DataArray, z, sigma_min_frac: float = 0.05
                      ) -> xr.DataArray:
    """Paper-strict σ_min floor (Couvreux et al. 2010, Eq. 2b).

    σ_min(z, t) = sigma_min_frac · (1/z) · ∫₀ᶻ σ_C(z', t) dz'

    Parameters
    ----------
    sigma_C : xr.DataArray
        σ-over-horizontal of the Couvreux tracer; must have a ``z`` dim.
    z : array-like
        Vertical coordinate (m).  Must be strictly increasing and start at 0.
    sigma_min_frac : float
        Floor fraction; paper uses 0.05.

    Returns
    -------
    xr.DataArray with the same dims as ``sigma_C``.
    """
    z_arr = np.asarray(z)
    z_ax  = sigma_C.dims.index("z")
    sc    = sigma_C.values

    # cumulative ∫ σ_C dz along z, starting at 0
    cum = cumulative_trapezoid(sc, z_arr, axis=z_ax, initial=0.0)

    # broadcast-shape 1/z along the z axis
    shape = [1] * sc.ndim
    shape[z_ax] = len(z_arr)
    z_rs = z_arr.reshape(shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        sm = sigma_min_frac * cum / np.where(z_rs > 0, z_rs, np.nan)

    return xr.DataArray(sm, dims=sigma_C.dims, coords=sigma_C.coords,
                        name="sigma_min")


def _per_time_z_b_z_t(ds_3d, stats):
    """Per-dump-time z_b, z_t from the nearest stats time step."""
    t_3d = ds_3d.time.values.astype("datetime64[ns]").astype("float64")
    t_st = stats["t_local"].values.astype("datetime64[ns]").astype("float64")
    z_b  = np.empty(len(t_3d))
    z_t  = np.empty(len(t_3d))
    ql_frac_all = stats["ql_frac"].values                                   # (nt_st, nz)
    z_grid      = stats["z"].values
    for i, t in enumerate(t_3d):
        idx      = int(np.argmin(np.abs(t_st - t)))
        z_b[i]   = compute_z_sl(stats, time_idx=idx)
        z_t[i]   = cloud_top_za(z_grid, ql_frac_all[idx])
    return z_b, z_t


def build_couvreux_mask(ds_3d: xr.Dataset,
                        m: float = 1.0,
                        sigma_min_frac: float = 0.05,
                        stats: xr.Dataset | None = None,
                        ql_threshold: float = 1e-5,
                        ) -> dict[str, xr.DataArray]:
    """Offline paper-strict Couvreux updraft mask.

    Implements Couvreux et al. (2010) Eqs. 2 and 3:
        thr(z, t) = ⟨C⟩_xy(z, t) + m · max(σ_C(z, t), σ_min(z, t))
        mask      = (C > thr) AND (w > 0)                                 [below z_ref]
                  = (C > thr) AND (w > 0) AND (q_l > ql_threshold)        [above z_ref]

    with z_ref = z_b + (z_t − z_b)/4.  σ_min floor (compute_sigma_min) is
    **essential** — without it, the above-BL zero-mean/zero-σ region would
    produce a threshold below every tracer value and a_up → ~1.

    Parameters
    ----------
    ds_3d : xr.Dataset
        Must contain ``couvreux`` and either ``w_cc`` or ``w`` (on zh).  If
        ``w_cc`` is absent it's interpolated from ``w``.
    stats : xr.Dataset, optional
        Domain-mean stats for z_b, z_t lookup; falls back to fixed 1000/2000 m
        if ``None`` (not recommended).

    Returns
    -------
    dict[str, xr.DataArray] with keys
        mask_paper        — Eq 2/3: tracer + w>0 (and q_l above z_ref)
        mask_tracer_only  — tracer threshold only, no w or q_l criterion
        mask_cloudy_up    — mask_paper ∧ (q_l > ql_threshold)
        mask_clear_up     — mask_paper ∧ (q_l ≤ ql_threshold)
    """
    if "couvreux" not in ds_3d:
        raise ValueError("ds_3d is missing 'couvreux'; include it in "
                         "load_3d_nc(variables=...) or run 3d_to_nc.py with "
                         "`-v ... couvreux`.")

    C  = ds_3d["couvreux"]
    ql = ds_3d["ql"] if "ql" in ds_3d else xr.zeros_like(C)
    w  = ds_3d["w_cc"] if "w_cc" in ds_3d else ds_3d["w"].interp(zh=ds_3d["z"])

    # Horizontal mean and std of the tracer per (z, t)
    C_mean  = C.mean(["x", "y"])
    sigma_C = C.std(["x", "y"])

    # Paper-strict floor and effective σ
    sigma_min = compute_sigma_min(sigma_C, ds_3d["z"].values, sigma_min_frac)
    sigma_eff = xr.where(sigma_C > sigma_min, sigma_C, sigma_min)

    thr = C_mean + m * sigma_eff                                            # (time, z)

    tracer_hit = C > thr                                                    # (time, z, y, x)

    # Transition height z_ref per time
    if stats is not None:
        z_b_arr, z_t_arr = _per_time_z_b_z_t(ds_3d, stats)
    else:
        z_b_arr = np.full(len(ds_3d.time), 1000.0)
        z_t_arr = np.full(len(ds_3d.time), 2000.0)
    z_b_da = xr.DataArray(z_b_arr, dims="time", coords={"time": ds_3d.time})
    z_t_da = xr.DataArray(z_t_arr, dims="time", coords={"time": ds_3d.time})
    z_ref  = z_b_da + 0.25 * (z_t_da - z_b_da)                              # (time,)

    below_ref = ds_3d["z"] < z_ref                                          # (time, z)
    cloudy    = ql > ql_threshold                                           # (time, z, y, x)

    # Eq 2 applies below z_ref; Eq 3 (add q_l>0) applies above
    mask_rising        = tracer_hit & (w > 0.0)
    mask_paper         = mask_rising & (below_ref | cloudy)

    return dict(
        mask_paper       = mask_paper,
        mask_tracer_only = tracer_hit,
        mask_cloudy_up   = mask_paper & cloudy,
        mask_clear_up    = mask_paper & (~cloudy),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §2 — H1: updraft mass flux
# ══════════════════════════════════════════════════════════════════════════════

def updraft_mass_flux(ds_3d: xr.Dataset,
                      masks: dict[str, xr.DataArray],
                      rho: float = RHO_REF,
                      ) -> xr.Dataset:
    """Per-mask a_up, w_up, M_up.

    Returns a single Dataset with per-mask fields suffixed by the short mask
    name (e.g. ``a_up_paper``, ``w_up_cloudy_up``, ``M_up_clear_up``).
    """
    w = ds_3d["w_cc"] if "w_cc" in ds_3d else ds_3d["w"].interp(zh=ds_3d["z"])
    out: dict[str, xr.DataArray] = {}
    for key, mask in masks.items():
        short = key.replace("mask_", "")
        a  = mask.mean(["x", "y"])
        wm = w.where(mask).mean(["x", "y"])
        out[f"a_up_{short}"]  = a.rename(f"a_up_{short}")
        out[f"w_up_{short}"]  = wm.rename(f"w_up_{short}")
        out[f"M_up_{short}"]  = (rho * a * wm).rename(f"M_up_{short}")
    return xr.Dataset(out)


# ══════════════════════════════════════════════════════════════════════════════
# §3 — H2: tracer-dilution entrainment
# ══════════════════════════════════════════════════════════════════════════════

def entrainment_rate_tracer(ds_3d: xr.Dataset,
                            masks: dict[str, xr.DataArray],
                            stats: xr.Dataset,
                            tau: float = 900.0,
                            w_min: float = 0.05,
                            ) -> xr.Dataset:
    """Tracer-dilution entrainment rate ε(z, t) inside cloudy updrafts.

    Χ = C_cloud / C_undil,  C_undil = C_cloud(z_b) · exp(−Δt(z)/τ),
    Δt(z, t) = ∫_{z_b}^{z} dz'/w_cloud(z', t).

    ε(z, t) = −(1/χ) · dχ/dz (Romps 2010).

    Parameters
    ----------
    tau : float
        Tracer decay time scale in seconds (matches the emission config).
    w_min : float
        m/s floor on w_cloud to keep 1/w finite in weakly rising parcels.

    Returns
    -------
    xr.Dataset with (time, z) fields: C_cloud, w_cloud, dt_trans, C_undil,
    chi, eps.
    """
    mask_cu = masks["mask_cloudy_up"]
    C  = ds_3d["couvreux"]
    w  = ds_3d["w_cc"] if "w_cc" in ds_3d else ds_3d["w"].interp(zh=ds_3d["z"])
    z  = ds_3d["z"]
    z_v = z.values

    C_cloud = C.where(mask_cu).mean(["x", "y"])                             # (time, z)
    w_cloud = w.where(mask_cu).mean(["x", "y"])

    z_b_arr, _ = _per_time_z_b_z_t(ds_3d, stats)

    n_t, n_z = C_cloud.sizes["time"], C_cloud.sizes["z"]
    dt_trans = np.full((n_t, n_z), np.nan)
    C_undil  = np.full((n_t, n_z), np.nan)
    chi_arr  = np.full((n_t, n_z), np.nan)

    w_vals = w_cloud.values
    C_vals = C_cloud.values

    for i in range(n_t):
        zb = z_b_arr[i]
        if not np.isfinite(zb) or zb <= 0:
            continue
        # First z-level at or above cloud base
        i_cb = int(np.searchsorted(z_v, zb))
        if i_cb >= n_z - 1:
            continue

        w_slc = w_vals[i, i_cb:]
        z_slc = z_v[i_cb:]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_w = np.where(w_slc > w_min, 1.0 / w_slc, np.nan)

        # Cumulative trapezoid of 1/w along z (starting at 0 at cloud base)
        dt_slc = np.zeros_like(z_slc)
        dz     = np.diff(z_slc)
        trap   = 0.5 * (inv_w[:-1] + inv_w[1:]) * dz
        dt_slc[1:] = np.nancumsum(trap)
        dt_trans[i, i_cb:] = dt_slc

        C_zb = C_vals[i, i_cb]
        if not np.isfinite(C_zb) or C_zb <= 0:
            continue
        C_und_slc = C_zb * np.exp(-dt_slc / tau)
        C_undil[i, i_cb:] = C_und_slc

        with np.errstate(divide="ignore", invalid="ignore"):
            chi_arr[i, i_cb:] = np.where(C_und_slc > 0,
                                         C_vals[i, i_cb:] / C_und_slc,
                                         np.nan)

    coords = {"time": ds_3d.time, "z": z}
    chi_da = xr.DataArray(chi_arr, dims=["time", "z"], coords=coords, name="chi")
    # ε = −(1/χ) dχ/dz via xarray finite-difference differentiation
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_da = (-1.0 / chi_da) * chi_da.differentiate("z")
    eps_da = eps_da.rename("eps")

    return xr.Dataset({
        "C_cloud":  C_cloud,
        "w_cloud":  w_cloud,
        "dt_trans": xr.DataArray(dt_trans, dims=["time", "z"], coords=coords,
                                 name="dt_trans"),
        "C_undil":  xr.DataArray(C_undil,  dims=["time", "z"], coords=coords,
                                 name="C_undil"),
        "chi":      chi_da,
        "eps":      eps_da,
    })


# ══════════════════════════════════════════════════════════════════════════════
# §3b — H2: Siebesma–Cuijpers / Rio (2010) entrainment from tracer contrast
# ══════════════════════════════════════════════════════════════════════════════

def entrainment_rate_siebesma(ds_3d: xr.Dataset,
                              masks: dict[str, xr.DataArray],
                              mass_flux_ds: xr.Dataset,
                              tau: float = 900.0,
                              w_min: float = 0.05,
                              dpsi_min: float = 1e-9,
                              m_floor: float = 1e-3,
                              ) -> xr.Dataset:
    """Siebesma–Cuijpers entrainment from the in-updraft / slab tracer contrast.

    Steady plume budget in the parcel-Lagrangian form, with a first-order
    tracer decay sink:

        ∂C_u/∂z = ε (C̄ − C_u) − C_u / (τ w_u)

    so

        ε_SC = (∂C_u/∂z + C_u / (τ w_u)) / (C̄ − C_u),
        δ_SC = ε_SC − (1/M) ∂M/∂z   (continuity).

    Reference: Siebesma & Cuijpers (1995), Rio et al. (2010, BLM 135). No
    "undiluted" reference is constructed (cf. ``entrainment_rate_tracer``),
    so the env-tracer bias that breaks the Romps formulation cancels here:
    the (C̄ − C_u) denominator is the directly-observed contrast between
    in-updraft and slab.
    """
    mask_cu = masks["mask_cloudy_up"]
    C  = ds_3d["couvreux"]
    z  = ds_3d["z"]
    w  = ds_3d["w_cc"] if "w_cc" in ds_3d else ds_3d["w"].interp(zh=z)

    # Slab and in-updraft means
    C_slab = C.mean(["x", "y"])
    C_u    = C.where(mask_cu).mean(["x", "y"])
    w_u    = w.where(mask_cu).mean(["x", "y"])
    w_u_safe = xr.where(np.abs(w_u) < w_min, np.nan, w_u)

    # ε_SC
    dC_u_dz  = C_u.differentiate("z")
    contrast = C_slab - C_u
    contrast_safe = xr.where(np.abs(contrast) < dpsi_min, np.nan, contrast)
    eps_sc = (dC_u_dz + C_u / (tau * w_u_safe)) / contrast_safe

    # δ_SC from mass continuity, gated on M to avoid blowup outside cloud layer
    M = mass_flux_ds["M_up_cloudy_up"]
    M_safe = xr.where(M < m_floor, np.nan, M)
    dM_dz  = M_safe.differentiate("z")
    delta_sc = eps_sc - dM_dz / M_safe

    return xr.Dataset({
        "C_slab":   C_slab.rename("C_slab"),
        "eps_sc":   eps_sc.rename("eps_sc"),
        "delta_sc": delta_sc.rename("delta_sc"),
    })


# ══════════════════════════════════════════════════════════════════════════════
# §4 — Ensemble wrapper
# ══════════════════════════════════════════════════════════════════════════════

def compute_ensemble(rs, rt: str,
                           tau: float = 900.0,
                           variables: list[str] | None = None,
                           ) -> tuple[xr.Dataset, xr.Dataset]:
    """Run H1 and H2 diagnostics over all reps of one RT type.

    Parameters
    ----------
    rs : RunSet
        From ``cass_analysis.RunSet`` / ``catalog.make_runset``.
    rt : str
        Radiation type key (``"2stream"`` or ``"raytracer"``).
    tau : float
        Tracer decay time for H2.
    variables : list, optional
        Passed to ``load_3d_nc``; defaults to thl, qt, ql, w, b, couvreux.

    Returns
    -------
    (mean_ds, std_ds) — ensemble mean and std across reps, aligned on the
    shortest time axis.
    """
    if variables is None:
        variables = ["thl", "qt", "ql", "w", "b", "couvreux"]

    per_rep = []
    for rep_dir in rs.dirs.get(rt, []):
        ds_3d = load_3d_nc(rep_dir, variables=variables)
        stats = load_stats(rep_dir)
        masks = build_couvreux_mask(ds_3d, stats=stats)
        mf    = updraft_mass_flux(ds_3d, masks)
        ent_R = entrainment_rate_tracer(ds_3d, masks, stats, tau=tau)
        ent_S = entrainment_rate_siebesma(ds_3d, masks, mf, tau=tau)
        per_rep.append(xr.merge([mf, ent_R, ent_S]))

    if not per_rep:
        raise ValueError(f"No reps found for rt={rt!r} in RunSet {rs.name}")

    nt_min  = min(ds.sizes["time"] for ds in per_rep)
    per_rep = [ds.isel(time=slice(None, nt_min)) for ds in per_rep]

    # Strip datetime coord for mean/std (matches load_stats_ensemble contract)
    t_coord = per_rep[0].coords["time"]
    prepped = [ds.assign_coords(time=np.arange(nt_min)) for ds in per_rep]
    stacked = xr.concat(prepped, dim="rep")

    mean_ds = stacked.mean("rep").assign_coords(time=t_coord.values)
    std_ds  = stacked.std("rep", ddof=0).assign_coords(time=t_coord.values)
    return mean_ds, std_ds
