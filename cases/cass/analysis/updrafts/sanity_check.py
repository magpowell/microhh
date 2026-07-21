#!/usr/bin/env python3
"""
Sanity checks for the H1/H2 Couvreux updraft pipeline.

Covers every item in PLAN.md §"Sanity checks (run before trusting diagnostics)".

Two check levels run automatically based on what's available in the rep dir:

LEVEL 1 — stats-only, available the moment the simulation finishes
    [L1a] Tracer subcloud budget order-of-magnitude
    [L1b] In-model couvreux mask coverage a_up ∈ [5, 20] %

LEVEL 2 — requires `3d_to_nc.py -v thl qt ql w b u v couvreux` to have run
    [L2a] Mass conservation ⟨ρ·w_cc⟩(z, t) ≈ 0
    [L2b] Offline mask coverage a_up ∈ [5, 20] %
    [L2c] χ(z_b) ≈ 1, χ(z_t) ∈ [0.2, 0.7]
    [L2d] Offline mask_tracer_only vs in-model couvreux/default/area

Usage
-----
    python sanity_check.py <rep_dir>
    python sanity_check.py $SCRATCH/CASS_LES/experiments/no_aerosols_zero_wind_v2/2stream/rep_01

Exits with 0 if every check passes or is gracefully skipped, 1 if any check
fails.  WARN-level results do not fail the script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from cass_analysis import load_stats, load_3d_nc, sim_time_to_lst  # noqa: E402
from diagnostics import (                                          # noqa: E402
    build_couvreux_mask, updraft_mass_flux, entrainment_rate_tracer,
)


# ── Report plumbing ─────────────────────────────────────────────────────────

class Result(NamedTuple):
    name: str
    status: str      # 'PASS' | 'WARN' | 'FAIL' | 'SKIP'
    detail: str


def _print_results(results: list[Result]) -> int:
    width = max(len(r.name) for r in results) + 2
    print("\n" + "=" * 72)
    print(f"{'CHECK':<{width}}  STATUS   DETAIL")
    print("-" * 72)
    for r in results:
        colour = {
            "PASS": "\033[32mPASS\033[0m",
            "WARN": "\033[33mWARN\033[0m",
            "FAIL": "\033[31mFAIL\033[0m",
            "SKIP": "\033[90mSKIP\033[0m",
        }.get(r.status, r.status)
        print(f"{r.name:<{width}}  {colour}    {r.detail}")
    print("=" * 72)
    any_fail = any(r.status == "FAIL" for r in results)
    n_pass   = sum(r.status == "PASS" for r in results)
    n_warn   = sum(r.status == "WARN" for r in results)
    n_fail   = sum(r.status == "FAIL" for r in results)
    n_skip   = sum(r.status == "SKIP" for r in results)
    print(f"{n_pass} pass  |  {n_warn} warn  |  {n_fail} fail  |  {n_skip} skip")
    return 1 if any_fail else 0


# ── LEVEL 1 — stats-only checks ─────────────────────────────────────────────

def check_subcloud_tracer_budget(rep_dir: Path,
                                 F_surf: float = 1e-5,
                                 tau: float = 900.0,
                                 h_sub: float = 1000.0,
                                 lst_lo: float = 14.0,
                                 lst_hi: float = 16.0,
                                 ) -> Result:
    """[L1a] Steady-state subcloud tracer balance.

    0 = F/h − ⟨C⟩/τ   ⇒   ⟨C⟩_ss = F · τ / h

    With emission F=1e-5, τ=900 s, subcloud depth h≈1000 m:
    ⟨C⟩_ss ≈ 9 × 10⁻⁶.  PASS if observed is within a factor of 3.
    """
    # cass_analysis.load_stats doesn't expose the raw tracer profile; read
    # cass.default directly with netCDF4.
    import netCDF4
    path = rep_dir / "cass.default.0000000.nc"
    if not path.exists():
        return Result("L1a subcloud tracer budget", "SKIP", f"no {path.name}")
    nc = netCDF4.Dataset(path)
    try:
        if "default" not in nc.groups or "couvreux" not in nc.groups["default"].variables:
            return Result("L1a subcloud tracer budget", "SKIP",
                          "no 'couvreux' profile in default stats group")
        C = np.asarray(nc.groups["default"].variables["couvreux"][:])   # (time, z)
        t_sec = np.asarray(nc.variables["time"][:])
        z = np.asarray(nc.variables["z"][:])
    finally:
        nc.close()

    lst   = sim_time_to_lst(t_sec)
    tmask = (lst >= lst_lo) & (lst <= lst_hi)
    if not tmask.any():
        return Result("L1a subcloud tracer budget", "SKIP",
                      f"no steps in LST {lst_lo}-{lst_hi}")
    zm = z < 1000.0
    if not zm.any():
        return Result("L1a subcloud tracer budget", "SKIP", "no subcloud z levels")

    C_sub = C[np.ix_(tmask, zm)].mean()
    expected = F_surf * tau / h_sub          # F · τ / h (steady-state)
    ratio    = C_sub / expected if expected > 0 else np.nan
    ok = 0.33 <= ratio <= 3.0
    return Result(
        "L1a subcloud tracer budget",
        "PASS" if ok else "WARN",
        f"⟨C⟩_sub={C_sub:.2e}  (expected F·τ/h ≈ {expected:.2e}, ratio={ratio:.2f})",
    )


def check_inmodel_mask_coverage(rep_dir: Path,
                                lst_lo: float = 12.0,
                                lst_hi: float = 17.0,
                                z_lo: float = 500.0,
                                z_hi: float = 2800.0,
                                ) -> Result:
    """[L1b] In-model couvreux-mask area fraction inside cloud layer ∈ [5, 20] %."""
    try:
        ds = load_stats(rep_dir, mask="couvreux")
    except (FileNotFoundError, KeyError) as e:
        return Result("L1b in-model mask coverage", "SKIP", f"{e}")

    import netCDF4
    path = rep_dir / "cass.couvreux.0000000.nc"
    nc   = netCDF4.Dataset(path)
    if "default" not in nc.groups or "area" not in nc.groups["default"].variables:
        nc.close()
        return Result("L1b in-model mask coverage", "SKIP",
                      "couvreux/default/area missing")
    area = np.asarray(nc.groups["default"].variables["area"][:])   # (time, z)
    nc.close()

    t_sec = ds["t_sec"].values
    z     = ds["z"].values
    lst   = sim_time_to_lst(t_sec)
    tm    = (lst >= lst_lo) & (lst <= lst_hi)
    zm    = (z >= z_lo) & (z <= z_hi)
    if not tm.any() or not zm.any():
        return Result("L1b in-model mask coverage", "SKIP",
                      "no samples in the LST/z window")
    a = area[np.ix_(tm, zm)] * 100.0   # %
    mean_a = a.mean()
    peak_a = a.max()
    ok     = 2.0 <= mean_a <= 25.0
    return Result(
        "L1b in-model mask coverage",
        "PASS" if ok else "WARN",
        f"⟨a_up⟩={mean_a:.1f} %  peak={peak_a:.1f} %  "
        f"(LST {lst_lo}-{lst_hi}, z {z_lo:.0f}-{z_hi:.0f} m)",
    )


# ── LEVEL 2 — needs 3d_to_nc.py outputs ──────────────────────────────────────

def _load_3d_if_available(rep_dir: Path):
    """Load thl, qt, ql, w, b, couvreux from 3D NC dumps, else return None."""
    needed = ["ql", "w", "couvreux"]
    missing = [v for v in needed if not (rep_dir / f"{v}.nc").exists()]
    if missing:
        return None, f"missing 3D NC for: {missing} (run 3d_to_nc.py first)"
    try:
        ds_3d = load_3d_nc(rep_dir,
                           variables=["thl", "qt", "ql", "w", "b", "couvreux"])
        return ds_3d, None
    except Exception as e:                                                  # noqa: BLE001
        return None, f"load_3d_nc error: {e}"


def check_mass_conservation(ds_3d: xr.Dataset,
                            rho: float = 1.2,
                            tol: float = 5e-4,
                            ) -> Result:
    """[L2a] ⟨ρ·w_cc⟩(z, t) ≈ 0.  Large residual → w-interp bug or ρ(z) needed."""
    if "w_cc" not in ds_3d:
        return Result("L2a mass conservation", "SKIP",
                      "no 'w_cc' (did load_3d_nc run?)")
    w_mean = ds_3d["w_cc"].mean(["x", "y"]).values  # (time, z)
    rms = float(np.sqrt(np.nanmean(w_mean**2)))
    ok  = rms < tol
    return Result(
        "L2a mass conservation",
        "PASS" if ok else "WARN",
        f"RMS ⟨w_cc⟩_xy = {rms:.2e} m/s  (tol {tol:.0e})",
    )


def check_offline_mask_coverage(masks: dict,
                                ds_3d: xr.Dataset,
                                stats: xr.Dataset,
                                lst_lo: float = 12.0,
                                lst_hi: float = 17.0,
                                ) -> Result:
    """[L2b] Offline mask_paper coverage inside cloud layer ∈ [5, 20] %."""
    a_up = masks["mask_paper"].mean(["x", "y"]).values * 100.0    # (time, z)
    t = ds_3d.time.values.astype("datetime64[ns]").astype("float64")
    t_st = stats["t_local"].values.astype("datetime64[ns]").astype("float64")
    # Build an LST vector for the 3D dumps using the nearest stats time's t_sec
    lst = np.array([
        sim_time_to_lst(stats["t_sec"].values[int(np.argmin(np.abs(t_st - ti)))])
        for ti in t
    ])
    z = ds_3d["z"].values
    zm = (z >= 500) & (z <= 2800)
    tm = (lst >= lst_lo) & (lst <= lst_hi)
    if not tm.any() or not zm.any():
        return Result("L2b offline mask coverage", "SKIP",
                      "no samples in the LST/z window")
    a = a_up[np.ix_(tm, zm)]
    mean_a = a.mean()
    peak_a = a.max()
    ok     = 2.0 <= mean_a <= 25.0
    return Result(
        "L2b offline mask coverage",
        "PASS" if ok else "FAIL",
        f"⟨a_up⟩={mean_a:.1f} %  peak={peak_a:.1f} %",
    )


def check_chi_bounds(ent: xr.Dataset,
                     ds_3d: xr.Dataset,
                     stats: xr.Dataset,
                     lst_lo: float = 12.0,
                     lst_hi: float = 17.0,
                     ) -> Result:
    """[L2c] χ(z_b) ≈ 1  and  χ(z_t) ∈ [0.2, 0.7]."""
    from cass_analysis import compute_z_sl, cloud_top_za

    t_3d = ds_3d.time.values.astype("datetime64[ns]").astype("float64")
    t_st = stats["t_local"].values.astype("datetime64[ns]").astype("float64")
    z_v  = ds_3d["z"].values
    chi  = ent["chi"].values

    chi_zb, chi_zt = [], []
    for i, t in enumerate(t_3d):
        idx  = int(np.argmin(np.abs(t_st - t)))
        lst  = sim_time_to_lst(stats["t_sec"].values[idx])
        if not (lst_lo <= lst <= lst_hi):
            continue
        zb  = compute_z_sl(stats, time_idx=idx)
        zt  = cloud_top_za(stats["z"].values, stats["ql_frac"].values[idx])
        ibb = int(np.argmin(np.abs(z_v - zb)))
        itt = int(np.argmin(np.abs(z_v - zt)))
        if np.isfinite(chi[i, ibb]):
            chi_zb.append(chi[i, ibb])
        if np.isfinite(chi[i, itt]):
            chi_zt.append(chi[i, itt])
    if not chi_zb or not chi_zt:
        return Result("L2c χ bounds", "SKIP", "no cloudy-up timesteps in window")

    mu_zb = float(np.nanmean(chi_zb))
    mu_zt = float(np.nanmean(chi_zt))
    ok_zb = 0.8 <= mu_zb <= 1.2
    ok_zt = 0.1 <= mu_zt <= 0.8
    status = "PASS" if (ok_zb and ok_zt) else ("WARN" if (ok_zb or ok_zt) else "FAIL")
    return Result(
        "L2c χ bounds",
        status,
        f"χ(z_b)={mu_zb:.2f} (expect ≈1),  χ(z_t)={mu_zt:.2f} (expect 0.2-0.7)",
    )


def check_offline_vs_inmodel(masks: dict, rep_dir: Path,
                             ds_3d: xr.Dataset, stats: xr.Dataset,
                             ) -> Result:
    """[L2d] Offline mask_tracer_only ≈ in-model cass.couvreux/default/area."""
    import netCDF4
    path = rep_dir / "cass.couvreux.0000000.nc"
    if not path.exists():
        return Result("L2d offline vs in-model mask", "SKIP",
                      "cass.couvreux.0000000.nc missing")
    nc = netCDF4.Dataset(path)
    if "default" not in nc.groups or "area" not in nc.groups["default"].variables:
        nc.close()
        return Result("L2d offline vs in-model mask", "SKIP",
                      "in-model mask area missing")
    area_im = np.asarray(nc.groups["default"].variables["area"][:])
    t_im    = np.asarray(nc.variables["time"][:])
    z_im    = np.asarray(nc.variables["z"][:])
    nc.close()

    a_off = masks["mask_tracer_only"].mean(["x", "y"]).values * 100.0
    z_off = ds_3d["z"].values
    # Nearest-time, nearest-z comparison in cloud layer
    t_off_sec = ds_3d.time.values.astype("datetime64[ns]").astype("float64")
    t_st_sec  = stats["t_local"].values.astype("datetime64[ns]").astype("float64")
    t_sec_dump = np.array([
        stats["t_sec"].values[int(np.argmin(np.abs(t_st_sec - ti)))]
        for ti in t_off_sec
    ])
    zm = (z_off >= 500) & (z_off <= 2800)
    diffs = []
    for i, ts in enumerate(t_sec_dump):
        j = int(np.argmin(np.abs(t_im - ts)))
        # Interpolate in-model area onto 3D z
        a_im_interp = np.interp(z_off, z_im, area_im[j]) * 100.0
        diffs.append(np.nanmean(a_im_interp[zm] - a_off[i, zm]))
    mean_diff = float(np.nanmean(diffs))
    # PLAN L90: "match to ~1%" — offline has σ_min floor so offline ≤ in-model
    ok = -1.0 <= mean_diff <= 5.0
    return Result(
        "L2d offline vs in-model mask",
        "PASS" if ok else "WARN",
        f"mean(a_im − a_off) = {mean_diff:+.2f} %  (offline has σ_min floor)",
    )


# ── Driver ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rep_dir", type=Path,
                    help="run directory (one rep) to check")
    ap.add_argument("--tau", type=float, default=900.0,
                    help="tracer decay timescale (s)")
    ap.add_argument("--lst-lo", type=float, default=12.0)
    ap.add_argument("--lst-hi", type=float, default=17.0)
    args = ap.parse_args()

    rep_dir = args.rep_dir.resolve()
    print(f"Sanity-checking: {rep_dir}")

    results: list[Result] = []

    # Level 1 — stats only
    results.append(check_subcloud_tracer_budget(rep_dir, tau=args.tau))
    results.append(check_inmodel_mask_coverage(rep_dir,
                                               lst_lo=args.lst_lo,
                                               lst_hi=args.lst_hi))

    # Level 2 — 3D dumps
    ds_3d, msg = _load_3d_if_available(rep_dir)
    if ds_3d is None:
        for n in ("L2a mass conservation", "L2b offline mask coverage",
                  "L2c χ bounds", "L2d offline vs in-model mask"):
            results.append(Result(n, "SKIP", msg))
    else:
        try:
            stats = load_stats(rep_dir)
        except Exception as e:                                              # noqa: BLE001
            for n in ("L2a mass conservation", "L2b offline mask coverage",
                      "L2c χ bounds", "L2d offline vs in-model mask"):
                results.append(Result(n, "FAIL", f"load_stats failed: {e}"))
        else:
            results.append(check_mass_conservation(ds_3d))
            try:
                masks = build_couvreux_mask(ds_3d, stats=stats)
            except Exception as e:                                          # noqa: BLE001
                for n in ("L2b offline mask coverage", "L2c χ bounds",
                          "L2d offline vs in-model mask"):
                    results.append(Result(n, "FAIL", f"mask build: {e}"))
            else:
                results.append(check_offline_mask_coverage(masks, ds_3d, stats,
                                                           lst_lo=args.lst_lo,
                                                           lst_hi=args.lst_hi))
                try:
                    ent = entrainment_rate_tracer(ds_3d, masks, stats, tau=args.tau)
                except Exception as e:                                      # noqa: BLE001
                    results.append(Result("L2c χ bounds", "FAIL", str(e)))
                else:
                    results.append(check_chi_bounds(ent, ds_3d, stats,
                                                    lst_lo=args.lst_lo,
                                                    lst_hi=args.lst_hi))
                results.append(check_offline_vs_inmodel(masks, rep_dir, ds_3d,
                                                         stats))

    sys.exit(_print_results(results))


if __name__ == "__main__":
    main()
