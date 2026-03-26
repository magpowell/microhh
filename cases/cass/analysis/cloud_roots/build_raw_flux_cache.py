#!/usr/bin/env python3
"""build_raw_flux_cache.py

Computes raw_flux_profile_cache.nc for every rep of the sweep experiments
(cs_veg, soil_moisture, wind_u).  Run via submit_build_raw_flux_cache.sh.

Usage:
    python build_raw_flux_cache.py [--n-par N] [--dry-run]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRATCH        = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell"))
LES_ROOT       = SCRATCH / "CASS_LES"
COMPOSITE_ROOT = LES_ROOT / "analysis" / "cloud_root_composite"
ANALYSIS_DIR   = Path(__file__).resolve().parent.parent   # .../analysis/
N_REPS         = 4


def build_cases():
    cases = []

    for val in ["cs_veg_0", "cs_veg_41840", "cs_veg_418400"]:
        for rt in ["2stream", "raytracer"]:
            cases.append(dict(
                run_root  = LES_ROOT / "experiments" / "cs_veg" / val / rt,
                comp_expt = f"cs_veg/{val}",
                comp_rt   = rt,
            ))

    for val in ["theta_0p1", "theta_0p2", "theta_0p3", "theta_0p4"]:
        for rt in ["2stream", "raytracer"]:
            cases.append(dict(
                run_root  = LES_ROOT / "experiments" / "soil_moisture" / val / rt,
                comp_expt = f"soil_moisture/{val}",
                comp_rt   = rt,
            ))

    for val in ["u_0p0", "u_2p5", "u_5p0", "u_7p5", "u_10p0"]:
        for rt in ["2stream", "raytracer"]:
            run_root = LES_ROOT / "experiments" / "wind_u" / val / rt
            if not any((run_root / f"rep_{i:02d}").exists() for i in range(1, N_REPS + 1)):
                print(f"  SKIP wind_u/{val}/{rt}: no reps found")
                continue
            cases.append(dict(
                run_root  = run_root,
                comp_expt = f"wind_u/{val}",
                comp_rt   = rt,
            ))

    return cases


def process_rep(run_dir: Path, cache_path: Path, label: str, dry_run: bool):
    sys.path.insert(0, str(ANALYSIS_DIR))
    import xarray as xr
    from cass_analysis import (
        load_stats, load_3d_nc, load_xy_files,
        compute_normalized_cloud_root_profiles,
    )

    # Invalidate stale cache
    if cache_path.exists():
        with xr.open_dataset(str(cache_path)) as chk:
            if "f_free_thl" not in chk or "t_hours" not in chk:
                print(f"  {label}: cache outdated, recomputing", flush=True)
                if not dry_run:
                    cache_path.unlink()
            else:
                print(f"  {label}: cache hit", flush=True)
                return label, "hit"

    if not (run_dir / "thl.nc").exists():
        print(f"  SKIP {label}: thl.nc not found", flush=True)
        return label, "skip"

    if dry_run:
        print(f"  [dry] {label}", flush=True)
        return label, "dry"

    print(f"  {label}: computing ...", flush=True)
    s     = load_stats(run_dir)
    ds_3d = load_3d_nc(run_dir, variables=["thl", "qt", "ql", "w"])
    ds_xy = load_xy_files(run_dir, variables=["qlqi_path"])
    res   = compute_normalized_cloud_root_profiles(ds_3d, ds_xy, s)
    ds_3d.close()

    if res["f_cloud_thl"].shape[0] == 0:
        print(f"  {label}: WARNING no valid timesteps", flush=True)
        return label, "empty"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({
        "f_cloud_thl":  xr.DataArray(res["f_cloud_thl"],  dims=["time", "zeta"]),
        "f_cloud_qv":   xr.DataArray(res["f_cloud_qv"],   dims=["time", "zeta"]),
        "f_free_thl":   xr.DataArray(res["f_free_thl"],   dims=["time", "zeta"]),
        "f_free_qv":    xr.DataArray(res["f_free_qv"],    dims=["time", "zeta"]),
        "f_domain_thl": xr.DataArray(res["f_domain_thl"], dims=["time", "zeta"]),
        "f_domain_qv":  xr.DataArray(res["f_domain_qv"],  dims=["time", "zeta"]),
        "z_sl":         xr.DataArray(res["z_sl"],         dims=["time"]),
        "cloud_frac":   xr.DataArray(res["cloud_frac"],   dims=["time"]),
        "t_hours":      xr.DataArray(res["t_hours"],      dims=["time"]),
    }, coords={"zeta": res["zeta"]}).to_netcdf(str(cache_path))

    print(f"  {label}: cached ({res['f_cloud_thl'].shape[0]} timesteps)", flush=True)
    return label, "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-par", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = [
        (case["run_root"] / f"rep_{i:02d}",
         COMPOSITE_ROOT / case["comp_expt"] / case["comp_rt"] / f"rep_{i:02d}" / "raw_flux_profile_cache.nc",
         f"{case['comp_expt']}/{case['comp_rt']}/rep_{i:02d}")
        for case in build_cases()
        for i in range(1, N_REPS + 1)
    ]

    print(f"Reps to process: {len(tasks)}  n_par={args.n_par}")

    counts = dict(done=0, hit=0, skip=0, empty=0, fail=0)
    with ProcessPoolExecutor(max_workers=args.n_par) as pool:
        futures = {
            pool.submit(process_rep, rd, cp, lbl, args.dry_run): lbl
            for rd, cp, lbl in tasks
        }
        for fut in as_completed(futures):
            try:
                _, status = fut.result()
                counts[status if status in counts else "done"] += 1
            except Exception as exc:
                print(f"  FAIL {futures[fut]}: {exc}", flush=True)
                counts["fail"] += 1

    print("Summary: " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
