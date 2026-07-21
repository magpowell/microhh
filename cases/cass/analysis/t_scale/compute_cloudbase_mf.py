#!/usr/bin/env python3
"""
Per-cloud cloud-base mass flux, tagged by tobac track age & birth-LST.

The 3-D dumps (w, ql) are hourly — too coarse to follow an individual cloud's
age. But the *population* present at any one dump spans a range of ages: each
tobac-tracked cell carries a known birth frame, so for a cloud present at dump
time T its age = T − birth_time is known exactly (dumps land on tobac frames).
Pooling (age, birth-LST, cloud-base mass flux) over all dumps × reps gives an
age-resolved MF distribution as a *population* reconstruction.

Caveat (length-biased / inspection-paradox): a random snapshot oversamples
long-lived clouds, so the absolute MF(age) shape is distorted (flattened at
old age). The trustworthy quantity is the 3D−1D contrast at each age, with
binning/cohort identical (same bias on both sides).

Method, per (expt, rt, rep):
  1. load_3d_nc(ql, w)  → ql(t,z,y,x), w_cc(t,z,y,x) on cell centres, hourly.
  2. cloud_track_features.nc → per cell: birth frame/LST; per (cell,frame):
     centroid (x_m,y_m), equiv radius req_m, age_s.
  3. rhoref(z) from cass.default (default group, anelastic base state).
  4. For each dump time T (= tobac frame T/60): for every tracked cloud
     present at that frame, take a disk footprint of radius req_m about its
     centroid (doubly-periodic). Per footprint column find cloud base
     z_b = lowest z with ql > QL_THR; the column contributes
     rhoref[z_b]·w_cc[z_b] iff it is cloudy AND updraft (w>0). The cloud's
     cloud-base mass flux M = area-mean over the footprint (i.e. ρ·a_up·w_up).
  5. Emit one row per (cloud, dump): age_s, blst, M, a_up, n_cols.

Output: $SCRATCH/CASS_LES/analysis/cloudbase_mf/{expt}/{rt}/rep_{NN}/cb_mf.nc

Usage:
    python compute_cloudbase_mf.py --expt no_aerosols_zero_wind --rt raytracer --rep 1
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CASS_ANALYSIS = Path("/global/homes/m/mpowell/repos/microhh/cases/cass/analysis")
sys.path.insert(0, str(CASS_ANALYSIS))
from cass_analysis import load_3d_nc  # noqa: E402

LST0    = 5.5      # h : sim t=0 → 05:30 LST
QL_THR  = 1e-5     # kg/kg : in-cloud threshold


def run_dir_for(les_root: Path, expt: str, rt: str, rep: int) -> Path:
    if expt == "base":
        return les_root / "base" / rt / f"rep_{rep:02d}"
    return les_root / "experiments" / expt / rt / f"rep_{rep:02d}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expt", required=True)
    p.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--ql-thr", type=float, default=QL_THR)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    SCRATCH  = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell"))
    LES_ROOT = SCRATCH / "CASS_LES"
    run_dir  = run_dir_for(LES_ROOT, args.expt, args.rt, args.rep)
    feat_p   = (LES_ROOT / "analysis" / "lifetime" / args.expt / args.rt
                / f"rep_{args.rep:02d}" / "cloud_track_features.nc")
    out_dir  = (LES_ROOT / "analysis" / "cloudbase_mf" / args.expt / args.rt
                / f"rep_{args.rep:02d}")
    out_p    = out_dir / "cb_mf.nc"

    if out_p.exists() and not args.force:
        print(f"[skip] {out_p} exists (--force to overwrite)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading 3-D dumps  {run_dir}")
    ds3 = load_3d_nc(run_dir, variables=["ql", "w"])      # ql(t,z,y,x), w_cc added
    ql  = ds3["ql"]
    wcc = ds3["w_cc"]
    z   = ds3["z"].values
    x   = ds3["x"].values
    y   = ds3["y"].values
    _tv = ds3["time"].values                              # may be float s or datetime64
    if np.issubdtype(_tv.dtype, np.datetime64):
        tdump = (_tv - _tv[0]) / np.timedelta64(1, "s")   # → seconds from start
    else:
        tdump = _tv.astype(float)
    dx  = float(x[1] - x[0])
    nx, ny = len(x), len(y)

    # ρ_ref(z) base state (anelastic, time-invariant)
    dstat = xr.open_dataset(run_dir / "cass.default.0000000.nc",
                            group="default", decode_times=False)
    rhoref = dstat["rhoref"].values
    z_stat = dstat["z"].values
    dstat.close()
    rho_z = np.interp(z, z_stat, rhoref)                  # ρ on 3-D z grid

    print(f"Reading tracks     {feat_p}")
    ft = xr.open_dataset(feat_p).to_dataframe()
    # frame is the tobac/xy frame index; dumps land on frames = T/60
    dt_xy = 60.0
    ft["dframe"] = (ft["t_sim_s"] / dt_xy).round().astype(int)
    birth = ft.groupby("cell")["t_sim_s"].min().rename("bt")
    ft = ft.merge(birth, on="cell")
    ft["blst"] = ft["bt"] / 3600.0 + LST0

    rows = []
    for it, T in enumerate(tdump):
        fr = int(round(T / dt_xy))
        cl = ft[ft["dframe"] == fr]
        if len(cl) == 0:
            continue
        # Bring this dump's fields into memory once (z,y,x); index by position
        ql3 = ql.isel(time=it).values          # (z,y,x)
        w3  = wcc.isel(time=it).values         # (z,y,x)
        cloudy = ql3 > args.ql_thr             # (z,y,x)
        has_cloud = cloudy.any(axis=0)         # (y,x) any cloud in column
        # cloud-base z-index per column (first z with ql>thr); 0 where none
        zb_idx = np.argmax(cloudy, axis=0)     # (y,x)
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        w_base   = w3[zb_idx, jj, ii]                      # (y,x) w at cloud base
        rho_base = rho_z[zb_idx]                            # (y,x)
        # contributing columns: cloudy AND updraft
        contrib  = has_cloud & (w_base > 0.0)
        mf_col   = np.where(contrib, rho_base * w_base, 0.0)  # (y,x) kg m-2 s-1

        for _, c in cl.iterrows():
            cxi = int(round((c["x_m"] - x[0]) / dx)) % nx
            cyi = int(round((c["y_m"] - y[0]) / dx)) % ny
            rad = max(1, int(round(c["req_m"] / dx)))
            di  = np.arange(-rad, rad + 1)
            mask2 = (di[:, None]**2 + di[None, :]**2) <= rad * rad
            yy = (cyi + di) % ny
            xx = (cxi + di) % nx
            fp_mf  = mf_col[np.ix_(yy, xx)][mask2]          # footprint MF
            fp_con = contrib[np.ix_(yy, xx)][mask2]
            n_fp   = mask2.sum()
            M      = float(fp_mf.sum() / n_fp)              # ρ·a_up·w_up
            a_up   = float(fp_con.sum() / n_fp)
            rows.append(dict(age_s=float(c["age_s"]), blst=float(c["blst"]),
                             M=M, a_up=a_up, n_fp=int(n_fp),
                             t_dump_s=float(T)))
    ds3.close()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  no cloud/dump matches — writing empty")
    ds_out = xr.Dataset.from_dataframe(df) if len(df) else xr.Dataset(
        coords={"index": np.arange(0)})
    ds_out.attrs.update(expt=args.expt, rt=args.rt, rep=args.rep,
                        ql_thr=args.ql_thr, lst0=LST0,
                        n_dumps=int(len(tdump)), n_rows=int(len(df)))
    ds_out.to_netcdf(out_p)
    print(f"Saved {out_p}   ({len(df)} cloud-dump rows)")
    if len(df):
        for lo, hi, nm in [(12, 14, "morn"), (15, 18, "aft")]:
            s = df[(df.blst >= lo) & (df.blst < hi)]
            print(f"  {nm} 12-14/15-18: n={len(s):5d}  <M>={s.M.mean():.4f} kg m-2 s-1")


if __name__ == "__main__":
    main()
