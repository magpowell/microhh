#!/usr/bin/env python3
"""
cloud_root_composite_prep.py  —  L&P 2014-style cloud-root composite preparation.

For each 3D dump time in the LST window [LST_MIN_H, LST_MAX_H]:
  1. Find cloud objects whose chord length in the slice direction >= MIN_CHORD_M
  2. Extract xz-slices (y=cy fixed) and yz-slices (x=cx fixed) through each centroid
  3. Compute perturbation fields: w'θ_l', w'q_v', w', θ_l', q_t', q_l
  4. Non-dimensionalise: x → (x − centroid) / L_chord,  z → z / z_sl(t)
  5. Bilinear-interpolate onto standard grid  XL_GRID × ZND_GRID
  6. Accumulate per-orientation event stacks and write to NetCDF

Outputs (in --output-dir):
  events_xz.nc   dims: (event, z_nd, xL)   — xz-slice events
  events_yz.nc   dims: (event, z_nd, xL)   — yz-slice events

  Both files share the same variable set:
    w_thl       w'θ_l'         K m s⁻¹
    w_qv        w'q_v'         g kg⁻¹ m s⁻¹
    w_prime     w'             m s⁻¹
    ql          q_l            kg kg⁻¹
    thl_prime   θ_l'           K
    qt_prime    q_t'           g kg⁻¹
  Plus per-event metadata:  L_m, z_sl_m, cx, cy, dump_t (float64 ns epoch)

Prerequisite: run 3d_to_nc.py in the rep directory first:
    cd $RUN_DIR && python 3d_to_nc.py -v thl qt ql w

References:
    Lohou, F. and Patton, E. G. (2014). Surface Energy Balance and Buoyancy Response
    to Shallow Cumulus Shading. J. Atmos. Sci., 71, 665–682.
    doi:10.1175/JAS-D-13-0145.1

Usage:
    python cloud_root_composite_prep.py \\
        --run-dir /pscratch/sd/m/mpowell/CASS_LES/base/2stream/rep_01 \\
        --output-dir /pscratch/sd/m/mpowell/CASS_LES/analysis/cloud_root_composite/base/2stream/rep_01
"""

import argparse
import sys
import time as _time
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ── analysis utilities (sibling module) ───────────────────────────────────────
_ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_ANALYSIS_DIR))
from cass_analysis import (
    load_stats, compute_z_sl,
    load_xy_files, load_3d_nc,
    cloud_mask_2d, find_cloud_objects,
    chord_length_1d, interp_event_to_std_grid,
    XL_GRID, ZND_GRID, COMPOSITE_VARS,
)

# ── Tunable parameters ────────────────────────────────────────────────────────
LST_MIN_H   = 11.5    # start of compositing window [hours past midnight, local time]
LST_MAX_H   = 15.5    # end   of compositing window
MIN_CHORD_M = 1000.0  # minimum chord length in the slice direction [m]
PREFILTER_L =  500.0  # effective-diameter pre-filter for find_cloud_objects [m]
                      # (objects smaller than this can't have chord >= MIN_CHORD_M)
FIELDS_3D   = ["thl", "qt", "ql", "w"]   # variables to load from 3d_to_nc output


def _in_lst_window(t_local_arr, lst_min=LST_MIN_H, lst_max=LST_MAX_H):
    """Bool mask: True for dump times within [lst_min, lst_max] hours LST."""
    hours = np.array([t.hour + t.minute / 60 for t in pd.DatetimeIndex(t_local_arr)])
    return (hours >= lst_min) & (hours <= lst_max)


def _slice_fields(ds_t: xr.Dataset, cy: int, cx: int,
                  orientation: str) -> tuple[np.ndarray, ...]:
    """Extract and compute perturbation fields for one slice.

    Parameters
    ----------
    ds_t        : 3D dataset at a single time (already .load()-ed)
    cy, cx      : centroid row / column indices
    orientation : 'y' → isel(y=cy);  'x' → isel(x=cx)

    Returns
    -------
    (w_thl, w_qv, w_prime, ql, thl_prime, qt_prime)   — each shape (nz, nhoriz)
    (horiz_m, z_m)  — coordinate arrays in metres
    """
    sl = ds_t.isel(y=cy) if orientation == "y" else ds_t.isel(x=cx)

    # numpy arrays (nz, nhoriz) — xarray already ensured correct dim order
    def _v(name):
        return sl[name].values  # (nz, nhoriz)

    w_thl    = _v("w_prime") * _v("thl_prime")
    w_qv     = _v("w_prime") * _v("qv_prime") * 1e3   # g/kg m/s
    w_prime  = _v("w_prime")
    ql       = _v("ql")
    thl_p    = _v("thl_prime")
    qt_p     = _v("qt_prime") * 1e3                   # g/kg

    horiz_m = (sl["x"] if orientation == "y" else sl["y"]).values.astype(float)
    z_m     = sl["z"].values.astype(float)

    return (w_thl, w_qv, w_prime, ql, thl_p, qt_p), horiz_m, z_m


def process_rep(run_dir: Path, output_dir: Path, verbose: bool = True) -> dict:
    """Process one rep directory.

    Returns
    -------
    dict with keys 'xz' and 'yz', each mapping to the number of events saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = _time.perf_counter()

    # ── Load metadata ─────────────────────────────────────────────────────────
    stats  = load_stats(run_dir)
    ds_xy  = load_xy_files(run_dir, variables=["qlqi_path"])
    dx     = float(ds_xy.x[1] - ds_xy.x[0])
    dy     = float(ds_xy.y[1] - ds_xy.y[0])
    xy_tf  = ds_xy.time.values.astype("datetime64[ns]").astype(np.float64)
    st_tf  = stats["t_local"].values.astype("datetime64[ns]").astype(np.float64)

    # ── Load 3D data (lazy) ───────────────────────────────────────────────────
    ds_3d    = load_3d_nc(run_dir, variables=FIELDS_3D)
    t_local  = pd.DatetimeIndex(ds_3d.time.values)
    valid_idx = np.where(_in_lst_window(t_local))[0]

    if verbose:
        print(f"  {run_dir}:")
        print(f"    {len(valid_idx)} dump times in "
              f"[{LST_MIN_H:.1f}–{LST_MAX_H:.1f}] LST  "
              f"(dx={dx:.0f}m, dy={dy:.0f}m)")

    events_xz: list[dict] = []
    events_yz: list[dict] = []

    for tidx in valid_idx:
        t3d_raw = ds_3d.time.values[tidx]
        t3d_f   = np.datetime64(t3d_raw, "ns").astype(np.float64)

        # Load this dump time into memory
        ds_t = ds_3d.isel(time=tidx).load()

        # Cloud mask from nearest xy time
        xy_tidx = int(np.argmin(np.abs(xy_tf - t3d_f)))
        mask_2d = cloud_mask_2d(
            ds_xy["qlqi_path"].isel(time=xy_tidx).values
        )

        # z_sl from nearest stats time
        st_tidx = int(np.argmin(np.abs(st_tf - t3d_f)))
        z_sl    = compute_z_sl(stats, time_idx=st_tidx)

        # z coordinates (fixed per simulation)
        z_m  = ds_t["z"].values.astype(float)
        z_nd = z_m / z_sl     # z/z_sl — common vertical axis for this event

        # Find cloud objects (loose effective-diameter pre-filter)
        labeled, props = find_cloud_objects(mask_2d, dx, dy, min_L=PREFILTER_L)

        n_new = {"xz": 0, "yz": 0}

        for obj in props:
            cy, cx, lbl = obj["cy"], obj["cx"], obj["label"]

            for orientation, evt_list, key in [
                ("y", events_xz, "xz"),
                ("x", events_yz, "yz"),
            ]:
                # Chord length in the slice direction
                chord = chord_length_1d(labeled, lbl, cy, cx, dx, dy, orientation)
                if chord < MIN_CHORD_M:
                    continue

                # Horizontal coordinates and centroid position in slice direction
                fields, horiz_m, zm = _slice_fields(ds_t, cy, cx, orientation)

                if orientation == "y":
                    cloud_pix_m = horiz_m[labeled[cy, :] == lbl]
                else:
                    cloud_pix_m = horiz_m[labeled[:, cx] == lbl]

                centroid_m = float(cloud_pix_m.mean())
                x_nd = (horiz_m - centroid_m) / chord   # x/L, monotone increasing

                # Interpolate each field onto standard grid
                (w_thl, w_qv, w_prime, ql, thl_p, qt_p) = fields
                evt: dict = {}
                for name, arr in zip(
                    COMPOSITE_VARS,
                    (w_thl, w_qv, w_prime, ql, thl_p, qt_p),
                ):
                    evt[name] = interp_event_to_std_grid(arr, x_nd, z_nd)

                evt["L_m"]    = float(chord)
                evt["z_sl_m"] = float(z_sl)
                evt["cx"]     = float(cx)
                evt["cy"]     = float(cy)
                evt["dump_t"] = float(t3d_f)
                evt_list.append(evt)
                n_new[key] += 1

        if verbose:
            print(f"    {str(t3d_raw)[:16]}  z_sl={z_sl:.0f} m  "
                  f"+{n_new['xz']} xz  +{n_new['yz']} yz  events")

    # ── Write output ──────────────────────────────────────────────────────────
    result = {}
    for key, evt_list in [("xz", events_xz), ("yz", events_yz)]:
        if not evt_list:
            print(f"  WARNING: no {key} events found — output not written")
            result[key] = 0
            continue
        out_path = output_dir / f"events_{key}.nc"
        _save_events(out_path, evt_list)
        result[key] = len(evt_list)
        if verbose:
            print(f"    → {len(evt_list)} {key} events saved to {out_path}")

    elapsed = _time.perf_counter() - t0
    if verbose:
        print(f"  Done in {elapsed:.1f} s")
    return result


def _save_events(path: Path, evt_list: list[dict]):
    """Stack list of event dicts and write to NetCDF."""
    data_vars = {}

    for vn in COMPOSITE_VARS:
        arr = np.stack([e[vn] for e in evt_list], axis=0)  # (n, n_znd, n_xL)
        data_vars[vn] = xr.DataArray(arr, dims=["event", "z_nd", "xL"])

    for mk in ("L_m", "z_sl_m", "cx", "cy", "dump_t"):
        data_vars[mk] = xr.DataArray(
            np.array([e[mk] for e in evt_list], dtype=np.float64),
            dims=["event"],
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "xL":    ("xL",    XL_GRID,  {"long_name": "x/L or y/L",
                                           "units":     "1"}),
            "z_nd":  ("z_nd",  ZND_GRID, {"long_name": "z/z_sl",
                                           "units":     "1"}),
            "event": ("event", np.arange(len(evt_list))),
        },
        attrs={
            "min_chord_m": MIN_CHORD_M,
            "lst_min_h":   LST_MIN_H,
            "lst_max_h":   LST_MAX_H,
            "reference":   "Lohou & Patton (2014) JAS doi:10.1175/JAS-D-13-0145.1",
        },
    )
    if path.exists():
        path.unlink()
    ds.to_netcdf(str(path))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir",    required=True,
                        help="Rep run directory (must contain 3d_to_nc output)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write events_xz.nc / events_yz.nc")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    process_rep(
        run_dir    = Path(args.run_dir),
        output_dir = Path(args.output_dir),
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()
