#!/usr/bin/env python3
"""
cloud_root_composite_debug.py  —  Quick test of the composite pipeline.

Runs on the debug run (64×64 domain) with a single dump time. Prints
diagnostic info, saves a small events file, and makes a quick plot.

Requires 3d_to_nc output in the debug run directory. Run from anywhere.

Usage:
    python cloud_root_composite_debug.py [--rt 2stream|raytracer]
                                         [--dump-idx -1]
                                         [--no-plot]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

from cass_analysis import (
    load_stats, compute_z_sl,
    load_xy_files, load_3d_nc,
    cloud_mask_2d, find_cloud_objects,
    chord_length_1d, interp_event_to_std_grid,
    load_composite_events, composite_mean_std,
    plot_cloud_root_composite,
    XL_GRID, ZND_GRID, COMPOSITE_VARS,
)
from cloud_root_composite_prep import (
    _in_lst_window, _slice_fields, _save_events,
    LST_MIN_H, LST_MAX_H, MIN_CHORD_M, PREFILTER_L, FIELDS_3D,
)

SCRATCH = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell")) / "CASS_LES"
DEBUG_ROOT = SCRATCH / "debug" / "no_aerosols"

OUTPUT_DIR = SCRATCH / "analysis" / "cloud_root_composite" / "debug" / "no_aerosols"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rt", default="2stream", choices=["2stream", "raytracer"])
    parser.add_argument("--dump-idx", type=int, default=-1,
                        help="3D dump time index to process (-1 = last)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = DEBUG_ROOT / args.rt / "rep_01"
    out_dir = OUTPUT_DIR / args.rt / "rep_01"

    print(f"=== cloud_root_composite_debug ===")
    print(f"RT      : {args.rt}")
    print(f"run_dir : {run_dir}")
    print(f"out_dir : {out_dir}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    stats  = load_stats(run_dir)
    ds_xy  = load_xy_files(run_dir, variables=["qlqi_path"])
    dx = float(ds_xy.x[1] - ds_xy.x[0])
    dy = float(ds_xy.y[1] - ds_xy.y[0])
    xy_tf  = ds_xy.time.values.astype("datetime64[ns]").astype(float)
    st_tf  = stats["t_local"].values.astype("datetime64[ns]").astype(float)

    ds_3d = load_3d_nc(run_dir, variables=FIELDS_3D)
    t_local = list(ds_3d.time.values)

    print(f"Grid     : dx={dx:.0f}m, dy={dy:.0f}m")
    print(f"3D dumps : {len(t_local)}")
    print(f"  all times: {[str(t)[:16] for t in t_local]}")
    print()

    # Select dump time (ignore LST filter for debug — just use args.dump_idx)
    tidx  = args.dump_idx % len(t_local)
    t3d   = ds_3d.time.values[tidx]
    t3d_f = np.datetime64(t3d, "ns").astype(float)
    print(f"Processing dump [{tidx}] : {str(t3d)[:16]}")

    ds_t    = ds_3d.isel(time=tidx).load()
    xy_tidx = int(np.argmin(np.abs(xy_tf - t3d_f)))
    st_tidx = int(np.argmin(np.abs(st_tf - t3d_f)))
    mask_2d = cloud_mask_2d(ds_xy["qlqi_path"].isel(time=xy_tidx).values)
    z_sl    = compute_z_sl(stats, time_idx=st_tidx)
    z_m     = ds_t["z"].values.astype(float)
    z_nd    = z_m / z_sl

    print(f"z_sl = {z_sl:.0f} m  |  cloud fraction = {mask_2d.mean():.2%}")

    # ── Find objects ──────────────────────────────────────────────────────────
    labeled, props = find_cloud_objects(mask_2d, dx, dy, min_L=PREFILTER_L)
    print(f"\nFound {len(props)} objects (L >= {PREFILTER_L:.0f}m):")
    for obj in props[:5]:
        lbl = obj["label"]
        chord_x = chord_length_1d(labeled, lbl, obj["cy"], obj["cx"], dx, dy, "y")
        chord_y = chord_length_1d(labeled, lbl, obj["cy"], obj["cx"], dx, dy, "x")
        print(f"  label={lbl}  L_eff={obj['L']:.0f}m  "
              f"chord_x={chord_x:.0f}m  chord_y={chord_y:.0f}m  "
              f"area={obj['area_m2']/1e6:.2f}km²  "
              f"centroid=({obj['cy']},{obj['cx']})")

    # ── Process events ────────────────────────────────────────────────────────
    events_xz: list[dict] = []
    events_yz: list[dict] = []

    for obj in props:
        cy, cx, lbl = obj["cy"], obj["cx"], obj["label"]
        for orientation, evt_list, key in [("y", events_xz, "xz"), ("x", events_yz, "yz")]:
            chord = chord_length_1d(labeled, lbl, cy, cx, dx, dy, orientation)
            if chord < MIN_CHORD_M:
                print(f"  [{key}] label={lbl}: chord={chord:.0f}m < {MIN_CHORD_M:.0f}m  SKIP")
                continue
            fields, horiz_m, zm = _slice_fields(ds_t, cy, cx, orientation)
            cloud_pix_m = horiz_m[(labeled[cy, :] == lbl) if orientation == "y"
                                  else (labeled[:, cx] == lbl)]
            centroid_m = float(cloud_pix_m.mean())
            x_nd = (horiz_m - centroid_m) / chord

            evt: dict = {}
            for name, arr in zip(COMPOSITE_VARS, fields):
                if arr is None:
                    continue
                evt[name] = interp_event_to_std_grid(arr, x_nd, z_nd)

            evt["L_m"]    = float(chord)
            evt["z_sl_m"] = float(z_sl)
            evt["cx"]     = float(cx)
            evt["cy"]     = float(cy)
            evt["dump_t"] = float(t3d_f)
            evt_list.append(evt)
            print(f"  [{key}] label={lbl}: chord={chord:.0f}m  centroid={centroid_m:.0f}m  ✓")

    print(f"\nEvents: {len(events_xz)} xz,  {len(events_yz)} yz")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, evts in [("xz", events_xz), ("yz", events_yz)]:
        if evts:
            out_path = out_dir / f"events_{key}.nc"
            _save_events(out_path, evts)
            print(f"Saved: {out_path}")

    # ── Quick plot ────────────────────────────────────────────────────────────
    if not args.no_plot and (events_xz or events_yz):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for row, (key, evts) in enumerate([("xz", events_xz), ("yz", events_yz)]):
            ax_l, ax_r = axes[row]
            if not evts:
                ax_l.set_visible(False); ax_r.set_visible(False); continue
            ds_ev = load_composite_events(out_dir / f"events_{key}.nc")
            comp  = composite_mean_std(ds_ev)
            plot_cloud_root_composite(ax_l, ax_r, comp,
                                      label=f"{args.rt} {key}")
            ax_l.set_title(f"{key}  —  w'θ_l'  ({len(evts)} events, debug)")
            ax_r.set_title(f"{key}  —  w'q_v'")
        plt.suptitle(f"DEBUG composite — {args.rt}  {str(t3d)[:16]}", y=1.01)
        plt.tight_layout()
        plot_path = out_dir / "debug_composite.png"
        plt.savefig(str(plot_path), dpi=100, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()

    print("\nDebug run complete.")


if __name__ == "__main__":
    main()
