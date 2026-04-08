#!/usr/bin/env python3
"""
cloud_root_composite_average.py  —  Average per-event composite files across reps.

Reads  events_xz.nc / events_yz.nc  for all reps of one (expt, rt) pair,
computes per-rep means, then averages across reps.

Strategy:  per-rep mean first, then mean-of-means across reps.
This gives equal weight to each rep regardless of event count — avoids
reps with many events dominating the composite.

Outputs (in --composite-dir/{expt}/{rt}/):
  composite_xz.nc
  composite_yz.nc

  Each contains, for every variable in COMPOSITE_VARS:
    <var>_mean     (z_nd, xL)   — mean across reps of per-rep means
    <var>_std      (z_nd, xL)   — std  across reps of per-rep means
    <var>_sem      (z_nd, xL)   — standard error of the mean (std / sqrt(n_reps))
    <var>_rep_mean (n_reps, z_nd, xL) — per-rep means (for diagnostics)
  Plus:
    n_events_per_rep   (n_reps,)  — event count per rep
    n_reps             scalar

Usage:
    python cloud_root_composite_average.py \\
        --expt base \\
        --rt   2stream \\
        --composite-dir /pscratch/sd/m/mpowell/CASS_LES/analysis/cloud_root_composite
"""

import argparse
import sys
import numpy as np
import xarray as xr
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))
from cass_analysis import COMPOSITE_VARS, XL_GRID, ZND_GRID


def average_reps(composite_dir: Path, expt: str, rt: str,
                 n_reps: int = 4, verbose: bool = True) -> dict[str, xr.Dataset]:
    """Load per-rep event files, compute per-rep means, average across reps.

    Returns dict with keys 'xz' and 'yz', each an xr.Dataset with the
    averaged composite fields.
    """
    base_dir = composite_dir / expt / rt
    results = {}

    for orient in ("xz", "yz"):
        rep_means = []
        n_per_rep = []

        for rep in range(1, n_reps + 1):
            events_path = base_dir / f"rep_{rep:02d}" / f"events_{orient}.nc"
            if not events_path.exists():
                if verbose:
                    print(f"  SKIP (not found): {events_path}")
                continue

            ds = xr.open_dataset(str(events_path))
            n_ev = ds.sizes["event"]
            n_per_rep.append(n_ev)

            # Per-rep mean over events
            rep_mean_vars = {}
            for vn in COMPOSITE_VARS:
                if vn not in ds:
                    continue
                arr = ds[vn].values   # (n_events, n_znd, n_xL)
                rep_mean_vars[vn] = np.nanmean(arr, axis=0)   # (n_znd, n_xL)
            rep_means.append(rep_mean_vars)
            ds.close()

            if verbose:
                print(f"  {orient}  rep_{rep:02d}: {n_ev} events")

        if not rep_means:
            if verbose:
                print(f"  No {orient} data found — skipping")
            continue

        n_reps_found = len(rep_means)
        out = {}

        for vn in COMPOSITE_VARS:
            if vn not in rep_means[0]:
                continue
            stack = np.stack([r[vn] for r in rep_means], axis=0)   # (n_reps, n_znd, n_xL)
            out[f"{vn}_mean"]     = xr.DataArray(np.nanmean(stack, axis=0),
                                                  dims=["z_nd", "xL"])
            out[f"{vn}_std"]      = xr.DataArray(np.nanstd(stack, axis=0),
                                                  dims=["z_nd", "xL"])
            out[f"{vn}_sem"]      = xr.DataArray(
                np.nanstd(stack, axis=0) / np.sqrt(n_reps_found),
                dims=["z_nd", "xL"],
            )
            out[f"{vn}_rep_mean"] = xr.DataArray(stack,
                                                   dims=["rep", "z_nd", "xL"])

        out["n_events_per_rep"] = xr.DataArray(
            np.array(n_per_rep, dtype=np.int32), dims=["rep"]
        )
        out["n_reps"] = xr.DataArray(np.int32(n_reps_found))

        ds_out = xr.Dataset(
            out,
            coords={
                "xL":    ("xL",    XL_GRID,                   {"long_name": "x/L or y/L"}),
                "z_nd":  ("z_nd",  ZND_GRID,                  {"long_name": "z/z_sl"}),
                "rep":   ("rep",   np.arange(n_reps_found) + 1),
            },
            attrs={
                "expt":        expt,
                "rt":          rt,
                "orientation": orient,
                "n_reps":      n_reps_found,
            },
        )
        results[orient] = ds_out

    return results


def average_reps_windowed(composite_dir: Path, expt: str, rt: str,
                          windows: list[tuple[float, float]],
                          window_labels: list[str],
                          n_reps: int = 4,
                          verbose: bool = True) -> dict[int, dict[str, xr.Dataset]]:
    """Like average_reps, but bins events into time windows before averaging.

    Each event's ``dump_t`` (float64 ns-epoch) is converted to LST, then
    assigned to a window.  Per-rep means are computed within each window,
    then averaged across reps.

    Parameters
    ----------
    windows : list of (lst_lo, lst_hi) tuples — inclusive lower, exclusive upper.
    window_labels : matching list of human-readable labels.

    Returns
    -------
    dict  {window_index: {'xz': xr.Dataset, 'yz': xr.Dataset, 'label': str}}
    """
    from sw_surface_composite import dump_t_to_lst

    base_dir = composite_dir / expt / rt
    n_win = len(windows)
    results = {}

    for orient in ("xz", "yz"):
        # Collect per-rep, per-window means
        # rep_win_means[wi][rep_idx] = {vn: (n_znd, n_xL)}
        rep_win_means = {wi: [] for wi in range(n_win)}
        rep_win_counts = {wi: [] for wi in range(n_win)}

        for rep in range(1, n_reps + 1):
            events_path = base_dir / f"rep_{rep:02d}" / f"events_{orient}.nc"
            if not events_path.exists():
                if verbose:
                    print(f"  SKIP (not found): {events_path}")
                continue

            ds = xr.open_dataset(str(events_path))
            dump_t_arr = ds["dump_t"].values  # (n_events,) float64 ns-epoch

            # Convert all dump_t to LST and assign windows
            lst_arr = np.array([dump_t_to_lst(dt) for dt in dump_t_arr])

            for wi, (lo, hi) in enumerate(windows):
                mask = (lst_arr >= lo) & (lst_arr < hi)
                n_ev_win = int(mask.sum())
                rep_win_counts[wi].append(n_ev_win)

                if n_ev_win == 0:
                    rep_win_means[wi].append(None)
                    continue

                rep_mean_vars = {}
                for vn in COMPOSITE_VARS:
                    if vn not in ds:
                        continue
                    arr = ds[vn].values[mask]  # (n_events_win, n_znd, n_xL)
                    rep_mean_vars[vn] = np.nanmean(arr, axis=0)
                rep_win_means[wi].append(rep_mean_vars)

                if verbose:
                    print(f"  {orient}  rep_{rep:02d}  win {wi} "
                          f"({window_labels[wi]}): {n_ev_win} events")

            ds.close()

        # Average across reps for each window
        for wi in range(n_win):
            valid_reps = [r for r in rep_win_means[wi] if r is not None]
            if not valid_reps:
                continue

            n_reps_found = len(valid_reps)
            out = {}

            for vn in COMPOSITE_VARS:
                if vn not in valid_reps[0]:
                    continue
                stack = np.stack([r[vn] for r in valid_reps], axis=0)
                out[f"{vn}_mean"] = xr.DataArray(
                    np.nanmean(stack, axis=0), dims=["z_nd", "xL"])
                out[f"{vn}_std"] = xr.DataArray(
                    np.nanstd(stack, axis=0), dims=["z_nd", "xL"])
                out[f"{vn}_sem"] = xr.DataArray(
                    np.nanstd(stack, axis=0) / np.sqrt(n_reps_found),
                    dims=["z_nd", "xL"])

            valid_counts = [c for c in rep_win_counts[wi] if c > 0]
            out["n_events_per_rep"] = xr.DataArray(
                np.array(valid_counts, dtype=np.int32), dims=["rep"])
            out["n_reps"] = xr.DataArray(np.int32(n_reps_found))

            ds_out = xr.Dataset(
                out,
                coords={
                    "xL":   ("xL",   XL_GRID,  {"long_name": "x/L or y/L"}),
                    "z_nd": ("z_nd", ZND_GRID, {"long_name": "z/z_sl"}),
                    "rep":  ("rep",  np.arange(n_reps_found) + 1),
                },
                attrs={
                    "expt": expt, "rt": rt, "orientation": orient,
                    "n_reps": n_reps_found,
                    "window_label": window_labels[wi],
                    "window_lst_lo": windows[wi][0],
                    "window_lst_hi": windows[wi][1],
                },
            )

            if wi not in results:
                results[wi] = {"label": window_labels[wi]}
            results[wi][orient] = ds_out

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--expt",          required=True,
                        help="Experiment name (e.g. 'base', 'cs_veg/cs_veg_42000')")
    parser.add_argument("--rt",            required=True,
                        help="Radiation type: '2stream' or 'raytracer'")
    parser.add_argument("--composite-dir", required=True,
                        help="Root composite directory "
                             "(contains {expt}/{rt}/rep_NN/events_*.nc)")
    parser.add_argument("--n-reps", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    comp_dir = Path(args.composite_dir)
    verbose  = not args.quiet

    if verbose:
        print(f"Averaging {args.expt}/{args.rt}  ({args.n_reps} reps)")

    results = average_reps(comp_dir, args.expt, args.rt, args.n_reps, verbose)

    out_dir = comp_dir / args.expt / args.rt
    out_dir.mkdir(parents=True, exist_ok=True)

    for orient, ds in results.items():
        out_path = out_dir / f"composite_{orient}.nc"
        if out_path.exists():
            out_path.unlink()
        ds.to_netcdf(str(out_path))
        n_ev = int(ds["n_events_per_rep"].values.sum())
        if verbose:
            print(f"  → composite_{orient}.nc  "
                  f"({ds.attrs['n_reps']} reps, {n_ev} total events)")

    if verbose:
        print("Done.")


if __name__ == "__main__":
    main()
