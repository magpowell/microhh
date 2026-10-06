"""Cloud detection, tracking and merge/split bookkeeping with tobac on the one-minute cloud water path.

python track_tobac.py --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1 [--solar 12 16.1] [--d-max 500] [--ms-distance 500]
Output: .../<rt>/rep_NN/tobac_features.csv (one row per feature per frame, with cell, area and merge/split ids),
        .../<rt>/rep_NN/tobac_cells.csv (one row per cell: first and last frame, lifetime, largest area, merge/split flags)
Definitions: a feature is a contiguous region of water path above THRESHOLD with at least N_MIN cells (size floor at
detection, no smoothing), periodic in x and y; features are linked by position (trackpy) within D_MAX per minute;
merges and splits by tobac.merge_split_MEST within MS_DISTANCE.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import tobac
import tobac.merge_split  # noqa: F401

from les_io import Run
from snapshot import out_path, run_dir

THRESHOLD = 1.e-3        # g m-2: any cloud water (float32 water path above zero)
N_MIN = 16               # cells
D_MAX = 500.             # m per minute
MS_DISTANCE = 500.       # m


def load_qlp(rd, solar=None):
    """Water path [g m-2] as (time, y, x) with a synthetic datetime axis (tobac parses time as dates)."""
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        q = ds["qlqi_path"]
        if q.ndim == 4 and q.shape[1] == 1:
            q = q.squeeze(q.dims[1], drop=True)
        t = ds["time"].values.astype(float)
        dxy = float(ds["x"].values[1] - ds["x"].values[0])
        run = Run(rd)
        lst = run.lst(t)
        keep = np.ones(t.size, dtype=bool) if solar is None else (lst >= solar[0]) & (lst <= solar[1])
        q = (q.isel({q.dims[0]: np.flatnonzero(keep)}) * 1000.).astype("float32").load()
    t, lst = t[keep], lst[keep]
    dt = float(t[1] - t[0])
    q = q.assign_coords({q.dims[0]: np.datetime64("2000-01-01") + (t * 1e9).astype("timedelta64[ns]")})
    return q, t, lst, dxy, dt


def detect_track(q, dxy, dt, d_max=D_MAX, ms_distance=MS_DISTANCE, n_min=N_MIN, memory=0):
    ny, nx = q.shape[1], q.shape[2]
    pbc = dict(PBC_flag="both", min_h1=0, max_h1=ny, min_h2=0, max_h2=nx)
    feats = tobac.feature_detection_multithreshold(q, dxy=dxy, threshold=[THRESHOLD], target="maximum", position_threshold="center",
                                                   sigma_threshold=0., n_min_threshold=n_min, PBC_flag="both")
    mask, feats = tobac.segmentation_2D(feats, q, dxy=dxy, threshold=THRESHOLD, PBC_flag="both")
    tracks = tobac.linking_trackpy(feats, q, dt=dt, dxy=dxy, d_max=d_max, stubs=1, time_cell_min=0., memory=memory, **pbc)
    ms = tobac.merge_split.merge_split_MEST(tracks, dxy, distance=ms_distance, **pbc)
    return tracks, ms


def cells_table(tr, dt, dxy):
    """One row per cell (tobac track); cell -1 (unlinked features) kept as single-frame cells with negative ids."""
    tr = tr.copy()
    lone = tr.cell < 0
    tr.loc[lone, "cell"] = -1 - np.arange(int(lone.sum()))
    g = tr.groupby("cell")
    c = pd.DataFrame(dict(frame_first=g.frame.min(), frame_last=g.frame.max(), n_frames=g.frame.nunique(), area_max=g.area.max(),
                          area_first=g.area.first(), num_max=g.num.max()))
    c["lifetime"] = c.n_frames * dt
    return tr, c.reset_index()


def merge_split_flags(c, ms):
    """Mark cells that take part in a merge or split according to merge_split_MEST (parent tracks with more than one cell)."""
    parent = pd.Series(np.asarray(ms["cell_parent_track_id"].values), index=np.asarray(ms["cell"].values))
    c["track"] = parent.reindex(c.cell).values
    size = c.groupby("track").cell.transform("size")
    c["in_multi_cell_track"] = (size > 1) & c.track.notna()
    return c


def analyse(expt, rt, rep, solar, d_max, ms_distance, n_min=N_MIN, memory=0, tag=""):
    rd = run_dir(expt, rt, rep)
    q, t, lst, dxy, dt = load_qlp(rd, solar)
    tracks, ms = detect_track(q, dxy, dt, d_max, ms_distance, n_min, memory)
    tracks["area"] = tracks["num"].astype(float) * dxy * dxy
    tracks["t_sec"] = t[tracks.frame.values.astype(int)]
    tracks["lst"] = lst[tracks.frame.values.astype(int)]
    tr, c = cells_table(tracks, dt, dxy)
    c = merge_split_flags(c, ms)
    c["lst_first"] = lst[c.frame_first.values.astype(int)]
    out = out_path(expt, rt, rep, 0).parent
    out.mkdir(parents=True, exist_ok=True)
    keep = [k for k in ("frame", "t_sec", "lst", "feature", "cell", "hdim_1", "hdim_2", "num", "ncells", "area") if k in tr]
    tr[keep].to_csv(out / f"tobac_features{tag}.csv", index=False)
    c.to_csv(out / f"tobac_cells{tag}.csv", index=False)
    return tr, c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    ap.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--solar", type=float, nargs=2, default=None, help="solar-hour window, default the whole run")
    ap.add_argument("--d-max", type=float, default=D_MAX)
    ap.add_argument("--ms-distance", type=float, default=MS_DISTANCE)
    ap.add_argument("--n-min", type=int, default=N_MIN, help="cells a feature needs at detection")
    ap.add_argument("--memory", type=int, default=0, help="frames a cell may be missing and still be linked")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tr, c = analyse(a.expt, a.rt, a.rep, a.solar, a.d_max, a.ms_distance, a.n_min, a.memory, a.tag)
    per_frame = tr.groupby("frame").size()
    u = c[~c.in_multi_cell_track]
    print(f"{a.rt} rep_{a.rep:02d}: features per frame median {int(per_frame.median())}, cells {len(c)}, unlinked {int((c.cell < 0).sum())}, "
          f"cells outside merge/split tracks {len(u)}: lifetime mean {u.lifetime.mean() / 60:.1f} min, p99 {u.lifetime.quantile(.99) / 60:.1f}, max {u.lifetime.max() / 60:.0f}; "
          f"births per hour 12-16 LT: {c[(c.lst_first >= 12) & (c.lst_first < 16)].groupby(np.floor(c.lst_first)).size().to_dict()}", flush=True)
