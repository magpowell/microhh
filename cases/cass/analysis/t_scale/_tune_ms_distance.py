#!/usr/bin/env python3
"""
One-off: tune merge_split_MEST `distance` on a single rep.

Runs the expensive feature-detection + trackpy linking ONCE, then sweeps
several `distance` values through merge_split_MEST and reports the
family-size and family-lifetime distribution for each, so we can pick a
non-degenerate value before the full batch.

Not a reusable tool — delete after the distance is chosen.
"""
import sys
import warnings
import numpy as np
sys.path.insert(0, "/global/homes/m/mpowell/repos/microhh/cases/cass/analysis/t_scale")

from compute_cloud_lifetimes import load_qlp, detect_and_track, aggregate_tracks
import tobac
import tobac.merge_split  # noqa: F401
import pandas as pd

REP = ("/pscratch/sd/m/mpowell/CASS_LES/experiments/"
       "no_aerosols_zero_wind/raytracer/rep_01/qlqi_path.xy.nc")

qlp, t_sim_s, dxy, dt = load_qlp(REP)
sim_min = (len(t_sim_s) * dt) / 60.0
print(f"dxy={dxy} m  dt={dt} s  sim={sim_min:.0f} min  ({len(t_sim_s)} frames)")

print("Feature detection + linking (once) ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    features, tracks = detect_and_track(qlp, dxy, dt, threshold=1.0,
                                        min_area_m2=2500.0, v_max=20.0, memory=1)
ts = aggregate_tracks(tracks, dt)
pt = ts["lifetime_s"].values / 60.0
print(f"per-track: n={len(ts)}  mean={pt.mean():.2f}  median={np.median(pt):.1f}  "
      f"p90={np.percentile(pt,90):.1f}  max={pt.max():.0f} min\n")

def families_for(distance):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = tobac.merge_split.merge_split_MEST(tracks, dxy, distance=distance)
    pv = "cell_parent_track_id"
    c2f = dict(zip(np.asarray(d["cell"].values), np.asarray(d[pv].values)))
    df = tracks[tracks["cell"] >= 0].copy()
    df["family"] = df["cell"].map(c2f)
    df = df.dropna(subset=["family"]); df["family"] = df["family"].astype(int)
    df = df[df["family"] >= 0]
    g = df.groupby("family")
    ff = g["frame"].min().values; fl = g["frame"].max().values
    ncell = g["cell"].nunique().values
    life = (fl - ff) * dt / 60.0 + dt / 60.0
    return life, ncell

print(f"{'distance':>10} | {'n_fam':>6} {'cells/fam':>16} {'max_cells':>9} | "
      f"{'life mean':>9} {'median':>7} {'p90':>6} {'max':>6}  {'max/sim':>7}")
print("-" * 92)
for dist in [None, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0]:
    life, ncell = families_for(dist)
    tag = "default" if dist is None else f"{dist:.0f} m"
    print(f"{tag:>10} | {len(life):>6} "
          f"{ncell.mean():>6.2f}±{ncell.std():>5.1f} {int(ncell.max()):>9} | "
          f"{life.mean():>8.2f} {np.median(life):>7.1f} "
          f"{np.percentile(life,90):>6.1f} {life.max():>6.0f}  "
          f"{life.max()/sim_min:>6.1%}")
