#!/usr/bin/env python3
"""
Compute Lagrangian cloud lifetimes via tobac feature-tracking on LWP fields.

Tracks contiguous LWP (qlqi_path) features through the xy time series and
emits one record per track with its first/last appearance, lifetime, peak LWP,
and time-averaged equivalent radius. Use this to compare against the Eulerian
T_τ produced by compute_T_scale.py — Lagrangian lifetimes are typically
longer because the tracker follows clouds across the domain.

Pipeline:
    1. feature_detection_multithreshold (single threshold by default, but the
       --thresholds CLI accepts a list for sensitivity tests).
    2. segmentation_2D (watershed) to attach pixel area to each feature.
    3. linking_trackpy to chain features across frames.
    4. Per-track aggregation → lifetime distribution + time series.

Output:  $SCRATCH/CASS_LES/analysis/lifetime/{expt}/{rt}/rep_{rep:02d}/
            cloud_tracks.nc          (one record per track)
            cloud_lifetime_ts.nc     (per-LST-bin distribution stats)

Usage:
    python compute_cloud_lifetimes.py --expt no_aerosols_zero_wind --rt raytracer --rep 1
    python compute_cloud_lifetimes.py --expt wind_sun --rt raytracer --rep 2 --threshold 5
    python compute_cloud_lifetimes.py --expt base --rt 2stream --rep 1
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import tobac
import tobac.merge_split  # noqa: F401  (submodule must be imported explicitly)

LST_OFFSET = 5.5   # h: simulation t=0 → 05:30 LST


def solar_azimuth_deg(t_sec, lat_deg, lon_deg, doy_start=205, hour_utc_start=12.0):
    """Solar azimuth (deg from N, clockwise). NOAA algorithm; matches cass_input.py.
    CASS: t=0 → day 205.5 = July 24, 12:00 UTC (defaults). Scalar or array t_sec."""
    hour_utc_cont = hour_utc_start + np.asarray(t_sec, dtype=float) / 3600.0
    doy_cont = doy_start + hour_utc_cont / 24.0
    gamma = 2.0 * np.pi / 365.0 * (doy_cont - 1.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    hour_utc = hour_utc_cont % 24.0
    tst = hour_utc * 60.0 + eqtime + 4.0 * lon_deg
    ha = np.radians(tst / 4.0 - 180.0)
    lat = np.radians(lat_deg)
    az = np.arctan2(-np.cos(decl) * np.sin(ha),
                    np.sin(decl) * np.cos(lat) - np.cos(decl) * np.sin(lat) * np.cos(ha))
    return np.degrees(np.mod(az, 2.0 * np.pi))


def load_qlp(path: Path) -> tuple[xr.DataArray, np.ndarray, float, float]:
    """Load qlqi_path (kg m-2) as g m-2 DataArray with (time, y, x).

    tobac stringifies the time coordinate and parses it as a calendar date,
    so the raw seconds axis (e.g. 49800.0) raises DateParseError. We attach a
    synthetic datetime64 axis (uniform spacing — all tobac needs; dt is also
    passed explicitly to linking) and return the original seconds separately
    for the output time axis.
    """
    ds  = xr.open_dataset(path, decode_times=False)
    qlp = ds["qlqi_path"]
    if qlp.ndim == 4 and qlp.shape[1] == 1:    # singleton z dimension
        qlp = qlp.squeeze(qlp.dims[1], drop=True)
    qlp = (qlp * 1000.0).astype("float32")     # kg m-2 → g m-2
    qlp.attrs.update(units="g m-2", long_name="liquid+ice water path")
    dxy = float(ds["x"].values[1] - ds["x"].values[0])
    t_sim_s = ds["time"].values.astype(float)  # seconds since sim start
    dt  = float(t_sim_s[1] - t_sim_s[0])
    ds.close()
    # Synthetic datetime axis (epoch arbitrary; only spacing matters to tobac)
    t_dt = np.datetime64("2003-07-24T00:00:00") + \
           (t_sim_s * 1e9).astype("timedelta64[ns]")
    qlp = qlp.assign_coords({qlp.dims[0]: t_dt})
    return qlp, t_sim_s, dxy, dt


def detect_and_track(qlp: xr.DataArray, dxy: float, dt: float,
                     threshold: float, min_area_m2: float,
                     v_max: float, memory: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run feature detection → segmentation → tracking. Returns (features, tracks)."""
    # tobac v1.6 accepts xarray DataArrays.
    features = tobac.feature_detection_multithreshold(
        qlp, dxy=dxy,
        threshold=[threshold],
        target="maximum",
        position_threshold="center",
        sigma_threshold=0.5,
        n_min_threshold=int(np.ceil(min_area_m2 / (dxy * dxy))),
    )
    if features is None or len(features) == 0:
        return None, None

    tracks = tobac.linking_trackpy(
        features, qlp, dxy=dxy, dt=dt,
        v_max=v_max,
        memory=memory,
        adaptive_step=0.95,
        adaptive_stop=0.2,
    )
    return features, tracks


def aggregate_tracks(tracks: pd.DataFrame, dt: float) -> pd.DataFrame:
    """One row per cell (cloud trajectory): first/last time, lifetime, peak LWP, etc."""
    # tobac assigns cell = -1 for unlinked features; drop them.
    df = tracks[tracks["cell"] >= 0].copy()
    g = df.groupby("cell")
    agg = pd.DataFrame({
        "cell"          : g["cell"].first().values,
        "frame_first"   : g["frame"].min().values,
        "frame_last"    : g["frame"].max().values,
        "n_frames"      : g["frame"].count().values,
        "time_first_s"  : g["time"].min().values.astype("datetime64[ns]").astype(float) / 1e9
                          if np.issubdtype(g["time"].min().values.dtype, np.datetime64)
                          else g["time"].min().values,
        "time_last_s"   : g["time"].max().values.astype("datetime64[ns]").astype(float) / 1e9
                          if np.issubdtype(g["time"].max().values.dtype, np.datetime64)
                          else g["time"].max().values,
    })
    agg["lifetime_s"] = (agg["frame_last"] - agg["frame_first"]) * dt + dt
    return agg.reset_index(drop=True)


def per_feature_size(tracks: pd.DataFrame, dxy: float, dt: float) -> pd.DataFrame:
    """One row per linked feature with size and age along its track.

    Enables size-at-formation vs size-later analysis, and (via x_m, y_m
    centroid) spatial joins to surface fields (e.g. testing whether
    afternoon survivors sit in sunlit shadow-gaps). `num` is the pixel
    count of the feature at the detection threshold (tobac); area = num·dxy²,
    equivalent radius = sqrt(area/π), L = 2·r_eq (matches base_comparison).
    `age_s` is time since the track's first detection.
    """
    df = tracks[tracks["cell"] >= 0].copy()
    if "num" not in df.columns:          # safety: older tobac may name it differently
        alt = next((c for c in ("ncells", "npix", "area") if c in df.columns), None)
        if alt is None:
            raise RuntimeError(f"no pixel-count column in tracks; cols={list(df.columns)}")
        df = df.rename(columns={alt: "num"})
    df["t_sim_s"]  = df["frame"].astype(float) * dt
    df["area_m2"]  = df["num"].astype(float) * dxy * dxy
    df["req_m"]    = np.sqrt(df["area_m2"] / np.pi)
    df["L_m"]      = 2.0 * df["req_m"]
    f0 = df.groupby("cell")["frame"].transform("min")
    df["age_s"]    = (df["frame"] - f0).astype(float) * dt
    # Centroid in metres. tobac carries physical 'x'/'y' from the DataArray
    # coords; fall back to grid-index columns × dxy. qlp dims are (time,y,x)
    # → hdim_1↔y (north), hdim_2↔x (east).
    if {"x", "y"} <= set(df.columns):
        df["x_m"] = df["x"].astype(float)
        df["y_m"] = df["y"].astype(float)
    else:
        df["x_m"] = df["hdim_2"].astype(float) * dxy
        df["y_m"] = df["hdim_1"].astype(float) * dxy
    return df[["cell", "frame", "t_sim_s", "age_s",
               "num", "area_m2", "req_m", "L_m", "x_m", "y_m"]].reset_index(drop=True)


def aggregate_families(tracks: pd.DataFrame, dxy: float, dt: float,
                       distance: float | None = None
                       ) -> tuple[pd.DataFrame, "xr.Dataset"]:
    """Merge/split-aware *family* lifetimes via tobac.merge_split_MEST.

    A per-track (cell) lifetime is cut at every merge/split event, so the
    fewer-larger 3D regime shows shorter per-track lifetimes even when the
    cloud *system* persists longer. This groups cells into merge/split-
    connected families and reports the lifetime of the whole family
    (first appearance of any member → last disappearance of any descendant).
    """
    d = tobac.merge_split.merge_split_MEST(tracks, dxy, distance=distance)
    # cell → parent track(family) id. Be defensive about the variable name.
    cand = [v for v in d.data_vars if "cell" in v and "track" in v and "parent" in v]
    parent_var = "cell_parent_track_id" if "cell_parent_track_id" in d.data_vars \
                 else (cand[0] if cand else None)
    if parent_var is None:
        raise RuntimeError(f"merge_split output lacks a cell→track map; "
                           f"vars={list(d.data_vars)}")
    cell2fam = dict(zip(np.asarray(d["cell"].values),
                        np.asarray(d[parent_var].values)))

    df = tracks[tracks["cell"] >= 0].copy()
    df["family"] = df["cell"].map(cell2fam)
    df = df.dropna(subset=["family"])
    df["family"] = df["family"].astype(int)
    df = df[df["family"] >= 0]                       # drop unassigned (-1)

    g = df.groupby("family")
    fam = pd.DataFrame({
        "family"      : g["family"].first().values,
        "frame_first" : g["frame"].min().values,
        "frame_last"  : g["frame"].max().values,
        "n_cells"     : g["cell"].nunique().values,
        "n_features"  : g["frame"].count().values,
    })
    fam["lifetime_s"] = (fam["frame_last"] - fam["frame_first"]) * dt + dt
    return fam.reset_index(drop=True), d


def _track_xy_m(df_cell: pd.DataFrame, dxy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_east_m, y_north_m) centroid series for one cell, ordered by frame.

    Prefers tobac projection coords; falls back to grid-index columns × dxy.
    qlp dims are (time, y, x) so hdim_1↔y(north), hdim_2↔x(east).
    """
    df_cell = df_cell.sort_values("frame")
    if {"projection_x_coordinate", "projection_y_coordinate"} <= set(df_cell.columns):
        x_e = df_cell["projection_x_coordinate"].values.astype(float)
        y_n = df_cell["projection_y_coordinate"].values.astype(float)
    else:
        x_e = df_cell["hdim_2"].values.astype(float) * dxy
        y_n = df_cell["hdim_1"].values.astype(float) * dxy
    return x_e, y_n


def shadow_advection_check(tracks: pd.DataFrame, track_summary: pd.DataFrame,
                           dt: float, dxy: float, lat: float, lon: float,
                           min_frames: int = 5, min_disp_m: float = 200.0
                           ) -> pd.DataFrame:
    """Per-track cloud-motion bearing vs the anti-solar (shadow) bearing.

    For each cell: net displacement → motion bearing (deg from N, CW); solar
    azimuth at the track's mid-time → shadow bearing = atan2(-sinφ, -cosφ);
    signed lag = motion − shadow wrapped to [-180, 180].

    Tracks shorter than `min_frames` or with net displacement < `min_disp_m`
    are flagged (bearing ill-defined for near-stationary / very short clouds).
    """
    df = tracks[tracks["cell"] >= 0]
    rows = []
    for cell, dfc in df.groupby("cell"):
        x_e, y_n = _track_xy_m(dfc, dxy)
        nfr = len(x_e)
        dx = x_e[-1] - x_e[0]
        dy = y_n[-1] - y_n[0]
        disp = float(np.hypot(dx, dy))
        # mid-time of the track in sim seconds (frame index × dt)
        f0 = int(dfc["frame"].min()); f1 = int(dfc["frame"].max())
        t_mid_s = 0.5 * (f0 + f1) * dt
        sun_az = float(solar_azimuth_deg(t_mid_s, lat, lon))
        # Shadow (anti-solar) bearing: direction the wind should push the cloud
        shadow_brg = np.degrees(np.arctan2(-np.sin(np.radians(sun_az)),
                                           -np.cos(np.radians(sun_az)))) % 360.0
        motion_brg = (np.degrees(np.arctan2(dx, dy)) % 360.0) if disp > 0 else np.nan
        lag = ((motion_brg - shadow_brg + 180.0) % 360.0) - 180.0 \
              if np.isfinite(motion_brg) else np.nan
        speed = disp / max((f1 - f0) * dt, dt)
        rows.append(dict(cell=int(cell), n_frames=nfr, disp_m=disp,
                         speed_m_s=speed, sun_az_deg=sun_az,
                         shadow_bearing_deg=shadow_brg,
                         motion_bearing_deg=motion_brg,
                         lag_deg=lag, t_mid_s=t_mid_s,
                         usable=bool(nfr >= min_frames and disp >= min_disp_m)))
    out = pd.DataFrame(rows)
    return out.merge(track_summary[["cell", "lifetime_s"]], on="cell", how="left")


def lifetime_timeseries(track_summary: pd.DataFrame, t_sim_s: np.ndarray) -> xr.Dataset:
    """For each LST bin, summarise the lifetime distribution of clouds *active in that frame*."""
    # A track is "active" at frame f if frame_first <= f <= frame_last.
    nt = len(t_sim_s)
    mean_life = np.full(nt, np.nan)
    med_life  = np.full(nt, np.nan)
    p90_life  = np.full(nt, np.nan)
    n_active  = np.zeros(nt, dtype=int)
    if len(track_summary) == 0:
        return xr.Dataset(
            {"lifetime_mean":   (("time",), mean_life),
             "lifetime_median": (("time",), med_life),
             "lifetime_p90":    (("time",), p90_life),
             "n_active":        (("time",), n_active)},
            coords={"time": t_sim_s},
        )
    ff = track_summary["frame_first"].values
    fl = track_summary["frame_last"].values
    lt = track_summary["lifetime_s"].values
    for f in range(nt):
        mask = (ff <= f) & (fl >= f)
        if mask.any():
            mean_life[f] = float(lt[mask].mean())
            med_life[f]  = float(np.median(lt[mask]))
            p90_life[f]  = float(np.percentile(lt[mask], 90))
            n_active[f]  = int(mask.sum())
    t_lst = t_sim_s / 3600.0 + LST_OFFSET
    return xr.Dataset(
        {"lifetime_mean":   (("time",), mean_life),
         "lifetime_median": (("time",), med_life),
         "lifetime_p90":    (("time",), p90_life),
         "n_active":        (("time",), n_active),
         "t_lst_h":         (("time",), t_lst)},
        coords={"time": t_sim_s},
        attrs={"description":
               "Per-frame statistics of cloud lifetimes (over tracks active in that frame)"},
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expt", required=True,
                   help="experiment name (e.g. no_aerosols_zero_wind, base, wind_sun)")
    p.add_argument("--rt", required=True, choices=["2stream", "raytracer"])
    p.add_argument("--rep", type=int, required=True, help="rep number (1-based)")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="LWP feature-detection threshold in g/m² (default 1.0)")
    p.add_argument("--min-area", type=float, default=2500.0,
                   help="minimum feature area in m² (default 2500 = 1 grid cell at dx=50m)")
    p.add_argument("--v-max", type=float, default=20.0,
                   help="trackpy max plausible cloud velocity in m/s (default 20)")
    p.add_argument("--memory", type=int, default=1,
                   help="trackpy memory: frames a cloud can disappear (default 1)")
    p.add_argument("--shadow-check", action="store_true",
                   help="also compute per-track motion bearing vs anti-solar (shadow) "
                        "bearing — verifies clouds advect toward their shadows (wind_sun)")
    p.add_argument("--lat", type=float, default=36.5, help="site latitude (deg N)")
    p.add_argument("--lon", type=float, default=-97.5, help="site longitude (deg E)")
    p.add_argument("--merge-split", action="store_true",
                   help="also compute merge/split-aware family lifetimes "
                        "(tobac.merge_split_MEST) — cloud-system longevity, not per-cell")
    p.add_argument("--ms-distance", type=float, default=None,
                   help="merge/split max linking distance in m (default: tobac "
                        "internal, ~25*dxy — usually over-merges; tune this)")
    p.add_argument("--force", action="store_true", help="overwrite existing output")
    args = p.parse_args()

    SCRATCH  = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell"))
    LES_ROOT = SCRATCH / "CASS_LES"
    # base lives at $LES_ROOT/base/; everything else under experiments/
    if args.expt == "base":
        in_path = LES_ROOT / "base" / args.rt / f"rep_{args.rep:02d}" / "qlqi_path.xy.nc"
    else:
        in_path = (LES_ROOT / "experiments" / args.expt
                   / args.rt / f"rep_{args.rep:02d}" / "qlqi_path.xy.nc")

    out_dir = (LES_ROOT / "analysis" / "lifetime"
               / args.expt / args.rt / f"rep_{args.rep:02d}")
    out_tracks = out_dir / "cloud_tracks.nc"
    out_ts     = out_dir / "cloud_lifetime_ts.nc"
    out_shadow = out_dir / "cloud_shadow_check.nc"
    out_family = out_dir / "cloud_family_lifetimes.nc"
    out_feat   = out_dir / "cloud_track_features.nc"

    if out_tracks.exists() and out_ts.exists() and not args.force:
        print(f"[skip] outputs exist in {out_dir} (--force to overwrite)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading  {in_path}")
    qlp, t_sim_s, dxy, dt = load_qlp(in_path)
    nt, ny, nx = qlp.shape
    print(f"Grid: {nt} × {ny} × {nx}  |  dx={dxy:.1f} m  dt={dt:.1f} s  |  "
          f"threshold={args.threshold} g/m²  min_area={args.min_area:.0f} m²")

    print("Detecting features + tracking ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        features, tracks = detect_and_track(
            qlp, dxy, dt,
            threshold=args.threshold,
            min_area_m2=args.min_area,
            v_max=args.v_max,
            memory=args.memory,
        )
    if tracks is None:
        print("  No features detected — writing empty outputs")
        track_summary = pd.DataFrame(columns=["cell","frame_first","frame_last","n_frames",
                                              "time_first_s","time_last_s","lifetime_s"])
    else:
        track_summary = aggregate_tracks(tracks, dt)
        n_unlinked = int((tracks["cell"] < 0).sum())
        print(f"  features: {len(features):>6}   tracks: {len(track_summary):>6}   "
              f"unlinked features: {n_unlinked}")

    # Per-frame lifetime time series (t_sim_s is the original seconds axis)
    ts = lifetime_timeseries(track_summary, t_sim_s)

    # Write tracks summary
    ds_tracks = xr.Dataset.from_dataframe(track_summary) if len(track_summary) else \
                xr.Dataset(coords={"index": np.arange(0)})
    ds_tracks.attrs.update(expt=args.expt, rt=args.rt, rep=args.rep,
                           threshold_g_m2=args.threshold,
                           min_area_m2=args.min_area,
                           v_max_m_s=args.v_max,
                           memory=args.memory,
                           dt_s=dt, dx_m=dxy)
    ds_tracks.to_netcdf(out_tracks)
    print(f"Saved   {out_tracks}")

    ts.attrs = dict(ds_tracks.attrs)
    ts.to_netcdf(out_ts)
    print(f"Saved   {out_ts}")

    # Per-feature size along tracks (size-at-formation vs size-later analysis)
    if tracks is not None and len(track_summary):
        feat = per_feature_size(tracks, dxy, dt)
        ds_feat = xr.Dataset.from_dataframe(feat)
        ds_feat.attrs.update(ds_tracks.attrs)
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
               for v in ("t_sim_s", "age_s", "area_m2", "req_m", "L_m",
                         "x_m", "y_m")}
        ds_feat.to_netcdf(out_feat, encoding=enc)
        print(f"Saved   {out_feat}")
        # quick size-at-formation vs later, by cloud age
        a0 = feat.loc[feat["age_s"] == 0, "L_m"]
        print(f"  L at formation (age 0):  mean={a0.mean():6.1f}  median={a0.median():6.1f} m  "
              f"(n={len(a0)})")
        for amin in (300, 600, 1200):
            sub = feat.loc[np.isclose(feat["age_s"], amin), "L_m"]
            if len(sub):
                print(f"  L at age {amin//60:>2d} min:      mean={sub.mean():6.1f}  "
                      f"median={sub.median():6.1f} m  (n={len(sub)})")

    # Summary print
    if len(track_summary):
        life_min = track_summary["lifetime_s"].values / 60.0
        print(f"\nLifetime distribution (n={len(life_min)} tracks):")
        print(f"  mean   = {life_min.mean():6.2f} min")
        print(f"  median = {np.median(life_min):6.2f} min")
        print(f"  p90    = {np.percentile(life_min, 90):6.2f} min")
        print(f"  max    = {life_min.max():6.2f} min")

    # Optional: verify clouds advect toward their shadows
    if args.shadow_check and tracks is not None and len(track_summary):
        sc = shadow_advection_check(tracks, track_summary, dt, dxy,
                                    lat=args.lat, lon=args.lon)
        ds_sc = xr.Dataset.from_dataframe(sc)
        ds_sc.attrs.update(ds_tracks.attrs)
        ds_sc.attrs["lat"] = args.lat
        ds_sc.attrs["lon"] = args.lon
        ds_sc.to_netcdf(out_shadow)
        print(f"Saved   {out_shadow}")
        good = sc[sc["usable"]]
        if len(good):
            lag = good["lag_deg"].values
            within = lambda d: 100.0 * np.mean(np.abs(lag) <= d)
            print(f"\nShadow-advection check  (n_usable={len(good)} / {len(sc)} tracks; "
                  f"|disp|≥200 m, ≥5 frames):")
            print(f"  mean |lag|        = {np.mean(np.abs(lag)):6.1f}°")
            print(f"  median signed lag = {np.median(lag):+6.1f}°")
            print(f"  within 30° of anti-solar = {within(30):4.0f}%")
            print(f"  within 45° of anti-solar = {within(45):4.0f}%")
            print(f"  within 90° (correct half-plane) = {within(90):4.0f}%")
        else:
            print("\nShadow-advection check: no usable tracks "
                  "(all too short or near-stationary)")

    # Optional: merge/split-aware family lifetimes (cloud-system longevity)
    if args.merge_split and tracks is not None and len(track_summary):
        print("\nMerge/split tracking (tobac.merge_split_MEST) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fam, _d = aggregate_families(tracks, dxy, dt,
                                         distance=args.ms_distance)
        ds_fam = xr.Dataset.from_dataframe(fam)
        ds_fam.attrs.update(ds_tracks.attrs)
        ds_fam.to_netcdf(out_family)
        print(f"Saved   {out_family}")
        if len(fam):
            fl = fam["lifetime_s"].values / 60.0
            pt = track_summary["lifetime_s"].values / 60.0
            print(f"  families: {len(fam)}  (vs {len(track_summary)} per-track cells; "
                  f"mean {fam['n_cells'].mean():.2f} cells/family)")
            print(f"  family lifetime  mean={fl.mean():6.2f}  median={np.median(fl):6.2f}  "
                  f"p90={np.percentile(fl,90):6.2f}  max={fl.max():6.2f} min")
            print(f"  (per-track  was  mean={pt.mean():6.2f}  median={np.median(pt):6.2f} min)")
        else:
            print("  no families produced")


if __name__ == "__main__":
    main()
