"""Horizontal drift of clouds and of their roots from the one-minute fields.

The drift of a cloud is the displacement of its centroid between consecutive frames in which it neither merges nor
splits, for clouds of at least DMIN equivalent diameter. It is rotated into the solar azimuth of that time: along is
positive toward the sun, across is positive to the left of that direction. A cloud that grows on one side moves its
centroid without moving itself; the drift includes that.

The root of a cloud is marked in a disk around the cloud centroid (WINDOW times the equivalent radius) by the
centroid of one of
  flux   the surface flux deficit, mean of H + LE minus H + LE where that is positive (the shaded surface),
  conv   rising air at ZCONV, from the one-minute 3D files (w at the surface, as in the xy cross-sections, is zero).
The drift of the whole pattern comes from the cross-correlation of consecutive fields; the largest clouds dominate
it, so it is reported on its own. With wind=True the mean wind of the cloud layer is subtracted from every drift.

python drift.py --run <run dir> [--tmin 23700 --tmax 38400] [--wind] [--conv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import track as tr
from composite import solar_azimuth
from les_io import Run

CP, LV = 1005., 2.501e6
DMIN = 500.          # smallest equivalent diameter [m]
WINDOW = 1.5         # radius of the root window in equivalent radii
ZCONV = 100.         # height of the rising-air marker [m]
MIN_STEPS = 5
FRAC_MIN = 1.e-3     # cloud layer: levels with cloud fraction above this


def wrap(d, size):
    """Displacement on a periodic axis, in [-size / 2, size / 2)."""
    return (d + 0.5 * size) % size - 0.5 * size


def diameter(area):
    return 2. * np.sqrt(np.asarray(area, dtype=float) / np.pi)


def sun_direction(run, t):
    """Horizontal unit vectors (x east, y north): toward the sun, and to the left of that direction."""
    az = solar_azimuth(run, np.asarray(t, dtype=float))[0]
    return (np.sin(az), np.cos(az)), (-np.cos(az), np.sin(az))


def steps(feats, xsize, ysize, dmin=DMIN):
    """One row per cloud and pair of consecutive frames in which the cloud has one predecessor and one successor
    and is at least dmin across in both. Positions are those of the two frames."""
    a = feats[feats["n_succ"] <= 1]
    b = feats[feats["n_pred"] <= 1].assign(frame=lambda f: f["frame"] - 1)
    m = a.merge(b, on=["track", "frame"], suffixes=("", "_next"))
    m = m[np.minimum(diameter(m["area"]), diameter(m["area_next"])) >= dmin]
    dt = m["time_next"] - m["time"]
    return pd.DataFrame(dict(track=m["track"], frame=m["frame"], time=0.5 * (m["time"] + m["time_next"]), dt=dt,
                             x=m["x"], y=m["y"], x_next=m["x_next"], y_next=m["y_next"],
                             area=m["area"], area_next=m["area_next"],
                             u=wrap(m["x_next"] - m["x"], xsize) / dt, v=wrap(m["y_next"] - m["y"], ysize) / dt)
                        ).sort_values(["frame", "track"]).reset_index(drop=True)


def window_centroid(w, xc, yc, radius, dx, dy):
    """Offset (x, y) from (xc, yc) of the centroid of the weights w >= 0 inside a periodic disk; nan without weight."""
    ny, nx = w.shape
    ni, nj = int(np.ceil(radius / dx)) + 1, int(np.ceil(radius / dy)) + 1
    i0, j0 = int(np.floor(xc / dx)), int(np.floor(yc / dy))
    ii, jj = np.arange(i0 - ni, i0 + ni + 1), np.arange(j0 - nj, j0 + nj + 1)
    ox = (ii + 0.5) * dx - xc
    oy = (jj + 0.5) * dy - yc
    p = w[np.ix_(jj % ny, ii % nx)] * (ox[None, :]**2 + oy[:, None]**2 <= radius**2)
    s = p.sum()
    if s <= 0.:
        return np.nan, np.nan
    return float((p * ox[None, :]).sum() / s), float((p * oy[:, None]).sum() / s)


def root_steps(st, marker, dx, dy, xsize, ysize, window=WINDOW):
    """Offset of the root from the cloud centroid in the first frame (rx, ry) and drift of the root (ur, vr) for
    every step. marker(frame) returns the field of weights of that frame."""
    out = np.full((len(st), 4), np.nan)
    cache = {}

    def field(f):
        if f not in cache:
            for g in [g for g in cache if g < f - 1]:
                del cache[g]
            cache[f] = marker(f)
        return cache[f]

    for k, r in enumerate(st.itertuples(index=False)):
        f = int(r.frame)
        a = window_centroid(field(f), r.x, r.y, window * 0.5 * diameter(r.area), dx, dy)
        b = window_centroid(field(f + 1), r.x_next, r.y_next, window * 0.5 * diameter(r.area_next), dx, dy)
        out[k] = (a[0], a[1], (wrap(r.x_next - r.x, xsize) + b[0] - a[0]) / r.dt,
                  (wrap(r.y_next - r.y, ysize) + b[1] - a[1]) / r.dt)
    return st.assign(rx=out[:, 0], ry=out[:, 1], ur=out[:, 2], vr=out[:, 3])


def project(st, run, pairs=(("u", "v", ""),)):
    """Adds the components along and across the sun direction for every (x name, y name, suffix)."""
    e, n = sun_direction(run, st["time"].values)
    new = {}
    for a, b, s in pairs:
        new["along" + s] = st[a] * e[0] + st[b] * e[1]
        new["across" + s] = st[a] * n[0] + st[b] * n[1]
    return st.assign(**new)


def per_track(st, min_steps=MIN_STEPS):
    """Mean over the steps of every cloud followed for at least min_steps steps."""
    g = st.groupby("track")
    t = g[[c for c in st.columns if c not in ("track", "frame")]].mean()
    t["n"] = g.size()
    return t[t["n"] >= min_steps].reset_index()


def pattern_shift(a, b, dx, dy):
    """Displacement (x, y) that carries field a into field b, from the peak of their cross-correlation."""
    fa, fb = np.fft.rfft2(a - a.mean()), np.fft.rfft2(b - b.mean())
    c = np.fft.irfft2(np.conj(fa) * fb, s=a.shape)
    j, i = np.unravel_index(np.argmax(c), c.shape)

    def sub(cm, c0, cp):
        d = cm - 2. * c0 + cp
        return 0. if d == 0. else 0.5 * (cm - cp) / d

    ny, nx = a.shape
    sj = j + sub(c[(j - 1) % ny, i], c[j, i], c[(j + 1) % ny, i])
    si = i + sub(c[j, (i - 1) % nx], c[j, i], c[j, (i + 1) % nx])
    return wrap(si * dx, nx * dx), wrap(sj * dy, ny * dy)


def layer_mean_wind(time, u, v, frac, t):
    """Mean wind (u, v) over the levels with cloud fraction above FRAC_MIN, interpolated to the times t."""
    m = frac > FRAC_MIN
    n = np.maximum(m.sum(axis=1), 1)
    return np.interp(t, time, (u * m).sum(axis=1) / n), np.interp(t, time, (v * m).sum(axis=1) / n)


def cloud_layer_wind(run_dir, t):
    f = sorted(Path(run_dir).glob("cass.default.*.nc"))[0]
    time = np.round(xr.open_dataset(f, decode_times=False)["time"].values)
    d = xr.open_dataset(f, group="default", decode_times=False)
    frac = xr.open_dataset(f, group="thermo", decode_times=False)["ql_frac"].values
    return layer_mean_wind(time, d["u"].values, d["v"].values, frac, t)


def flux_deficit(run):
    """Marker of the shaded surface at every frame: mean of H + LE minus H + LE, where positive [W m-2]."""
    rho = float(np.fromfile(run.dir / "rhoref.0000000", dtype="<f8")[run.ktot])
    h, time = tr.load(run.dir, "thl_fluxbot")[:2]
    le = tr.load(run.dir, "qt_fluxbot")[0]
    f = rho * (CP * h.astype(float) + LV * le.astype(float))
    return np.maximum(f.mean(axis=(1, 2), keepdims=True) - f, 0.), time


def rising_air(run, time, z=ZCONV):
    """Marker function of rising air at the half level nearest z, from the one-minute 3D files."""
    k = int(np.argmin(np.abs(run.zh[:-1] - z)))
    n = run.jtot * run.itot

    def marker(f):
        w = np.fromfile(run.dir / f"w_hf.{int(round(time[f])):07d}", dtype="<f4", count=n, offset=4 * n * k)
        return np.maximum(w.reshape(run.jtot, run.itot).astype(float), 0.)
    return marker


def analyse(run_dir, tmin=None, tmax=None, thr=0., wind=False, conv=False, dmin=DMIN):
    """Steps of clouds with their drift, per-track means, and the drift of the pattern."""
    run = Run(run_dir)
    path, time, x, y = tr.load(run.dir, "qlqi_path")
    base, top, core = (tr.load(run.dir, v)[0] for v in ("qlqi_base", "qlqi_top", "qlqicore_max_thv_prime"))
    k = (time >= (time[0] if tmin is None else tmin)) & (time <= (time[-1] if tmax is None else tmax))
    k0 = int(np.argmax(k))
    path, base, top, core, time = path[k], base[k], top[k], core[k], time[k]
    feats, _ = tr.track(path, base, top, core, time, run.dx, run.dy, thr)
    st = steps(feats, run.xsize, run.ysize, dmin)
    pairs = [("u", "v", "")]
    deficit = flux_deficit(run)[0]
    markers = [("flux", lambda f: deficit[k0 + f])] + ([("conv", rising_air(run, time))] if conv else [])
    for name, marker in markers:
        r = root_steps(st, marker, run.dx, run.dy, run.xsize, run.ysize)
        st = st.assign(**{f"rx_{name}": r["rx"], f"ry_{name}": r["ry"], f"u_{name}": r["ur"], f"v_{name}": r["vr"]})
        pairs += [(f"u_{name}", f"v_{name}", f"_{name}"), (f"rx_{name}", f"ry_{name}", f"_off_{name}")]
    dt = np.diff(time)
    ps = np.array([pattern_shift(path[f], path[f + 1], run.dx, run.dy) for f in range(len(time) - 1)]) / dt[:, None]
    pat = pd.DataFrame(dict(time=0.5 * (time[1:] + time[:-1]), u=ps[:, 0], v=ps[:, 1]))
    pat = pat[path[:-1].max(axis=(1, 2)) > thr].reset_index(drop=True)
    if wind:
        for d, cols in ((st, [p for p in pairs if not p[2].startswith("_off")]), (pat, [("u", "v", "")])):
            uw, vw = cloud_layer_wind(run.dir, d["time"].values)
            for a, b, _ in cols:
                d[a], d[b] = d[a] - uw, d[b] - vw
    st = project(st, run, pairs)
    return st, per_track(st), project(pat, run)


def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan), x.size


def report(run_dir, tmin, tmax, wind, conv):
    st, t, pat = analyse(run_dir, tmin, tmax, wind=wind, conv=conv)
    print(f"{run_dir}\n{len(st)} steps of {st['track'].nunique()} clouds of at least {DMIN:.0f} m; "
          f"{len(t)} clouds followed for {MIN_STEPS} steps or more" + ("; cloud-layer wind subtracted" if wind else ""))
    rows = [("cloud drift", t, "", "m s-1"), ("root drift, flux deficit", t, "_flux", "m s-1"),
            ("root offset, flux deficit", t, "_off_flux", "m")]
    if conv:
        rows += [("root drift, rising air", t, "_conv", "m s-1"), ("root offset, rising air", t, "_off_conv", "m")]
    rows += [("pattern drift", pat, "", "m s-1")]
    for name, d, s, unit in rows:
        out, n = [], 0
        for c in ("along", "across"):
            m, se, n = mean_se(d[c + s]) if len(d) else (np.nan, np.nan, 0)
            out.append(f"{c} {m:8.3f} +- {se:.3f}")
        print(f"  {name:26s} {out[0]}   {out[1]} {unit}  (n = {n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--wind", action="store_true", help="subtract the mean wind of the cloud layer")
    ap.add_argument("--conv", action="store_true", help="root from rising air in the one-minute 3D files")
    a = ap.parse_args()
    report(a.run, a.tmin, a.tmax, a.wind, a.conv)
