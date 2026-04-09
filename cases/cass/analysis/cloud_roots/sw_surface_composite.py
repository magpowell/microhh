#!/usr/bin/env python3
"""
Compute cloud-object-composite of surface SW radiation (Q_sw) in the
chord-normalised coordinate system of existing cloud events.

Reads events_xz.nc / events_yz.nc (output of cloud_root_composite_prep.py)
and the surface SW xy files, extracts chord-normalised Q profiles per event,
groups into time windows, and saves composite means per window.

For each event the chord-normalised SW profile is extracted at the matching
SW snapshot and interpolated to XL_GRID (200 pts, −1 to 1).

Q sources:
  raytracer : sw_flux_sfc_dir_rt + sw_flux_sfc_dif_rt  (true surface dir+dif)
  2stream   : sw_flux_dn[:, 0, :, :]                   (downwelling at zh=0)

Output saved to {comp-dir}/sw_composite.nc alongside the events files.

Usage:
    python sw_surface_composite.py \\
        --run-dir  $SCRATCH/CASS_LES/experiments/no_aerosols_zero_wind/raytracer/rep_01 \\
        --rt       raytracer \\
        --comp-dir $SCRATCH/CASS_LES/analysis/cloud_root_composite/no_aerosols_zero_wind/raytracer/rep_01
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── Shared imports from parent analysis package ──────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cass_analysis import zenith_angle, CASS_LAT as LAT, CASS_DOY as DOY, LST_OFFSET, XL_GRID, dump_t_to_lst

# ── Grid / time constants ──────────────────────────────────────────────────────
DT_XY = 60     # seconds per xy snapshot

# ── Equal-zenith-angle windows ────────────────────────────────────────────────


def compute_equal_zenith_windows(lat=LAT, doy=DOY,
                                  lst_range=(11.5, 17.0), n_windows=3):
    """Compute *n_windows* equal-zenith-angle bands spanning *lst_range*.

    The band containing the minimum zenith angle (solar noon) is always
    the first band.  Returns (windows, labels) where *windows* is a list
    of (lst_lo, lst_hi) tuples and *labels* a matching list of strings.
    """
    lst = np.linspace(lst_range[0], lst_range[1], 5000)
    zen = zenith_angle(lst, lat, doy)
    zen_min, zen_max = float(zen.min()), float(zen.max())
    bw = (zen_max - zen_min) / n_windows
    zen_edges = [zen_min + i * bw for i in range(n_windows + 1)]

    windows, labels = [], []
    for i in range(n_windows):
        mask = (zen >= zen_edges[i]) & (zen <= zen_edges[i + 1])
        lst_in = lst[mask]
        if len(lst_in) == 0:
            continue
        lo, hi = float(lst_in.min()), float(lst_in.max())
        windows.append((lo, hi))
        def _hm(h):
            return f'{int(h):02d}:{int(round((h % 1) * 60)):02d}'
        labels.append(
            f'{_hm(lo)}\u2013{_hm(hi)}  '
            f'(\u03b8_z {zen_edges[i]:.0f}\u2013{zen_edges[i+1]:.0f}\u00b0)'
        )
    return windows, labels


WINDOWS, WINDOW_LABELS = compute_equal_zenith_windows()
N_WIN = len(WINDOWS)

# XL_GRID imported from cass_analysis


# ── Solar geometry helpers ─────────────────────────────────────────────────────

def solar_angles(lst_h, lat=LAT, doy=DOY):
    """Return (elevation_deg, azimuth_from_N_deg_CW) for an LST hour.

    Uses simple formulas (declination, hour angle); no equation-of-time
    correction (< 2 min error for July 24, negligible for our purposes).
    """
    decl  = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + doy))))
    lat_r = np.radians(lat)
    ha    = np.radians(15.0 * (lst_h - 12.0))          # hour angle (negative AM)

    sin_el = (np.sin(lat_r) * np.sin(decl)
              + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    el = float(np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0))))

    el_r = np.radians(el)
    cos_az = ((np.sin(decl) - np.sin(el_r) * np.sin(lat_r))
              / (np.cos(el_r) * np.cos(lat_r) + 1e-12))
    az = float(np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0))))
    if lst_h > 12.0:          # afternoon: azimuth > 180° (W side)
        az = 360.0 - az
    return el, az             # elevation [°], azimuth from N clockwise [°]


# ── Time-matching helpers ──────────────────────────────────────────────────────

# dump_t_to_lst imported from cass_analysis


def dump_t_to_xy_tidx(dump_t_ns, dt=DT_XY):
    """float64 ns-epoch → index into the xy snapshot time axis."""
    lst_h = dump_t_to_lst(dump_t_ns)
    sim_s = (lst_h - LST_OFFSET) * 3600.0
    return int(round(sim_s / dt))


def window_idx(lst_h):
    """Return window index (0–3) for an LST hour, or -1 if outside all windows."""
    for i, (lo, hi) in enumerate(WINDOWS):
        if lo <= lst_h < hi:
            return i
    return -1


# ── Profile extraction ─────────────────────────────────────────────────────────

def extract_chord_profile(line, centroid_idx, L_m, cell_size):
    """Chord-normalise a 1D surface profile and interpolate to XL_GRID.

    Parameters
    ----------
    line : ndarray (n,)   — 1D surface field (Q or LWP) along the chord axis
    centroid_idx : int    — pixel index of the chord centroid
    L_m : float           — chord length [m]
    cell_size : float     — grid spacing [m] (dx or dy)

    Returns
    -------
    ndarray (200,) on XL_GRID; NaN outside the range covered by the grid.
    """
    n = len(line)
    # Roll so the centroid lands at position n//2
    shift  = n // 2 - int(centroid_idx)
    rolled = np.roll(line, shift)

    # Chord-normalised axis after rolling
    x_m    = np.arange(n) * cell_size + cell_size / 2.0
    xL_raw = (x_m - x_m[n // 2]) / L_m   # centroid → xL = 0

    return np.interp(XL_GRID, xL_raw, rolled, left=np.nan, right=np.nan)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir',  required=True,
                   help='Rep run directory (has sw_flux_*.xy.nc, qlqi_path.xy.nc)')
    p.add_argument('--rt',       required=True, choices=['2stream', 'raytracer'],
                   help='Radiation scheme used in this run')
    p.add_argument('--comp-dir', required=True,
                   help='Composite directory (has events_xz.nc, events_yz.nc; output saved here)')
    p.add_argument('--force',    action='store_true',
                   help='Overwrite existing sw_composite.nc')
    args = p.parse_args()

    run_dir  = Path(args.run_dir)
    comp_dir = Path(args.comp_dir)
    out_path = comp_dir / 'sw_composite.nc'

    if out_path.exists() and not args.force:
        print(f'[skip] {out_path} already exists  (--force to overwrite)')
        return

    # ── Load SW files (lazy) ──────────────────────────────────────────────────
    if args.rt == 'raytracer':
        ds_dir = xr.open_dataset(run_dir / 'sw_flux_sfc_dir_rt.xy.nc',
                                 decode_times=False)
        ds_dif = xr.open_dataset(run_dir / 'sw_flux_sfc_dif_rt.xy.nc',
                                 decode_times=False)
        # Q[t, y, x] = dir + dif
        def _load_Q(tidx):
            return (ds_dir['sw_flux_sfc_dir_rt'][tidx].values
                    + ds_dif['sw_flux_sfc_dif_rt'][tidx].values)   # (ny, nx)
    else:
        ds_sw = xr.open_dataset(run_dir / 'sw_flux_dn.xy.nc',
                                decode_times=False)
        # Q[t, zh=0, y, x]
        def _load_Q(tidx):
            return ds_sw['sw_flux_dn'][tidx, 0].values             # (ny, nx)

    ds_lwp = xr.open_dataset(run_dir / 'qlqi_path.xy.nc',
                              decode_times=False)

    # Grid spacing (assume square; x=y)
    if args.rt == 'raytracer':
        x_vals = ds_dir['x'].values
    else:
        x_vals = ds_sw['x'].values
    y_vals = ds_lwp['y'].values
    dx = float(x_vals[1] - x_vals[0])
    dy = float(y_vals[1] - y_vals[0])

    # ── Domain-mean SW↓ per window (time-averaged, not event-averaged) ────────
    # Use one orientation's event times to identify unique snapshots per window.
    # (Both orientations share the same dump times, so this is orientation-agnostic.)
    _sample_events = None
    for _ef in ['events_xz.nc', 'events_yz.nc']:
        _ep = comp_dir / _ef
        if _ep.exists():
            _sample_events = xr.open_dataset(_ep, decode_times=False)
            break

    Q_domain_mean = np.full(N_WIN, np.nan, dtype=np.float32)
    if _sample_events is not None:
        _dt_arr = _sample_events['dump_t'].values
        # Map unique snapshot times to windows
        _snap_to_win = {}   # tidx → set of window indices
        _snap_to_lst = {}
        for dt in _dt_arr:
            lst_h = dump_t_to_lst(dt)
            wi = window_idx(lst_h)
            if wi < 0:
                continue
            tidx = dump_t_to_xy_tidx(dt)
            _snap_to_win.setdefault(tidx, set()).add(wi)
            _snap_to_lst[tidx] = lst_h

        # Load Q for unique snapshots and compute domain mean
        _Q_dm_accum = {wi: [] for wi in range(N_WIN)}
        for tidx, wins in sorted(_snap_to_win.items()):
            Q_2d = _load_Q(tidx)
            dm = float(np.nanmean(Q_2d))
            for wi in wins:
                _Q_dm_accum[wi].append(dm)

        for wi in range(N_WIN):
            if _Q_dm_accum[wi]:
                Q_domain_mean[wi] = float(np.mean(_Q_dm_accum[wi]))
        _sample_events.close()
        print(f'\nDomain-mean SW↓ per window:')
        for wi in range(N_WIN):
            print(f'  window {wi}: {Q_domain_mean[wi]:.1f} W m⁻²  '
                  f'({len(_Q_dm_accum[wi])} snapshots)')

    # ── Process each orientation ──────────────────────────────────────────────
    results = {}

    for orient_label, events_file, fixed_dim, chord_dim, cell_size in [
        ('xz', 'events_xz.nc', 'cy', 'cx', dx),   # orientation='y': fix y=cy, chord along x
        ('yz', 'events_yz.nc', 'cx', 'cy', dy),   # orientation='x': fix x=cx, chord along y
    ]:
        ev_path = comp_dir / events_file
        if not ev_path.exists():
            print(f'  [{orient_label}] no {events_file}, skipping')
            continue

        print(f'\nProcessing {orient_label} events  ({ev_path})')
        ds_ev = xr.open_dataset(ev_path, decode_times=False)

        dump_t_arr = ds_ev['dump_t'].values     # (n_events,) float64 ns-epoch
        cx_arr     = ds_ev['cx'].values.astype(int)
        cy_arr     = ds_ev['cy'].values.astype(int)
        L_m_arr    = ds_ev['L_m'].values

        n_ev = len(dump_t_arr)

        # Pre-cache unique Q snapshots (only ~5 unique dump times per rep)
        unique_tidx = sorted(set(dump_t_to_xy_tidx(dt) for dt in dump_t_arr))
        print(f'  {n_ev} events, {len(unique_tidx)} unique SW snapshots')
        Q_cache   = {ti: _load_Q(ti)   for ti in unique_tidx}
        LWP_cache = {ti: ds_lwp['qlqi_path'][ti].values for ti in unique_tidx}

        # Accumulators: sum of profiles and event counts per window
        Q_sum   = np.zeros((N_WIN, len(XL_GRID)))
        LWP_sum = np.zeros((N_WIN, len(XL_GRID)))
        n_count = np.zeros(N_WIN, dtype=int)

        for i in range(n_ev):
            lst_h = dump_t_to_lst(dump_t_arr[i])
            wi    = window_idx(lst_h)
            if wi < 0:
                continue

            tidx   = dump_t_to_xy_tidx(dump_t_arr[i])
            Q_2d   = Q_cache[tidx]        # (ny, nx)
            LWP_2d = LWP_cache[tidx]      # (ny, nx) kg m-2

            cx   = cx_arr[i]
            cy   = cy_arr[i]
            L_m  = L_m_arr[i]

            # Extract 1D chord profile and chord-normalise
            if orient_label == 'xz':
                # chord along x, fixed y=cy
                Q_line   = Q_2d[cy, :]
                LWP_line = LWP_2d[cy, :]
                chord_px = cx
            else:
                # chord along y, fixed x=cx
                Q_line   = Q_2d[:, cx]
                LWP_line = LWP_2d[:, cx]
                chord_px = cy

            Q_prof   = extract_chord_profile(Q_line,   chord_px, L_m, cell_size)
            LWP_prof = extract_chord_profile(LWP_line, chord_px, L_m, cell_size)

            # Accumulate (treat NaN as missing)
            valid = ~np.isnan(Q_prof)
            Q_sum[wi, valid]   += Q_prof[valid]
            LWP_sum[wi, valid] += LWP_prof[valid]
            n_count[wi]        += 1

        # Average
        Q_mean   = np.where(n_count[:, None] > 0,
                            Q_sum   / np.maximum(n_count[:, None], 1), np.nan)
        LWP_mean = np.where(n_count[:, None] > 0,
                            LWP_sum / np.maximum(n_count[:, None], 1), np.nan)

        results[orient_label] = dict(
            Q_mean=Q_mean.astype(np.float32),
            LWP_mean=LWP_mean.astype(np.float32),
            n_events=n_count,
        )
        for wi in range(N_WIN):
            print(f'  window {wi} ({WINDOW_LABELS[wi]}): {n_count[wi]} events')

        ds_ev.close()

    if args.rt == 'raytracer':
        ds_dir.close(); ds_dif.close()
    else:
        ds_sw.close()
    ds_lwp.close()

    if not results:
        print('No orientation data produced — nothing saved.')
        return

    # ── Compute solar angles at window midpoints ──────────────────────────────
    win_mid   = [0.5 * (lo + hi) for lo, hi in WINDOWS]
    sol_elev  = np.array([solar_angles(m)[0] for m in win_mid], dtype=np.float32)
    sol_az    = np.array([solar_angles(m)[1] for m in win_mid], dtype=np.float32)
    sol_zen   = 90.0 - sol_elev

    win_lo = np.array([w[0] for w in WINDOWS], dtype=np.float32)
    win_hi = np.array([w[1] for w in WINDOWS], dtype=np.float32)

    # ── Build output dataset ──────────────────────────────────────────────────
    data_vars = {
        'solar_elev_deg': xr.DataArray(
            sol_elev, dims=['window'],
            attrs={'units': 'degrees', 'long_name': 'Solar elevation at window midpoint'}),
        'solar_zen_deg': xr.DataArray(
            sol_zen, dims=['window'],
            attrs={'units': 'degrees', 'long_name': 'Solar zenith angle at window midpoint'}),
        'solar_az_deg': xr.DataArray(
            sol_az, dims=['window'],
            attrs={'units': 'degrees', 'long_name': 'Solar azimuth (from N, CW) at window midpoint'}),
        'window_lst_lo': xr.DataArray(win_lo, dims=['window'],
                                      attrs={'units': 'h', 'long_name': 'Window start LST'}),
        'window_lst_hi': xr.DataArray(win_hi, dims=['window'],
                                      attrs={'units': 'h', 'long_name': 'Window end LST'}),
        'Q_domain_mean': xr.DataArray(
            Q_domain_mean, dims=['window'],
            attrs={'units': 'W m-2',
                   'long_name': 'Domain-mean surface SW↓ (time-averaged per window)'}),
    }

    for orient_label, d in results.items():
        sfx = orient_label   # 'xz' or 'yz'
        data_vars[f'Q_mean_{sfx}'] = xr.DataArray(
            d['Q_mean'], dims=['window', 'xL'],
            attrs={'units': 'W m-2',
                   'long_name': f'Mean surface SW ({sfx}) in chord coords'})
        data_vars[f'LWP_mean_{sfx}'] = xr.DataArray(
            d['LWP_mean'], dims=['window', 'xL'],
            attrs={'units': 'kg m-2',
                   'long_name': f'Mean LWP ({sfx}) in chord coords'})
        data_vars[f'n_events_{sfx}'] = xr.DataArray(
            d['n_events'], dims=['window'],
            attrs={'long_name': f'Event count per window ({sfx})'})

    ds_out = xr.Dataset(
        data_vars,
        coords={
            'xL':     ('xL',     XL_GRID.astype(np.float32),
                        {'long_name': 'chord-normalised horizontal distance x/L'}),
            'window': ('window', np.arange(N_WIN, dtype=int),
                        {'long_name': 'time window index'}),
        },
        attrs={
            'rt':            args.rt,
            'run_dir':       str(run_dir),
            'comp_dir':      str(comp_dir),
            'window_labels': str(WINDOW_LABELS),
            'lat':           LAT,
            'doy':           DOY,
        },
    )

    enc = {v: {'dtype': 'float32', 'zlib': True, 'complevel': 4}
           for v in data_vars if ds_out[v].dtype == np.float32}
    ds_out.to_netcdf(out_path, encoding=enc)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
