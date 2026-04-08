#!/usr/bin/env python3
"""
Compute sliding-window integral time scale T_τ of LWP for one simulation rep.

    T_τ(t) = dt * ∫₀^{first_zero} ρ(τ) dτ    (Taylor 1921, Stull 1988)

where ρ is the normalized ACF and the upper limit is the first zero crossing.
The ACF is estimated over a symmetric window of `--window` steps centred on t.

Saves T_scale(time, y, x) and T_scale_mean(time) to:
  $SCRATCH/CASS_LES/analysis/timescale/{expt}/{rt}/rep_{rep:02d}/T_scale.nc

Usage:
    python compute_T_scale.py --expt no_aerosols_zero_wind --rt raytracer --rep 1
    python compute_T_scale.py --expt base --rt 2stream --rep 2
"""

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

LST_OFFSET = 5.5   # h: simulation t = 0 → 05:30 LST


def acf_fft(X, nlags):
    """Batch FFT-based normalised ACF for all columns of X (window × N).

    Returns rho of shape (nlags+1, N).
    Zero-variance columns (all-cloud or all-clear throughout the window) → NaN.
    """
    Xd   = X - X.mean(axis=0, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * X.shape[0] - 1)))   # next power-of-2
    F    = np.fft.rfft(Xd, n=nfft, axis=0)
    acov = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:X.shape[0]]  # (window, N)
    var  = acov[0:1, :]   # lag-0 = variance
    with np.errstate(invalid='ignore'):
        rho = np.where(var > 0, acov / var, np.nan)
    return rho[:nlags + 1, :]   # (nlags+1, N)


def compute_T_scale(lwp, dt, window, nlags):
    """Sliding-window integral time scale for lwp (T, ny, nx).

    Returns T_scale (T, ny, nx) in seconds; edge steps (first/last window//2)
    remain NaN because the full window is unavailable there.
    """
    T_len, ny, nx = lwp.shape
    N    = ny * nx
    half = window // 2
    X    = lwp.reshape(T_len, N)          # (T, N)

    lag_idx = np.arange(nlags + 1)[:, None]   # (nlags+1, 1)
    T_scale = np.full((T_len, ny, nx), np.nan, dtype=np.float32)

    for t in range(half, T_len - half):
        if t % 100 == 0:
            print(f'  t = {t}/{T_len}', flush=True)

        X_win = X[t - half : t + half]           # (window, N)
        rho   = acf_fft(X_win, nlags)            # (nlags+1, N)

        first_zero    = np.argmax(rho <= 0, axis=0)   # (N,)
        never_crosses = np.all(rho > 0, axis=0)
        first_zero    = np.where(never_crosses, nlags + 1, first_zero)

        mask       = lag_idx < first_zero[None, :]    # (nlags+1, N)
        T_scale[t] = (dt * np.nansum(rho * mask, axis=0)).reshape(ny, nx)

    return T_scale


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--expt',   required=True,
                   help='Experiment name, e.g. no_aerosols_zero_wind or base')
    p.add_argument('--rt',     required=True, choices=['2stream', 'raytracer'])
    p.add_argument('--rep',    required=True, type=int, help='Rep number (1-based)')
    p.add_argument('--window', default=120,   type=int,
                   help='Sliding window width in timesteps (default 120 = 2 h at dt=60 s)')
    p.add_argument('--nlags',  default=30,    type=int,
                   help='Max lag for ACF integration (default 30 = 30 min at dt=60 s)')
    p.add_argument('--dt',     default=60,    type=int,
                   help='Seconds per snapshot (default 60)')
    p.add_argument('--force',  action='store_true',
                   help='Overwrite existing output')
    args = p.parse_args()

    SCRATCH = os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell')
    LES_ROOT = Path(SCRATCH) / 'CASS_LES'

    # base lives at $LES_ROOT/base/; all others under experiments/
    if args.expt == 'base':
        in_path = LES_ROOT / 'base' / args.rt / f'rep_{args.rep:02d}' / 'qlqi_path.xy.nc'
    else:
        in_path = (LES_ROOT / 'experiments' / args.expt
                   / args.rt / f'rep_{args.rep:02d}' / 'qlqi_path.xy.nc')

    out_dir  = (LES_ROOT / 'analysis' / 'timescale'
                / args.expt / args.rt / f'rep_{args.rep:02d}')
    out_path = out_dir / 'T_scale.nc'

    if out_path.exists() and not args.force:
        print(f'[skip] {out_path} already exists  (--force to overwrite)')
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading  {in_path}')
    ds      = xr.open_dataset(in_path, decode_times=False)
    lwp     = ds.qlqi_path.values.astype(np.float32)   # (T, ny, nx)
    t_sim_s = ds.time.values
    x_vals  = ds.x.values
    y_vals  = ds.y.values
    ds.close()

    T_len, ny, nx = lwp.shape
    print(f'Grid: {T_len} × {ny} × {nx}  |  '
          f'window = {args.window} steps = {args.window * args.dt / 3600:.1f} h  |  '
          f'nlags = {args.nlags} steps = {args.nlags * args.dt / 60:.0f} min')

    T_scale      = compute_T_scale(lwp, args.dt, args.window, args.nlags)
    t_lst_h      = t_sim_s / 3600.0 + LST_OFFSET
    T_scale_mean = np.nanmean(T_scale, axis=(1, 2))

    ds_out = xr.Dataset(
        {
            'T_scale': xr.DataArray(
                T_scale, dims=['time', 'y', 'x'],
                attrs={'units': 's',
                       'long_name': 'Integral time scale of LWP (sliding window)'},
            ),
            'T_scale_mean': xr.DataArray(
                T_scale_mean, dims=['time'],
                attrs={'units': 's', 'long_name': 'Domain-mean integral time scale'},
            ),
            't_lst_h': xr.DataArray(
                t_lst_h, dims=['time'],
                attrs={'units': 'h', 'long_name': 'Local standard time'},
            ),
        },
        coords={'time': t_sim_s, 'x': x_vals, 'y': y_vals},
        attrs={
            'expt': args.expt,
            'rt': args.rt,
            'rep': args.rep,
            'window_steps': args.window,
            'window_h': args.window * args.dt / 3600,
            'nlags': args.nlags,
            'dt_s': args.dt,
        },
    )
    # float32 encoding to keep file size manageable
    enc = {v: {'dtype': 'float32', 'zlib': True, 'complevel': 4}
           for v in ['T_scale', 'T_scale_mean', 't_lst_h']}
    ds_out.to_netcdf(out_path, encoding=enc)
    print(f'Saved   {out_path}')


if __name__ == '__main__':
    main()
