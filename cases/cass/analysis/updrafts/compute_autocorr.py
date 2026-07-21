"""Per-rep cached compute of 2D horizontal autocorrelation of the
Couvreux paper updraft mask, for the L(z) coherence-length diagnostic.

For each (z, t) the doubly periodic 2D autocorrelation of the binary
mask is computed via FFT, then radially averaged into n_bins of r.
Result is C(time, z, r), saved as netCDF per (rt, rep). Skips reps with
an existing cache file.

Same maths as cell E in `mechanism.ipynb`, lifted out for batch use.
"""
import sys, time
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.fft import fft2, ifft2, fftshift

CASS_ROOT = Path('/global/homes/m/mpowell/repos/microhh/cases/cass')
sys.path.insert(0, str(CASS_ROOT / 'analysis'))
sys.path.insert(0, str(CASS_ROOT / 'analysis' / 'updrafts'))

from cass_analysis import load_3d_nc, load_stats
from diagnostics  import build_couvreux_mask

EXPT      = 'no_aerosols_zero_wind_v2'
LES_ROOT  = Path('/pscratch/sd/m/mpowell/CASS_LES/experiments') / EXPT
CACHE_DIR = Path('/pscratch/sd/m/mpowell/CASS_LES/analysis/cache/autocorr') / EXPT
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 4
N_BINS = 64


def autocorr_radial(mask: xr.DataArray, n_bins: int = N_BINS) -> xr.Dataset:
    nt = mask.sizes['time']; nz = mask.sizes['z']
    ny = mask.sizes['y'];    nx = mask.sizes['x']
    dx = float(np.median(np.diff(mask.x.values)))
    dy = float(np.median(np.diff(mask.y.values)))

    rx = (np.arange(nx) - nx // 2) * dx
    ry = (np.arange(ny) - ny // 2) * dy
    Rx, Ry = np.meshgrid(rx, ry)
    rgrid = np.hypot(Rx, Ry).ravel()
    r_max = min(nx * dx, ny * dy) / 2
    edges   = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(rgrid, edges) - 1, 0, n_bins - 1)
    counts  = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)

    out = np.full((nt, nz, n_bins), np.nan, dtype=np.float32)
    for it in range(nt):
        t0 = time.time()
        m   = mask.isel(time=it).transpose('z', 'y', 'x').values.astype(np.float32)
        mp  = m - m.mean(axis=(-2, -1), keepdims=True)
        var = mp.var(axis=(-2, -1))
        F   = fft2(mp, axes=(-2, -1))
        R   = np.real(ifft2(F * np.conj(F), axes=(-2, -1))) / (ny * nx)
        Cs  = fftshift(R, axes=(-2, -1)).reshape(nz, -1)
        for iz in range(nz):
            if var[iz] <= 0: continue
            s = np.bincount(bin_idx, weights=Cs[iz], minlength=n_bins)
            out[it, iz] = (s / counts) / var[iz]
        print(f'    t={it+1}/{nt}  {time.time()-t0:.1f}s', flush=True)
    return xr.Dataset(
        {'C': (('time', 'z', 'r'), out)},
        coords={'time': mask.time, 'z': mask.z, 'r': centers},
    )


def per_rep(rt: str, rep_idx: int):
    cache = CACHE_DIR / f'{rt}_rep_{rep_idx:02d}.nc'
    if cache.exists():
        print(f'  CACHE HIT {cache.name}', flush=True)
        return
    rd = LES_ROOT / rt / f'rep_{rep_idx:02d}'
    if not (rd / 'couvreux.nc').exists():
        print(f'  miss: {rd}/couvreux.nc', flush=True)
        return
    t0 = time.time()
    print(f'\n=== {rt}/{rd.name} ===', flush=True)
    print('  loading 3D + building mask ...', flush=True)
    ds_3d = load_3d_nc(rd, variables=['ql', 'w', 'couvreux'])
    stats = load_stats(rd)
    masks = build_couvreux_mask(ds_3d, stats=stats)
    print(f'  mask shape: {dict(masks["mask_paper"].sizes)}', flush=True)
    ds_ac = autocorr_radial(masks['mask_paper'])
    ds_ac.to_netcdf(cache)
    print(f'  wrote {cache}   total {time.time()-t0:.1f}s', flush=True)


def main():
    print(f'EXPT={EXPT}   CACHE_DIR={CACHE_DIR}', flush=True)
    for rt in ('2stream', 'raytracer'):
        for i in range(1, N_REPS + 1):
            per_rep(rt, i)
    print('\nALL DONE', flush=True)


if __name__ == '__main__':
    main()
