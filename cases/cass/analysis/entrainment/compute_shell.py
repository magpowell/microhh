#!/usr/bin/env python3
"""Heus & Jonker (2008) cloud/shell/subsiding-shell region diagnostics.

For each column at every z level, classify cells into four regions:

    core            : q_l > QL_THRESH  AND  w > 0
    cloud_shell     : q_l > QL_THRESH  AND  w ≤ 0         (in-cloud downdraft)
    subsiding_shell : q_l < QL_THRESH  AND  w < 0
                       AND within N_DIL cells of any cloudy cell
    env             : everything else

Per (expt, rt, rep): compute per-snapshot, per-region, per-z profiles of
area fraction, <w>, M = ρ·A·<w>, and mean(θ_l, q_t, q_l, b).

Output: $SCRATCH/CASS_LES/analysis/shell/{expt}/{rt}/rep_{rep:02d}/shell.nc
"""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr
import netCDF4
from scipy import ndimage

LST_OFFSET = 5.5

QL_THRESH = 1e-5     # kg/kg — cloud definition (H&J use 0.01 g/kg = 1e-5)
N_DIL     = 2        # subsiding-shell: within N_DIL cells of any cloud

_STRUCT = ndimage.generate_binary_structure(3, 1)   # 6-connectivity


def load_3d_snapshot(run_dir, tidx):
    """Load thl, qt, ql, w (interp to cell centres), b at dump tidx."""
    run_dir = Path(run_dir)
    thl_ds = xr.open_dataset(run_dir / 'thl.nc', decode_times=False)
    qt_ds  = xr.open_dataset(run_dir / 'qt.nc',  decode_times=False)
    ql_ds  = xr.open_dataset(run_dir / 'ql.nc',  decode_times=False)
    w_ds   = xr.open_dataset(run_dir / 'w.nc',   decode_times=False)
    b_ds   = xr.open_dataset(run_dir / 'b.nc',   decode_times=False)

    thl = thl_ds['thl'].isel(time=tidx).values.astype(np.float32)
    qt  = qt_ds['qt'].isel(time=tidx).values.astype(np.float32)
    ql  = ql_ds['ql'].isel(time=tidx).values.astype(np.float32)
    w_h = w_ds['w'].isel(time=tidx).values.astype(np.float32)
    b   = b_ds['b'].isel(time=tidx).values.astype(np.float32)

    z = thl_ds['z'].values.astype(np.float32)
    zh = w_ds['zh'].values.astype(np.float32)
    t_sec = float(thl_ds['time'].values[tidx])

    # Interp w from zh to z
    w_cc = np.empty_like(thl, dtype=np.float32)
    for k in range(len(z)):
        if k == 0:
            w_cc[k] = w_h[0]
        elif k + 1 < len(zh):
            w_cc[k] = 0.5 * (w_h[k] + w_h[k + 1])
        else:
            w_cc[k] = w_h[-1]

    thl_ds.close(); qt_ds.close(); ql_ds.close(); w_ds.close(); b_ds.close()
    return dict(thl=thl, qt=qt, ql=ql, w=w_cc, b=b, z=z, t_sec=t_sec)


def load_rho(run_dir, t_sec):
    stats = netCDF4.Dataset(Path(run_dir) / 'cass.default.0000000.nc')
    t_st = np.asarray(stats.variables['time'][:])
    tidx = int(np.argmin(np.abs(t_st - t_sec)))
    rho = np.asarray(stats.groups['thermo'].variables['rho'][tidx]).astype(np.float32)
    stats.close()
    return rho


def partition_regions(ql, w):
    """Return dict of four boolean 3D masks."""
    cloudy   = ql > QL_THRESH
    core     = cloudy & (w > 0)
    c_shell  = cloudy & (w <= 0)
    # Dilate cloudy mask in 3-D; subsiding shell = non-cloudy dry downdraft
    # that is adjacent to any cloud within N_DIL cells.
    dilated  = ndimage.binary_dilation(cloudy, structure=_STRUCT, iterations=N_DIL)
    s_shell  = (~cloudy) & (w < 0) & dilated
    env      = ~(core | c_shell | s_shell)
    return dict(core=core, cloud_shell=c_shell, subsiding_shell=s_shell, env=env)


def per_region_profiles(masks, fields, rho_1d):
    """Per-region, per-z means + mass flux.  fields: dict name→(nz,ny,nx)."""
    nz = fields['w'].shape[0]
    ncell_xy = fields['w'].shape[1] * fields['w'].shape[2]
    out = {}
    for rname, mask in masks.items():
        area_frac = mask.sum(axis=(1, 2)).astype(np.float64) / ncell_xy
        means = {}
        for fname, arr in fields.items():
            m = np.full(nz, np.nan, dtype=np.float32)
            for k in range(nz):
                sel = mask[k]
                if sel.any():
                    m[k] = arr[k][sel].mean()
            means[fname] = m
        M = rho_1d.astype(np.float32) * area_frac.astype(np.float32) * means['w']
        out[rname] = dict(area_frac=area_frac.astype(np.float32), M=M, **means)
    return out


def process_rep(run_dir, out_path, lst_lo=11.0, lst_hi=17.0):
    run_dir = Path(run_dir)
    with xr.open_dataset(run_dir / 'thl.nc', decode_times=False) as ds:
        t_sec_all = ds['time'].values
        z_1d = ds['z'].values.astype(np.float32)
    lst_all = t_sec_all / 3600.0 + LST_OFFSET
    tidx_list = np.where((lst_all >= lst_lo) & (lst_all <= lst_hi))[0]
    print(f'  snapshots in LST [{lst_lo},{lst_hi}]: {len(tidx_list)} '
          f'(LST={lst_all[tidx_list]})', flush=True)

    regions = ['core', 'cloud_shell', 'subsiding_shell', 'env']
    fields_out = ['area_frac', 'w', 'M', 'thl', 'qt', 'ql', 'b']

    nz = len(z_1d)
    n_snap = len(tidx_list)
    data = {r: {f: np.full((n_snap, nz), np.nan, dtype=np.float32)
                for f in fields_out} for r in regions}
    lst_snap = np.empty(n_snap, dtype=np.float32)
    t_sec_snap = np.empty(n_snap, dtype=np.float64)

    for si, tidx in enumerate(tidx_list):
        snap = load_3d_snapshot(run_dir, tidx)
        t_sec = snap['t_sec']
        lst = t_sec / 3600.0 + LST_OFFSET
        lst_snap[si] = lst
        t_sec_snap[si] = t_sec
        print(f'  [{si+1}/{n_snap}] LST={lst:.2f}', flush=True)

        rho_1d = load_rho(run_dir, t_sec)
        masks = partition_regions(snap['ql'], snap['w'])
        prof = per_region_profiles(
            masks,
            dict(w=snap['w'], thl=snap['thl'], qt=snap['qt'], ql=snap['ql'], b=snap['b']),
            rho_1d,
        )
        for r in regions:
            for f in fields_out:
                data[r][f][si] = prof[r][f]
        del snap, masks, prof

    data_vars = {}
    for r in regions:
        for f in fields_out:
            data_vars[f'{r}_{f}'] = (('snap', 'z'), data[r][f])
    data_vars['lst']   = (('snap',), lst_snap)
    data_vars['t_sec'] = (('snap',), t_sec_snap)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'z': z_1d},
        attrs=dict(
            description='Heus & Jonker 2008 cloud/shell/subsiding-shell diagnostics',
            ql_thresh=QL_THRESH, n_dil=N_DIL,
            lst_window=f'{lst_lo}-{lst_hi}',
        ),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--expt', required=True)
    ap.add_argument('--rt',   required=True, choices=['2stream', 'raytracer'])
    ap.add_argument('--rep',  required=True, type=int)
    ap.add_argument('--lst-lo', type=float, default=11.0)
    ap.add_argument('--lst-hi', type=float, default=17.0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    scratch = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
    les_root = scratch / 'CASS_LES'
    if args.expt == 'base':
        run_dir = les_root / 'base' / args.rt / f'rep_{args.rep:02d}'
    else:
        run_dir = les_root / 'experiments' / args.expt / args.rt / f'rep_{args.rep:02d}'
    out_path = (les_root / 'analysis' / 'shell' / args.expt / args.rt
                / f'rep_{args.rep:02d}' / 'shell.nc')

    if out_path.exists() and not args.force:
        print(f'EXISTS: {out_path} — use --force to overwrite')
        return
    print(f'INPUT  : {run_dir}')
    print(f'OUTPUT : {out_path}')
    process_rep(run_dir, out_path, lst_lo=args.lst_lo, lst_hi=args.lst_hi)


if __name__ == '__main__':
    main()
