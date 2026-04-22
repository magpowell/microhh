#!/usr/bin/env python3
"""Diagnose per-updraft lateral entrainment ε(z) following Gentine et al. (2016).

For one (experiment, rt, rep) triple, identify individual cumulus updrafts at
each 3D dump time in the LST window, compute:

    h  = c_p * (thl * Π) + g * z + L_v * q_t          (warm, q_i = 0)
    ε(z) = -(1 / (h_u - h_env)) * dh_u/dz

where h_u is the updraft-cross-section mean and h_env is the clear-column mean
after excluding a 2-cell subsiding shell around any updraft.

Writes per-sample arrays (one entry per (updraft × valid z-level)) to:
    $SCRATCH/CASS_LES/analysis/entrainment/{expt}/{rt}/rep_{rep:02d}/entrainment.nc
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import netCDF4
from scipy import ndimage

# Physical constants
CP      = 1005.0
G       = 9.81
LV      = 2.5e6
RD_CP   = 0.286          # R_d / c_p (dry air)
P_REF   = 1.0e5          # Pa
EPS_V   = 0.608          # R_v/R_d - 1
LST_OFFSET = 5.5

# Algorithm thresholds (Gentine et al. 2016 defaults)
QL_THRESH         = 1e-5       # kg/kg (defines cloudy levels within a plume)
W_THRESH          = 1.0        # m/s
MIN_CELLS         = 8          # min pixels in updraft object
MIN_LEVELS        = 3          # min vertical extent of an updraft
MIN_CLOUDY_LEVELS = 3          # plume must contain ≥ this many cloudy levels
SHELL_CELLS       = 2          # env-exclusion dilation in 3D cells
DH_THRESH         = 500.0      # J/kg — minimum |h_u - h_env| for ε computation


# ── Field loading ────────────────────────────────────────────────────────────

def load_3d_snapshot(run_dir, tidx):
    """Load thl, qt, ql, w (interpolated to cell centres) at 3D-dump index tidx.

    Returns dict with arrays of shape (nz, ny, nx) plus z (nz,), t_sec scalar.
    """
    run_dir = Path(run_dir)
    thl_ds = xr.open_dataset(run_dir / 'thl.nc', decode_times=False)
    qt_ds  = xr.open_dataset(run_dir / 'qt.nc',  decode_times=False)
    ql_ds  = xr.open_dataset(run_dir / 'ql.nc',  decode_times=False)
    w_ds   = xr.open_dataset(run_dir / 'w.nc',   decode_times=False)

    thl = thl_ds['thl'].isel(time=tidx).values.astype(np.float32)  # (z, y, x)
    qt  = qt_ds['qt'].isel(time=tidx).values.astype(np.float32)
    ql  = ql_ds['ql'].isel(time=tidx).values.astype(np.float32)
    # w is on zh (staggered); interpolate to z by averaging adjacent half-levels.
    w_h = w_ds['w'].isel(time=tidx).values.astype(np.float32)      # (zh, y, x)
    # w has dims (time, zh, y, x) or (time, y, x, zh); normalise
    if 'zh' in w_ds['w'].dims and w_ds['w'].dims.index('zh') != 0:
        # Put zh first
        w_h = np.transpose(
            w_h,
            axes=[list(w_ds['w'].dims).index(d) - 1 for d in ('zh', 'y', 'x')]
        )

    z   = thl_ds['z'].values.astype(np.float32)
    zh  = w_ds['zh'].values.astype(np.float32)
    x   = thl_ds['x'].values.astype(np.float32)
    y   = thl_ds['y'].values.astype(np.float32)
    t_sec = float(thl_ds['time'].values[tidx])

    # Interpolate w from zh to z using linear interpolation
    # w_h shape: (nzh, ny, nx); zh may have nzh == nz or nz+1. MicroHH typically has
    # nzh == nz (half levels between z points). We map w(z_i) = 0.5*(w_h(zh_i) + w_h(zh_{i+1}))
    # If zh has same length as z, shift by +0.5 — interpolate.
    w_cc = np.empty_like(thl, dtype=np.float32)
    for k in range(len(z)):
        # Find nearest lower zh index
        if k == 0:
            w_cc[k] = w_h[0]
        elif k < len(zh) - 1:
            w_cc[k] = 0.5 * (w_h[k] + w_h[k + 1]) if (k + 1) < len(zh) else w_h[k]
        else:
            w_cc[k] = w_h[-1]

    thl_ds.close(); qt_ds.close(); ql_ds.close(); w_ds.close()

    return dict(thl=thl, qt=qt, ql=ql, w=w_cc, z=z, x=x, y=y, t_sec=t_sec)


def load_pressure_rho(run_dir, t_sec):
    """Return p(z), rho(z) at the stats time closest to t_sec (shape (nz,))."""
    stats = netCDF4.Dataset(Path(run_dir) / 'cass.default.0000000.nc')
    t_st = np.asarray(stats.variables['time'][:])
    tidx = int(np.argmin(np.abs(t_st - t_sec)))
    p   = np.asarray(stats.groups['thermo'].variables['phydro'][tidx]).astype(np.float32)
    rho = np.asarray(stats.groups['thermo'].variables['rho'][tidx]).astype(np.float32)
    stats.close()
    return p, rho


def compute_h(thl, qt, z_1d, p_1d):
    """Frozen MSE: h = c_p * (thl * Π) + g*z + L_v*q_t  (J/kg), q_i = 0 here.

    Π(z) = (p(z)/P_ref)^(R_d/c_p).
    Broadcasts (nz,) profiles against (nz, ny, nx) 3D arrays.
    """
    Pi = (p_1d / P_REF) ** RD_CP
    Pi = Pi.astype(np.float32)
    h = (CP * thl * Pi[:, None, None]
         + G * z_1d.astype(np.float32)[:, None, None]
         + LV * qt)
    return h


def compute_thv(thl, qt, ql, p_1d):
    """Return (θ, θ_v) from θ_l, q_t, q_l (warm, q_i = 0).

        θ  = θ_l + (L_v/c_p) q_l / Π
        θ_v = θ (1 + ε_v q_v − q_l),   q_v = q_t − q_l
    """
    Pi = ((p_1d / P_REF) ** RD_CP).astype(np.float32)
    theta = thl + (LV / CP) * ql / Pi[:, None, None]
    qv = qt - ql
    thv = theta * (1.0 + EPS_V * qv - ql)
    return theta, thv


# ── Updraft identification and ε computation ─────────────────────────────────

_STRUCT_3D = ndimage.generate_binary_structure(3, 1)   # 6-connectivity
_STRUCT_SHELL = ndimage.generate_binary_structure(3, 1)


def find_updrafts(ql, w):
    """Label contiguous strong-updraft regions (w > W_THRESH), retain only those
    containing ≥ MIN_CLOUDY_LEVELS cloudy levels.  Plume then extends through
    the entire subcloud + cloud updraft column.  Returns labels, valid_ids.
    """
    mask = w > W_THRESH
    labels, n_obj = ndimage.label(mask, structure=_STRUCT_3D)
    if n_obj == 0:
        return labels, []

    valid = []
    for lbl in range(1, n_obj + 1):
        lmask = labels == lbl
        if lmask.sum() < MIN_CELLS:
            continue
        zk = np.where(lmask.any(axis=(1, 2)))[0]
        if zk.size < MIN_LEVELS:
            continue
        # Must contain enough cloudy levels
        cloudy_per_level = (lmask & (ql > QL_THRESH)).any(axis=(1, 2))
        if cloudy_per_level.sum() < MIN_CLOUDY_LEVELS:
            continue
        valid.append(lbl)
    return labels, valid


def env_mean_fields(fields_3d, labels, ql):
    """Horizontal mean of each 3-D field over clear non-updraft columns per level.

    fields_3d : dict name → (nz, ny, nx) array.
    Returns dict name → (nz,) array of env means.
    """
    updraft_mask = labels > 0
    dilated = ndimage.binary_dilation(
        updraft_mask, structure=_STRUCT_SHELL, iterations=SHELL_CELLS)
    cloudy   = ql > 1e-6
    excluded = dilated | cloudy
    out = {}
    for name, arr in fields_3d.items():
        nz = arr.shape[0]
        env = np.empty(nz, dtype=np.float32)
        for k in range(nz):
            keep = ~excluded[k]
            env[k] = arr[k][keep].mean() if keep.sum() > 0 else np.nan
        out[name] = env
    return out


def per_updraft_profiles(h, theta, thv, qt, w, ql, labels, valid_ids,
                         dx, dy, z_1d, rho_1d):
    """Per-updraft h, θ_v, q_t, A, w, M profiles over the full updraft extent
    (subcloud + cloud).

    Extents recorded per plume:
      ub_idx, ut_idx  – bottom/top of w-updraft column
      cb_idx, ct_idx  – first/last cloudy level within the plume (ql > QL_THRESH)
    """
    nz = h.shape[0]
    out = []
    for lbl in valid_ids:
        mask = labels == lbl
        per_level_count = mask.sum(axis=(1, 2))
        present = per_level_count > 0
        k_present = np.where(present)[0]
        if k_present.size < MIN_LEVELS:
            continue
        cloudy_present = (mask & (ql > QL_THRESH)).any(axis=(1, 2))
        cloudy_k = np.where(cloudy_present)[0]
        if cloudy_k.size < MIN_CLOUDY_LEVELS:
            continue

        # Full-plume means (footprint A, w_u, M): over every w>W_THRESH cell.
        # Core thermodynamic means (h_u, θ, θ_v, q_t, q_l): cloudy cells only,
        # i.e. (plume footprint) ∩ (q_l > QL_THRESH) — this is what Kuang &
        # Bretherton 2006 / Gentine 2016 use for the MSE ε diagnostic.
        h_u     = np.full(nz, np.nan, dtype=np.float32)
        theta_u = np.full(nz, np.nan, dtype=np.float32)
        thv_u   = np.full(nz, np.nan, dtype=np.float32)
        qt_u    = np.full(nz, np.nan, dtype=np.float32)
        ql_u    = np.full(nz, np.nan, dtype=np.float32)
        w_u     = np.full(nz, np.nan, dtype=np.float32)
        A       = np.full(nz, np.nan, dtype=np.float32)
        for k in k_present:
            sel_full = mask[k]                               # full plume footprint
            sel_core = sel_full & (ql[k] > QL_THRESH)        # cloudy-core subset
            w_u[k] = w[k][sel_full].mean()
            A[k]   = per_level_count[k] * dx * dy
            if sel_core.any():
                h_u[k]     = h[k][sel_core].mean()
                theta_u[k] = theta[k][sel_core].mean()
                thv_u[k]   = thv[k][sel_core].mean()
                qt_u[k]    = qt[k][sel_core].mean()
                ql_u[k]    = ql[k][sel_core].mean()
        M = rho_1d.astype(np.float32) * A * w_u

        ub_idx = int(k_present[0])
        ut_idx = int(k_present[-1])
        cb_idx = int(cloudy_k[0])
        ct_idx = int(cloudy_k[-1])
        out.append(dict(
            label=int(lbl),
            h_u=h_u, theta_u=theta_u, thv_u=thv_u,
            qt_u=qt_u, ql_u=ql_u,
            w_u=w_u, A=A, M=M,
            ub_idx=ub_idx, ut_idx=ut_idx,
            cb_idx=cb_idx, ct_idx=ct_idx,
            w_max=float(np.nanmax(w_u)),
            M_cb=float(M[cb_idx]) if np.isfinite(M[cb_idx]) else np.nan,
            M_peak=float(np.nanmax(M)),
        ))
    return out


def entrainment_samples(plumes, env, z_1d):
    """Compute ε, dM/dz, dL/dz per plume per interior z-level.  Returns flat arrays.

    env : dict with h, thl, qt env-mean profiles (each (nz,)).
    L   : 2*sqrt(A/π), effective plume diameter.
    δ   : ε − (1/M) dM/dz (Siebesma & Cuijpers 1995).
    """
    s = {k: [] for k in (
        'z', 'eps', 'M', 'pid', 'L', 'dMdz', 'dLdz',
        'frac_dMdz', 'frac_dLdz', 'delta',
        'hu', 'henv', 'thvu', 'thvenv', 'qtu', 'qtenv',
        'thetau', 'thetaenv', 'qlu',
    )}
    h_env     = env['h']
    thv_env   = env['thv']
    qt_env    = env['qt']
    theta_env = env['theta']
    for pid, p in enumerate(plumes):
        h_u    = p['h_u'];   thv_u   = p['thv_u']; qt_u = p['qt_u']
        theta_u = p['theta_u']; ql_u = p['ql_u']
        A      = p['A'];     M       = p['M']
        L     = 2.0 * np.sqrt(np.where(A > 0, A / np.pi, np.nan))
        # Sample the full updraft interior (subcloud + cloud). ε is computed
        # only where the MSE contrast is large enough; elsewhere it is NaN
        # but M and other fields are still recorded for the histograms.
        ub, ut = p['ub_idx'], p['ut_idx']
        for k in range(ub + 1, ut):
            if not np.isfinite(M[k]) or M[k] <= 0:
                continue
            dz = z_1d[k + 1] - z_1d[k - 1]
            # ε from MSE budget — requires cloudy-core h_u at k±1 and
            # h_env at k, plus enough MSE contrast to avoid noise.
            if (np.isfinite(h_u[k - 1]) and np.isfinite(h_u[k + 1])
                    and np.isfinite(h_env[k])):
                dh = h_u[k] - h_env[k]
                if np.isfinite(dh) and abs(dh) > DH_THRESH:
                    eps = -(h_u[k + 1] - h_u[k - 1]) / dz / dh
                    if not np.isfinite(eps):
                        eps = np.nan
                else:
                    eps = np.nan
            else:
                eps = np.nan

            if np.isfinite(M[k - 1]) and np.isfinite(M[k + 1]) and M[k] > 0:
                dMdz = (M[k + 1] - M[k - 1]) / dz
                frac_dMdz = dMdz / M[k]
            else:
                dMdz = np.nan; frac_dMdz = np.nan
            if np.isfinite(L[k - 1]) and np.isfinite(L[k + 1]) and L[k] > 0:
                dLdz = (L[k + 1] - L[k - 1]) / dz
                frac_dLdz = dLdz / L[k]
            else:
                dLdz = np.nan; frac_dLdz = np.nan
            delta = (eps - frac_dMdz
                     if (np.isfinite(eps) and np.isfinite(frac_dMdz))
                     else np.nan)

            s['z'].append(z_1d[k]);      s['eps'].append(eps)
            s['M'].append(M[k]);         s['pid'].append(pid)
            s['L'].append(L[k])
            s['dMdz'].append(dMdz);      s['dLdz'].append(dLdz)
            s['frac_dMdz'].append(frac_dMdz); s['frac_dLdz'].append(frac_dLdz)
            s['delta'].append(delta)
            s['hu'].append(h_u[k]);         s['henv'].append(h_env[k])
            s['thvu'].append(thv_u[k]);     s['thvenv'].append(thv_env[k])
            s['thetau'].append(theta_u[k]); s['thetaenv'].append(theta_env[k])
            s['qtu'].append(qt_u[k]);       s['qtenv'].append(qt_env[k])
            s['qlu'].append(ql_u[k])
    return {k: np.asarray(v, dtype=np.float32) if k != 'pid'
            else np.asarray(v, dtype=np.int32) for k, v in s.items()}


# ── Main per-rep driver ──────────────────────────────────────────────────────

def process_rep(run_dir, out_path, lst_lo=11.0, lst_hi=17.0):
    run_dir = Path(run_dir)
    # Get 3D dump times from thl.nc
    with xr.open_dataset(run_dir / 'thl.nc', decode_times=False) as ds:
        t_sec_all = ds['time'].values
        x = ds['x'].values
        y = ds['y'].values
        z = ds['z'].values
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    lst_all = t_sec_all / 3600.0 + LST_OFFSET
    tidx_list = np.where((lst_all >= lst_lo) & (lst_all <= lst_hi))[0]
    print(f'  snapshots in LST [{lst_lo},{lst_hi}]: {len(tidx_list)} '
          f'(LST={lst_all[tidx_list]})', flush=True)

    acc_keys = ['z', 'eps', 'M', 'pid', 'L',
                'dMdz', 'dLdz', 'frac_dMdz', 'frac_dLdz', 'delta',
                'hu', 'henv', 'thvu', 'thvenv', 'qtu', 'qtenv',
                'thetau', 'thetaenv', 'qlu']
    acc = {k: [] for k in acc_keys}
    acc['snap'] = []
    all_lst = []
    all_cb_z, all_ct_z, all_ub_z, all_ut_z = [], [], [], []
    all_wmax, all_Mcb, all_Mpeak = [], [], []

    next_plume_id = 0
    for snap_i, tidx in enumerate(tidx_list):
        t_sec = float(t_sec_all[tidx])
        lst = float(lst_all[tidx])
        print(f'  [{snap_i+1}/{len(tidx_list)}] tidx={tidx} t={t_sec:.0f}s LST={lst:.2f}',
              flush=True)

        snap = load_3d_snapshot(run_dir, tidx)
        p_1d, rho_1d = load_pressure_rho(run_dir, t_sec)
        h             = compute_h(snap['thl'], snap['qt'], snap['z'], p_1d)
        theta, thv    = compute_thv(snap['thl'], snap['qt'], snap['ql'], p_1d)

        labels, valid = find_updrafts(snap['ql'], snap['w'])
        print(f'    {len(valid)} updrafts passed size/depth filters', flush=True)
        if not valid:
            del snap, h, theta, thv, labels
            continue

        env = env_mean_fields(
            {'h': h, 'theta': theta, 'thv': thv, 'qt': snap['qt']},
            labels, snap['ql'],
        )
        plumes = per_updraft_profiles(
            h, theta, thv, snap['qt'], snap['w'], snap['ql'],
            labels, valid, dx, dy, snap['z'], rho_1d)

        samp = entrainment_samples(plumes, env, snap['z'])
        n_s = len(samp['z'])
        print(f'    {n_s} ε samples (interior levels)', flush=True)

        samp['pid'] = samp['pid'] + next_plume_id
        for pid_local, p in enumerate(plumes):
            all_lst.append(lst)
            all_cb_z.append(float(snap['z'][p['cb_idx']]))
            all_ct_z.append(float(snap['z'][p['ct_idx']]))
            all_ub_z.append(float(snap['z'][p['ub_idx']]))
            all_ut_z.append(float(snap['z'][p['ut_idx']]))
            all_wmax.append(p['w_max'])
            all_Mcb.append(p['M_cb'])
            all_Mpeak.append(p['M_peak'])

        for k in acc_keys:
            acc[k].append(samp[k])
        acc['snap'].append(np.full(n_s, snap_i, dtype=np.int32))
        next_plume_id += len(plumes)

        del snap, h, theta, thv, labels, plumes

    def _cat(lst, dtype):
        return np.concatenate(lst) if lst else np.array([], dtype=dtype)
    z_arr        = _cat(acc['z'],         np.float32)
    eps_arr      = _cat(acc['eps'],       np.float32)
    M_arr        = _cat(acc['M'],         np.float32)
    pid_arr      = _cat(acc['pid'],       np.int32)
    snap_arr     = _cat(acc['snap'],      np.int32)
    hu_arr       = _cat(acc['hu'],        np.float32)
    henv_arr     = _cat(acc['henv'],      np.float32)
    thvu_arr     = _cat(acc['thvu'],      np.float32)
    thvenv_arr   = _cat(acc['thvenv'],    np.float32)
    qtu_arr      = _cat(acc['qtu'],       np.float32)
    qtenv_arr    = _cat(acc['qtenv'],     np.float32)
    thetau_arr   = _cat(acc['thetau'],    np.float32)
    thetaenv_arr = _cat(acc['thetaenv'],  np.float32)
    qlu_arr      = _cat(acc['qlu'],       np.float32)
    L_arr        = _cat(acc['L'],         np.float32)
    dMdz_arr     = _cat(acc['dMdz'],      np.float32)
    dLdz_arr     = _cat(acc['dLdz'],      np.float32)
    fdMdz_arr    = _cat(acc['frac_dMdz'], np.float32)
    fdLdz_arr    = _cat(acc['frac_dLdz'], np.float32)
    delta_arr    = _cat(acc['delta'],     np.float32)

    lst_arr   = np.asarray(all_lst,   dtype=np.float32)
    cb_z_arr  = np.asarray(all_cb_z,  dtype=np.float32)
    ct_z_arr  = np.asarray(all_ct_z,  dtype=np.float32)
    ub_z_arr  = np.asarray(all_ub_z,  dtype=np.float32)
    ut_z_arr  = np.asarray(all_ut_z,  dtype=np.float32)
    wmax_arr  = np.asarray(all_wmax,  dtype=np.float32)
    Mcb_arr   = np.asarray(all_Mcb,   dtype=np.float32)
    Mpeak_arr = np.asarray(all_Mpeak, dtype=np.float32)

    ds = xr.Dataset(
        data_vars=dict(
            # Per-sample (updraft × interior z-level)
            z           = (('sample',), z_arr),
            epsilon     = (('sample',), eps_arr),
            delta       = (('sample',), delta_arr),          # detrainment (1/m)
            M           = (('sample',), M_arr),
            L           = (('sample',), L_arr),              # plume diameter (m)
            dMdz        = (('sample',), dMdz_arr),           # kg/s/m
            dLdz        = (('sample',), dLdz_arr),           # m/m (dimensionless)
            frac_dMdz   = (('sample',), fdMdz_arr),          # (1/M) dM/dz, 1/m
            frac_dLdz   = (('sample',), fdLdz_arr),          # (1/L) dL/dz, 1/m
            plume_id    = (('sample',), pid_arr),
            snap_id     = (('sample',), snap_arr),
            h_u         = (('sample',), hu_arr),
            h_env       = (('sample',), henv_arr),
            thv_u       = (('sample',), thvu_arr),
            thv_env     = (('sample',), thvenv_arr),
            theta_u     = (('sample',), thetau_arr),
            theta_env   = (('sample',), thetaenv_arr),
            qt_u        = (('sample',), qtu_arr),
            qt_env      = (('sample',), qtenv_arr),
            ql_u        = (('sample',), qlu_arr),
            # Per-plume summary
            plume_lst   = (('plume',), lst_arr),
            plume_cb_z  = (('plume',), cb_z_arr),
            plume_ct_z  = (('plume',), ct_z_arr),
            plume_ub_z  = (('plume',), ub_z_arr),
            plume_ut_z  = (('plume',), ut_z_arr),
            plume_wmax  = (('plume',), wmax_arr),
            plume_Mcb   = (('plume',), Mcb_arr),
            plume_Mpeak = (('plume',), Mpeak_arr),
        ),
        attrs=dict(
            description='Per-updraft lateral entrainment diagnostics (Gentine 2016).',
            lst_window=f'{lst_lo}-{lst_hi}',
            ql_thresh=QL_THRESH, w_thresh=W_THRESH,
            min_cells=MIN_CELLS, min_levels=MIN_LEVELS,
            shell_cells=SHELL_CELLS, dh_thresh=DH_THRESH,
        ),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f'  wrote {out_path} ({len(eps_arr)} samples, {len(lst_arr)} plumes)',
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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
    out_path = (les_root / 'analysis' / 'entrainment' / args.expt / args.rt
                / f'rep_{args.rep:02d}' / 'entrainment.nc')

    if out_path.exists() and not args.force:
        print(f'EXISTS: {out_path} — use --force to overwrite')
        return
    print(f'INPUT  : {run_dir}')
    print(f'OUTPUT : {out_path}')
    process_rep(run_dir, out_path, lst_lo=args.lst_lo, lst_hi=args.lst_hi)


if __name__ == '__main__':
    main()
