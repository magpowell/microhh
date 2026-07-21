"""Per-rep cached compute of Couvreux-mask per-column scalars for v2.

For each (t, x, y) column with at least one (Couvreux mask & ql>QL_TH) cell:
  z_cb       — z[lowest such cell]
  z_ct       — z[highest such cell]
  z_LNB      — first b sign flip + → ≤0 going up from z_cb (linear interp)
  w_cb,w_ct,w_LNB — w_cc evaluated at the corresponding z
  b_surface  — b at the lowest model level
  b_cb       — b at z_cb cell
  b_0p2zi    — b at 0.2*z_cb (subcloud depth proxy), linear interp
  w_0p2zi    — w_cc at 0.2*z_cb, linear interp

Cached as pickle per (rt, rep). Skipped if cache exists.
"""
import sys, time, pickle, argparse
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid

CASS_ROOT = Path('/global/homes/m/mpowell/repos/microhh/cases/cass')
sys.path.insert(0, str(CASS_ROOT / 'analysis'))
sys.path.insert(0, str(CASS_ROOT / 'analysis' / 'updrafts'))

from cass_analysis import load_3d_nc, load_stats
from diagnostics  import _per_time_z_b_z_t

EXPT      = 'no_aerosols_zero_wind_v2'
LES_ROOT  = Path('/pscratch/sd/m/mpowell/CASS_LES/experiments') / EXPT
CACHE_DIR = Path('/pscratch/sd/m/mpowell/CASS_LES/analysis/cache/couvreux_columns') / EXPT
CACHE_DIR.mkdir(parents=True, exist_ok=True)

QL_TH      = 1e-5
LST_OFFSET = 5.5


def per_rep(rt, rep_idx):
    cache = CACHE_DIR / f'{rt}_rep_{rep_idx:02d}.pkl'
    if cache.exists():
        print(f'  CACHE HIT {cache.name}', flush=True)
        return cache
    rd = LES_ROOT / rt / f'rep_{rep_idx:02d}'
    if not (rd / 'couvreux.nc').exists():
        print(f'  miss: {rd}/couvreux.nc', flush=True)
        return None
    t0 = time.time()
    print(f'  loading 3D fields: {rd.name} ({rt}) ...', flush=True)
    ds = load_3d_nc(rd, variables=['b', 'w', 'ql', 'couvreux'])
    stats = load_stats(rd)
    z   = ds.z.values
    t_dt = ds.time.values
    nt  = t_dt.size
    nz, ny, nx = z.size, ds.sizes['y'], ds.sizes['x']

    # Robust LST per timestep
    st_t_sec = stats['t_sec'].values
    st_t_lcl = stats['t_local'].values.astype('datetime64[ns]').astype(np.int64)
    t_3d_lcl = t_dt.astype('datetime64[ns]').astype(np.int64)
    lst = np.empty(nt)
    for it in range(nt):
        ji  = int(np.argmin(np.abs(st_t_lcl - t_3d_lcl[it])))
        lst[it] = float(st_t_sec[ji]) / 3600.0 + LST_OFFSET

    # ─── Materialise full numpy arrays once, build mask in pure numpy ────
    print(f'  materialising arrays ...', flush=True); t1 = time.time()
    C_all  = ds['couvreux'].values
    print(f'    couvreux loaded  {time.time()-t1:.1f}s  shape={C_all.shape}'
          f'  {C_all.nbytes/1e9:.1f} GB', flush=True); t1 = time.time()
    w_all  = (ds['w_cc'].values if 'w_cc' in ds
              else ds['w'].interp(zh=ds['z']).values)
    print(f'    w_cc loaded      {time.time()-t1:.1f}s', flush=True); t1 = time.time()
    ql_all = ds['ql'].values
    print(f'    ql loaded        {time.time()-t1:.1f}s', flush=True); t1 = time.time()
    b_all  = ds['b'].values
    print(f'    b loaded         {time.time()-t1:.1f}s', flush=True); t1 = time.time()

    # mask_paper = (C > thr) & (w > 0) & (below_ref | cloudy)
    # thr(t,z) = mean_xy(C) + max(std_xy(C), sigma_min(t,z))
    C_mean  = C_all.mean(axis=(2, 3))                              # (nt, nz)
    sigma_C = C_all.std (axis=(2, 3))                              # (nt, nz)
    # sigma_min via cumulative ∫σ_C dz, axis = z (axis=1)
    cum     = cumulative_trapezoid(sigma_C, z, axis=1, initial=0.0)
    z_pos   = np.where(z > 0, z, np.nan)
    sigma_min  = 0.05 * cum / z_pos[None, :]
    sigma_eff  = np.where(sigma_C > sigma_min, sigma_C, sigma_min)
    thr        = C_mean + sigma_eff                                # (nt, nz)
    tracer_hit = C_all > thr[:, :, None, None]                     # (nt, nz, ny, nx)
    del C_all

    # z_ref(t) per timestep from stats
    z_b_arr, z_t_arr = _per_time_z_b_z_t(ds, stats)
    z_ref            = z_b_arr + 0.25 * (z_t_arr - z_b_arr)        # (nt,)
    below_ref        = z[None, :] < z_ref[:, None]                 # (nt, nz)
    cloudy           = ql_all > QL_TH                              # (nt, nz, ny, nx)

    mask_all = tracer_hit & (w_all > 0.0) & (below_ref[:, :, None, None] | cloudy)
    del tracer_hit
    print(f'  mask built  {time.time()-t1:.1f}s  '
          f'mean_active_frac={mask_all.mean():.4f}', flush=True)

    print(f'  scanning columns (nt={nt}) ...', flush=True)
    pools = {k: {'values': [], 'lst': []}
             for k in ('z_cb', 'z_LNB', 'z_ct',
                       'w_cb', 'w_LNB', 'w_ct',
                       'b_surface', 'b_cb', 'b_0p2zi',
                       'w_0p2zi')}
    yi, xi = np.indices((ny, nx))

    for it in range(nt):
        m_t  = mask_all[it]
        ql_t = ql_all[it]
        b_t  = b_all[it]
        w_t  = w_all[it]

        active = m_t & (ql_t > QL_TH)                  # (nz, ny, nx) bool
        any_act = active.any(axis=0)                   # (ny, nx)
        if not any_act.any():
            continue
        z_cb_idx = np.argmax(active, axis=0)
        z_ct_idx = (nz - 1) - np.argmax(active[::-1], axis=0)

        bt = b_t
        flip = (bt[:-1] > 0) & (bt[1:] <= 0)
        zarr = np.arange(flip.shape[0])[:, None, None]
        flip = flip & (zarr >= z_cb_idx[None, :, :])
        has_flip = flip.any(axis=0)
        i_lnb = np.where(has_flip, np.argmax(flip, axis=0), 0)

        b_lo = bt[i_lnb, yi, xi]
        b_hi = bt[i_lnb + 1, yi, xi]
        z_lo = z[i_lnb]
        z_hi = z[i_lnb + 1]
        denom = b_hi - b_lo
        z_lnb = np.where(np.abs(denom) > 1e-12,
                         z_lo - b_lo * (z_hi - z_lo) / denom,
                         z_lo)
        wt = w_t
        w_lo = wt[i_lnb, yi, xi]
        w_hi = wt[i_lnb + 1, yi, xi]
        denom_z = z_hi - z_lo
        w_lnb_a = np.where(np.abs(denom_z) > 1e-9,
                           w_lo + (z_lnb - z_lo) / denom_z * (w_hi - w_lo),
                           w_lo)

        w_cb_a = wt[z_cb_idx, yi, xi]
        w_ct_a = wt[z_ct_idx, yi, xi]
        z_cb_a = z[z_cb_idx]
        z_ct_a = z[z_ct_idx]
        b_surface_a = bt[0, yi, xi]
        b_cb_a      = bt[z_cb_idx, yi, xi]

        # b and w at 0.2 * z_cb per column (linear interp between cells)
        z_target = 0.2 * z_cb_a                                           # (ny, nx)
        k_lo     = np.clip(np.searchsorted(z, z_target, side='right') - 1,
                           0, nz - 2)
        frac     = (z_target - z[k_lo]) / (z[k_lo + 1] - z[k_lo])
        b_0p2zi_a = bt[k_lo, yi, xi] + frac * (bt[k_lo + 1, yi, xi] - bt[k_lo, yi, xi])
        w_0p2zi_a = wt[k_lo, yi, xi] + frac * (wt[k_lo + 1, yi, xi] - wt[k_lo, yi, xi])

        sel  = any_act
        sel2 = any_act & has_flip

        lst_t = lst[it]
        n1, n2 = int(sel.sum()), int(sel2.sum())
        for k, arr in [('z_cb', z_cb_a), ('z_ct', z_ct_a),
                       ('w_cb', w_cb_a), ('w_ct', w_ct_a),
                       ('b_surface', b_surface_a), ('b_cb', b_cb_a),
                       ('b_0p2zi', b_0p2zi_a),
                       ('w_0p2zi', w_0p2zi_a)]:
            pools[k]['values'].append(arr[sel])
            pools[k]['lst'   ].append(np.full(n1, lst_t))
        for k, arr in [('z_LNB', z_lnb), ('w_LNB', w_lnb_a)]:
            pools[k]['values'].append(arr[sel2])
            pools[k]['lst'   ].append(np.full(n2, lst_t))
        print(f'    t_idx={it}  LST={lst_t:.2f}  active_cols={n1:6d}  has_LNB={n2:6d}', flush=True)

    out = {k: {sub: np.concatenate(v[sub]) if v[sub] else np.array([])
               for sub in ('values', 'lst')}
           for k, v in pools.items()}
    with open(cache, 'wb') as fh:
        pickle.dump(out, fh)
    print(f'  done  {time.time()-t0:.1f}s   →  {cache}', flush=True)
    return cache


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rt')
    ap.add_argument('--rep', type=int)
    args = ap.parse_args()
    if args.rt and args.rep:
        per_rep(args.rt, args.rep)
    else:
        for rt in ('2stream', 'raytracer'):
            for ri in range(1, 5):
                per_rep(rt, ri)
