#!/usr/bin/env python3
"""Compute and cache H1/H2 updraft diagnostics for one (expt, rt, rep) tuple.

Wrapper for the same pipeline mechanism.ipynb cell 1 runs, but invokable
as a CLI so it can be parallelised across reps via sbatch
(see ``submit_h1h2.sh``).

Output: ``$SCRATCH/CASS_LES/analysis/cache/h1_h2/<expt>/<rt>_<rep>.nc``
with per-mask a_up / w_up / M_up + Romps eps + Siebesma–Cuijpers
eps_sc / delta_sc / C_slab.
"""
import argparse
import os
import sys
from pathlib import Path

import xarray as xr

# Resolve sibling modules (parent of slice_prep is the updrafts dir)
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))                       # diagnostics.py
sys.path.insert(0, str(_HERE.parent))                # cass_analysis.py

from cass_analysis import load_3d_nc, load_stats     # noqa: E402
from diagnostics import (                            # noqa: E402
    build_couvreux_mask, updraft_mass_flux,
    entrainment_rate_tracer, entrainment_rate_siebesma,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expt', required=True)
    ap.add_argument('--rt', required=True, choices=['2stream', 'raytracer'])
    ap.add_argument('--rep', required=True,
                    help="rep_NN or NN")
    ap.add_argument('--tau', type=float, default=900.0)
    ap.add_argument('--force', action='store_true',
                    help="Recompute even if cache exists with all expected fields")
    args = ap.parse_args()

    rep = args.rep if args.rep.startswith('rep_') else f'rep_{int(args.rep):02d}'

    scratch  = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
    rep_dir  = scratch / f'CASS_LES/experiments/{args.expt}/{args.rt}/{rep}'
    cache_d  = scratch / f'CASS_LES/analysis/cache/h1_h2/{args.expt}'
    cache_d.mkdir(parents=True, exist_ok=True)
    cache    = cache_d / f'{args.rt}_{rep}.nc'

    if not (rep_dir / 'couvreux.nc').exists():
        print(f'[{args.rt}/{rep}] missing couvreux.nc — run 3d_to_nc.py first.',
              file=sys.stderr)
        return 2

    if cache.exists() and not args.force:
        ds_chk = xr.open_dataset(cache)
        has_all = all(v in ds_chk.variables
                      for v in ('eps', 'eps_sc', 'delta_sc', 'M_up_cloudy_up'))
        ds_chk.close()
        if has_all:
            print(f'[{args.rt}/{rep}] cache OK: {cache}')
            return 0
        print(f'[{args.rt}/{rep}] cache stale — recomputing.')
        cache.unlink()

    print(f'[{args.rt}/{rep}] computing → {cache}')
    ds_3d = load_3d_nc(rep_dir, variables=['ql', 'w', 'couvreux'])
    stats = load_stats(rep_dir)
    masks = build_couvreux_mask(ds_3d, stats=stats)
    mf    = updraft_mass_flux(ds_3d, masks)
    ent_R = entrainment_rate_tracer(ds_3d, masks, stats, tau=args.tau)
    ent_S = entrainment_rate_siebesma(ds_3d, masks, mf, tau=args.tau)
    out   = xr.merge([mf, ent_R, ent_S])
    out.to_netcdf(cache)
    print(f'[{args.rt}/{rep}] wrote {cache}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
