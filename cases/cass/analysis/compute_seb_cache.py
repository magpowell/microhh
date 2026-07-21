#!/usr/bin/env python
"""Standalone driver to (re)compute the cloud-conditioned SEB pickle cache
used by base_comparison.ipynb. Mirrors the logic of `_load_or_compute_conditioned`
in the notebook, so the resulting pickle files are drop-in for the cell.

Usage
-----
    python compute_seb_cache.py [--expt EXPT_KEY] [--n-reps N] [--rt {2stream,raytracer,both}] [--force]

Outputs
-------
    $SCRATCH/CASS_LES/analysis/seb_cache/experiments_<EXPT>__{2stream,raytracer}.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cass_analysis import conditioned_means_ensemble  # noqa: E402
from catalog import make_runset, CASS_ROOT            # noqa: E402

VARS = {
    '2stream':   ['thl_fluxbot', 'qt_fluxbot', 'sw_flux_dn',
                  'lw_flux_dn', 'lw_flux_up'],
    'raytracer': ['thl_fluxbot', 'qt_fluxbot', 'sw_flux_sfc_rt', 'sw_flux_dn',
                  'lw_flux_dn', 'lw_flux_up'],
}

CACHE_DIR = CASS_ROOT / 'analysis' / 'seb_cache'


def _materialise(result):
    """Force dask compute on every DataArray so the pickle is self-contained."""
    mean_d, std_d = result
    for d in (mean_d, std_d):
        for kind in d:
            d[kind] = {var: da.compute() for var, da in d[kind].items()}
    return mean_d, std_d


def _is_cache_fresh(cache_file: Path, needed_vars: set) -> bool:
    if not cache_file.exists():
        return False
    try:
        with open(cache_file, 'rb') as fh:
            cached = pickle.load(fh)
    except Exception as e:
        print(f'  [cache unreadable: {e!r}] {cache_file.name}')
        return False
    if 'domain' not in cached[0]:
        return False
    cached_vars = set(cached[0]['domain'].keys())
    return needed_vars.issubset(cached_vars)


def compute_one(expt: str, rt: str, n_reps: int, force: bool) -> Path:
    rs = make_runset(expt, n_reps=n_reps)
    rep_dirs = rs.dirs[rt]
    needed = set(VARS[rt])
    cache_file = CACHE_DIR / f'experiments_{expt}__{rt}.pkl'

    if not force and _is_cache_fresh(cache_file, needed):
        print(f'[skip] cache fresh: {cache_file}')
        return cache_file

    print(f'[compute] {expt} / {rt}  (n_reps={len(rep_dirs)},  vars={sorted(needed)})')
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = _materialise(conditioned_means_ensemble(rep_dirs, variables=VARS[rt]))
    dt = time.time() - t0
    print(f'  compute time: {dt/60:.1f} min')

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as fh:
        pickle.dump(result, fh)
    print(f'  wrote: {cache_file}')
    return cache_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expt', default='no_aerosols_zero_wind',
                    help='experiment key (default: no_aerosols_zero_wind)')
    ap.add_argument('--n-reps', type=int, default=4)
    ap.add_argument('--rt', choices=['2stream', 'raytracer', 'both'], default='both')
    ap.add_argument('--force', action='store_true',
                    help='ignore existing cache and recompute')
    args = ap.parse_args()

    rts = ['2stream', 'raytracer'] if args.rt == 'both' else [args.rt]

    print(f'CASS_ROOT  = {CASS_ROOT}')
    print(f'CACHE_DIR  = {CACHE_DIR}')
    print(f'expt       = {args.expt}')
    print(f'rts        = {rts}')
    print(f'n_reps     = {args.n_reps}')
    print(f'force      = {args.force}')
    print()

    for rt in rts:
        compute_one(args.expt, rt, args.n_reps, args.force)
        print()

    print('OK.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
