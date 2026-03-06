"""
Check every .pickle file in the LS2D_ERA5/cass tree, print its CDS request state,
and delete those that are 'rejected' (or 'failed') so the next run re-submits them.

Safe to run while the screen session is active — it only deletes rejected pickles.
"""

import glob
import os
import sys
import dill as pickle
import requests

ERA5_DIR = '/pscratch/sd/m/mpowell/LS2D_ERA5/cass'

pkl_files = sorted(glob.glob(f'{ERA5_DIR}/**/*.pickle', recursive=True))
print(f'Found {len(pkl_files)} pickle files\n')

n_ok      = 0
n_deleted = 0
n_error   = 0

for pf in pkl_files:
    try:
        with open(pf, 'rb') as f:
            req = pickle.load(f)
        req.update()
        state = req.reply.get('state', 'unknown')
    except requests.exceptions.HTTPError as e:
        # Request no longer exists on CDS — treat as rejected
        print(f'  HTTP error (stale/expired) — DELETING: {pf}')
        os.remove(pf)
        n_deleted += 1
        continue
    except Exception as e:
        print(f'  ERROR loading {os.path.basename(pf)}: {e} — leaving in place')
        n_error += 1
        continue

    if state in ('queued', 'running'):
        print(f'  {state:10s}: {os.path.basename(pf)}')
        n_ok += 1
    elif state == 'completed':
        # Completed but .nc not downloaded yet — unusual, leave it
        print(f'  {state:10s}: {os.path.basename(pf)}  (nc not yet downloaded?)')
        n_ok += 1
    else:
        # 'rejected', 'failed', or anything else — delete so it gets re-submitted
        print(f'  {state:10s} — DELETING: {os.path.basename(pf)}')
        os.remove(pf)
        n_deleted += 1

print(f'\nDone. Active: {n_ok}  Deleted: {n_deleted}  Errors: {n_error}')
