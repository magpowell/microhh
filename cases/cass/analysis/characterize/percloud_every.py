"""Per-cloud tables (cloud_w, dilution) every N minutes of the 60 s fields, one field load per frame.

python percloud_every.py --expt no_aerosols_zero_wind_v3 --rt 2stream --rep 1 [--every 5] [--solar 11.9 16.1]
"""
import argparse

import cloud_w
import dilution
from les_io import Run
from snapshot import frames_every, load, run_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    ap.add_argument("--rt", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--every", type=float, default=5.)
    ap.add_argument("--solar", type=float, nargs=2, default=(11.9, 16.1))
    a = ap.parse_args()
    for t in frames_every(Run(run_dir(a.expt, a.rt, a.rep)), a.every, a.solar):
        run = Run(run_dir(a.expt, a.rt, a.rep))
        f, _ = load(run, t)
        c = cloud_w.analyse(a.expt, a.rt, a.rep, t, fields=(f, run))
        d = dilution.analyse(a.expt, a.rt, a.rep, t, fields=(f, run))
        print(f"{a.rt} rep_{a.rep:02d} t={t} clouds={c.sizes['cloud']} fitted={int((d['core_n_levels'].values >= dilution.MIN_LEVELS).sum())}", flush=True)
