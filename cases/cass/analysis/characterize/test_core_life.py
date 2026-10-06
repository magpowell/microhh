"""Exact check of core_life.py on constructed frames: two pulses in one shell, one pulse that outlives a shell split."""
import numpy as np

import core_life as cl
import track as tr

N, DX, T = 20, 100., 8


def _fields():
    path = np.zeros((T, N, N), dtype=np.float32); core = np.full((T, N, N), tr.FILL, dtype=np.float32)
    for f in range(1, 6):                       # shell A, frames 1-5
        path[f, 2:8, 2:8] = 1.
    for f in (1, 2, 3):                         # pulse 1 in A
        core[f, 3:5, 3:5] = 0.1
    core[5, 5:7, 5:7] = 0.1                     # pulse 2 in A, after a frame without core
    for f in range(1, 7):                       # shell B, frames 1-6, split at frame 4 into a small left and a large right piece
        path[f, 12:18, 12:18] = 1.
        if f >= 4:
            path[f, 12:18, 14] = 0.
        core[f, 14:16, 12:14] = 0.1             # one pulse that stays in the small left piece
    return path, core


def _track(path, core):
    time = 60. * np.arange(T)
    zeros = np.zeros_like(path)
    fc, tc = tr.track(path, zeros, zeros, core, time, DX, DX)
    pk = np.where(cl.core_mask(path, core), path, 0.)
    fk, tk = tr.track(pk, zeros, zeros, core, time, DX, DX)
    return fc, tc, fk, tk, time


def test_pulses_and_shells():
    path, core = _fields()
    fc, tc, fk, tk, time = _track(path, core)
    m = cl.build(path, core, fc, fk)
    lst = 12. + time / 3600.
    p = cl.pulses(m, tk, lst, DX, DX).sort_values("frame_first").reset_index(drop=True)
    s = cl.shells(m, tc, lst, DX, DX)
    assert p.frames.tolist() == [3, 6, 1] and p.n_shells.tolist() == [1, 2, 1] and p.shell_changed.tolist() == [False, True, False]
    assert p.untouched.all() and np.allclose(p.shell_D, 2. * np.sqrt(np.array([36., 36., 36.]) * DX * DX / np.pi))
    A = s[s.n_pulses == 2].iloc[0]
    assert A.active == 4 and A.frames == 5 and A.lead == 0 and A.decay == 0 and A.longest == 3
    B = s[(s.n_pulses == 1) & (s.splits_out == 1)].iloc[0]         # the original B keeps the large right piece: core frames 1-3 only
    assert B.active == 3 and B.frames == 6 and B.decay == 3 and B.longest == 3
    Bp = s[(s.n_pulses == 1) & (s.splits_out == 0)].iloc[0]        # the split-off left piece hosts the core at frames 4-6
    assert Bp.active == 3 and Bp.frames == 3 and Bp.lead == 0 and Bp.decay == 0 and Bp.longest == 3
    assert len(s) == 3 and (s.n_pulses > 0).all()
