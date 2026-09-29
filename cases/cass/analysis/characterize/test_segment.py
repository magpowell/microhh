"""Exact checks of segment.py on constructed water path fields."""
import numpy as np

import masks as mk
import segment as sg

N = 64


def _blob(cx, cy, r):
    jj, ii = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    dx = np.minimum(abs(ii - cx), N - abs(ii - cx))
    dy = np.minimum(abs(jj - cy), N - abs(jj - cy))
    return np.maximum(1. - (dx**2 + dy**2) / r**2, 0.)


def _square(cx, cy, h):
    m = np.zeros((N, N), dtype=bool)
    m[np.ix_(np.arange(cy - h, cy + h + 1) % N, np.arange(cx - h, cx + h + 1) % N)] = True
    return m


def _split(lwp, core):
    obj, nobj = mk.label_periodic(lwp > 0.)
    seed, nseed = sg.seeds(core)
    return obj, nobj, nseed, sg.split(obj, nobj, seed, nseed, lwp)


def test_two_equal_cores_divide_at_the_midline():
    lwp = np.maximum(_blob(24, 32, 10), _blob(38, 32, 10))          # overlap between x = 28 and x = 34
    core = _square(24, 32, 1) | _square(38, 32, 1)
    obj, nobj, nseed, (sub, nsub, parent, nse) = _split(lwp, core)
    assert nobj == 1 and nseed == 2 and nsub == 2 and nse[0] == 2 and (parent == 1).all()
    a = np.bincount(sub.ravel())[1:]
    assert a.sum() == (lwp > 0).sum() and abs(int(a[0]) - int(a[1])) <= 21      # the 21 columns on x = 31 go to one side
    assert len(set(sub[32, 22:31])) == 1 and len(set(sub[32, 32:41])) == 1 and sub[32, 25] != sub[32, 37]


def test_split_across_the_periodic_boundary():
    lwp = np.maximum(_blob(58, 3, 10), _blob(8, 3, 10))             # one object through x = 0 and y = 0
    core = _square(58, 3, 1) | _square(8, 3, 1)
    obj, nobj, nseed, (sub, nsub, parent, nse) = _split(lwp, core)
    assert nobj == 1 and nsub == 2
    assert sub[3, 58] != sub[3, 8] and sub[3, 58] == sub[N - 3, 58] and sub[3, 8] == sub[N - 3, 8]
    a = np.bincount(sub.ravel())[1:]
    assert a.sum() == (lwp > 0).sum() and abs(int(a[0]) - int(a[1])) <= 21


def test_objects_with_one_or_no_core_stay_whole():
    lwp = np.maximum(_blob(12, 12, 6), _blob(45, 45, 8))
    core = _square(45, 45, 1)
    core[12, 12] = True                                              # one column, below CORE_MIN
    obj, nobj, nseed, (sub, nsub, parent, nse) = _split(lwp, core)
    assert nobj == 2 and nseed == 1 and nsub == 2 and sorted(nse) == [0, 1]
    assert np.array_equal(sub > 0, obj > 0) and sorted(np.bincount(sub.ravel())[1:]) == sorted(np.bincount(obj.ravel())[1:])


def test_closing_bridges_a_one_cell_gap():
    core = np.zeros((N, N), dtype=bool)
    core[20:23, 20:22] = True
    core[20:23, 23:25] = True                                        # gap at x = 22
    assert sg.seeds(core, close=True)[1] == 1 and sg.seeds(core, close=False)[1] == 2
    assert (sg.seeds(core, close=True)[0] > 0).sum() == core.sum()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
