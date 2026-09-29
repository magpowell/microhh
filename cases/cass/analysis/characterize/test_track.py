"""Analytic checks of track.py on constructed cloud fields. Counts and times are exact by construction."""
import numpy as np

import track as tr

N, DX, DT = 64, 50., 60.
NOCLOUD = -1.e9


def _box(f, j0, j1, i0, i1, base=1000., top=1500.):
    jj, ii = np.arange(j0, j1) % N, np.arange(i0, i1) % N
    f["path"][np.ix_(jj, ii)] = 0.1
    f["base"][np.ix_(jj, ii)] = base
    f["top"][np.ix_(jj, ii)] = top


def _frames(nt):
    return [dict(path=np.zeros((N, N)), base=np.full((N, N), NOCLOUD), top=np.full((N, N), NOCLOUD)) for _ in range(nt)]


def _run(fr):
    a = {k: np.array([f[k] for f in fr]) for k in ("path", "base", "top")}
    core = np.full_like(a["path"], 9.97e36)
    return tr.track(a["path"], a["base"], a["top"], core, np.arange(len(fr)) * DT, DX, DX)


def test_single_cloud_drifting_across_the_boundary():
    fr = _frames(12)
    for f in range(2, 8):                       # present in frames 2..7, moving 2 cells per frame through x = 0
        _box(fr[f], 60, 66, 58 + 2 * f, 64 + 2 * f, top=1000. + 100. * f)
    feats, t = _run(fr)
    assert len(t) == 1 and len(feats) == 6
    r = t.iloc[0]
    assert r.lifetime == 6 * DT and r.birth == "new" and r.death == "gone"
    assert r.area_max == 36 * DX * DX and r.depth_max == 700. and r.age_depth_max == 5 * DT
    assert r.merges_in == 0 and r.splits_out == 0 and r.family_size == 1


def test_merge_keeps_the_larger_cloud():
    fr = _frames(10)
    for f in range(0, 4):
        _box(fr[f], 10, 20, 10, 20)             # large, 100 cells
        if f >= 1:
            _box(fr[f], 12, 16, 22, 26)         # small, 16 cells, born at frame 1, one cell gap
    for f in range(4, 8):
        _box(fr[f], 10, 20, 10, 26)             # merged
    feats, t = _run(fr)
    assert len(t) == 2
    big, small = t.sort_values("area_max", ascending=False).iloc[0], t.sort_values("area_max").iloc[0]
    assert big.birth == "start" and big.death == "gone" and big.lifetime == 8 * DT and big.merges_in == 1
    assert small.birth == "new" and small.death == "merge" and small.lifetime == 3 * DT
    assert big.family == small.family and big.family_lifetime == 8 * DT and big.family_size == 2


def test_split_starts_a_new_track():
    fr = _frames(10)
    for f in range(1, 4):
        _box(fr[f], 10, 20, 10, 30)
    for f in range(4, 9):
        _box(fr[f], 10, 20, 10, 22)             # larger part continues
        if f < 7:
            _box(fr[f], 10, 20, 24, 30)         # smaller part, frames 4..6
    feats, t = _run(fr)
    assert len(t) == 2
    big, small = t.sort_values("lifetime", ascending=False).iloc[0], t.sort_values("lifetime").iloc[0]
    assert big.birth == "new" and big.lifetime == 8 * DT and big.splits_out == 1 and big.death == "gone"
    assert small.birth == "split" and small.death == "gone" and small.lifetime == 3 * DT


def test_record_end_is_flagged():
    fr = _frames(5)
    for f in range(2, 5):
        _box(fr[f], 30, 34, 30, 34)
    _, t = _run(fr)
    assert len(t) == 1 and t.iloc[0].death == "end" and t.iloc[0].lifetime == 3 * DT


def test_separate_clouds_stay_separate():
    fr = _frames(6)
    for f in range(1, 5):
        _box(fr[f], 5, 9, 5, 9)
        _box(fr[f], 40, 46, 40, 46, top=2000.)
    _, t = _run(fr)
    assert len(t) == 2 and t.family.nunique() == 2 and set(t.depth_max) == {500., 1000.}
    assert (t.lifetime == 4 * DT).all() and (t.birth == "new").all() and (t.death == "gone").all()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
