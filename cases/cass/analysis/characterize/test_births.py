"""Exact checks of births.py on a constructed field: periodic distances, nearest cloud, sun-relative displacement, reference."""
import numpy as np

import births as bi

N, DX = 20, 100.
X = (np.arange(N) + 0.5) * DX


def _field():
    path = np.zeros((N, N), dtype=np.float32)
    path[2:6, 16:20] = 1.       # cloud A: 4 x 4 columns at the right edge (wraps toward x = 0)
    path[12:14, 8:10] = 1.      # cloud B: 2 x 2 = 4 cells, below the 16-cell threshold
    path[15:19, 3:7] = 1.       # cloud C: 4 x 4 columns
    return path


def test_periodic_distance_and_nearest():
    path = _field()
    lab, n, cen, diam = bi.existing(path, DX, DX, min_area=16 * DX * DX)
    assert n == 3 and set(np.unique(lab[lab > 0])) == {lab[3, 18], lab[16, 4]} and lab[12, 8] == 0
    sun = np.array([0., 1.])    # sun to the north: along = displacement in y
    d, near, along, across = bi.geometry(lab, n, cen, path, sun, X, X, DX, DX)
    assert d[3, 18] == 0. and d[3, 1] == 2. * DX and near[3, 1] == lab[3, 18]       # across the periodic edge
    assert d[16, 9] == 3. * DX and near[16, 9] == lab[16, 4]
    assert np.isclose(d[9, 9], np.hypot(3., 6.) * DX)                                # diagonal to C's corner (row 15, col 6)
    assert np.isclose(along[19, 4], X[19] - cen[lab[16, 4] - 1, 1]) and np.isclose(across[16, 9], -(X[9] - cen[lab[16, 4] - 1, 0]))
    assert np.isclose(along[0, 4], (X[0] + N * DX) - cen[lab[16, 4] - 1, 1])        # wraps to the north of C


def test_reference_by_brute_force():
    path = _field()
    lab, n, cen, diam = bi.existing(path, DX, DX, min_area=16 * DX * DX)
    d, near, along, across = bi.geometry(lab, n, cen, path, np.array([0., 1.]), X, X, DX, DX)
    clear = path <= 0.
    ref = bi.reference(d, along, clear, radii=(250., 1000.), sun_radius=1000.)
    assert ref["n_clear"] == N * N - 16 - 4 - 16
    assert np.isclose(ref["ref_within_250"], (clear & (d <= 250.)).sum() / ref["n_clear"])
    close = clear & (d <= 1000.)
    assert np.isclose(ref["ref_sunward"], (close & (along > 0.)).sum() / close.sum())
    assert 0.3 < ref["ref_sunward"] < 0.7
