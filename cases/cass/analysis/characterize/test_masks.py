"""Analytic checks of masks.py. Tolerances are about 3x the errors measured on 2026-09-28."""
import numpy as np
from scipy import ndimage
import masks as mk


def _disk(n, R, cx, cy):
    jj, ii = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    dx = np.minimum(abs(ii - cx), n - abs(ii - cx))
    dy = np.minimum(abs(jj - cy), n - abs(jj - cy))
    return dx**2 + dy**2 <= R**2


def test_w_to_full_linear_profile():
    zh = np.arange(257) * 25.
    z = 0.5 * (zh[:-1] + zh[1:])
    w = mk.w_to_full((0.01 * zh[:-1])[:, None, None] * np.ones((1, 4, 4)))
    assert np.abs(w[:-1, 0, 0] - 0.01 * z[:-1]).max() < 1.e-12
    assert w[0, 0, 0] == 0.01 * z[0]            # k = 0 is the mean of the surface and first half level
    assert w[-1, 0, 0] == 0.5 * 0.01 * zh[-2]   # w = 0 at the domain top


def test_wrapped_disk_is_one_object():
    n, dx = 256, 50.
    for R, cx, cy, tolD in [(10.3, 3.2, 250.7, 1.5e-3), (30., 128., 128., 5.e-4)]:  # measured 4.4e-4, 1.0e-4
        m = _disk(n, R, cx, cy)
        lab, k = mk.label_periodic(m)
        assert k == 1 and mk.object_areas(lab, k)[0] == m.sum()
        D = mk.equivalent_diameter(mk.object_areas(lab, k), dx, dx)[0]
        assert abs(D / (2. * R * dx) - 1.) < tolD
        c = mk.periodic_centroids(lab, k, dx, dx)[0] / dx
        err = (c - np.array([cx, cy]) + n / 2) % n - n / 2
        assert np.abs(err).max() < 0.2           # measured 0.06 px
    assert ndimage.label(_disk(n, 10.3, 3.2, 250.7), structure=np.ones((3, 3)))[1] == 4


def test_corner_diagonal_and_separate_objects():
    m = np.zeros((8, 8), bool); m[0, 0] = m[7, 7] = True
    assert mk.label_periodic(m)[1] == 1
    m = np.zeros((8, 8), bool); m[0, 0] = m[4, 4] = True
    assert mk.label_periodic(m)[1] == 2


def test_core_clear_columns_and_cloud_base():
    ql = np.zeros((6, 4, 4)); w = np.zeros((6, 4, 4)); thv = np.full((6, 4, 4), 300.)
    ql[2:5, 1, 1] = 1e-3; w[2:5, 1, 1] = 1.; thv[2:5, 1, 1] = 301.      # core column
    ql[3, 2, 2] = 1e-3; w[3, 2, 2] = 1.; thv[3, 2, 2] = 299.            # cloudy updraft, negatively buoyant
    c = mk.core(ql, w, thv)
    assert c.sum() == 3 and c[2:5, 1, 1].all()
    assert mk.cloudy_updraft(ql, w).sum() == 4
    col = mk.clear_columns(ql)
    assert col.sum() == 14 and not col[1, 1] and not col[2, 2]
    frac = c.mean(axis=(1, 2))
    assert mk.cloud_base_index(frac, 1.e-4) == 2 and mk.cloud_base_index(frac, 0.5) == -1
    m, n = mk.masked_mean(thv, c)
    assert m[2] == 301. and np.isnan(m[0]) and n[2] == 1


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
