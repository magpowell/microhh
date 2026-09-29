"""Exact and calibrated checks of cluster.py. Poisson tolerance: 4 standard errors of the measured spread."""
import numpy as np

import cluster as cl

L = 1000.


def _lattice(m):
    a = L / m
    g = (np.arange(m) + 0.5) * a
    x, y = np.meshgrid(g, g)
    return x.ravel(), y.ravel(), a


def test_nearest_neighbour_across_the_boundary():
    x, y = np.array([5., 995., 500.]), np.array([500., 500., 100.])
    d = cl.nn_distance(x, y, L, L)
    assert np.allclose(d[:2], 10.) and abs(d[2] - np.hypot(495., 400.)) < 1.e-9


def test_lattice_is_regular():
    x, y, a = _lattice(20)
    d = cl.nn_distance(x, y, L, L)
    assert np.allclose(d, a) and abs(cl.iorg(d, L, L) - np.exp(-np.pi)) < 1.e-12


def test_tight_pairs_are_clustered():
    x, y, a = _lattice(20)
    eps = 1.
    x2, y2 = np.r_[x, x + eps], np.r_[y, y]
    d = cl.nn_distance(x2, y2, L, L)
    assert np.allclose(d, eps) and abs(cl.iorg(d, L, L) - np.exp(-800. / L**2 * np.pi * eps**2)) < 1.e-12


def test_random_points_give_one_half():
    rng = np.random.default_rng(3)
    v = np.array([cl.iorg(cl.nn_distance(*cl.random_points(400, L, L, rng), L, L), L, L) for _ in range(200)])
    assert abs(v.mean() - 0.5) < 4. * v.std(ddof=1) / np.sqrt(v.size)       # measured mean 0.4995, sd 0.016


def test_disks_do_not_overlap_and_look_regular():
    rng = np.random.default_rng(4)
    D = np.r_[np.full(10, 120.), np.full(190, 40.)]
    x, y = cl.random_disks(D, L, L, rng)
    dx = np.abs(x[:, None] - x[None, :]); dx = np.minimum(dx, L - dx)
    dy = np.abs(y[:, None] - y[None, :]); dy = np.minimum(dy, L - dy)
    gap = np.hypot(dx, dy) - 0.5 * (D[:, None] + D[None, :]) + 1.e9 * np.eye(D.size)
    assert gap.min() >= 0.
    v = np.mean([cl.iorg(cl.nn_distance(*cl.random_disks(D, L, L, rng), L, L), L, L) for _ in range(30)])
    assert v < 0.5


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
