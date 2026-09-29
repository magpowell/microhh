"""Analytic checks of pressure.py. Tolerances are about 3x the errors measured on 2026-09-28."""
import numpy as np
import pressure as pr

NZ, NY, NX, D = 24, 20, 28, 50.
RNG = np.random.default_rng(1)
Z = (np.arange(NZ) + 0.5) * D
RHO = 1.15 * np.exp(-Z / 9000.)
RHOH = 1.15 * np.exp(-np.arange(NZ + 1) * D / 9000.)


def _random_T():
    Tw = RNG.normal(size=(NZ + 1, NY, NX)); Tw[0] = Tw[-1] = 0.
    return RNG.normal(size=(NZ, NY, NX)), RNG.normal(size=(NZ, NY, NX)), Tw


def test_projection_removes_divergence():
    Tu, Tv, Tw = _random_T()
    pi = pr.project(Tu, Tv, Tw, RHO, RHOH, D, D, D)
    au = Tu - (pi - np.roll(pi, 1, axis=2)) / D
    av = Tv - (pi - np.roll(pi, 1, axis=1)) / D
    aw = Tw - pr.grad_z(pi, D)
    div = pr.divergence(au, av, aw, RHO, RHOH, D, D, D)
    assert np.abs(div).max() < 1.e-13 * np.abs(pr.divergence(Tu, Tv, Tw, RHO, RHOH, D, D, D)).max() * 30   # measured 2e-15 relative
    assert np.abs(aw.mean(axis=(1, 2))).max() < 1.e-14                       # no mean vertical acceleration


def test_gradient_field_is_recovered():
    p0 = RNG.normal(size=(NZ, NY, NX))
    Tu = (p0 - np.roll(p0, 1, axis=2)) / D
    Tv = (p0 - np.roll(p0, 1, axis=1)) / D
    Tw = pr.grad_z(p0, D)
    pi = pr.project(Tu, Tv, Tw, RHO, RHOH, D, D, D)
    d = (pi - pi.mean()) - (p0 - p0.mean())
    assert np.abs(d).max() < 1.e-10 * np.abs(p0).max()                        # measured 3e-13


def test_horizontally_uniform_buoyancy_gives_no_acceleration():
    b = np.zeros((NZ, NY, NX)); b[5:12] = 0.01
    bh = pr.full_to_half(b)
    pi = pr.project(None, None, bh, RHO, RHOH, D, D, D)
    assert np.abs(bh - pr.grad_z(pi, D))[1:-1].max() < 1.e-15


def test_effective_buoyancy_of_a_sphere():
    n, R = 96, 8.
    z, y, x = np.meshgrid(*(np.arange(n) + 0.5,) * 3, indexing="ij")
    b = ((z - n / 2) ** 2 + (y - n / 2) ** 2 + (x - n / 2) ** 2 <= R**2).astype(float)
    one, oneh = np.ones(n), np.ones(n + 1)
    bh = pr.full_to_half(b)
    pi = pr.project(None, None, bh, one, oneh, 1., 1., 1.)
    beta = pr.half_to_full(bh - pr.grad_z(pi, 1.))
    c = beta[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1].mean()
    assert abs(c - 2. / 3.) < 0.03          # unbounded-domain value 2/3; measured error from images and grid below


def test_advection_of_uniform_flow_vanishes_and_conserves_momentum():
    u = np.full((NZ, NY, NX), 3.); v = np.full((NZ, NY, NX), -2.); w = np.zeros((NZ + 1, NY, NX))
    Tu, Tv, Tw = pr.advection(u, v, w, RHO, RHOH, D, D, D)
    assert max(np.abs(Tu).max(), np.abs(Tv).max(), np.abs(Tw).max()) < 1.e-13
    Tu_, Tv_, Tw_ = _random_T()                      # a divergence-free flow built by projection
    pi = pr.project(Tu_, Tv_, Tw_, RHO, RHOH, D, D, D)
    u = Tu_ - (pi - np.roll(pi, 1, axis=2)) / D
    v = Tv_ - (pi - np.roll(pi, 1, axis=1)) / D
    w = Tw_ - pr.grad_z(pi, D)
    Tu, Tv, Tw = pr.advection(u, v, w, RHO, RHOH, D, D, D)
    mom = (RHO[:, None, None] * Tu).sum() / (RHO[:, None, None] * np.abs(Tu)).sum()
    assert abs(mom) < 1.e-13                         # flux form: domain-integrated u momentum is conserved


if __name__ == "__main__":
    import sys
    if "--measure" in sys.argv:
        n, R = 96, 8.
        z, y, x = np.meshgrid(*(np.arange(n) + 0.5,) * 3, indexing="ij")
        b = ((z - n / 2) ** 2 + (y - n / 2) ** 2 + (x - n / 2) ** 2 <= R**2).astype(float)
        bh = pr.full_to_half(b)
        pi = pr.project(None, None, bh, np.ones(n), np.ones(n + 1), 1., 1., 1.)
        beta = pr.half_to_full(bh - pr.grad_z(pi, 1.))
        print("sphere centre effective buoyancy:", beta[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1].mean())
        Tu, Tv, Tw = _random_T(); pi = pr.project(Tu, Tv, Tw, RHO, RHOH, D, D, D)
        au = Tu - (pi - np.roll(pi, 1, axis=2)) / D; av = Tv - (pi - np.roll(pi, 1, axis=1)) / D; aw = Tw - pr.grad_z(pi, D)
        print("div after / before:", np.abs(pr.divergence(au, av, aw, RHO, RHOH, D, D, D)).max() / np.abs(pr.divergence(Tu, Tv, Tw, RHO, RHOH, D, D, D)).max())
        p0 = RNG.normal(size=(NZ, NY, NX)); pi = pr.project((p0 - np.roll(p0, 1, axis=2)) / D, (p0 - np.roll(p0, 1, axis=1)) / D, pr.grad_z(p0, D), RHO, RHOH, D, D, D)
        print("gradient recovery rel err:", np.abs((pi - pi.mean()) - (p0 - p0.mean())).max() / np.abs(p0).max())
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
