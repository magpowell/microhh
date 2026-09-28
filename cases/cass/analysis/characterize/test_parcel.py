"""Analytic checks of parcel.py. Tolerances are about 3x the errors measured on 2026-09-28."""
import numpy as np
import parcel as pc

Z = (np.arange(200) + 0.5) * 25.


def _plume(eps, a=0.012, b=-2.e-6, x0=0.016, z0=Z[40]):
    """Exact solution of dx/dz = -eps (x - env), env = a + b z."""
    env = a + b * Z
    c = x0 - (a + b * z0 - b / eps)
    return env, a + b * Z - b / eps + c * np.exp(-eps * (Z - z0))


def test_bulk_entrainment_recovers_known_eps():
    for eps in (5.e-4, 1.e-3, 2.e-3):
        env, x = _plume(eps)
        loc, mean, med, integ = pc.bulk_entrainment(Z, x, env, 44, 100)
        assert abs(mean / eps - 1.) < 6.e-4     # measured 2.1e-4 at eps = 2e-3 (centred difference, dz = 25 m)
        assert abs(integ / eps - 1.) < 4.e-4    # measured 1.2e-4 at eps = 2e-3 (trapezoid)


def test_relax_matches_exact_solution():
    eps = 1.e-3
    env, x = _plume(eps)
    got = pc.relax(Z, env, x[40], 40, 120, eps)
    assert np.abs(got[40:121] - x[40:121]).max() < 3.e-7   # second-order scheme; measured 9.0e-8 on x ~ 0.016


def test_cin_of_a_uniform_negative_layer():
    B = np.where(Z < Z[40] + 300., -0.01, 0.02)
    cin, lfc, found = pc.cin_lfc(Z, B, 40, 150)
    k1 = 40 + np.flatnonzero(B[40:] > 0)[0]
    exact = 0.01 * (Z[k1 - 1] - Z[40]) + 0.5 * 0.01 * 25.   # last trapezoid runs from -0.01 to 0 after clipping
    assert found and lfc == Z[k1] and abs(cin - exact) < 1.e-12
    assert abs(np.sqrt(2. * cin) - np.sqrt(2. * exact)) < 1.e-12


def test_cin_zero_when_buoyant_at_base_and_flag_when_no_lfc():
    assert pc.cin_lfc(Z, np.full(Z.size, 0.01), 40, 150) == (0., float(Z[40]), True)
    cin, lfc, found = pc.cin_lfc(Z, np.full(Z.size, -0.01), 40, 150)
    assert not found and abs(cin - 0.01 * (Z[150] - Z[40])) < 1.e-12


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
