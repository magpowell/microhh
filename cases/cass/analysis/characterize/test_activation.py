"""Analytic checks of activation.py."""
import numpy as np

import activation as ac
import parcel as pc

Z = (np.arange(120) + 0.5) * 25.


def test_vectorised_cin_matches_scalar():
    rng = np.random.default_rng(0)
    B = rng.normal(0., 0.01, (200, Z.size)).cumsum(axis=1) * 0.2 + rng.normal(-0.005, 0.01, (200, 1))
    B[:20] = -np.abs(B[:20])                       # parcels with no LFC
    cin, found = ac.cin_lfc_many(Z, B)
    for i in range(B.shape[0]):
        c, _, f = pc.cin_lfc(Z, B[i], 0, Z.size - 1)
        assert abs(cin[i] - c) <= 1.e-13 * max(c, 1.) and bool(found[i]) == f   # measured 6.3e-16 relative


def test_active_fraction_for_uniform_w():
    w = np.linspace(0., 4., 400001)
    cin = np.full(w.size, 0.5)
    f = ac.active_fraction(w, cin, np.ones(w.size, bool))
    assert abs(f - (1. - np.sqrt(2. * 0.5) / 4.)) < 1.e-5      # exact 0.75; grid spacing 1e-5 m/s
    assert ac.active_fraction(w, cin, np.zeros(w.size, bool)) == 0.


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
