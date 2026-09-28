"""Analytic checks of thermo.py. Tolerances are about 3x the errors measured on 2026-09-28."""
import numpy as np
import thermo as th

CASES = [(285., 85000., 1e-3), (278., 75000., 2e-3), (295., 95000., 5e-4), (274., 70000., 1e-5)]


def _state(T, p, ql):
    exn = th.exner(p)
    return (T - th.Lv / th.cp * ql) / exn, th.qsat_liq(p, T) + ql, exn


def test_esat_reference_point():
    assert float(th.esat_liq(th.T0)) == 611.21


def test_esat_against_buck():
    T = np.linspace(263.15, 313.15, 501)
    Tc = T - th.T0
    buck = 611.21 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))
    assert np.abs(th.esat_liq(T) / buck - 1.).max() < 1.5e-3  # measured 4.8e-4


def test_sat_adjust_roundtrip_model_tolerance():
    for T, p, ql in CASES:
        thl, qt, exn = _state(T, p, ql)
        a = th.sat_adjust(thl, qt, p, exn)
        assert abs(float(a["T"]) - T) < 2.e-5      # measured 7.3e-6 (Newton stops at 1e-5 relative step)
        assert abs(float(a["ql"]) - ql) < 1.5e-8   # measured 4.3e-9


def test_sat_adjust_roundtrip_converged():
    for T, p, ql in CASES:
        thl, qt, exn = _state(T, p, ql)
        a = th.sat_adjust(thl, qt, p, exn, tol=1.e-14, nitermax=50)
        assert abs(float(a["T"]) - T) < 1.e-10
        assert abs(float(a["ql"]) - ql) < 1.e-15


def test_sat_adjust_unsaturated_is_identity():
    exn = th.exner(90000.)
    a = th.sat_adjust(300., 5.e-3, 90000., exn)
    assert float(a["T"]) == 300. * exn and float(a["ql"]) == 0.


def test_T_from_mse_roundtrip():
    for T, p, z, ql in [(285., 85000., 1500., 1e-3), (290., 90000., 1000., 0.)]:
        qv = th.qsat_liq(p, T) if ql > 0 else 0.6 * th.qsat_liq(p, T)
        Ti, qli = th.T_from_mse(th.mse(T, z, qv), qv + ql, z, p)
        assert abs(float(Ti) - T) < 1.e-9 and abs(float(qli) - ql) < 1.e-15


def test_theta_v_matches_virtual_temperature():
    T, p, qv, ql = 285., 85000., 8e-3, 1e-3
    exn = th.exner(p)
    thl = (T - th.Lv / th.cp * ql) / exn
    assert abs(float(th.theta_v(thl, qv + ql, ql, 0., exn) * exn - th.virtual_temperature(T, qv, ql))) < 1.e-10


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
