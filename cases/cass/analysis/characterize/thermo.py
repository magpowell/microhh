"""MicroHH moist thermodynamics, ported from include/constants.h and include/thermo_moist_functions.h."""
import numpy as np

Rd, Rv, cp, Lv, Lf = 287.04, 461.5, 1005., 2.501e6, 3.337e5
grav, p0, T0 = 9.81, 1.e5, 273.15
Ls = Lv + Lf
ep = Rd / Rv
RvRd = Rv / Rd

_C = (611.21, 44.393067270, 1.4279398448, 2.6415206946e-2, 3.0291749160e-4, 2.1159987257e-6,
      7.5015702516e-9, -1.5604873363e-12, -9.9726710231e-14, -4.8165754883e-17, 1.3839187032e-18)


def esat_liq(T):
    x = np.clip(np.asarray(T, dtype=float) - T0, -75., 50.)
    e = np.full_like(x, _C[10])
    for c in _C[9::-1]:
        e = c + x * e
    return e


def esat_ice(T):
    x = np.clip(np.asarray(T, dtype=float) - T0, -100., 50.)
    return 611.15 * np.exp(22.452 * x / (272.55 + x))


def _q_from_e(p, e):
    return ep * e / (p - (1. - ep) * e)


def qsat_liq(p, T):
    return _q_from_e(p, esat_liq(T))


def qsat_ice(p, T):
    return _q_from_e(p, esat_ice(T))


def water_fraction(T):
    return np.clip((np.asarray(T, dtype=float) - 233.15) / (T0 - 233.15), 0., 1.)


def qsat(p, T):
    a = water_fraction(T)
    return a * qsat_liq(p, T) + (1. - a) * qsat_ice(p, T)


def _dqsatdT(p, T, esat, L):
    e = esat(T)
    den = p - e * (1. - ep)
    return (ep / den + (1. - ep) * ep * e / den**2) * L * e / (Rv * T**2)


def dqsatdT_liq(p, T):
    return _dqsatdT(p, T, esat_liq, Lv)


def dqsatdT_ice(p, T):
    return _dqsatdT(p, T, esat_ice, Ls)


def exner(p):
    return (np.asarray(p, dtype=float) / p0) ** (Rd / cp)


def sat_adjust(thl, qt, p, exn, tol=1.e-5, nitermax=10):
    """Vectorised port of sat_adjust. Returns dict(T, ql, qi, qs, n_cold)."""
    shape = np.broadcast(thl, qt, p, exn).shape
    thl, qt, p, exn = (np.atleast_1d(np.array(a, dtype=float)) for a in np.broadcast_arrays(thl, qt, p, exn))
    tl = thl * exn
    T = tl.copy()
    qs = qsat_liq(p, tl)
    ql = np.zeros_like(tl)
    qi = np.zeros_like(tl)
    sat = qt > qs
    warm = sat & (tl >= T0)
    cold = sat & (tl < T0)

    def newton(sel, is_cold):
        tlw, qtw, pw = tl[sel], qt[sel], p[sel]
        tnr = tlw.copy()
        old = np.full_like(tnr, 1.e9)
        for _ in range(nitermax):
            act = np.abs(tnr - old) / old > tol
            if not act.any():
                break
            old = np.where(act, tnr, old)
            if is_cold:
                aw = water_fraction(tnr)
                ai = 1. - aw
                dadT = np.where((aw > 0.) & (aw < 1.), 0.025, 0.)
                q = qsat(pw, tnr)
                f = tnr - tlw - aw * Lv / cp * qtw - ai * Ls / cp * qtw + aw * Lv / cp * q + ai * Ls / cp * q
                fp = (1. - dadT * Lv / cp * qtw + dadT * Ls / cp * qtw + dadT * Lv / cp * q - dadT * Ls / cp * q
                      + aw * Lv / cp * dqsatdT_liq(pw, tnr) + ai * Ls / cp * dqsatdT_ice(pw, tnr))
            else:
                q = qsat_liq(pw, tnr)
                f = tnr - tlw - Lv / cp * (qtw - q)
                fp = 1. + Lv / cp * dqsatdT_liq(pw, tnr)
            tnr = np.where(act, tnr - f / fp, tnr)
        return tnr

    if warm.any():
        tw = newton(warm, False)
        q = qsat_liq(p[warm], tw)
        T[warm], qs[warm], ql[warm] = tw, q, np.maximum(0., qt[warm] - q)
    if cold.any():
        tc = newton(cold, True)
        q = qsat(p[cold], tc)
        aw = water_fraction(tc)
        c = np.maximum(0., qt[cold] - q)
        T[cold], qs[cold], ql[cold], qi[cold] = tc, q, aw * c, (1. - aw) * c
    return dict(T=T.reshape(shape), ql=ql.reshape(shape), qi=qi.reshape(shape), qs=qs.reshape(shape),
                n_cold=int(cold.sum()))


def theta_v(thl, qt, ql, qi, exn):
    th = thl + Lv * ql / (cp * exn) + Ls * qi / (cp * exn)
    return th * (1. - (1. - RvRd) * qt - RvRd * (ql + qi))


def buoyancy(thv, thvref):
    return grav * (thv - thvref) / thvref


def virtual_temperature(T, qv, qc):
    return T * (1. + (RvRd - 1.) * qv - qc)


def mse(T, z, qv):
    return cp * T + grav * z + Lv * qv


def mse_sat(T, z, p):
    return cp * T + grav * z + Lv * qsat_liq(p, T)


def T_from_mse(h, qt, z, p, tol=1.e-10, nitermax=50):
    """Invert h = cp T + g z + Lv qv (liquid only). Returns T, ql."""
    shape = np.broadcast(h, qt, z, p).shape
    h, qt, z, p = (np.atleast_1d(np.array(a, dtype=float)) for a in np.broadcast_arrays(h, qt, z, p))
    T = (h - grav * z - Lv * qt) / cp
    sat = qt > qsat_liq(p, T)
    if sat.any():
        hs, zs, ps = h[sat], z[sat], p[sat]
        t = T[sat].copy()
        for _ in range(nitermax):
            f = cp * t + grav * zs + Lv * qsat_liq(ps, t) - hs
            dt = f / (cp + Lv * dqsatdT_liq(ps, t))
            t = t - dt
            if np.max(np.abs(dt)) < tol:
                break
        T[sat] = t
    ql = np.where(sat, qt - qsat_liq(p, T), 0.)
    return T.reshape(shape), np.maximum(ql, 0.).reshape(shape)
