"""Checks of the bulk entrainment and detrainment estimator of entrainment.py on plumes with prescribed rates."""
import numpy as np

import entrainment as en

DZ = 25.
ZH = np.arange(0., 4001., DZ)
Z = 0.5 * (ZH[:-1] + ZH[1:])


def plume(eps, delta, gamma=-2.e-6, phi0=0.012, H=np.inf, w=2., n=4000):
    """Steady top-hat plume: dphi_c/dz = -eps (phi_c - phi_e), dM/dz = (eps - delta) M, integrated on a fine grid."""
    zf = np.linspace(0., ZH[-1], n * 4 + 1)
    e, d = eps(zf), delta(zf)
    phi_e = 0.010 + gamma * zf
    phi_c = np.empty_like(zf); phi_c[0] = phi0
    for i in range(zf.size - 1):                       # midpoint steps
        h = zf[i + 1] - zf[i]
        k1 = -e[i] * (phi_c[i] - phi_e[i])
        mid = phi_c[i] + 0.5 * h * k1
        k2 = -0.5 * (e[i] + e[i + 1]) * (mid - 0.5 * (phi_e[i] + phi_e[i + 1]))
        phi_c[i + 1] = phi_c[i] + h * k2
    lnM = np.concatenate([[0.], np.cumsum(0.5 * ((e - d)[1:] + (e - d)[:-1]) * np.diff(zf))])
    M = 0.05 * np.exp(lnM)
    rho = 1.1 * np.exp(-Z / H) if np.isfinite(H) else np.full(Z.size, 1.1)
    f = lambda x, zz: np.interp(zz, zf, x)
    return dict(M=f(M, ZH), phi_c=f(phi_c, Z), phi_e=f(phi_e, Z), rho=rho, a=f(M, Z) / (rho * w))


def test_recovers_constant_rates():
    p = plume(lambda z: np.full_like(z, 4.e-4), lambda z: np.full_like(z, 2.5e-3))
    eps, delta = en.layer_rates(Z, ZH, p["a"], p["M"], p["phi_c"], p["phi_e"], 1000., 1250.)
    assert abs(eps - 4.e-4) < 2.e-6 and abs(delta - 2.5e-3) < 2.e-6      # measured error 4e-7: interpolation of phi_c to the edges


def test_recovers_the_layer_mean_of_varying_rates():
    e = lambda z: 9.e-4 * np.exp(-z / 1500.)
    d = lambda z: 1.e-3 + 1.e-6 * z
    p = plume(e, d)
    for z1 in (500., 1500., 2500.):
        eps, delta = en.layer_rates(Z, ZH, p["a"], p["M"], p["phi_c"], p["phi_e"], z1, z1 + 250.)
        zz = np.linspace(z1, z1 + 250., 501)
        # eps is weighted by (phi_c - phi_e); over 250 m that differs from the plain mean by under 1 %
        assert abs(eps / e(zz).mean() - 1.) < 0.01
        assert abs((eps - delta) - (e(zz) - d(zz)).mean()) < 1.e-8      # the mass-flux part is exact


def test_a_half_level_shift_of_the_mass_flux_is_detected():
    p = plume(lambda z: np.full_like(z, 4.e-4), lambda z: 1.e-3 + 2.e-6 * z)
    _, good = en.layer_rates(Z, ZH, p["a"], p["M"], p["phi_c"], p["phi_e"], 1000., 1250.)
    shifted = np.interp(ZH + 0.5 * DZ, ZH, p["M"])                       # full-level values labelled as half levels
    _, bad = en.layer_rates(Z, ZH, p["a"], shifted, p["phi_c"], p["phi_e"], 1000., 1250.)
    truth = 1.e-3 + 2.e-6 * 1125.
    assert abs(good - truth) < 2.e-6 and abs(bad - truth) > 2.e-5       # a 12.5 m shift moves delta by 2.5e-5


def test_source_and_tendency_residuals_restore_eps():
    eps0, w, S = 4.e-4, 2., 2.e-7
    p = plume(lambda z: np.full_like(z, eps0), lambda z: np.full_like(z, 2.e-3), w=w)
    k = (Z > 1000.) & (Z < 1250.)
    # a source S in the core changes the steady profile to dphi_c/dz = -eps (phi_c - phi_e) + S / w
    zf = np.linspace(0., ZH[-1], 16001); phi_e = 0.010 - 2.e-6 * zf; pc = np.empty_like(zf); pc[0] = 0.012
    for i in range(zf.size - 1):
        h = zf[i + 1] - zf[i]
        k1 = -eps0 * (pc[i] - phi_e[i]) + S / w
        k2 = -eps0 * (pc[i] + 0.5 * h * k1 - 0.5 * (phi_e[i] + phi_e[i + 1])) + S / w
        pc[i + 1] = pc[i] + h * k2
    phi_c = np.interp(Z, zf, pc)
    raw, _ = en.layer_rates(Z, ZH, p["a"], p["M"], phi_c, p["phi_e"], 1000., 1250.)
    src, tnd, atnd = en.layer_residuals(Z, ZH, p["a"], p["rho"], np.full(ZH.size, w), phi_c, p["phi_e"], np.full(Z.size, S), np.zeros(Z.size), np.zeros(Z.size), 1000., 1250.)
    assert abs(raw - eps0) > 2.e-5 and abs(raw + src - eps0) < 1.e-7 and tnd == 0. and atnd == 0.      # the source moves eps by 3.1e-5


def test_layer_estimator_is_unbiased_under_noise():
    rng = np.random.default_rng(1)
    p = plume(lambda z: np.full_like(z, 2.e-4), lambda z: np.full_like(z, 2.e-3))
    lay = [en.layer_rates(Z, ZH, p["a"], p["M"], p["phi_c"] + rng.normal(0., 2.e-5, Z.size), p["phi_e"], 1000., 1250.)[0] for _ in range(400)]
    # noise of 0.6 % of the core excess per level gives a scatter of 11 % of eps per layer; the mean stays on the truth
    assert abs(np.mean(lay) - 2.e-4) < 3. * np.std(lay) / 20. and 1.5e-5 < np.std(lay) < 3.e-5


def test_bulk_rate_of_two_plumes():
    """Equal detrainment rates: the bulk rate is the mass-flux-weighted mean. A more diluted plume that also detrains
    more: the bulk rate falls below it, because the detrained air is not at the mean core value (Romps 2010)."""
    k = (Z > 1000.) & (Z < 1250.)
    out = []
    for delta_b in (2.e-3, 4.e-3):
        a = plume(lambda z: np.full_like(z, 2.e-4), lambda z: np.full_like(z, 2.e-3))
        b = plume(lambda z: np.full_like(z, 1.2e-3), lambda z: np.full_like(z, delta_b))
        M, aa = a["M"] + b["M"], a["a"] + b["a"]
        phi_c = (a["a"] * a["phi_c"] + b["a"] * b["phi_c"]) / aa
        eps, _ = en.layer_rates(Z, ZH, aa, M, phi_c, a["phi_e"], 1000., 1250.)
        Ma, Mb = np.interp(Z[k], ZH, a["M"]), np.interp(Z[k], ZH, b["M"])
        out.append(eps / ((Ma * 2.e-4 + Mb * 1.2e-3).sum() / (Ma + Mb).sum()))
    assert abs(out[0] - 1.) < 0.01 and out[1] < 0.9


def test_decaying_tracer_recovers_eps():
    eps0, w, tau = 4.e-4, 2.5, 900.
    zf = np.linspace(0., ZH[-1], 16001); Ce = 1.e-6 * np.exp(-zf / 800.); C = np.empty_like(zf); C[0] = 8.e-6
    rhs = lambda c, ce: -eps0 * (c - ce) - c / (tau * w)
    for i in range(zf.size - 1):
        h = zf[i + 1] - zf[i]
        k1 = rhs(C[i], Ce[i])
        C[i + 1] = C[i] + h * rhs(C[i] + 0.5 * h * k1, 0.5 * (Ce[i] + Ce[i + 1]))
    Cc, Cen = np.interp(Z, zf, C), np.interp(Z, zf, Ce)
    p = plume(lambda z: np.full_like(z, eps0), lambda z: np.full_like(z, 2.e-3), w=w)
    raw, _ = en.layer_rates(Z, ZH, p["a"], p["M"], Cc, Cen, 1000., 1250.)
    src, _, _ = en.layer_residuals(Z, ZH, p["a"], p["rho"], np.full(ZH.size, w), Cc, Cen, -Cc / tau, np.zeros(Z.size), np.zeros(Z.size), 1000., 1250.)
    assert raw > 2. * eps0 and abs(raw + src - eps0) < 1.e-6           # without the sink the tracer gives more than twice eps
