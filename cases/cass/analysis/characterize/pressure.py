"""Anelastic pressure reconstruction on MicroHH's staggered grid (uniform dx, dy, dz; periodic in x, y; walls in z).

pi is perturbation pressure divided by density, so that Dw/Dt = b - d(pi)/dz. Arrays are (z, y, x); u sits on the
west face, v on the south face and w on the bottom face of a cell; rho is on full levels, rhoh on half levels (nz + 1).
"""
import numpy as np


def _k2(n, d):
    return (2. - 2. * np.cos(2. * np.pi * np.fft.fftfreq(n))) / d**2


def project(Tu, Tv, Tw, rho, rhoh, dx, dy, dz):
    """Pressure pi that makes (Tu, Tv, Tw) - grad(pi) satisfy anelastic continuity with no flow through the walls.

    Tw has nz + 1 half levels; its wall values are ignored. Tu or Tv may be None (zero).
    """
    nz, ny, nx = Tw.shape[0] - 1, Tw.shape[1], Tw.shape[2]
    F = rhoh[:, None, None] * Tw
    F[0] = F[-1] = 0.
    R = (F[1:] - F[:-1]) / dz
    if Tu is not None:
        R += rho[:, None, None] * (np.roll(Tu, -1, axis=2) - Tu) / dx
    if Tv is not None:
        R += rho[:, None, None] * (np.roll(Tv, -1, axis=1) - Tv) / dy
    Rh = np.fft.rfft2(R, axes=(1, 2))
    K2 = _k2(ny, dy)[:, None] + _k2(nx, dx)[None, :nx // 2 + 1]
    a = np.append(0., rhoh[1:nz]) / dz**2            # coefficient of pi[k-1]
    c = np.append(rhoh[1:nz], 0.) / dz**2            # coefficient of pi[k+1]
    K2s = K2.copy()
    K2s[0, 0] = 1.                                   # placeholder; the mean mode is integrated below
    cp = np.empty((nz,) + K2.shape)
    d = Rh
    b = -(a[0] + c[0]) - rho[0] * K2s
    cp[0] = c[0] / b
    d[0] = d[0] / b
    for k in range(1, nz):
        b = -(a[k] + c[k]) - rho[k] * K2s - a[k] * cp[k - 1]
        cp[k] = c[k] / b
        d[k] = (d[k] - a[k] * d[k - 1]) / b
    for k in range(nz - 2, -1, -1):
        d[k] -= cp[k] * d[k + 1]
    pi = np.fft.irfft2(d, s=(ny, nx), axes=(1, 2))
    pi -= pi.mean(axis=(1, 2), keepdims=True)
    Twm = Tw.mean(axis=(1, 2))
    pbar = np.concatenate([[0.], np.cumsum(Twm[1:nz] * dz)])   # d(pibar)/dz = mean Tw on interior half levels
    return pi + pbar[:, None, None]


def grad_z(pi, dz):
    """d(pi)/dz on the nz + 1 half levels, zero at the walls."""
    g = np.zeros((pi.shape[0] + 1,) + pi.shape[1:])
    g[1:-1] = (pi[1:] - pi[:-1]) / dz
    return g


def half_to_full(f):
    return 0.5 * (f[:-1] + f[1:])


def full_to_half(f):
    """Full levels to the nz + 1 half levels; wall values are zero."""
    h = np.zeros((f.shape[0] + 1,) + f.shape[1:])
    h[1:-1] = 0.5 * (f[:-1] + f[1:])
    return h


def grad_h_centre(pi, dx, dy):
    """Horizontal pressure gradient averaged to cell centres."""
    gx = (pi - np.roll(pi, 1, axis=2)) / dx
    gy = (pi - np.roll(pi, 1, axis=1)) / dy
    return 0.5 * (gx + np.roll(gx, -1, axis=2)), 0.5 * (gy + np.roll(gy, -1, axis=1))


def divergence(u, v, wh, rho, rhoh, dx, dy, dz):
    """Anelastic mass divergence; wh has nz + 1 half levels."""
    return (rho[:, None, None] * ((np.roll(u, -1, axis=2) - u) / dx + (np.roll(v, -1, axis=1) - v) / dy)
            + (rhoh[1:, None, None] * wh[1:] - rhoh[:-1, None, None] * wh[:-1]) / dz)


def advection(u, v, w, rho, rhoh, dx, dy, dz):
    """Second-order flux-form advective tendencies (Tu, Tv, Tw); w and Tw have nz + 1 half levels."""
    r, rh = rho[:, None, None], rhoh[:, None, None]
    xm = lambda f: 0.5 * (f + np.roll(f, 1, axis=2))      # to the west face / corner
    ym = lambda f: 0.5 * (f + np.roll(f, 1, axis=1))
    xp = lambda f: 0.5 * (f + np.roll(f, -1, axis=2))     # face to centre
    yp = lambda f: 0.5 * (f + np.roll(f, -1, axis=1))
    ddx = lambda f: (f - np.roll(f, 1, axis=2)) / dx      # centre to west face
    ddy = lambda f: (f - np.roll(f, 1, axis=1)) / dy
    ddxp = lambda f: (np.roll(f, -1, axis=2) - f) / dx    # face to centre
    ddyp = lambda f: (np.roll(f, -1, axis=1) - f) / dy

    uh, vh = full_to_half(u), full_to_half(v)             # u, v on half levels, zero flux at walls
    Tu = -(ddx(xp(u) ** 2) + ddyp(xm(v) * ym(u)))
    fz = rh * xm(w) * uh
    Tu -= (fz[1:] - fz[:-1]) / (r * dz)
    Tv = -(ddxp(ym(u) * xm(v)) + ddy(yp(v) ** 2))
    fz = rh * ym(w) * vh
    Tv -= (fz[1:] - fz[:-1]) / (r * dz)
    Tw = np.zeros_like(w)
    ww = r * half_to_full(w) ** 2
    Tw[1:-1] = -(ddxp(xm(w) * uh)[1:-1] + ddyp(ym(w) * vh)[1:-1] + (ww[1:] - ww[:-1]) / (rh[1:-1] * dz))
    return Tu, Tv, Tw
