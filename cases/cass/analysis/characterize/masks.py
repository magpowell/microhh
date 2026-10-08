"""Masks and object labelling on MicroHH fields. Arrays are (z, y, x)."""
import numpy as np
from scipy import ndimage


def w_to_full(w_h):
    """w on half levels zh[0..ktot-1] (zh[0] = surface) to full levels; w = 0 at the domain top."""
    w = np.empty(w_h.shape, dtype=float)
    w[:-1] = 0.5 * (w_h[:-1] + w_h[1:])
    w[-1] = 0.5 * w_h[-1]
    return w


def slab_mean(f):
    return f.mean(axis=(-2, -1), keepdims=True)


def core(ql, w, thv, ql_thr=0.):
    """Cloudy, buoyant and rising (Siebesma et al. 2003, as in Drueke et al. 2020). The model's qlcore mask has no w condition."""
    return (ql > ql_thr) & (w > 0.) & (thv > slab_mean(thv))


def cloudy_updraft(ql, w, ql_thr=0.):
    return (ql > ql_thr) & (w > 0.)


def clear_columns(ql, ql_thr=0.):
    return ql.max(axis=0) <= ql_thr


def cloud_base_index(frac, thr=1.e-4):
    """Lowest level index where frac > thr, or -1."""
    k = np.flatnonzero(frac > thr)
    return int(k[0]) if k.size else -1


def masked_mean(f, m):
    """Level means of f over mask m (z, y, x); NaN where the mask is empty."""
    n = m.sum(axis=(-2, -1))
    s = np.where(m, f, 0.).sum(axis=(-2, -1))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def column_mean(f, col):
    """Level means of f (z, y, x) over the 2D column mask col."""
    return f[:, col].mean(axis=1)


def label_periodic(mask):
    """8-connected labels of a 2D mask on a doubly periodic domain. Returns labels (0 = background), n."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for first, last in ((lab[:, 0], lab[:, -1]), (lab[0, :], lab[-1, :])):
        for s in (-1, 0, 1):
            a, b = first, np.roll(last, s)
            for u, v in set(zip(a[(a > 0) & (b > 0)].tolist(), b[(a > 0) & (b > 0)].tolist())):
                union(u, v)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq = np.unique(roots[1:])
    remap = np.zeros(n + 1, dtype=int)
    remap[uniq] = np.arange(1, uniq.size + 1)
    return remap[roots][lab], int(uniq.size)


def object_areas(lab, n):
    return np.bincount(lab.ravel(), minlength=n + 1)[1:]


def equivalent_diameter(area_px, dx, dy):
    return 2. * np.sqrt(area_px * dx * dy / np.pi)


def periodic_centroids(lab, n, dx, dy):
    """Centroids [m] of labelled objects using circular means."""
    ny, nx = lab.shape
    j, i = np.nonzero(lab)
    l = lab[j, i]
    out = np.empty((n, 2))
    for ax, (idx, m, d) in enumerate(((i, nx, dx), (j, ny, dy))):
        ang = 2. * np.pi * (idx + 0.5) / m
        c = np.bincount(l, weights=np.cos(ang), minlength=n + 1)[1:]
        s = np.bincount(l, weights=np.sin(ang), minlength=n + 1)[1:]
        out[:, ax] = (np.arctan2(s, c) % (2. * np.pi)) / (2. * np.pi) * m * d
    return out
