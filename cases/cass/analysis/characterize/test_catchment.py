"""Exact checks of the convergence-point and reach bookkeeping of catchment.py on constructed force profiles."""
import numpy as np

import catchment as ca

XL = np.linspace(-1., 1., 200)


def test_crossings_interpolate_and_classify():
    x, kind = ca.crossings(XL, -np.sin(np.pi * (XL + 0.2)))     # converging at -0.2, diverging at 0.8
    assert np.allclose(x, [-0.2, 0.8], atol=1.e-4) and kind.tolist() == [-1., 1.]


def test_reach_to_the_edge_and_to_an_outward_turn():
    r = ca.reach(XL, -np.sin(np.pi * (XL + 0.2)))
    assert np.isclose(r["r_c"], -0.2, atol=1.e-4) and np.isclose(r["reach_sun"], 0.8, atol=1.e-4) and np.isclose(r["reach_shadow"], 1.0, atol=1.e-4)
    assert r["edge_sun"] and not r["edge_shadow"]
    r = ca.reach(XL, -XL)                                        # symmetric root, inward everywhere
    assert np.isclose(r["r_c"], 0., atol=1.e-4) and np.isclose(r["reach_sun"], 1.) and np.isclose(r["reach_shadow"], 1.) and r["edge_sun"] and r["edge_shadow"]


def test_reach_picks_the_convergence_nearest_the_root():
    f = -np.sin(2. * np.pi * (XL + 0.1))                         # converging at -0.1 and 0.9 (and -1.1 outside), diverging at 0.4, -0.6
    r = ca.reach(XL, f)
    assert np.isclose(r["r_c"], -0.1, atol=1.e-4) and np.isclose(r["reach_sun"], 0.5, atol=1.e-4) and np.isclose(r["reach_shadow"], 0.5, atol=1.e-4)
    assert np.isnan(ca.reach(XL, np.ones(XL.size))["r_c"])
