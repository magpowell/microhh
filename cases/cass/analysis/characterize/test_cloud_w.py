"""Exact checks of cloud_w.py on a constructed field with two clouds."""
import numpy as np

import cloud_w as cw

NZ, N, DX, DZ = 40, 32, 50., 25.
Z = (np.arange(NZ) + 0.5) * DZ


def _field():
    qc = np.zeros((NZ, N, N)); w = np.full((NZ, N, N), -0.1); thv = np.full((NZ, N, N), 300.)
    # cloud A: wraps across x = 0, levels 10..29, buoyant updraft of 2 m/s in levels 10..14 then 4 m/s above
    a = (slice(10, 30), slice(4, 8), [30, 31, 0, 1])
    qc[10:30, 4:8, :][:, :, [30, 31, 0, 1]] = 1.e-4
    w[10:15, 4:8, :][:, :, [30, 31, 0, 1]] = 2.
    w[15:30, 4:8, :][:, :, [30, 31, 0, 1]] = 4.
    thv[10:30, 4:8, :][:, :, [30, 31, 0, 1]] = 301.
    # cloud B: levels 12..19, rising at 1 m/s but not buoyant
    qc[12:20, 20:24, 10:14] = 1.e-4
    w[12:20, 20:24, 10:14] = 1.
    thv[12:20, 20:24, 10:14] = 299.
    return dict(qc=qc, w=w, thv=thv)


def test_two_clouds():
    t = cw.table(_field(), Z, DX, DX, kb=10, nlev=4)
    assert t["depth"].size == 2
    A, B = int(np.argmax(t["depth"])), int(np.argmin(t["depth"]))
    assert t["depth"][A] == 19 * DZ and t["depth"][B] == 7 * DZ
    assert t["area"][A] == 16 * DX * DX and t["core_cols"][A] == 16 and t["core_cols"][B] == 0
    assert t["n_core_level"][A] == 16 and t["w_core_level"][A] == 2.
    assert t["n_core_layer"][A] == 80 and t["w_core_layer"][A] == 2.
    assert t["n_core_own"][A] == 80 and t["w_core_own"][A] == 2. and t["z_core_own"][A] == Z[10]
    assert t["n_core_layer"][B] == 0 and np.isnan(t["w_core_layer"][B]) and np.isnan(t["w_core_own"][B])
    assert t["n_cu_level"][B] == 0 and t["n_cu_layer"][B] == 16 * 3 and t["w_cu_layer"][B] == 1.
    assert t["n_cu_own"][B] == 16 * 5 and t["z_cu_own"][B] == Z[12]


def test_layer_straddling_a_speed_change():
    t = cw.table(_field(), Z, DX, DX, kb=13, nlev=4)     # levels 13..17: two at 2 m/s, three at 4 m/s
    A = int(np.argmax(t["depth"]))
    assert t["n_core_layer"][A] == 80 and abs(t["w_core_layer"][A] - 3.2) < 1.e-12
    assert t["w_core_own"][A] == 2.


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
