"""Check the root fractions against Table 8.4 of the IFS documentation Cy41r2 (per cent per soil layer)."""
import numpy as np

import cass_land_composite as lc

TABLE_8_4 = {1: (24, 41, 31, 4), 2: (35, 38, 23, 4), 3: (26, 39, 29, 6), 4: (26, 38, 29, 7), 5: (24, 38, 31, 7),
             6: (25, 34, 27, 14), 7: (27, 27, 27, 9), 9: (47, 45, 8, 0), 10: (24, 41, 31, 4), 11: (17, 31, 33, 19),
             13: (25, 34, 27, 11), 16: (23, 36, 30, 11), 17: (23, 36, 30, 11), 18: (19, 35, 36, 10), 19: (19, 35, 36, 10)}


def test_root_fractions_match_the_ifs_table():
    """Rows 7 and 13 of the printed table sum to 90 and 97 per cent and are left out. Measured worst case 0.67."""
    worst = 0.
    for i, want in TABLE_8_4.items():
        r = 100. * lc.root_frac(lc.TABLE_8_1[i][5], lc.TABLE_8_1[i][6])
        assert abs(r.sum() - 100.) < 1.e-9
        if sum(want) in (99, 100, 101):
            worst = max(worst, np.abs(r - np.array(want)).max())
    assert worst < 0.75
    return worst


def test_layer_interfaces():
    zh = np.zeros(5)
    for k in range(4):
        zh[k + 1] = zh[k] + 2. * (-lc.Z_SOIL[k] - zh[k])
    assert np.allclose(zh, [0., 0.07, 0.28, 1.0, 2.89])


if __name__ == "__main__":
    test_layer_interfaces()
    print("ok test_layer_interfaces")
    print("largest difference from Table 8.4 [per cent]:", round(test_root_fractions_match_the_ifs_table(), 2))
