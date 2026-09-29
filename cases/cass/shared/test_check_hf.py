"""Exact checks of check_hf.py on constructed run folders."""
import tempfile
from pathlib import Path

import numpy as np

import check_hf as ch

INI = """[grid]
itot=8
jtot=6
ktot=20
zsize=500
[time]
endtime=1800
[dump]
hf_starttime=600
hf_endtime=1200
hf_sampletime=300
hf_zmax=250
hf_dumplist=u,p
"""


def _run(d, spoil=None, drop=None, extra=None):
    d = Path(d)
    (d / "cass.ini").write_text(INI)
    rng = np.random.default_rng(0)
    for v in ("u", "p"):
        for t in (600, 900, 1200):
            full = rng.normal(size=(20, 6, 8))
            if t in (600, 1200):
                full.astype("<f8").tofile(d / f"{v}.{t:07d}")
            hf = full[:10].astype("<f4")
            if spoil == (v, t):
                hf[3, 2, 1] = np.nextafter(hf[3, 2, 1], np.float32(10.))
            if drop != (v, t):
                hf.tofile(d / f"{v}_hf.{t:07d}")
    if extra:
        np.zeros((10, 6, 8), dtype="<f4").tofile(d / f"u_hf.{extra:07d}")
    return d


def test_clean_run_passes():
    with tempfile.TemporaryDirectory() as d:
        r = ch.check_stream(_run(d))
    assert r["levels"] == 10 and r["expected_times"] == 3 and r["compared"] == 4
    assert not (r["missing"] or r["unexpected"] or r["wrong_size"] or r["not_identical"])


def test_one_changed_bit_is_found():
    with tempfile.TemporaryDirectory() as d:
        r = ch.check_stream(_run(d, spoil=("p", 1200)))
    assert r["not_identical"] == ["p_hf.0001200"]


def test_missing_and_unexpected_files_are_found():
    with tempfile.TemporaryDirectory() as d:
        r = ch.check_stream(_run(d, drop=("u", 900), extra=1500))
    assert r["missing"] == ["u_hf.0000900"] and r["unexpected"] == ["u_hf.0001500"]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
