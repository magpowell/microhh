"""Exit status 1 when one of the given binary field files (float64) holds a number that is not finite.

python check_finite.py thl.0046800 qt.0046800 ...
"""
import sys

import numpy as np

CHUNK = 2**24


def finite(path):
    a = np.memmap(path, dtype="<f8", mode="r") if path.stat().st_size else np.zeros(0)
    return all(np.isfinite(a[i:i + CHUNK]).all() for i in range(0, a.size, CHUNK))


if __name__ == "__main__":
    from pathlib import Path
    bad = [f for f in sys.argv[1:] if not finite(Path(f))]
    for f in bad:
        print(f"not finite: {f}")
    sys.exit(1 if bad else 0)
