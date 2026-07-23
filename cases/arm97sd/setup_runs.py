#!/usr/bin/env python3
"""
Set up ARM97 shallow-to-deep run directories.

Modes:
  (default)    $SCRATCH/ARM97SD_LES/base/{2stream,raytracer}/rep_{01..NN}/
  --fit-test   $SCRATCH/ARM97SD_LES/fit_test/<grid>/raytracer/  one rep per
               grid candidate, endtime=1800, dumps/crosses off. Measures
               raytracer GPU memory (allocation-dominated) on hbm80g.
  --debug      $SCRATCH/ARM97SD_LES/debug/{2stream,raytracer}/rep_01/
               64x64 columns, full 12 h forcing.

Usage:
  python setup_runs.py [--fit-test | --debug] [--grid PRESET] [--dry-run]

Requires arm97sd_ls2d_input.nc (run preprocessing/arm97sd_ls2d_input.py
first and symlink into data/).
"""

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

# Derived from this file's own location (cases/arm97sd/setup_runs.py -> repo
# root), so the case travels between machines without editing. $MICROHH_DIR
# still wins if set, for out-of-tree checkouts.
MICROHH_DIR = Path(os.environ.get("MICROHH_DIR", Path(__file__).resolve().parents[2]))
CASE_DIR    = MICROHH_DIR / "cases" / "arm97sd"
if "SCRATCH" not in os.environ:
    raise SystemExit("SCRATCH is not set. Source the env script for this machine, "
                     "e.g. config/empireai_alpha_env.sh on Empire AI Alpha.")
SCRATCH     = Path(os.environ["SCRATCH"])
RUN_ROOT    = SCRATCH / "ARM97SD_LES"

# Python with xarray/netCDF4/scipy. Empire AI Alpha has no system scientific
# stack, so this is a mamba env; override with $XR_PY elsewhere.
XR_PY = os.environ.get("XR_PY", sys.executable)

# E3SM IOP forcing. On NERSC this lived in /global/cfs .../inputdata; it is also
# public on the LCRC E3SM inputdata mirror, so it is kept in-repo under
# shared_data/ rather than depending on a site-specific data mount.
IOP_FILE = Path(os.environ.get(
    "IOP_FILE", MICROHH_DIR / "shared_data" / "ARM97_iopfile_4scam.nc"))

RADS = ["2stream", "raytracer"]
REPS = [1, 2, 3, 4]

# Uniform dz everywhere (raytracer requirement); zsize fixed at 25.6 km,
# buffer starts at 20 km (set in arm97sd_base.ini).
GRID_PRESETS = {
    # name:          itot  jtot  ktot  xsize     ysize     (dx, dz noted)
    "dx200_k256": ("512", "512", "256", "102400", "102400"),  # 200 m, 100 m
    "dz80_k320":  ("512", "512", "320", "102400", "102400"),  # 200 m,  80 m
    "dx200_i640": ("640", "640", "256", "128000", "128000"),  # 200 m, 100 m
    "dx375_k256": ("512", "512", "256", "192000", "192000"),  # 375 m, 100 m
    "dz50_k512":  ("512", "512", "512", "102400", "102400"),  # 200 m,  50 m
    "dz50_i384":  ("384", "384", "512",  "76800",  "76800"),  # 200 m,  50 m, smaller domain
    "dz67_k384":  ("512", "512", "384", "102400", "102400"),  # 200 m, 66.7 m  <- production
    # 64^2 mini-domain BL-resolution ladder (morning BL only, no deep conv)
    "dbg64_k256": ("64", "64", "256", "12800", "12800"),   # dz=100
    "dbg64_k384": ("64", "64", "384", "12800", "12800"),   # dz=66.7
    "dbg64_k512": ("64", "64", "512", "12800", "12800"),   # dz=50
}

# Stretched vertical grids (non-uniform dz raytracer): dz0 up to z_fine, then
# geometric growth capped at dz_max, rescaled to land exactly on zsize.
# name: (itot, jtot, xsize, ysize, dz0, z_fine, growth, dz_max)
STRETCH_PRESETS = {
    "dz67s":      ("512", "512", "102400", "102400", 66.6667, 4000.0, 1.025, 400.0),
    "dbg64_s":    ("64",  "64",  "12800",  "12800",  66.6667, 4000.0, 1.025, 400.0),
}
ZSIZE = 25600.0


def stretched_z(dz0, z_fine, growth, dz_max, zsize=ZSIZE):
    import numpy as np
    dz = []
    z_now = 0.0
    while z_now + dz0 <= z_fine + 1e-6:
        dz.append(dz0)
        z_now += dz0
    d = dz0
    while z_now < zsize:
        d = min(d * growth, dz_max)
        dz.append(d)
        z_now += d
    dz = np.array(dz) * (zsize / z_now)
    zh = np.concatenate(([0.0], np.cumsum(dz)))
    return 0.5 * (zh[:-1] + zh[1:])


def stretch_grid_tuple(name):
    itot, jtot, xsize, ysize, dz0, z_fine, growth, dz_max = STRETCH_PRESETS[name]
    z = stretched_z(dz0, z_fine, growth, dz_max)
    return (itot, jtot, str(len(z)), xsize, ysize), z


FIT_TEST_GRIDS = list(GRID_PRESETS)

DEBUG_GRID = ("64", "64", "256", "12800", "12800")

RADIATION_FILES = {
    "coefficients_lw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-lw-g128.nc",
    "coefficients_sw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-sw-g112.nc",
    "cloud_coefficients_lw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-lw.nc",
    "cloud_coefficients_sw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-sw.nc",
}

SHARED_SCRIPTS = [
    ("arm97sd_input.py", CASE_DIR / "arm97sd_input.py"),
    ("3d_to_nc.py",       MICROHH_DIR / "python" / "3d_to_nc.py"),
    ("cross_to_nc.py",    MICROHH_DIR / "python" / "cross_to_nc.py"),
]

CASE_DATA = [
    ("ARM97_iopfile_4scam.nc", IOP_FILE),
    ("arm97sd_ls2d_input.nc",  CASE_DIR / "data" / "arm97sd_ls2d_input.nc"),
    ("van_genuchten_parameters.nc", MICROHH_DIR / "misc" / "van_genuchten_parameters.nc"),
]


def symlink(src: Path, dst: Path, dry_run: bool):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if not src.exists():
        print(f"  WARNING: source not found: {src}")
    if dry_run:
        print(f"  [dry] link {dst.name} -> {src}")
        return
    dst.symlink_to(src)
    print(f"  link {dst.name} -> {src}")


def merge_ini(rt: str, rndseed: int, grid: tuple, fit_test: bool,
              endtime: float = None) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
    )
    cfg.optionxform = str
    cfg.read([
        CASE_DIR / "config" / "arm97sd_base.ini",
        CASE_DIR / "config" / f"arm97sd_{rt}.ini",
    ])
    cfg.set("fields", "rndseed", str(rndseed))

    itot, jtot, ktot, xsize, ysize = grid
    cfg.set("grid", "itot", itot)
    cfg.set("grid", "jtot", jtot)
    cfg.set("grid", "ktot", ktot)
    cfg.set("grid", "xsize", xsize)
    cfg.set("grid", "ysize", ysize)

    if fit_test:
        et = f"{endtime:.0f}" if endtime else "1800"
        cfg.set("time", "endtime", et)
        cfg.set("time", "savetime", et)
        cfg.set("dump", "swdump", "0")
        cfg.set("cross", "swcross", "0")
        cfg.set("column", "swcolumn", "0")

    cfg.set("column", "coordinates[x]", str(float(xsize) / 2))
    cfg.set("column", "coordinates[y]", str(float(ysize) / 2))
    return cfg


def setup_run(run_dir: Path, rt: str, rndseed: int, grid: tuple,
              fit_test: bool, dry_run: bool, endtime: float = None,
              z_grid=None):
    print(f"\n--- {run_dir.relative_to(RUN_ROOT)} ---")
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rt, rndseed, grid, fit_test, endtime)
    if dry_run:
        print(f"  [dry] write arm97sd.ini (rt={rt}, seed={rndseed}, grid={grid})")
    else:
        with open(run_dir / "arm97sd.ini", "w") as fh:
            cfg.write(fh)
        print(f"  wrote arm97sd.ini (rt={rt}, seed={rndseed}, "
              f"grid={'x'.join(grid[:3])})")
        zgrid_file = run_dir / "zgrid.txt"
        if z_grid is not None:
            import numpy as np
            np.savetxt(zgrid_file, z_grid, fmt="%.8f")
            print(f"  wrote zgrid.txt ({len(z_grid)} stretched levels)")
        elif zgrid_file.exists():
            zgrid_file.unlink()

    symlink(MICROHH_DIR / "build_gpu" / "microhh", run_dir / "microhh", dry_run)
    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)
    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)
    for name, src in CASE_DATA:
        symlink(src, run_dir / name, dry_run)

    if dry_run:
        print("  [dry] python arm97sd_input.py")
        return

    print("  running arm97sd_input.py ...")
    result = subprocess.run(
        [XR_PY, "arm97sd_input.py"],
        cwd=run_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"  ERROR: arm97sd_input.py failed:\n{result.stderr[-2000:]}")
    print("  " + result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    all_grids = list(GRID_PRESETS) + list(STRETCH_PRESETS)
    parser.add_argument("--fit-test", action="store_true",
                        help="one raytracer run per grid preset, endtime=1800")
    parser.add_argument("--grids", nargs="+", choices=all_grids, default=None,
                        help="fit-test only: restrict to these presets")
    parser.add_argument("--endtime", type=float, default=None,
                        help="fit-test only: override endtime (default 1800 s)")
    parser.add_argument("--debug", action="store_true",
                        help="64x64 columns, rep_01 only")
    parser.add_argument("--grid", default="dz67_k384", choices=all_grids,
                        help="grid preset for production/debug runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def resolve(gname):
        if gname in STRETCH_PRESETS:
            return stretch_grid_tuple(gname)
        return GRID_PRESETS[gname], None

    if args.fit_test:
        for gname in (args.grids or FIT_TEST_GRIDS):
            grid, z_grid = resolve(gname)
            run_dir = RUN_ROOT / "fit_test" / gname / "raytracer"
            setup_run(run_dir, "raytracer", 1, grid,
                      fit_test=True, dry_run=args.dry_run, endtime=args.endtime,
                      z_grid=z_grid)
        print("\nDone. Submit with submit_fit_test.sh")
        return

    if args.debug:
        grid, z_grid = resolve(args.grid) if args.grid != "dz67_k384" else (DEBUG_GRID, None)
        for rt in RADS:
            run_dir = RUN_ROOT / "debug" / rt / "rep_01"
            setup_run(run_dir, rt, 1, grid,
                      fit_test=False, dry_run=args.dry_run, z_grid=z_grid)
        print("\nDone. Debug dirs ready.")
        return

    grid, z_grid = resolve(args.grid)
    for rt in RADS:
        for rep in REPS:
            run_dir = RUN_ROOT / "base" / rt / f"rep_{rep:02d}"
            setup_run(run_dir, rt, rep, grid,
                      fit_test=False, dry_run=args.dry_run, z_grid=z_grid)
    print("\nDone. Submit with submit_production.sh")


if __name__ == "__main__":
    main()
