#!/usr/bin/env python3
"""
Set up GoAmazon single-pulse run directories.

Modes:
  (default)    $SCRATCH/GOAMAZON_LES/base/{2stream,raytracer}/rep_{01..NN}/
  --fit-test   $SCRATCH/GOAMAZON_LES/fit_test/<grid>/raytracer/  one rep per
               grid candidate, endtime=1800, dumps/crosses off. Measures
               raytracer GPU memory (allocation-dominated) on hbm80g.
  --debug      $SCRATCH/GOAMAZON_LES/debug/{2stream,raytracer}/rep_01/
               64x64 columns, full 12 h forcing.

Usage:
  python setup_runs.py [--fit-test | --debug] [--grid PRESET] [--dry-run]

Requires goamazon_ls2d_input.nc (run preprocessing/goamazon_ls2d_input.py
first and symlink into data/).
"""

import argparse
import configparser
import os
import subprocess
from pathlib import Path

# Derived from this file's own location (cases/goamazon/setup_runs.py -> repo
# root), so the case travels between machines without editing. $MICROHH_DIR
# still wins if set, for out-of-tree checkouts.
MICROHH_DIR = Path(os.environ.get("MICROHH_DIR", Path(__file__).resolve().parents[2]))
CASE_DIR    = MICROHH_DIR / "cases" / "goamazon"
SCRATCH     = Path(os.environ.get("SCRATCH", f"/mnt/lustre/columbia/{os.environ.get('USER','')}"))
RUN_ROOT    = SCRATCH / "GOAMAZON_LES"

# Python with xarray/netCDF4/scipy. Empire AI Alpha has no system scientific
# stack, so this is a mamba env; override with $XR_PY elsewhere.
XR_PY = os.environ.get("XR_PY", f"{Path.home()}/miniforge3/envs/microhh/bin/python")

# E3SM IOP forcing. On NERSC this lived in /global/cfs .../inputdata; it is also
# public on the LCRC E3SM inputdata mirror, so it is kept in-repo under
# shared_data/ rather than depending on a site-specific data mount.
IOP_FILE = Path(os.environ.get(
    "IOP_FILE", MICROHH_DIR / "shared_data" / "GOAMAZON_singlepulse_iopfile_4scam.nc"))

RADS = ["2stream", "raytracer"]
REPS = [1, 2, 3, 4]

# Uniform dz everywhere (raytracer requirement); zsize fixed at 25.6 km,
# buffer starts at 20 km (set in goamazon_base.ini).
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
FIT_TEST_GRIDS = list(GRID_PRESETS)

DEBUG_GRID = ("64", "64", "256", "12800", "12800")

RADIATION_FILES = {
    "coefficients_lw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-lw-g128.nc",
    "coefficients_sw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-sw-g112.nc",
    "cloud_coefficients_lw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-lw.nc",
    "cloud_coefficients_sw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-sw.nc",
}

SHARED_SCRIPTS = [
    ("goamazon_input.py", CASE_DIR / "goamazon_input.py"),
    ("3d_to_nc.py",       MICROHH_DIR / "python" / "3d_to_nc.py"),
    ("cross_to_nc.py",    MICROHH_DIR / "python" / "cross_to_nc.py"),
]

CASE_DATA = [
    ("GOAMAZON_singlepulse_iopfile_4scam.nc", IOP_FILE),
    ("goamazon_ls2d_input.nc",  CASE_DIR / "data" / "goamazon_ls2d_input.nc"),
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
        CASE_DIR / "config" / "goamazon_base.ini",
        CASE_DIR / "config" / f"goamazon_{rt}.ini",
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
              fit_test: bool, dry_run: bool, endtime: float = None):
    print(f"\n--- {run_dir.relative_to(RUN_ROOT)} ---")
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rt, rndseed, grid, fit_test, endtime)
    if dry_run:
        print(f"  [dry] write goamazon.ini (rt={rt}, seed={rndseed}, grid={grid})")
    else:
        with open(run_dir / "goamazon.ini", "w") as fh:
            cfg.write(fh)
        print(f"  wrote goamazon.ini (rt={rt}, seed={rndseed}, "
              f"grid={'x'.join(grid[:3])})")

    symlink(MICROHH_DIR / "build_gpu" / "microhh", run_dir / "microhh", dry_run)
    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)
    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)
    for name, src in CASE_DATA:
        symlink(src, run_dir / name, dry_run)

    if dry_run:
        print("  [dry] python goamazon_input.py")
        return

    print("  running goamazon_input.py ...")
    result = subprocess.run(
        [XR_PY, "goamazon_input.py"],
        cwd=run_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"  ERROR: goamazon_input.py failed:\n{result.stderr[-2000:]}")
    print("  " + result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-test", action="store_true",
                        help="one raytracer run per grid preset, endtime=1800")
    parser.add_argument("--grids", nargs="+", choices=GRID_PRESETS, default=None,
                        help="fit-test only: restrict to these presets")
    parser.add_argument("--endtime", type=float, default=None,
                        help="fit-test only: override endtime (default 1800 s)")
    parser.add_argument("--debug", action="store_true",
                        help="64x64 columns, rep_01 only")
    parser.add_argument("--grid", default="dz67_k384", choices=GRID_PRESETS,
                        help="grid preset for production/debug runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.fit_test:
        for gname in (args.grids or FIT_TEST_GRIDS):
            run_dir = RUN_ROOT / "fit_test" / gname / "raytracer"
            setup_run(run_dir, "raytracer", 1, GRID_PRESETS[gname],
                      fit_test=True, dry_run=args.dry_run, endtime=args.endtime)
        print("\nDone. Submit with submit_fit_test.sh")
        return

    if args.debug:
        for rt in RADS:
            run_dir = RUN_ROOT / "debug" / rt / "rep_01"
            setup_run(run_dir, rt, 1, DEBUG_GRID,
                      fit_test=False, dry_run=args.dry_run)
        print("\nDone. Debug dirs ready.")
        return

    for rt in RADS:
        for rep in REPS:
            run_dir = RUN_ROOT / "base" / rt / f"rep_{rep:02d}"
            setup_run(run_dir, rt, rep, GRID_PRESETS[args.grid],
                      fit_test=False, dry_run=args.dry_run)
    print("\nDone. Submit with submit_production.sh")


if __name__ == "__main__":
    main()
