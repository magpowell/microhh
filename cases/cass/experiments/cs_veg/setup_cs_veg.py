#!/usr/bin/env python3
"""
Set up CASS cs_veg experiment run directories.

Creates:
  $SCRATCH/CASS_LES/experiments/cs_veg/cs_veg_{VALUE}/{2stream,raytracer}/rep_{01..04}/

Parameter values (J m-2 K-1):
  0, 41840, 418400, 4184000, 41840000  (= J&M 0/1/10/100/1000 cm water equivalent)

INI overlays on top of base:
  [land_surface] cs_veg = <VALUE>
  [land_surface] c_veg  = 1.0   (already set in base ini)
  [aerosol]      swaerosol = false

cass_input.py is called with --zero-winds.

Usage:
  python setup_cs_veg.py [--dry-run] [--values V [V ...]]
"""

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

MICROHH_DIR  = Path("/global/homes/m/mpowell/repos/microhh")
CASS_DIR     = MICROHH_DIR / "cases" / "cass"
SHARED_DIR   = CASS_DIR / "shared"
SCRATCH      = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell"))
EXP_SCRATCH  = SCRATCH / "CASS_LES" / "experiments" / "cs_veg"

RADS = ["2stream", "raytracer"]
REPS = [1, 2, 3, 4]

CS_VEG_VALUES = [0, 41840, 418400, 4184000, 41840000]

RADIATION_FILES = {
    "coefficients_lw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-lw-g128.nc",
    "coefficients_sw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-sw-g112.nc",
    "cloud_coefficients_lw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-lw.nc",
    "cloud_coefficients_sw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-sw.nc",
    "aerosol_optics.nc":        MICROHH_DIR / "rte-rrtmgp-cpp" / "data" / "aerosol_optics.nc",
}

SHARED_SCRIPTS = [
    ("cass_input.py",  SHARED_DIR / "cass_input.py"),
    ("cass_utils.py",  SHARED_DIR / "cass_utils.py"),
    ("cleanup_run.py", SHARED_DIR / "cleanup_run.py"),
    ("3d_to_nc.py",    MICROHH_DIR / "python" / "3d_to_nc.py"),
    ("cross_to_nc.py", MICROHH_DIR / "python" / "cross_to_nc.py"),
]

CASE_DATA = [
    "cass_snd.txt",
    "cass_sfc.txt",
    "cass_lsf.txt",
    "cass_ls2d_input.nc",
    "cass_cams_composite.nc",
    "van_genuchten_parameters.nc",
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


DEBUG_GRID = {"itot": "64", "jtot": "64", "xsize": "6400.", "ysize": "6400."}


def merge_ini(rndseed: int, rt: str, cs_veg: int, debug: bool = False) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
    )
    cfg.optionxform = str
    cfg.read([
        SHARED_DIR / "config" / "cass_base.ini",
        SHARED_DIR / "config" / f"cass_{rt}.ini",
    ])
    cfg.set("fields", "rndseed", str(rndseed))
    cfg.set("land_surface", "cs_veg", str(cs_veg))
    cfg.set("aerosol", "swaerosol", "false")
    if debug:
        for k, v in DEBUG_GRID.items():
            cfg.set("grid", k, v)
    try:
        xsize = float(cfg.get("grid", "xsize"))
        ysize = float(cfg.get("grid", "ysize"))
        cfg.set("column", "coordinates[x]", str(xsize / 2))
        cfg.set("column", "coordinates[y]", str(ysize / 2))
    except (configparser.NoOptionError, ValueError):
        pass
    return cfg


def setup_rep(cs_veg: int, rt: str, rep: int, dry_run: bool, debug: bool = False):
    run_root = SCRATCH / "CASS_LES" / "debug" / "cs_veg" if debug else EXP_SCRATCH
    run_dir = run_root / f"cs_veg_{cs_veg}" / rt / f"rep_{rep:02d}"
    print(f"\n--- cs_veg_{cs_veg}/{rt}/rep_{rep:02d} ---")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rndseed=rep, rt=rt, cs_veg=cs_veg, debug=debug)
    if dry_run:
        print(f"  [dry] write cass.ini  (rndseed={rep}, cs_veg={cs_veg})")
    else:
        with open(run_dir / "cass.ini", "w") as f:
            cfg.write(f)
        print(f"  wrote cass.ini  (rndseed={rep}, cs_veg={cs_veg})")

    symlink(MICROHH_DIR / "build_gpu" / "microhh", run_dir / "microhh", dry_run)

    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)

    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)

    for name in CASE_DATA:
        symlink(SHARED_DIR / "data" / name, run_dir / name, dry_run)

    if dry_run:
        print("  [dry] python cass_input.py --zero-winds")
        return

    print("  running cass_input.py --zero-winds ...")
    result = subprocess.run(
        [sys.executable, "cass_input.py", "--zero-winds"],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: cass_input.py failed:\n{result.stderr[-2000:]}")
        sys.exit(1)
    print("  cass_input.nc written")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing")
    parser.add_argument("--debug", action="store_true",
                        help="64x64 grid, rep_01 only, scratch under CASS_LES/debug/cs_veg/")
    parser.add_argument("--values", type=int, nargs="+", default=CS_VEG_VALUES,
                        metavar="V",
                        help=f"cs_veg values to set up (default: {CS_VEG_VALUES})")
    args = parser.parse_args()

    reps = [1] if args.debug else REPS
    run_root = SCRATCH / "CASS_LES" / "debug" / "cs_veg" if args.debug else EXP_SCRATCH
    print(f"Setting up cs_veg experiment runs in: {run_root}")
    if args.dry_run:
        print("(dry run — no changes will be made)\n")

    for cs_veg in args.values:
        for rt in RADS:
            for rep in reps:
                setup_rep(cs_veg, rt, rep, args.dry_run, args.debug)

    print(f"\nDone. Run submit_cs_veg.sh to launch the jobs.")


if __name__ == "__main__":
    main()
