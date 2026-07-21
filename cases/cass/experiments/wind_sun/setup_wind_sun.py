#!/usr/bin/env python3
"""
Set up CASS wind_sun experiment run directories.

Creates:
  $SCRATCH/CASS_LES/experiments/wind_sun/{2stream,raytracer}/rep_{01..04}/

Wind prescription:
  Target wind vector magnitude U = 5 m/s, direction = anti-solar (cloud
  advected toward its shadow), rotating with the solar azimuth through the
  diurnal cycle. Solar position computed from cass.ini lat/lon for day
  205.5 (July 24, 12:00 UTC at SGP = ~07:00 LST CDT).

  - u_nudge[t,z] = -U sin(phi_s(t))   uniform in z
  - v_nudge[t,z] = -U cos(phi_s(t))   uniform in z
  - u_geo, v_geo identical to u_nudge, v_nudge (Coriolis tendency ~ 0 when
    BL wind matches the prescription).
  - nudge timescale tau = 1800 s (constant in z) — strong enough to track
    16 deg/h azimuth rotation with ~8 deg lag.
  - time_ls resampled to 600 s spacing inside cass_input.py.

INI overlays on top of base:
  [aerosol]  swaerosol      = false
  [force]    swtimedep_geo  = true

cass_input.py is called with --sun-wind 5.0 --nudge-timescale 1800.

Usage:
  python setup_wind_sun.py [--dry-run] [--debug]
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
EXP_SCRATCH  = SCRATCH / "CASS_LES" / "experiments" / "wind_sun"

RADS = ["2stream", "raytracer"]
REPS = [1, 2, 3, 4]

WIND_MAGNITUDE  = 5.0       # m/s
NUDGE_TIMESCALE = 1800.0    # s

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

DEBUG_GRID = {"itot": "64", "jtot": "64", "xsize": "6400.", "ysize": "6400."}


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


def merge_ini(rndseed: int, rt: str, debug: bool = False) -> configparser.ConfigParser:
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
    cfg.set("aerosol", "swaerosol", "false")
    cfg.set("force", "swtimedep_geo", "true")
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


def setup_rep(rt: str, rep: int, dry_run: bool, debug: bool = False):
    run_root = SCRATCH / "CASS_LES" / "debug" / "wind_sun" if debug else EXP_SCRATCH
    run_dir = run_root / rt / f"rep_{rep:02d}"
    print(f"\n--- {rt}/rep_{rep:02d} ---")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rndseed=rep, rt=rt, debug=debug)
    if dry_run:
        print(f"  [dry] write cass.ini (rndseed={rep})")
    else:
        with open(run_dir / "cass.ini", "w") as f:
            cfg.write(f)
        print(f"  wrote cass.ini (rndseed={rep})")

    symlink(MICROHH_DIR / "build_gpu" / "microhh", run_dir / "microhh", dry_run)

    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)

    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)

    for name in CASE_DATA:
        symlink(SHARED_DIR / "data" / name, run_dir / name, dry_run)

    cmd = [sys.executable, "cass_input.py",
           "--sun-wind", str(WIND_MAGNITUDE),
           "--nudge-timescale", str(NUDGE_TIMESCALE)]
    if dry_run:
        print(f"  [dry] {' '.join(cmd[1:])}")
        return

    print(f"  running cass_input.py --sun-wind {WIND_MAGNITUDE} --nudge-timescale {NUDGE_TIMESCALE} ...")
    result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
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
                        help="64x64 grid, rep_01 only, scratch under CASS_LES/debug/wind_sun/")
    args = parser.parse_args()

    reps = [1] if args.debug else REPS
    run_root = SCRATCH / "CASS_LES" / "debug" / "wind_sun" if args.debug else EXP_SCRATCH
    print(f"Setting up wind_sun experiment runs in: {run_root}")
    if args.dry_run:
        print("(dry run — no changes will be made)\n")

    for rt in RADS:
        for rep in reps:
            setup_rep(rt, rep, args.dry_run, debug=args.debug)

    print(f"\nDone. Run submit_wind_sun.sh to launch the jobs.")


if __name__ == "__main__":
    main()
