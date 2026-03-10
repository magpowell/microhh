#!/usr/bin/env python3
"""
Set up CASS mean_state_nudge experiment run directories (raytracer only).

Creates:
  $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_{TIMESCALE}s/raytracer/rep_{01..04}/

The raytracer runs are nudged toward the ensemble-mean thl/qt profiles from the
completed no_aerosols 2stream runs (extract_nudge_profiles.py must be run first).

Configuration on top of no_aerosols base:
  - nudgelist = u,v,thl,qt  (thl and qt added)
  - timedeplist_nudge = u,v,thl,qt
  - nudgefac = 1/TIMESCALE (s^-1) via --nudge-thermo flag to cass_input.py
  - swaerosol = false
  - zero winds (--zero-winds)

Usage:
  python setup_mean_state_nudge.py \\
      --timescale 3600 \\
      --nudge-profiles $SCRATCH/CASS_LES/experiments/mean_state_nudge/nudge_profiles.nc \\
      [--dry-run] [--debug]
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
EXP_ROOT     = SCRATCH / "CASS_LES" / "experiments" / "mean_state_nudge"

REPS = [1, 2, 3, 4]

DEBUG_GRID = {"itot": "64", "jtot": "64", "xsize": "6400.", "ysize": "6400."}

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


def merge_ini(rndseed: int, timescale: float, debug: bool = False) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
    )
    cfg.optionxform = str
    cfg.read([
        SHARED_DIR / "config" / "cass_base.ini",
        SHARED_DIR / "config" / "cass_raytracer.ini",
    ])
    cfg.set("fields", "rndseed", str(rndseed))
    cfg.set("aerosol", "swaerosol", "false")
    # Add thl and qt to nudge lists
    cfg.set("force", "nudgelist", "u,v,thl,qt")
    cfg.set("force", "timedeplist_nudge", "u,v,thl,qt")
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


def setup_rep(rep: int, timescale: float, nudge_profiles: Path,
              run_root: Path, dry_run: bool, debug: bool = False):
    run_dir = run_root / "raytracer" / f"rep_{rep:02d}"
    print(f"\n--- mean_state_nudge/nudge_{timescale:.0f}s/raytracer/rep_{rep:02d} ---")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rndseed=rep, timescale=timescale, debug=debug)
    if dry_run:
        print(f"  [dry] write cass.ini  (rndseed={rep}, nudge thl+qt, timescale={timescale:.0f}s)")
    else:
        with open(run_dir / "cass.ini", "w") as f:
            cfg.write(f)
        print(f"  wrote cass.ini  (rndseed={rep}, nudge thl+qt, timescale={timescale:.0f}s)")

    symlink(MICROHH_DIR / "build_gpu" / "microhh", run_dir / "microhh", dry_run)

    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)

    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)

    for name in CASE_DATA:
        symlink(SHARED_DIR / "data" / name, run_dir / name, dry_run)

    if dry_run:
        print(f"  [dry] python cass_input.py --zero-winds "
              f"--nudge-thermo {nudge_profiles} {timescale:.0f}")
        return

    print("  running cass_input.py ...")
    result = subprocess.run(
        [sys.executable, "cass_input.py",
         "--zero-winds",
         "--nudge-thermo", str(nudge_profiles), str(timescale)],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: cass_input.py failed:\n{result.stderr[-2000:]}")
        sys.exit(1)
    print("  cass_input.nc written")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timescale", type=float, required=True,
                        help="Nudging timescale in seconds (e.g. 3600)")
    parser.add_argument("--nudge-profiles", required=True,
                        help="Path to nudge_profiles.nc (from extract_nudge_profiles.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing")
    parser.add_argument("--debug", action="store_true",
                        help="64x64 grid, rep_01 only, scratch under CASS_LES/debug/")
    args = parser.parse_args()

    nudge_profiles = Path(args.nudge_profiles)
    if not nudge_profiles.exists():
        print(f"ERROR: nudge_profiles.nc not found: {nudge_profiles}", file=sys.stderr)
        print("Run extract_nudge_profiles.py first.", file=sys.stderr)
        sys.exit(1)

    label = f"nudge_{args.timescale:.0f}s"
    if args.debug:
        run_root = SCRATCH / "CASS_LES" / "debug" / "mean_state_nudge" / label
    else:
        run_root = EXP_ROOT / label

    reps = [1] if args.debug else REPS
    print(f"Setting up mean_state_nudge/{label} runs in: {run_root}")
    if args.dry_run:
        print("(dry run — no changes will be made)\n")

    for rep in reps:
        setup_rep(rep, args.timescale, nudge_profiles, run_root, args.dry_run, args.debug)

    print(f"\nDone. Run submit_mean_state_nudge.sh --timescale {args.timescale:.0f} to launch jobs.")


if __name__ == "__main__":
    main()
