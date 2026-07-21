#!/usr/bin/env python3
"""
Set up GoAmazon composite shallow-cumulus run directories.

Grid is fixed -- CASS's own validated shallow-Cu grid (512x512x256,
dx=50m, dz=25m, 6.4 km domain), already proven to run the raytracer
comfortably on a single A100. No fit test needed (unlike goamazon/arm97sd's
deep-convection grid, which was memory-constrained).

Modes:
  (default)  $SCRATCH/GOAMAZON_SHCU_LES/base/{2stream,raytracer}/rep_{01..NN}/
  --debug    $SCRATCH/GOAMAZON_SHCU_LES/debug/{2stream,raytracer}/rep_01/
             128x128 columns (6.4x6.4 km, same dx=50m as production) --
             unlike the deep-conv cases, a meaningful smoke test here:
             shallow-Cu cells are small enough to develop realistically at
             this domain size, not just "does it crash".

Usage:
  python setup_runs.py [--debug] [--dry-run]

Requires goamazon_ls2d_input.nc (reused from the goamazon case) and the
sfc/snd/lsf composite forcing (Manco & Figueroa 2025) symlinked into data/.
"""

import argparse
import configparser
import os
import subprocess
from pathlib import Path

MICROHH_DIR = Path("/global/homes/m/mpowell/repos/microhh")
CASE_DIR    = MICROHH_DIR / "cases" / "goamazon_shcu"
SCRATCH     = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell"))
RUN_ROOT    = SCRATCH / "GOAMAZON_SHCU_LES"

XR_PY = "/global/homes/m/mpowell/.conda/envs/xr_env/bin/python"

RADS = ["2stream", "raytracer"]
REPS = [1, 2, 3, 4]

PROD_GRID  = ("512", "512", "256", "25600", "25600")   # dx=50m, dz=25m
DEBUG_GRID = ("128", "128", "256", "6400",  "6400")    # same dx/dz, smaller area

RADIATION_FILES = {
    "coefficients_lw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-lw-g128.nc",
    "coefficients_sw.nc":       MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-gas-sw-g112.nc",
    "cloud_coefficients_lw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-lw.nc",
    "cloud_coefficients_sw.nc": MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data" / "rrtmgp-clouds-sw.nc",
}

SHARED_SCRIPTS = [
    ("goamazon_shcu_input.py", CASE_DIR / "goamazon_shcu_input.py"),
    ("3d_to_nc.py",            MICROHH_DIR / "python" / "3d_to_nc.py"),
    ("cross_to_nc.py",         MICROHH_DIR / "python" / "cross_to_nc.py"),
]

CASE_DATA = [
    ("sfc", CASE_DIR / "data" / "sfc"),
    ("snd", CASE_DIR / "data" / "snd"),
    ("lsf", CASE_DIR / "data" / "lsf"),
    ("goamazon_ls2d_input.nc", CASE_DIR / "data" / "goamazon_ls2d_input.nc"),
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


def merge_ini(rt: str, rndseed: int, grid: tuple, debug: bool) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
    )
    cfg.optionxform = str
    cfg.read([
        CASE_DIR / "config" / "goamazon_shcu_base.ini",
        CASE_DIR / "config" / f"goamazon_shcu_{rt}.ini",
    ])
    cfg.set("fields", "rndseed", str(rndseed))

    itot, jtot, ktot, xsize, ysize = grid
    cfg.set("grid", "itot", itot)
    cfg.set("grid", "jtot", jtot)
    cfg.set("grid", "ktot", ktot)
    cfg.set("grid", "xsize", xsize)
    cfg.set("grid", "ysize", ysize)

    cfg.set("column", "coordinates[x]", str(float(xsize) / 2))
    cfg.set("column", "coordinates[y]", str(float(ysize) / 2))
    return cfg


def setup_run(run_dir: Path, rt: str, rndseed: int, grid: tuple, debug: bool, dry_run: bool):
    print(f"\n--- {run_dir.relative_to(RUN_ROOT)} ---")
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rt, rndseed, grid, debug)
    if dry_run:
        print(f"  [dry] write goamazon_shcu.ini (rt={rt}, seed={rndseed}, grid={grid})")
    else:
        with open(run_dir / "goamazon_shcu.ini", "w") as fh:
            cfg.write(fh)
        print(f"  wrote goamazon_shcu.ini (rt={rt}, seed={rndseed}, "
              f"grid={'x'.join(grid[:3])})")

    symlink(MICROHH_DIR / "build_gpu" / "microhh", run_dir / "microhh", dry_run)
    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)
    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)
    for name, src in CASE_DATA:
        symlink(src, run_dir / name, dry_run)

    if dry_run:
        print("  [dry] python goamazon_shcu_input.py")
        return

    print("  running goamazon_shcu_input.py ...")
    result = subprocess.run(
        [XR_PY, "goamazon_shcu_input.py"],
        cwd=run_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"  ERROR: goamazon_shcu_input.py failed:\n{result.stderr[-2000:]}")
    print("  " + result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true",
                        help="128x128 columns, rep_01 only")
    parser.add_argument("--debug-rep", type=int, default=1,
                        help="debug only: rndseed/rep number (default 1)")
    parser.add_argument("--debug-rt", nargs="+", choices=RADS, default=None,
                        help="debug only: restrict to these RT modes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.debug:
        rep = args.debug_rep
        for rt in (args.debug_rt or RADS):
            run_dir = RUN_ROOT / "debug" / rt / f"rep_{rep:02d}"
            setup_run(run_dir, rt, rep, DEBUG_GRID, debug=True, dry_run=args.dry_run)
        print("\nDone. Debug dirs ready.")
        return

    for rt in RADS:
        for rep in REPS:
            run_dir = RUN_ROOT / "base" / rt / f"rep_{rep:02d}"
            setup_run(run_dir, rt, rep, PROD_GRID, debug=False, dry_run=args.dry_run)
    print("\nDone. Submit with submit_production.sh")


if __name__ == "__main__":
    main()
