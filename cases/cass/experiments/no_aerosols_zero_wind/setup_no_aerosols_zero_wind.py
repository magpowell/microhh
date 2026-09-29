#!/usr/bin/env python3
"""
Set up CASS no_aerosols_zero_wind experiment run directories.

Creates:
  $SCRATCH/CASS_LES/experiments/no_aerosols_zero_wind/{2stream,raytracer}/rep_{01..04}/

Aerosols off, zero winds. Dual purpose:
  1. Direct science: zero-wind 2stream vs raytracer LWP differences.
  2. Source of nudge profiles for the mean_state_nudge experiment
     (extract from 2stream column output after these runs complete).

Versions:
  v2  6.4 km domain, u/v nudging on 3 h (archived on HPSS; the root on scratch holds restored data).
  v3  8 km domain, thl/qt nudging above 5 km on 1 h, pressure in the dumps, core mask and tendency
      statistics, third configuration raytracer_swmatch. Soil state, vegetation cover and leaf area
      index from ERA5 at the start hour; canopy parameters from the IFS table. See V3 below.

Usage:
  python setup_no_aerosols_zero_wind.py [--version v3] [--dry-run] [--debug] [--input-args ...]
"""

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

def _repo_root() -> Path:
    """Repo root from this file's location, so the case travels between machines.
    Override with $MICROHH_DIR (the Slurm bodies export it explicitly)."""
    if "MICROHH_DIR" in os.environ:
        return Path(os.environ["MICROHH_DIR"])
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "cass").is_dir() and (parent / "config").is_dir():
            return parent
    raise SystemExit("Cannot locate the microhh repo root from %s; set $MICROHH_DIR" % __file__)


if "SCRATCH" not in os.environ:
    raise SystemExit("SCRATCH is not set. Source the env script for this machine "
                     "(e.g. config/empireai_alpha_env.sh) or export SCRATCH.")

MICROHH_DIR = _repo_root()
# Binary to run: $MICROHH_EXEC if set (on Alpha the validated copy under
# ~/validated_builds/), else the in-tree GPU build.
MICROHH_EXEC = Path(os.environ.get("MICROHH_EXEC", MICROHH_DIR / "build_gpu" / "microhh"))
CASS_DIR     = MICROHH_DIR / "cases" / "cass"
SHARED_DIR   = CASS_DIR / "shared"
SCRATCH      = Path(os.environ["SCRATCH"])
EXP_NAME     = "no_aerosols_zero_wind"

RADS = {"v2": ["2stream", "raytracer"], "v3": ["2stream", "raytracer", "raytracer_swmatch"]}
REPS = [1, 2, 3, 4]

# (section, key, value) overrides of the merged ini.
V3 = [
    ("grid", "zsize", "8000"), ("grid", "ktot", "320"),
    ("buffer", "zstart", "6400"),
    ("boundary", "z0m", "0.035"),      # CASS value (not the IFS crop value 0.25)
    ("boundary", "z0h", "0.003"),      # choice, as in v2; within 20 % of the IFS crop value 0.0025
    ("force", "nudgelist", "thl,qt"), ("force", "timedeplist_nudge", "thl,qt"),
    ("stats", "swtendency", "1"),
    ("micro", "swmicrobudget", "1"),
    ("dump", "dumplist", "p,ql,b,qt,thl,w,couvreux"),
]
# High-frequency 3D stream (float32, below hf_zmax): 11:58 to 16:03 local solar time for a 12:00 UTC start.
V3_HF = [
    ("dump", "swhf", "1"), ("dump", "hf_starttime", "23700"), ("dump", "hf_endtime", "38400"),
    ("dump", "hf_zmax", "6000"), ("dump", "hf_dumplist", "u,v,w,thl,qt,p"),
]
V3_HF_60S = {("2stream", 1), ("raytracer", 1)}     # every other member samples at 300 s
V3_MASKS = ("couvreux", "wplus", "ql", "qlcore")
LAND = "cass_land_composite_12utc.nc"            # ERA5 soil and vegetation at the start hour (cass_land_composite.py)
V3_INPUT_ARGS = ["--zero-winds", "--nudge-scalars", "--exner-ls", "--taper-wls", "--land-composite", LAND]
V3_LAND_KEYS = ("c_veg", "lai", "rs_veg_min", "gD")


def exp_root(version: str, debug: bool, tag: str = "") -> Path:
    base = SCRATCH / "CASS_LES" / ("debug" if debug else "experiments")
    return base / (f"{EXP_NAME}_{version}" + (f"_{tag}" if tag else ""))

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

CASE_DATA_V3 = [LAND]
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


def merge_ini(rndseed: int, rt: str, debug: bool = False, version: str = "v3") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
    )
    cfg.optionxform = str
    cfg.read([
        SHARED_DIR / "config" / "cass_base.ini",
        SHARED_DIR / "config" / f"cass_{rt.split('_')[0]}.ini",
    ])
    if rt == "raytracer_swmatch":
        cfg.set("radiation", "swscalesfc_to_2str", "true")
    cfg.set("fields", "rndseed", str(rndseed))
    cfg.set("aerosol", "swaerosol", "false")
    # Add evisc to 3D dumps for SGS flux correction in cloud-root composites
    dumplist = cfg.get("dump", "dumplist")
    if "evisc" not in dumplist:
        cfg.set("dump", "dumplist", dumplist + ",evisc")
    # Couvreux (2010) passive tracer for updraft conditional sampling.
    # Tracer-only mask in-model; AND with w>0, ql>0 offline in xarray.
    slist = cfg.get("fields", "slist", fallback="").strip()
    if "couvreux" not in slist:
        cfg.set("fields", "slist", (slist + "," if slist else "") + "couvreux")
    # per-scalar surface-flux BC (magnitude arbitrary; mask uses relative threshold).
    # Value 1e-5 matches the LASSO / BOMEX convention in the MicroHH repo.
    cfg.set("boundary", "sbcbot[couvreux]", "flux")
    cfg.set("boundary", "sbot[couvreux]",   "1e-5")
    cfg.set("boundary", "sbctop[couvreux]", "neumann")
    cfg.set("boundary", "stop[couvreux]",   "0.")
    # Exponential relaxation sink. tau = 900 s (paper canonical; Couvreux 2010 BLM 134).
    # nstd_couvreux = 1 is paper canonical; LASSO uses 3 to compensate for the absent
    # sigma_min floor in MicroHH's Decay::get_mask -- revisit if mask is noisy aloft.
    if not cfg.has_section("decay"):
        cfg.add_section("decay")
    cfg.set("decay", "swdecay[couvreux]", "exponential")
    cfg.set("decay", "timescale",         "900.")
    cfg.set("decay", "nstd_couvreux",     "1.")
    # register couvreux/wplus/ql masks for conditional stats. Do NOT add "default" --
    # stats.cxx:561 auto-appends it, and listing it explicitly produces a duplicate
    # that leaves domain-mean stats (ql_cover, qlqi_path, thl, qt, ...) all zero/NaN.
    masklist = cfg.get("stats", "masklist", fallback="")
    for m in (V3_MASKS if version == "v3" else ("couvreux", "wplus", "ql")):
        if m not in masklist.split(","):
            masklist += ("," if masklist else "") + m
    cfg.set("stats", "masklist", masklist)
    # include couvreux in 3D dumps for cell-exact offline masking
    dumplist = cfg.get("dump", "dumplist")
    if "couvreux" not in dumplist:
        cfg.set("dump", "dumplist", dumplist + ",couvreux")
    if version == "v3":
        for sec, key, val in V3 + V3_HF:
            cfg.set(sec, key, val)
        cfg.set("dump", "hf_sampletime", "60" if (rt, rndseed) in V3_HF_60S else "300")
        import netCDF4
        with netCDF4.Dataset(SHARED_DIR / "data" / LAND) as f:
            for key in V3_LAND_KEYS:
                cfg.set("land_surface", key, f"{float(f.getncattr(key)):.4g}")
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


def setup_rep(rt: str, rep: int, dry_run: bool, debug: bool = False, version: str = "v3", input_args=None, tag=""):
    run_dir = exp_root(version, debug, tag) / rt / f"rep_{rep:02d}"
    print(f"\n--- no_aerosols_zero_wind_{version}/{rt}/rep_{rep:02d} ---")
    if not dry_run and any(run_dir.glob("*.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]")):
        raise SystemExit(f"{run_dir} already holds model output; refusing to set it up again")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    cfg = merge_ini(rndseed=rep, rt=rt, debug=debug, version=version)
    if dry_run:
        print(f"  [dry] write cass.ini  (rndseed={rep}, swaerosol=false, zero winds)")
    else:
        with open(run_dir / "cass.ini", "w") as f:
            cfg.write(f)
        print(f"  wrote cass.ini  (rndseed={rep}, swaerosol=false, zero winds)")

    symlink(MICROHH_EXEC, run_dir / "microhh", dry_run)

    for name, src in SHARED_SCRIPTS:
        symlink(src, run_dir / name, dry_run)

    for name, src in RADIATION_FILES.items():
        symlink(src, run_dir / name, dry_run)

    for name in CASE_DATA + (CASE_DATA_V3 if version == "v3" else []):
        symlink(SHARED_DIR / "data" / name, run_dir / name, dry_run)

    if input_args is None:
        input_args = V3_INPUT_ARGS if version == "v3" else ["--zero-winds"]
    if dry_run:
        print("  [dry] python cass_input.py " + " ".join(input_args))
        return

    print("  running cass_input.py " + " ".join(input_args) + " ...")
    result = subprocess.run(
        [sys.executable, "cass_input.py", *input_args],
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
                        help="64x64 grid, rep_01 only, scratch under CASS_LES/debug/no_aerosols_zero_wind_<version>/")
    parser.add_argument("--version", default="v3", choices=sorted(RADS))
    parser.add_argument("--rt", nargs="+", default=None, help="subset of the radiation configurations")
    parser.add_argument("--tag", default="", help="suffix of the debug folder, for test runs (needs --debug)")
    parser.add_argument("--input-args", nargs=argparse.REMAINDER, default=None,
                        help="arguments for cass_input.py, replacing the version default (must come last)")
    args = parser.parse_args()

    if args.tag and not args.debug:
        raise SystemExit("--tag is for debug runs only")
    reps = [1] if args.debug else REPS
    print(f"Setting up no_aerosols_zero_wind runs in: {exp_root(args.version, args.debug, args.tag)}")
    if args.dry_run:
        print("(dry run — no changes will be made)\n")

    for rt in (args.rt or RADS[args.version]):
        for rep in reps:
            setup_rep(rt, rep, args.dry_run, args.debug, args.version, args.input_args, args.tag)

    print("\nDone. Run submit_no_aerosols_zero_wind.sh to launch the jobs.")


if __name__ == "__main__":
    main()
