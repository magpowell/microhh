#!/usr/bin/env python3
"""
Setup complexity-reduction experiments for LES cumulus case 20140325_t03.

Three experiments, each with 3 random-seed reps × {rt, standard} = 6 runs:
  1. no_aerosols  — aerosols off (already the default in the source ini)
  2. no_gases     — aerosols off + all gas mixing ratios zeroed in input NC
  3. dark_ocean   — no_gases + surface albedo set to 0.07

Output base: /pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325/

Before running this script, generate the no-gases input file:
    python make_no_gases_nc.py
"""

import argparse
import configparser
from pathlib import Path

# === Paths ===
SOURCE_CASE = Path("/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2/20140325_t03")
OUTPUT_BASE = Path("/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325")
MICROHH_DIR = Path("/global/homes/m/mpowell/repos/microhh")

PATHS = {
    "microhh":       MICROHH_DIR / "build_gpu" / "microhh",
    "rrtmgp_data":   MICROHH_DIR / "rte-rrtmgp-cpp" / "rrtmgp-data",
    "cabauw_cases":  MICROHH_DIR / "cases" / "cabauw",
    "cross_to_nc":   MICROHH_DIR / "python" / "cross_to_nc.py",
    "microhh_tools": MICROHH_DIR / "python" / "microhh_tools.py",
}

RADIATION_FILES = {
    "cloud_coefficients_lw.nc": "rrtmgp-clouds-lw.nc",
    "cloud_coefficients_sw.nc": "rrtmgp-clouds-sw.nc",
    "coefficients_lw.nc":       "rrtmgp-gas-lw-g128.nc",
    "coefficients_sw.nc":       "rrtmgp-gas-sw-g112.nc",
}

LSM_FILES = ["van_genuchten_parameters.nc"]

RADIATION_CONFIG = {"rt": "rrtmgp_rt", "standard": "rrtmgp"}

N_REPS = 3
SEEDS = list(range(1, N_REPS + 1))

# Scripts to symlink from this directory into OUTPUT_BASE
REPO_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = [
    "setup_complexity.py",
    "make_no_gases_nc.py",
    "sbatch_complexity.sh",
    "submit_complexity.sh",
]


def create_symlink(source: Path, dest: Path):
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(source)


def setup_symlinks(sim_dir: Path, input_nc: Path):
    """Create all symlinks needed in a simulation directory."""
    create_symlink(PATHS["microhh"], sim_dir / "microhh")
    create_symlink(input_nc, sim_dir / "cabauw_input.nc")

    for dest_name, source_name in RADIATION_FILES.items():
        create_symlink(PATHS["rrtmgp_data"] / source_name, sim_dir / dest_name)

    for lsm_file in LSM_FILES:
        create_symlink(PATHS["cabauw_cases"] / lsm_file, sim_dir / lsm_file)

    create_symlink(PATHS["cross_to_nc"],   sim_dir / "cross_to_nc.py")
    create_symlink(PATHS["microhh_tools"], sim_dir / "microhh_tools.py")


def create_ini(rad_type: str, rndseed: int, experiment: str, out_path: Path):
    """Write a modified cabauw.ini for a complexity experiment run.

    Source: SOURCE_CASE/rt/cabauw.ini (already has swaerosol=false, swcross=1).
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(SOURCE_CASE / "rt" / "cabauw.ini")

    # Radiation type
    config.set("radiation", "swradiation", RADIATION_CONFIG[rad_type])

    # Seed and output settings
    config.set("fields", "rndseed", str(rndseed))
    config.set("dump",   "swdump",  "0")
    config.set("cross",  "swcross", "1")

    # Aerosols off (explicit, even though source already has this)
    config.set("aerosol", "swaerosol", "false")
    config.set("aerosol", "swtimedep", "false")

    # no_gases / dark_ocean: disable time-dependent gas reading
    if experiment in ("no_gases", "dark_ocean"):
        config.set("radiation", "timedeplist_gas", "")

    # dark_ocean: ocean surface properties.
    # swwater=1 is incompatible with swhomogeneous=1, so we replicate the
    # water-tile physics manually (mirrors what set_water_tiles kernel does).
    if experiment == "dark_ocean":
        # Radiation
        config.set("radiation", "sfc_alb_dir", "0.07")
        config.set("radiation", "sfc_alb_dif", "0.07")
        config.set("radiation", "emis_sfc",    "0.99")
        # Aerodynamic roughness (open ocean, ~COARE typical fixed value)
        config.set("boundary", "z0m", "0.0002")
        config.set("boundary", "z0h", "0.0002")
        # Land-surface: remove vegetation, set resistances to zero
        # (kernel explicitly zeros c_veg, rs_veg, rs_soil for water tiles)
        config.set("land_surface", "c_veg",          "0.0")
        config.set("land_surface", "lai",             "0.0")
        config.set("land_surface", "rs_veg_min",      "0")
        config.set("land_surface", "rs_soil_min",     "0")
        config.set("land_surface", "lambda_stable",   "0.0")
        config.set("land_surface", "lambda_unstable", "0.0")

    with open(out_path, "w") as f:
        config.write(f)


def setup_experiment(experiment: str, input_nc: Path):
    """Set up one experiment across all radiation types and seeds."""
    print(f"\n{'='*60}")
    print(f"Setting up: {experiment}")
    print(f"{'='*60}")

    for rad_type in ["rt", "standard"]:
        for seed in SEEDS:
            run_name = f"seed_{seed}"
            sim_dir  = OUTPUT_BASE / experiment / rad_type / run_name
            sim_dir.mkdir(parents=True, exist_ok=True)

            create_ini(rad_type, seed, experiment, sim_dir / "cabauw.ini")
            setup_symlinks(sim_dir, input_nc)
            print(f"  {rad_type}/{run_name}")


def setup_repo_symlinks():
    """Symlink scripts from this repo directory into OUTPUT_BASE."""
    print(f"\nLinking repo scripts into {OUTPUT_BASE}:")
    for script in REPO_SCRIPTS:
        source = REPO_SCRIPTS_DIR / script
        dest   = OUTPUT_BASE / script
        if not source.exists():
            print(f"  Warning: {source} not found, skipping")
            continue
        create_symlink(source, dest)
        print(f"  {script}")


def main():
    normal_nc   = SOURCE_CASE / "cabauw_input.nc"
    no_gases_nc = SOURCE_CASE / "cabauw_input_no_gases.nc"

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    setup_repo_symlinks()

    # Experiment 1: aerosols off, gases still present
    setup_experiment("no_aerosols", input_nc=normal_nc)

    # Check that the no-gases NC was pre-generated
    if not no_gases_nc.exists():
        print(f"\nWARNING: {no_gases_nc} not found.")
        print("Run first:  python make_no_gases_nc.py")
        print("Skipping no_gases and dark_ocean experiments.")
        return

    # Experiment 2: no gases, no aerosols
    setup_experiment("no_gases", input_nc=no_gases_nc)

    # Experiment 3: no gases, no aerosols, dark ocean surface
    setup_experiment("dark_ocean", input_nc=no_gases_nc)

    n_dirs = 3 * N_REPS * 2
    print(f"\nSetup complete. {n_dirs} simulation directories created under:")
    print(f"  {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
