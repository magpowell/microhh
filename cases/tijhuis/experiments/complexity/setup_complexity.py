#!/usr/bin/env python3
"""
Setup complexity-reduction experiments for LES cumulus case 20140325_t03.

Three experiments, each with 3 random-seed reps × {rt, standard} = 6 runs:
  1. no_aerosols      — aerosols off
  2. no_gases         — aerosols off + all gas mixing ratios zeroed in input NC
  3. prescribed_fluxes — aerosols off; surface fluxes prescribed from paper_results
                         ensemble mean H(t)/LE(t); tests 3D vs 1D heating rates

Output base: /pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325/

Before running this script, prepare the input files (run once):
    python fix_input_nc.py          # adds z + time_rad to timedep group
    python make_no_gases_nc.py
    python make_prescribed_flux_nc.py
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
    "cabauw_cases":  MICROHH_DIR / "misc",
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

# Per-experiment input NC (source ini is always SOURCE_CASE/rt/cabauw.ini for all)
EXPERIMENTS = {
    "no_aerosols":      SOURCE_CASE / "cabauw_input.nc",
    "no_gases":         SOURCE_CASE / "cabauw_input_no_gases.nc",
    "prescribed_fluxes": SOURCE_CASE / "cabauw_input_prescribed_flux.nc",
}

# Scripts to symlink from this directory into OUTPUT_BASE
REPO_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = [
    "setup_complexity.py",
    "make_no_gases_nc.py",
    "make_prescribed_flux_nc.py",
    "fix_input_nc.py",
    "sbatch_complexity.sh",
    "submit_complexity.sh",
    "sbatch_restart_complexity.sh",
    "submit_restart_complexity.sh",
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
    All experiments have aerosols off.
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

    # Aerosols off for all experiments (explicit, even though source already has this)
    config.set("aerosol", "swaerosol", "false")
    config.set("aerosol", "swtimedep", "false")

    # no_gases: remove timedeplist_gas entirely so MicroHH uses its default
    # empty list. Setting it to "" produces [""] in the parser, which triggers
    # an "Illegal string" exception in input_tools.h check_item().
    if experiment == "no_gases":
        config.remove_option("radiation", "timedeplist_gas")

    # prescribed_fluxes: bypass LSM with time-varying kinematic surface fluxes
    if experiment == "prescribed_fluxes":
        config.set("boundary", "swboundary",  "surface")
        config.set("boundary", "sbcbot[thl]", "flux")
        config.set("boundary", "sbcbot[qt]",  "flux")
        config.set("boundary", "sbot[thl]",   "0.")   # overridden by timedep at runtime
        config.set("boundary", "sbot[qt]",    "0.")   # overridden by timedep at runtime
        config.set("boundary", "swtimedep",   "1")
        config.set("boundary", "timedeplist", "thl_sbot,qt_sbot")
        config.set("boundary", "z0m", "0.0002")
        config.set("boundary", "z0h", "0.0002")

    with open(out_path, "w") as f:
        config.write(f)


def setup_experiment(experiment: str):
    """Set up one experiment across all radiation types and seeds."""
    print(f"\n{'='*60}")
    print(f"Setting up: {experiment}")
    print(f"{'='*60}")

    input_nc = EXPERIMENTS[experiment]
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
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    setup_repo_symlinks()

    skipped = []
    for experiment, input_nc in EXPERIMENTS.items():
        if not input_nc.exists():
            print(f"\nWARNING: {input_nc.name} not found, skipping {experiment}.")
            skipped.append(experiment)
            continue
        setup_experiment(experiment)

    n_done = len(EXPERIMENTS) - len(skipped)
    print(f"\nSetup complete. {n_done * N_REPS * 2} simulation directories created under:")
    print(f"  {OUTPUT_BASE}")
    if skipped:
        print(f"Skipped (missing input NC): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
