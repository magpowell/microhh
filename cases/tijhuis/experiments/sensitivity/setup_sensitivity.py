#!/usr/bin/env python3
"""
Setup sensitivity experiment for LES cumulus case 20140716_t03.

Experiments:
  1. seed_experiment: 5 random seed reps at full domain (512x512, 25.6km),
     rt + standard radiation.
  2. domain_experiment: 16 random seed reps at 1/4 domain (128x128, 6.4km),
     rt + standard radiation. 16 quarter-domains = same total area as 1 full domain.

All runs disable 3D dumps (swdump=0) and enable cross-sections.
"""

import configparser
from pathlib import Path

# === Paths ===
SOURCE_CASE = Path("/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2/20140716_t03")
OUTPUT_BASE = Path("/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS")
MICROHH_DIR = Path("/global/homes/m/mpowell/repos/microhh")

PATHS = {
    'microhh':       MICROHH_DIR / 'build_gpu' / 'microhh',
    'rrtmgp_data':   MICROHH_DIR / 'rte-rrtmgp-cpp' / 'rrtmgp-data',
    'cabauw_cases':  MICROHH_DIR / 'cases' / 'cabauw',
    'cross_to_nc':   MICROHH_DIR / 'python' / 'cross_to_nc.py',
    'microhh_tools': MICROHH_DIR / 'python' / 'microhh_tools.py',
}

RADIATION_FILES = {
    'cloud_coefficients_lw.nc': 'rrtmgp-clouds-lw.nc',
    'cloud_coefficients_sw.nc': 'rrtmgp-clouds-sw.nc',
    'coefficients_lw.nc':       'rrtmgp-gas-lw-g128.nc',
    'coefficients_sw.nc':       'rrtmgp-gas-sw-g112.nc',
}

LSM_FILES = ['van_genuchten_parameters.nc']

RADIATION_CONFIG = {'rt': 'rrtmgp_rt', 'standard': 'rrtmgp'}

# Scripts to symlink from repo into OUTPUT_BASE
REPO_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = [
    'setup_sensitivity.py',
    'sbatch_sensitivity.sh',
    'sbatch_restart.sh',
    'submit_all.sh',
    'submit_debug.sh',
    'submit_restart.sh',
]


def create_symlink(source: Path, dest: Path):
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(source)


def setup_symlinks(sim_dir: Path):
    """Create all symlinks needed in a simulation directory."""
    create_symlink(PATHS['microhh'], sim_dir / 'microhh')
    create_symlink(SOURCE_CASE / 'rt' / 'cabauw_input.nc', sim_dir / 'cabauw_input.nc')

    for dest_name, source_name in RADIATION_FILES.items():
        create_symlink(PATHS['rrtmgp_data'] / source_name, sim_dir / dest_name)

    for lsm_file in LSM_FILES:
        create_symlink(PATHS['cabauw_cases'] / lsm_file, sim_dir / lsm_file)

    create_symlink(PATHS['cross_to_nc'], sim_dir / 'cross_to_nc.py')
    create_symlink(PATHS['microhh_tools'], sim_dir / 'microhh_tools.py')


def create_ini(rad_type, rndseed, itot, jtot, xsize, ysize, out_path):
    """Create a modified cabauw.ini for a sensitivity run."""
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(SOURCE_CASE / 'cabauw.ini')

    config.set('radiation', 'swradiation', RADIATION_CONFIG[rad_type])
    config.set('grid', 'itot', str(itot))
    config.set('grid', 'jtot', str(jtot))
    config.set('grid', 'xsize', str(xsize))
    config.set('grid', 'ysize', str(ysize))
    config.set('fields', 'rndseed', str(rndseed))
    config.set('dump', 'swdump', '0')
    config.set('cross', 'swcross', '1')
    config.set('aerosol', 'swaerosol', 'false')
    config.set('aerosol', 'swtimedep', 'false')

    with open(out_path, 'w') as f:
        config.write(f)


def setup_experiment(name, n_reps, itot, jtot, xsize, ysize, seed_start=1):
    """Set up one experiment (seed or domain) for both radiation types."""
    print(f"\n{'='*60}")
    print(f"Setting up: {name} ({itot}x{jtot}, {xsize}x{ysize}m, {n_reps} reps)")
    print(f"{'='*60}")

    for rad_type in ['rt', 'standard']:
        for i in range(n_reps):
            seed = seed_start + i
            if name == 'seed_experiment':
                run_name = f"seed_{seed}"
            else:
                run_name = f"rep_{seed:02d}"

            sim_dir = OUTPUT_BASE / name / rad_type / run_name
            sim_dir.mkdir(parents=True, exist_ok=True)

            create_ini(rad_type, seed, itot, jtot, xsize, ysize,
                       sim_dir / 'cabauw.ini')
            setup_symlinks(sim_dir)
            print(f"  {rad_type}/{run_name}")


def setup_repo_symlinks():
    """Create symlinks in OUTPUT_BASE pointing to scripts in the repo."""
    print(f"\nLinking repo scripts into {OUTPUT_BASE}:")
    for script in REPO_SCRIPTS:
        source = REPO_SCRIPTS_DIR / script
        dest = OUTPUT_BASE / script
        if not source.exists():
            print(f"  Warning: {source} not found, skipping")
            continue
        create_symlink(source, dest)
        print(f"  {script} -> {source}")


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Symlink scripts from repo into scratch
    setup_repo_symlinks()

    # Experiment 1: vary random seed at full domain (8 reps, fills 2 nodes x 4 GPUs)
    setup_experiment('seed_experiment', n_reps=8,
                     itot=512, jtot=512, xsize=25600, ysize=25600)

    # Experiment 2: 1/4 domain with 16 reps (same total area as 1 full domain)
    setup_experiment('domain_experiment', n_reps=16,
                     itot=128, jtot=128, xsize=6400, ysize=6400)

    print(f"\nSetup complete. {16 + 32} simulation directories created under:")
    print(f"  {OUTPUT_BASE}")


if __name__ == '__main__':
    main()
