#!/usr/bin/env python3
"""
Setup script for CASS LES simulations.

This script sets up a scratch directory with the appropriate configuration
and symlinks for running CASS simulations with different physics options.

Usage:
    python setup_cass.py <config> [--scratch-dir DIR] [--rays-per-pixel N]

Configurations:
    forced    - Forced surface fluxes, no radiation or LSM
    2stream   - RRTMGP 2-stream radiation with LSM
    raytracer - RRTMGP-RT 3D ray tracer with LSM

Examples:
    python setup_cass.py forced
    python setup_cass.py 2stream --scratch-dir /pscratch/sd/m/mpowell/CASS_2stream
    python setup_cass.py raytracer --rays-per-pixel 512
"""

import argparse
import configparser
import os
import sys
from pathlib import Path


MICROHH_DIR = Path("/global/homes/m/mpowell/repos/microhh")

# Paths to various resources
PATHS = {
    'microhh': MICROHH_DIR / 'build_gpu' / 'microhh',
    'python_scripts': MICROHH_DIR / 'python',
    'rrtmgp_data': MICROHH_DIR / 'rte-rrtmgp-cpp' / 'rrtmgp-data',
    'rrtmgp_data_extra': MICROHH_DIR / 'rte-rrtmgp-cpp' / 'data',
    'cass': MICROHH_DIR / 'cases' / 'cass',
}

# Configuration file mapping
CONFIG_FILES = {
    'forced': 'cass_forced.ini',
    '2stream': 'cass_2stream.ini',
    'raytracer': 'cass_raytracer.ini',
}

# Python scripts to link
PYTHON_SCRIPTS = [
    'cass_utils.py',
    'cass_input.py',
    'cass_ls2d_input.py',
    'download_cams.py',
    'save_era5_profiles.py',
    '3d_to_nc.py',
    'cross_to_nc.py',
    'cleanup_run.py',
    'download_cass_data.sh',
    'sbatch_era5_download.sh',
]


# Radiation coefficient files
RADIATION_FILES = {
    'cloud_coefficients_lw.nc': 'rrtmgp-clouds-lw.nc',
    'cloud_coefficients_sw.nc': 'rrtmgp-clouds-sw.nc',
    'coefficients_lw.nc': 'rrtmgp-gas-lw-g128.nc',
    'coefficients_sw.nc': 'rrtmgp-gas-sw-g112.nc',
}

# LSM files
LSM_FILES = [
    'van_genuchten_parameters.nc',
]


def merge_ini_files(base_path: Path, overlay_path: Path) -> configparser.ConfigParser:
    """
    Merge two INI files, with overlay taking precedence.

    Handles the microhh INI format which uses:
    - Comments starting with # or ;
    - Keys with brackets like sbot[thl]
    """
    config = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=('#', ';'),
        inline_comment_prefixes=('#', ';'),
    )
    # Preserve case of keys
    config.optionxform = str

    # Read base first, then overlay (overlay overwrites base)
    config.read([base_path, overlay_path])

    return config


def write_ini_file(config: configparser.ConfigParser, output_path: Path,
                   rays_per_pixel: int = None, grid_size: int = None):
    """
    Write the merged config to a file.

    Optionally override rays_per_pixel for raytracer config.
    Also sets column coordinates based on grid size.
    """
    # Override rays_per_pixel if specified
    if rays_per_pixel is not None and config.has_section('radiation'):
        if config.get('radiation', 'swradiation', fallback='') == 'rrtmgp_rt':
            config.set('radiation', 'rays_per_pixel', str(rays_per_pixel))

    # Set column coordinates to center of domain
    if config.has_section('column') and config.has_section('grid'):
        try:
            xsize = float(config.get('grid', 'xsize'))
            ysize = float(config.get('grid', 'ysize'))
            config.set('column', 'coordinates[x]', str(xsize / 2))
            config.set('column', 'coordinates[y]', str(ysize / 2))
        except (configparser.NoOptionError, ValueError):
            pass

    with open(output_path, 'w') as f:
        config.write(f)

    print(f"  Created: {output_path}")


def create_symlink(source: Path, dest: Path, force: bool = True):
    """Create a symbolic link, optionally removing existing."""
    if dest.exists() or dest.is_symlink():
        if force:
            dest.unlink()
        else:
            print(f"  Skipping (exists): {dest.name}")
            return

    dest.symlink_to(source)
    print(f"  Linked: {dest.name} -> {source}")


def setup_scratch(config_name: str, scratch_dir: Path, rays_per_pixel: int = None):
    """
    Set up the scratch directory for a CASS simulation.

    Args:
        config_name: One of 'forced', '2stream', 'raytracer'
        scratch_dir: Path to scratch directory
        rays_per_pixel: Override for rays_per_pixel (raytracer only)
    """
    print(f"\nSetting up CASS '{config_name}' configuration in:")
    print(f"  {scratch_dir}\n")

    # Validate config name
    if config_name not in CONFIG_FILES:
        print(f"Error: Unknown configuration '{config_name}'")
        print(f"Valid options: {', '.join(CONFIG_FILES.keys())}")
        sys.exit(1)

    # Create scratch directory if needed
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Merge and write INI file
    print("Creating configuration file:")
    base_ini = PATHS['cass'] / 'cass_base.ini'
    overlay_ini = PATHS['cass'] / CONFIG_FILES[config_name]

    config = merge_ini_files(base_ini, overlay_ini)
    output_ini = scratch_dir / 'cass.ini'
    write_ini_file(config, output_ini, rays_per_pixel=rays_per_pixel)

    # Create symlinks
    print("\nCreating symlinks:")

    # microhh executable
    create_symlink(PATHS['microhh'], scratch_dir / 'microhh')

    # Python scripts - some from cass2, some from python/
    for script in PYTHON_SCRIPTS:
        if (PATHS['cass'] / script).exists():
            source = PATHS['cass'] / script
        elif (PATHS['python_scripts'] / script).exists():
            source = PATHS['python_scripts'] / script
        else:
            print(f"  Warning: Could not find {script}")
            continue
        create_symlink(source, scratch_dir / script)

    # sbatch script
    sbatch_source = PATHS['cass'] / 'sbatch_cass_les.sh'
    if sbatch_source.exists():
        create_symlink(sbatch_source, scratch_dir / 'sbatch_cass_les.sh')

    # Radiation files (only for radiation configs)
    if config_name in ('2stream', 'raytracer'):
        print("\nLinking radiation coefficient files:")
        for dest_name, source_name in RADIATION_FILES.items():
            source = PATHS['rrtmgp_data'] / source_name
            if source.exists():
                create_symlink(source, scratch_dir / dest_name)
            else:
                print(f"  Warning: {source_name} not found")
        aerosol_optics = PATHS['rrtmgp_data_extra'] / 'aerosol_optics.nc'
        if aerosol_optics.exists():
            create_symlink(aerosol_optics, scratch_dir / 'aerosol_optics.nc')
        else:
            print(f"  Warning: aerosol_optics.nc not found")

    # LSM files (only for LSM configs)
    if config_name in ('2stream', 'raytracer'):
        print("\nLinking LSM files:")
        for lsm_file in LSM_FILES:
            source = PATHS['cass'] / lsm_file
            if source.exists():
                create_symlink(source, scratch_dir / lsm_file)
            else:
                print(f"  Warning: {lsm_file} not found")

    print("\n" + "*"*60)
    print("Setup complete!")
    print("*"*60)
    print(f"\nConfiguration: {config_name}")
    if config_name == 'raytracer' and rays_per_pixel:
        print(f"Rays per pixel: {rays_per_pixel}")

def main():
    parser = argparse.ArgumentParser(
        description='Set up CASS LES simulation in scratch directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'config',
        choices=['forced', '2stream', 'raytracer'],
        help='Configuration to use'
    )
    parser.add_argument(
        '--scratch-dir', '-d',
        type=Path,
        default=None,
        help='Scratch directory (default: $SCRATCH/CASS_LES_<config>)'
    )
    parser.add_argument(
        '--rays-per-pixel', '-r',
        type=int,
        default=None,
        help='Override rays_per_pixel for raytracer config (default: 256)'
    )

    args = parser.parse_args()

    # Set default scratch directory
    if args.scratch_dir is None:
        scratch_base = os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell')
        args.scratch_dir = Path(scratch_base) / f'CASS_LES_{args.config}'

    # Warn if rays-per-pixel used with non-raytracer config
    if args.rays_per_pixel and args.config != 'raytracer':
        print(f"Warning: --rays-per-pixel ignored for '{args.config}' config")

    setup_scratch(args.config, args.scratch_dir, args.rays_per_pixel)


if __name__ == '__main__':
    main()
