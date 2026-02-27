#!/usr/bin/env python3

import argparse
import configparser
import netCDF4 as nc
from pathlib import Path

MICROHH_DIR = Path("/global/homes/m/mpowell/repos/microhh")

# Paths to various resources
PATHS = {
    'microhh': MICROHH_DIR / 'build_gpu' / 'microhh',
    'wrk_dir': Path("/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2"),
    'rrtmgp_data': MICROHH_DIR / 'rte-rrtmgp-cpp' / 'rrtmgp-data',
    'cabauw_cases': MICROHH_DIR / 'cases' / 'cabauw',
}

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

# Radiation config mapping
RADIATION_CONFIG = {
    'rt': 'rrtmgp_rt',
    'standard': 'rrtmgp',
}


def edit_ini_file(ini_path: Path, output_path: Path, radiation_type: str):
    """
    Copy and modify the ini file:
    - Set swradiation based on radiation type (rrtmgp_rt or rrtmgp)
    - Add dump section for output
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(ini_path)

    # Set radiation type
    if not config.has_section('radiation'):
        config.add_section('radiation')
    config.set('radiation', 'swradiation', RADIATION_CONFIG[radiation_type])

    # Add and enable dump section
    if not config.has_section('dump'):
        config.add_section('dump')
    config.set('dump', 'swdump', '1')
    config.set('dump', 'sampletime', '3600')
    config.set('dump', 'dumplist', 'T,ql,qi,qt,thl,w')

    # Enable cross section
    if not config.has_section('cross'):
        config.add_section('cross')
    config.set('cross', 'swcross', '1')

    # disable aerosols
    if not config.has_section('aerosol'):
        config.add_section('aerosol')
    config.set('aerosol', 'swaerosol', 'false')
    config.set('aerosol', 'swtimedep', 'false')

    with open(output_path, 'w') as f:
        config.write(f)

    print(f"  Created ini: {output_path}")


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


def setup_simulation_dir(date_dir: Path, radiation_type: str):
    """Set up a simulation directory for a given date and radiation type."""

    sim_dir = date_dir / radiation_type
    sim_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Setting up: {sim_dir}")
    print(f"{'='*60}")

    # Copy and modify ini file
    ini_source = date_dir / 'cabauw.ini'
    if ini_source.exists():
        edit_ini_file(ini_source, sim_dir / 'cabauw.ini', radiation_type)
    else:
        print(f"  Warning: {ini_source} not found")

    # Copy cabauw_input.nc (unchanged)
    input_source = date_dir / 'cabauw_input.nc'
    input_dest = sim_dir / 'cabauw_input.nc'
    if input_source.exists():
        if input_dest.exists():
            input_dest.unlink()
        src = nc.Dataset(input_source, "r")
        dst = nc.Dataset(input_dest, "w", format="NETCDF4")

        def copy_attrs(src_obj, dst_obj):
            for attr in src_obj.ncattrs():
                dst_obj.setncattr(attr, src_obj.getncattr(attr))

        # Copy root dimension and variable
        dst.createDimension("z", src.dimensions["z"].size)
        copy_attrs(src, dst)
        z_var = dst.createVariable("z", "f8", ("z",))
        z_var[:] = src.variables["z"][:]

        # Copy each group
        for grp_name, grp in src.groups.items():
            dst_grp = dst.createGroup(grp_name)
            copy_attrs(grp, dst_grp)

            # Copy local dimensions
            for dim_name, dim in grp.dimensions.items():
                dst_grp.createDimension(dim_name, dim.size)

            # Add local z dimension if any variable uses it
            needs_z = any("z" in v.dimensions for v in grp.variables.values())
            if needs_z and "z" not in grp.dimensions:
                dst_grp.createDimension("z", src.dimensions["z"].size)

            # Also add time_rad = time_ls for timedep group (for timedeplist_gas support)
            if grp_name == "timedep" and "time_ls" in grp.dimensions:
                dst_grp.createDimension("time_rad", grp.dimensions["time_ls"].size)
                tr = dst_grp.createVariable("time_rad", "f8", ("time_rad",))
                tr[:] = grp.variables["time_ls"][:]

            # Copy variables
            for var_name, var in grp.variables.items():
                dst_var = dst_grp.createVariable(var_name, var.dtype, var.dimensions)
                copy_attrs(var, dst_var)
                dst_var[:] = var[:]

        src.close()
        dst.close()
        print(f"  Copied a modified: cabauw_input.nc")
    else:
        print(f"  Warning: {input_source} not found")

    # Create symlinks
    print("\nCreating symlinks:")

    # microhh executable
    if PATHS['microhh'].exists():
        create_symlink(PATHS['microhh'], sim_dir / 'microhh')
    else:
        print(f"  Warning: microhh executable not found at {PATHS['microhh']}")

    # Radiation coefficient files
    print("\nLinking radiation coefficient files:")
    for dest_name, source_name in RADIATION_FILES.items():
        source = PATHS['rrtmgp_data'] / source_name
        if source.exists():
            create_symlink(source, sim_dir / dest_name)
        else:
            print(f"  Warning: {source_name} not found at {source}")

    # LSM files
    print("\nLinking LSM files:")
    for lsm_file in LSM_FILES:
        source = PATHS['cabauw_cases'] / lsm_file
        if source.exists():
            create_symlink(source, sim_dir / lsm_file)
        else:
            print(f"  Warning: {lsm_file} not found at {source}")

    print(f"\nSetup complete for {sim_dir.name}!")


def main():
    parser = argparse.ArgumentParser(
        description='Set up cumulus LES simulations for different dates and radiation configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dates', '-d',
        nargs='+',
        type=str,
        required=True,
        help='Date directories to process (e.g., 20140519_t03 20140716_t03)'
    )
    parser.add_argument(
        '--radiation', '-r',
        nargs='+',
        choices=['rt', 'standard'],
        default=['rt', 'standard'],
        help='Radiation configurations to set up (default: both rt and standard)'
    )
    parser.add_argument(
        '--base-dir', '-b',
        type=Path,
        default=PATHS['wrk_dir'],
        help=f'Base directory containing date folders (default: {PATHS["wrk_dir"]})'
    )

    args = parser.parse_args()

    # Process each date
    for date in args.dates:
        date_dir = args.base_dir / date
        if not date_dir.exists():
            print(f"Warning: {date_dir} does not exist, skipping")
            continue

        # Set up each radiation configuration
        for rad_type in args.radiation:
            setup_simulation_dir(date_dir, rad_type)

    print("\n" + "*"*60)
    print("All simulations set up!")
    print("*"*60)


if __name__ == '__main__':
    main()
