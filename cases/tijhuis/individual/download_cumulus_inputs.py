#!/usr/bin/env python3
"""
Download and set up the cumulus cases input files.

Downloads the input data from Zenodo, extracts it to scratch, and creates
symlinks to the run scripts from the repo.

Usage:
    python download_cumulus_inputs.py
"""

import subprocess
import sys
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/12783821/files/inputfiles_cumulus_cases.tar.gz?download=1"
SCRATCH_DIR = Path("/pscratch/sd/m/mpowell/inputfiles_cumulus_cases_2")
REPO_SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "run_cumulus_les.sh",
    "sbatch_cumulus_les.sh",
    "setup_cumulus_cases.py",
]


def main():
    # Create the scratch directory
    if SCRATCH_DIR.exists():
        print(f"Directory already exists: {SCRATCH_DIR}")
        print("Remove it first if you want a fresh download.")
        sys.exit(1)

    SCRATCH_DIR.mkdir(parents=True)
    print(f"Created: {SCRATCH_DIR}")

    # Download the tarball
    tarball = SCRATCH_DIR / "inputfiles_cumulus_cases.tar.gz"
    print(f"\nDownloading from Zenodo...")
    subprocess.run(
        ["curl", "-L", "-o", str(tarball), ZENODO_URL],
        check=True,
    )
    print(f"Downloaded: {tarball}")

    # Extract - the tarball contains an inputfiles_cumulus_cases/ directory,
    # so we strip that top-level component to extract directly into SCRATCH_DIR
    print(f"\nExtracting...")
    subprocess.run(
        ["tar", "xzf", str(tarball), "-C", str(SCRATCH_DIR), "--strip-components=1"],
        check=True,
    )
    print("Extraction complete.")

    # Remove the tarball
    tarball.unlink()
    print(f"Removed tarball.")

    # Create symlinks to repo scripts
    print(f"\nCreating symlinks to repo scripts:")
    for script in SCRIPTS:
        source = REPO_SCRIPTS_DIR / script
        dest = SCRATCH_DIR / script
        if not source.exists():
            print(f"  Warning: {source} not found, skipping")
            continue
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(source)
        print(f"  {dest.name} -> {source}")

    print(f"\nDone! Input files are in: {SCRATCH_DIR}")
    print(f"\nNext steps:")
    print(f"  cd {SCRATCH_DIR}")
    print(f"  python setup_cumulus_cases.py -d 20140519_t03 20140716_t03 ...")


if __name__ == "__main__":
    main()
