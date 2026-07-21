#!/bin/bash
# Set up and submit debug runs for wind_sun experiment.
# Single U=5 m/s, anti-solar tracking; 64x64 grid, rep_01 only.
# Usage: submit_debug_wind_sun.sh
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug wind_sun runs ..."
python "$SCRIPT_DIR/setup_wind_sun.py" --debug

# 2stream
SIM_DIR="$SCRATCH/CASS_LES/debug/wind_sun/2stream/rep_01"
sbatch \
    --constraint=gpu \
    --job-name="dbg_ws_2s" \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug wind_sun 2stream"

# raytracer
SIM_DIR="$SCRATCH/CASS_LES/debug/wind_sun/raytracer/rep_01"
sbatch \
    --constraint="gpu&hbm80g" \
    --job-name="dbg_ws_rt" \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug wind_sun raytracer"
