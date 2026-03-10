#!/bin/bash
# Set up and submit debug runs for no_aerosols_zero_wind experiment (rep_01, 64x64 grid).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug no_aerosols_zero_wind runs..."
python "$SCRIPT_DIR/setup_no_aerosols_zero_wind.py" --debug

# 2stream
SIM_DIR="$SCRATCH/CASS_LES/debug/no_aerosols_zero_wind/2stream/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_nazw_2s \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug no_aerosols_zero_wind 2stream"

# raytracer
SIM_DIR="$SCRATCH/CASS_LES/debug/no_aerosols_zero_wind/raytracer/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_nazw_rt \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug no_aerosols_zero_wind raytracer"
