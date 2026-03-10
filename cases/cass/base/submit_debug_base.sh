#!/bin/bash
# Set up and submit debug runs for the base case (rep_01 only, 64x64 grid).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug base runs..."
python "$SCRIPT_DIR/setup_base.py" --debug

# 2stream
SIM_DIR="$SCRATCH/CASS_LES/debug/base/2stream/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_base_2s \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug base 2stream ($(basename $SIM_DIR))"

# raytracer
SIM_DIR="$SCRATCH/CASS_LES/debug/base/raytracer/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_base_rt \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug base raytracer ($(basename $SIM_DIR))"
