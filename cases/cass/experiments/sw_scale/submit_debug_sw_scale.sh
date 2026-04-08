#!/bin/bash
# Set up and submit debug run for sw_scale experiment (rep_01, 64x64 grid).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug sw_scale run..."
python "$SCRIPT_DIR/setup_sw_scale.py" --debug

# raytracer only; debug domain is small enough that hbm80g is not required
SIM_DIR="$SCRATCH/CASS_LES/debug/sw_scale/raytracer/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_sw_scale_rt \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug sw_scale raytracer"
