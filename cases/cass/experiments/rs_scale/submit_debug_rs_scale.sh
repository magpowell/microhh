#!/bin/bash
# Set up and submit debug runs for rs_scale experiment (rep_01, 64x64, rs_scale=0.25 only).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug rs_scale runs..."
python "$SCRIPT_DIR/setup_rs_scale.py" --debug --values 0.25

# 2stream
SIM_DIR="$SCRATCH/CASS_LES/debug/rs_scale/rs_0p25/2stream/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_rs_2s \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug rs_scale 2stream"

# raytracer
SIM_DIR="$SCRATCH/CASS_LES/debug/rs_scale/rs_0p25/raytracer/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_rs_rt \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug rs_scale raytracer"
