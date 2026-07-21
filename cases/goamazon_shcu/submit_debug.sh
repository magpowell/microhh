#!/bin/bash
# Set up and submit debug runs (rep_01, 128x128 @ same dx=50m as production).
# Same pattern as cases/cass/base/submit_debug_base.sh -- one single-GPU
# debug job per RT mode.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/GOAMAZON_SHCU_LES/logs"

echo "Setting up debug runs..."
python "$SCRIPT_DIR/setup_runs.py" --debug

SIM_DIR="$SCRATCH/GOAMAZON_SHCU_LES/debug/2stream/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_shcu_2s \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SCRIPT_DIR/sbatch_debug.sh"
echo "Submitted debug 2stream ($(basename $SIM_DIR))"

SIM_DIR="$SCRATCH/GOAMAZON_SHCU_LES/debug/raytracer/rep_01"
sbatch \
    --constraint="gpu&hbm80g" \
    --job-name=dbg_shcu_rt \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SCRIPT_DIR/sbatch_debug.sh"
echo "Submitted debug raytracer ($(basename $SIM_DIR))"
