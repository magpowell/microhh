#!/bin/bash
# Set up and submit GoAmazon composite ShCu DEBUG runs on Empire AI Alpha.
# 128x128 at the same dx=50 m as production, one single-GPU job per RT mode.
#
# Run this before any production submission: it is the end-to-end smoke test
# for the Alpha build (sm_90 binary, netCDF/HDF5 stack, mamba input pipeline).
set -euo pipefail

ACCOUNT=${ACCOUNT:-columbia}
PARTITION=${PARTITION:-columbia}
SCRATCH=${SCRATCH:-/mnt/lustre/columbia/$USER}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROHH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
XR_PY=${XR_PY:-$HOME/miniforge3/envs/microhh/bin/python}

LOGS="$SCRATCH/GOAMAZON_SHCU_LES/logs"
mkdir -p "$LOGS"

echo "Setting up debug runs (SCRATCH=$SCRATCH)..."
"$XR_PY" "$SCRIPT_DIR/setup_runs.py" --debug

# 12 CPUs per GPU: Alpha nodes are 96 cores / 8 GPUs. Requesting a single GPU
# (rather than packing a whole node) is deliberate -- single-GPU jobs backfill,
# which is where the short queue waits on this cluster come from.
COMMON=(--account="$ACCOUNT" --partition="$PARTITION"
        --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=12
        --time=00:30:00)

for RT in 2stream raytracer; do
    SIM_DIR="$SCRATCH/GOAMAZON_SHCU_LES/debug/$RT/rep_01"
    jid=$(sbatch --parsable "${COMMON[@]}" \
        --job-name="dbg_shcu_${RT}" \
        --output="$LOGS/debug-${RT}-%j.out" \
        --error="$LOGS/debug-${RT}-%j.err" \
        --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
        "$SCRIPT_DIR/sbatch_runs_alpha.sh")
    echo "Submitted debug $RT: job $jid  ($SIM_DIR)"
done

echo
echo "Watch with:  squeue -u $USER"
echo "Logs in:     $LOGS"
