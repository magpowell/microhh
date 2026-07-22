#!/bin/bash
# Set up and submit arm97sd DEBUG runs on Empire AI Alpha.
# ARM SGP 27 June 1997 shallow-to-deep -- 16.5 h sim, 512x512x384.
# Debug uses 64x64 columns with the full forcing -- an end-to-end smoke test of
# the build, the input pipeline and the Slurm wiring before spending real time.
#
# NOTE on walltime: 30 min (the Perlmutter debug QOS cap) is NOT enough here.
# The raytracer is roughly 3x slower per step than 2stream, so DEBUG_WALLTIME
# defaults to 2 h. Requested walltime is charged against the account's shared
# GrpTRESMins, so do not inflate it further without reason.
set -euo pipefail

ACCOUNT=${ACCOUNT:-columbia}
PARTITION=${PARTITION:-columbia}
SCRATCH=${SCRATCH:-/mnt/lustre/columbia/$USER}
DEBUG_WALLTIME=${DEBUG_WALLTIME:-02:00:00}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROHH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
XR_PY=${XR_PY:-$HOME/miniforge3/envs/microhh/bin/python}

LOGS="$SCRATCH/ARM97SD_LES/logs"
mkdir -p "$LOGS"

echo "Setting up arm97sd debug runs (SCRATCH=$SCRATCH)..."
"$XR_PY" "$SCRIPT_DIR/setup_runs.py" --debug

# One GPU per job: single-GPU jobs backfill, which is where the short queue
# waits on this cluster come from. 12 CPUs/GPU (nodes are 96 cores / 8 GPUs).
COMMON=(--account="$ACCOUNT" --partition="$PARTITION"
        --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=12
        --time="$DEBUG_WALLTIME")

for RT in 2stream raytracer; do
    SIM_DIR="$SCRATCH/ARM97SD_LES/debug/$RT/rep_01"
    jid=$(sbatch --parsable "${COMMON[@]}" \
        --job-name="dbg_a97sd_${RT}" \
        --output="$LOGS/debug-${RT}-%j.out" \
        --error="$LOGS/debug-${RT}-%j.err" \
        --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
        "$SCRIPT_DIR/sbatch_runs_alpha.sh")
    echo "Submitted arm97sd debug $RT: job $jid"
done

echo
echo "Watch with:  squeue -u $USER"
echo "Logs in:     $LOGS"
