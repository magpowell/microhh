#!/bin/bash
# Submit GoAmazon composite ShCu PRODUCTION on Empire AI Alpha.
#
# One single-GPU job per (RT mode, rep), not node-packed pairs. Two reasons:
#
#  1. Alpha has 8 GPUs/node (Perlmutter had 4) and no `gpu_shared`-style QOS,
#     so a whole-node request would idle 7 GPUs the moment the first run ends.
#     Handoff lesson "never size a whole-node job around the fastest job in the
#     mix" applies harder here, not less.
#  2. Single-GPU jobs backfill. Measured 1-GPU waits in this partition are
#     ~30 min median with p25 = 0; larger requests wait materially longer.
#
# Cost note: the columbia account has a shared GrpTRESMins cap of 600,000
# GPU-minutes, charged on REQUESTED walltime, not used. --time below is sized
# to the expected runtime plus margin rather than the 7-day maximum, on
# purpose. Raising it wastes shared budget and hurts backfill.
#
# This is CASS's own grid (512x512x256, dx=50 m) and CASS's own comparable runs
# finished inside ~22 h, so 24 h should complete both RT modes outright. The
# restart chain is wired up anyway in case a rep runs long.
set -euo pipefail

ACCOUNT=${ACCOUNT:-columbia}
PARTITION=${PARTITION:-columbia}
SCRATCH=${SCRATCH:-/mnt/lustre/columbia/$USER}
NREPS=${NREPS:-4}
WALLTIME=${WALLTIME:-24:00:00}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROHH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
XR_PY=${XR_PY:-$HOME/miniforge3/envs/microhh/bin/python}

EXP="$SCRATCH/GOAMAZON_SHCU_LES/base"
LOGS="$SCRATCH/GOAMAZON_SHCU_LES/logs"
mkdir -p "$LOGS"

echo "Setting up production runs (SCRATCH=$SCRATCH, NREPS=$NREPS)..."
"$XR_PY" "$SCRIPT_DIR/setup_runs.py"

# 12 CPUs per GPU: Alpha nodes are 96 cores / 8 GPUs.
# No --gres GPU type filter: this grid fits comfortably in an H100's 80 GB, so
# constraining to the 6 H200 nodes would only lengthen the queue for no gain.
COMMON=(--account="$ACCOUNT" --partition="$PARTITION"
        --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=12
        --time="$WALLTIME")

for RT in 2stream raytracer; do
    for i in $(seq -w 1 "$NREPS"); do
        SIM_DIR="$EXP/$RT/rep_$i"
        [[ -d "$SIM_DIR" ]] || { echo "  skip $RT rep_$i (no dir)"; continue; }

        jid=$(sbatch --parsable "${COMMON[@]}" \
            --job-name="shcu_${RT}_$i" \
            --output="$LOGS/${RT}-rep${i}-%j.out" \
            --error="$LOGS/${RT}-rep${i}-%j.err" \
            --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
            "$SCRIPT_DIR/sbatch_runs_alpha.sh")
        echo "Submitted $RT rep_$i: job $jid"

        # Raytracer is the long mode; chain one restart per rep so a rep that
        # overruns WALLTIME resumes from its last hourly checkpoint instead of
        # being lost. afterany: the chain must also run if the parent timed out.
        if [[ "$RT" == "raytracer" ]]; then
            rid=$(sbatch --parsable "${COMMON[@]}" \
                --job-name="shcu_rst_$i" \
                --dependency=afterany:"$jid" \
                --output="$LOGS/restart-rep${i}-%j.out" \
                --error="$LOGS/restart-rep${i}-%j.err" \
                --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
                "$SCRIPT_DIR/sbatch_restart_alpha.sh")
            echo "  chained restart: job $rid (afterany:$jid)"
        fi
    done
done

echo
echo "Watch with:  squeue -u $USER"
echo "Logs in:     $LOGS"
