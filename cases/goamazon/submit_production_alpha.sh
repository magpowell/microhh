#!/bin/bash
# Submit goamazon PRODUCTION on Empire AI Alpha.
# GoAmazon day 278 single pulse -- 12 h sim, 512x512x256, dx=200 m.
#
# One single-GPU job per (RT mode, rep), not node-packed groups:
#   1. Alpha has 8 GPUs/node (Perlmutter had 4) and no gpu_shared-style QOS, so
#      a whole-node request idles GPUs as soon as the fastest run finishes.
#   2. Single-GPU jobs backfill; measured 1-GPU waits here are ~30 min median.
#
# Raytracer took 35-90 h per rep on Perlmutter A100s. WALLTIME defaults to 48 h
# with a chained restart per raytracer rep; a rep that overruns resumes from its
# last hourly checkpoint rather than being lost.
#
# Grid note: keep goamazon on the SAME grid as the completed Perlmutter reps
# (dx200_k256) unless deliberately starting a resolution-sensitivity variant --
# changing it breaks comparability with reps 01-04 already finished there.
#
# Cost note: the columbia account shares a GrpTRESMins cap charged on REQUESTED
# walltime, not used. Do not raise WALLTIME "just in case" -- it wastes shared
# budget and hurts backfill.
#
# GPU choice: the existing grid fits an H100's 80 GB (same as the A100s this was
# tuned for), so no --gres type filter -- constraining to the 6 H200 nodes would
# only lengthen the queue. Set GPU_TYPE=nvidia_h200 to force H200 for a
# larger-grid variant.
set -euo pipefail

ACCOUNT=${ACCOUNT:-columbia}
PARTITION=${PARTITION:-columbia}
SCRATCH=${SCRATCH:-/mnt/lustre/columbia/$USER}
NREPS=${NREPS:-4}
WALLTIME=${WALLTIME:-48:00:00}
TS_WALLTIME=${TS_WALLTIME:-16:00:00}
GPU_TYPE=${GPU_TYPE:-}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROHH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
XR_PY=${XR_PY:-$HOME/miniforge3/envs/microhh/bin/python}

EXP="$SCRATCH/GOAMAZON_LES/base"
LOGS="$SCRATCH/GOAMAZON_LES/logs"
mkdir -p "$LOGS"

if [[ -n "$GPU_TYPE" ]]; then GRES="gpu:${GPU_TYPE}:1"; else GRES="gpu:1"; fi

echo "Setting up goamazon production (SCRATCH=$SCRATCH, NREPS=$NREPS)..."
"$XR_PY" "$SCRIPT_DIR/setup_runs.py"

BASE=(--account="$ACCOUNT" --partition="$PARTITION"
      --gres="$GRES" --ntasks-per-node=1 --cpus-per-task=12)

for RT in 2stream raytracer; do
    # 2stream is much shorter; give it its own (smaller) walltime.
    if [[ "$RT" == "2stream" ]]; then W="$TS_WALLTIME"; else W="$WALLTIME"; fi

    for i in $(seq -w 1 "$NREPS"); do
        SIM_DIR="$EXP/$RT/rep_$i"
        [[ -d "$SIM_DIR" ]] || { echo "  skip $RT rep_$i (no dir)"; continue; }

        jid=$(sbatch --parsable "${BASE[@]}" --time="$W" \
            --job-name="goam_${RT}_$i" \
            --output="$LOGS/${RT}-rep${i}-%j.out" \
            --error="$LOGS/${RT}-rep${i}-%j.err" \
            --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
            "$SCRIPT_DIR/sbatch_runs_alpha.sh")
        echo "Submitted goamazon $RT rep_$i: job $jid  (walltime $W)"

        # Chain a restart for the long raytracer reps. afterany: the chain must
        # also run when the parent hits the wall, which is the case it exists for.
        if [[ "$RT" == "raytracer" ]]; then
            rid=$(sbatch --parsable "${BASE[@]}" --time="$WALLTIME" \
                --job-name="goam_rst_$i" \
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
