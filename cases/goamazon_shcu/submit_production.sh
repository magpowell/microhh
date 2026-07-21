#!/bin/bash
# Submit GoAmazon composite ShCu production as gpu_shared 2-GPU jobs (no
# idle GPUs; half-node charging; gpu_shared MaxWall is 48 h). This is
# CASS's own grid (512x512x256, dx=50m) and CASS's own runs (similar
# 13.9 h endtime, same physics complexity -- shallow-Cu, no deep tower)
# completed comfortably within a single ~22 h regular-queue window, so a
# 24 h shared-queue job should be ample for both RT modes -- no restart
# chain expected to be needed, but --time is generous and the chain script
# is still wired up in case a rep runs long.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/GOAMAZON_SHCU_LES/base"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/GOAMAZON_SHCU_LES/logs"

SHARED_ARGS=(--qos=shared --constraint="gpu&hbm80g" --gres=gpu:2
             --ntasks-per-node=2 --cpus-per-task=32)

# 2stream pair
sbatch "${SHARED_ARGS[@]}" \
    --time=24:00:00 \
    --job-name=shcu_2s \
    --export=ALL,SIM_DIRS="$EXP/2stream/rep_01:$EXP/2stream/rep_02" \
    "$SCRIPT_DIR/sbatch_runs.sh"
echo "Submitted goamazon_shcu 2stream pair (24 h shared)"

# raytracer pair + dependent restart chain
RT_JOB=$(sbatch --parsable "${SHARED_ARGS[@]}" \
    --time=24:00:00 \
    --job-name=shcu_rt \
    --export=ALL,SIM_DIRS="$EXP/raytracer/rep_01:$EXP/raytracer/rep_02" \
    "$SCRIPT_DIR/sbatch_runs.sh")
echo "Submitted goamazon_shcu raytracer pair (24 h shared): $RT_JOB"

sbatch "${SHARED_ARGS[@]}" \
    --time=24:00:00 \
    --job-name=shcu_rst \
    --dependency=afterany:$RT_JOB \
    --export=ALL,SIM_DIRS="$EXP/raytracer/rep_01:$EXP/raytracer/rep_02" \
    "$SCRIPT_DIR/sbatch_restart.sh"
echo "Submitted goamazon_shcu raytracer restart chain (afterany:$RT_JOB)"
