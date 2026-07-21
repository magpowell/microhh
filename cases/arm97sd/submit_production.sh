#!/bin/bash
# Submit ARM97SD production as gpu_shared 2-GPU jobs (no idle GPUs; half-node
# charging; gpu_shared MaxWall is 48 h):
#   - 2stream rep_01/02:   one 16 h job (completes; ~11 h expected)
#   - raytracer rep_01/02: one 48 h job + dependent 48 h restart chain
# Estimated raytracer total: 60-90 h (16.5 h sim; convective MC calls are
# ~7-8 min each on this grid, from the GoAmazon sibling case).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/ARM97SD_LES/base"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/ARM97SD_LES/logs"

SHARED_ARGS=(--qos=shared --constraint="gpu&hbm80g" --gres=gpu:2
             --ntasks-per-node=2 --cpus-per-task=32)

# 2stream pair
sbatch "${SHARED_ARGS[@]}" \
    --time=16:00:00 \
    --job-name=a97sd_2s \
    --export=ALL,SIM_DIRS="$EXP/2stream/rep_01:$EXP/2stream/rep_02" \
    "$SCRIPT_DIR/sbatch_runs.sh"
echo "Submitted arm97sd 2stream pair (16 h shared)"

# raytracer pair + dependent restart chain
RT_JOB=$(sbatch --parsable "${SHARED_ARGS[@]}" \
    --time=48:00:00 \
    --job-name=a97sd_rt \
    --export=ALL,SIM_DIRS="$EXP/raytracer/rep_01:$EXP/raytracer/rep_02" \
    "$SCRIPT_DIR/sbatch_runs.sh")
echo "Submitted arm97sd raytracer pair (48 h shared): $RT_JOB"

sbatch "${SHARED_ARGS[@]}" \
    --time=48:00:00 \
    --job-name=a97sd_rst \
    --dependency=afterany:$RT_JOB \
    --export=ALL,SIM_DIRS="$EXP/raytracer/rep_01:$EXP/raytracer/rep_02" \
    "$SCRIPT_DIR/sbatch_restart.sh"
echo "Submitted arm97sd raytracer restart chain (afterany:$RT_JOB)"
