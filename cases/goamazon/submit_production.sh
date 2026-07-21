#!/bin/bash
# Submit GoAmazon production: ONE 36 h job on one hbm80g node running
# 2stream rep_01/02 + raytracer rep_01/02 (one GPU each). If the raytracers
# hit the wall, chain with:
#   sbatch --qos=regular --constraint="gpu&hbm80g" --time=36:00:00 \
#       --job-name=goam_rst \
#       --export=ALL,SIM_DIRS="<raytracer dirs>" sbatch_restart.sh
# (gpu_regular MaxWall is 48 h.)
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/GOAMAZON_LES/base"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/GOAMAZON_LES/logs"

SIM_DIRS="$EXP/2stream/rep_01:$EXP/2stream/rep_02:$EXP/raytracer/rep_01:$EXP/raytracer/rep_02"

sbatch \
    --qos=regular \
    --constraint="gpu&hbm80g" \
    --time=36:00:00 \
    --job-name=goam_prod \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_runs.sh"
echo "Submitted goamazon production (2stream rep01/02 + raytracer rep01/02, 36 h)"
