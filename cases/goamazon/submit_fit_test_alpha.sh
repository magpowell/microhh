#!/bin/bash
# Raytracer grid fit test on Empire AI Alpha.
#
# Purpose (handoff s7): the Perlmutter production grid was memory-constrained to
# the A100's 80 GB, not chosen for science reasons. The empirical fit there
# measured raytracer memory at ~0.60 GB/Mcell, and dz=50 m at full domain
# (134 Mcells, 80.5 GB) came in at 98% of an A100 and was rejected as too risky.
#
# Alpha's H200s have 141 GB (measured 143,771 MiB on alphagpu20 -- note one
# Empire AI spec sheet wrongly says 80 GB). At the same 0.60 GB/Mcell that
# projects to ~235 Mcells, which would make the rejected dz=50 m grid land near
# 57% of the card instead of 98%. DO NOT trust that extrapolation: the constant
# was measured on A100/Perlmutter and architecture, driver and autotuner
# differences can shift it. That is what this test is for.
#
# Storage: effectively free. setup_runs.py --fit-test sets swdump=0 and
# swcross=0 and endtime=1800, so these runs write only stats + gpu_mem.log.
# Safe to run while the group scratch space is still pending.
#
# One GPU per grid candidate, run concurrently as independent jobs (Alpha has
# no gpu_shared equivalent, and single-GPU jobs backfill best).
#
# Afterwards, per directory:
#   max memory:  sort -t, -k2 -h <dir>/gpu_mem.log | tail -1
#   completion:  grep -c . <dir>/goamazon.out
set -euo pipefail

ACCOUNT=${ACCOUNT:-columbia}
PARTITION=${PARTITION:-columbia}
SCRATCH=${SCRATCH:-/mnt/lustre/columbia/$USER}
# H200 by default -- the whole point is measuring headroom beyond 80 GB.
# Set GPU_TYPE=nvidia_h100_80gb_hbm3 to reproduce the Perlmutter-class limit,
# or GPU_TYPE= (empty) to take whatever is free.
GPU_TYPE=${GPU_TYPE:-nvidia_h200}
WALLTIME=${WALLTIME:-01:00:00}
# Default ladder: current production grid, then progressively larger.
GRIDS=${GRIDS:-"dz67_k384 dz50_k512 dx200_i640 dz80_k320"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROHH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
XR_PY=${XR_PY:-$HOME/miniforge3/envs/microhh/bin/python}

FIT="$SCRATCH/GOAMAZON_LES/fit_test"
LOGS="$SCRATCH/GOAMAZON_LES/logs"
mkdir -p "$LOGS"

if [[ -n "$GPU_TYPE" ]]; then GRES="gpu:${GPU_TYPE}:1"; else GRES="gpu:1"; fi

echo "Setting up fit-test grids: $GRIDS"
for grid in $GRIDS; do
    "$XR_PY" "$SCRIPT_DIR/setup_runs.py" --fit-test --grid "$grid"
done

for grid in $GRIDS; do
    SIM_DIR="$FIT/$grid/raytracer"
    [[ -d "$SIM_DIR" ]] || { echo "  skip $grid (no dir)"; continue; }
    jid=$(sbatch --parsable \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --gres="$GRES" --ntasks-per-node=1 --cpus-per-task=12 \
        --time="$WALLTIME" \
        --job-name="fit_${grid}" \
        --output="$LOGS/fit-${grid}-%j.out" \
        --error="$LOGS/fit-${grid}-%j.err" \
        --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR",GPU_MEM_LOG=1 \
        "$SCRIPT_DIR/sbatch_runs_alpha.sh")
    echo "Submitted fit test $grid: job $jid  (${GPU_TYPE:-any GPU})"
done

echo
echo "When done:  for d in $FIT/*/raytracer; do"
echo "              echo -n \"\$(basename \$(dirname \$d)) \"; sort -t, -k2 -h \$d/gpu_mem.log | tail -1"
echo "            done"
