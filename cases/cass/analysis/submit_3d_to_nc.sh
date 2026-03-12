#!/bin/bash
# Run 3d_to_nc.py for all rep directories of a given experiment.
#
# This converts MicroHH binary field dumps to NetCDF, which is required
# before cloud_root_composite_prep.py can run.
#
# Usage:
#   bash submit_3d_to_nc.sh --expt base [--rt 2stream] [--dry-run]
#   bash submit_3d_to_nc.sh --expt cs_veg --val 42000 [--rt raytracer]
#
# With no --rt, processes both 2stream and raytracer.
# With no --val, processes the experiment root directly (for simple experiments).
#
# The script submits one small CPU batch job per rep (fast I/O, ~30 min each).
# Variables converted: thl qt ql w
#
# Prerequisite: simulations must be complete (microhh run finished).

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
LOG_DIR="$LES_ROOT/logs"

EXPT=""
RT_LIST=(2stream raytracer)
VAL=""
DRY_RUN=false
VARIABLES="thl qt ql w"

for arg in "$@"; do
    case $arg in
        --expt=*)   EXPT="${arg#*=}" ;;
        --rt=*)     RT_LIST=("${arg#*=}") ;;
        --val=*)    VAL="${arg#*=}" ;;
        --dry-run)  DRY_RUN=true ;;
        *)          echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

[[ -z "$EXPT" ]] && { echo "Error: --expt required"; exit 1; }
mkdir -p "$LOG_DIR"

# Build experiment root path
if [[ -n "$VAL" ]]; then
    EXPT_ROOT="$LES_ROOT/experiments/$EXPT/$VAL"
else
    EXPT_ROOT="$LES_ROOT/$EXPT"
    [[ -d "$EXPT_ROOT" ]] || EXPT_ROOT="$LES_ROOT/experiments/$EXPT"
fi

echo "3d_to_nc  expt=$EXPT  root=$EXPT_ROOT"

for rt in "${RT_LIST[@]}"; do
    for rep in 01 02 03 04; do
        run_dir="$EXPT_ROOT/$rt/rep_$rep"
        if [[ ! -d "$run_dir" ]]; then
            echo "  SKIP (not found): $run_dir"
            continue
        fi
        # Already done?
        if [[ -f "$run_dir/thl.nc" ]]; then
            echo "  SKIP (thl.nc exists): $run_dir"
            continue
        fi
        if $DRY_RUN; then
            echo "  [dry] sbatch 3d_to_nc  $rt/rep_$rep"
            continue
        fi
        sbatch \
            --qos=regular \
            --constraint=cpu \
            --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
            --time=01:00:00 \
            --job-name=3d_to_nc \
            --output="$LOG_DIR/3d_to_nc-%j.out" \
            --error="$LOG_DIR/3d_to_nc-%j.err" \
            -A m1266 \
            --export=ALL,RUN_DIR="$run_dir",VARS="$VARIABLES" \
            --wrap='cd "$RUN_DIR" && module load conda && conda activate xr_env && python 3d_to_nc.py -v $VARS'
        echo "  Submitted 3d_to_nc: $rt/rep_$rep"
    done
done
