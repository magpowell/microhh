#!/bin/bash
# submit_restart_complexity.sh — restart a complexity experiment from its
# latest common checkpoint.
#
# Usage:
#   ./submit_restart_complexity.sh <experiment> <rad_type> [options]
#
# Arguments:
#   experiment   : no_aerosols | no_gases | prescribed_fluxes
#   rad_type     : rt | standard
#
# Options:
#   --time T     : override restart time (seconds); default = auto-detect
#   --walltime W : SLURM wall time (default: 06:00:00)
#
# Examples:
#   ./submit_restart_complexity.sh no_aerosols standard
#   ./submit_restart_complexity.sh no_aerosols standard --time 61200 --walltime 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325"
SEEDS=(1 2 3)

# --- Parse arguments ---
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <experiment> <rad_type> [--time T] [--walltime W]"
    exit 1
fi

EXPERIMENT="$1"
RAD_TYPE="$2"
shift 2

RESTART_TIME=""
WALL_TIME="06:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time)    RESTART_TIME="$2"; shift 2 ;;
        --walltime) WALL_TIME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate experiment/rad_type
EXP_DIR="$OUTPUT_BASE/$EXPERIMENT/$RAD_TYPE"
if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: Directory not found: $EXP_DIR"
    exit 1
fi

# --- Auto-detect restart time (latest checkpoint common to all seeds) ---
if [[ -z "$RESTART_TIME" ]]; then
    COMMON_TIME=""
    for seed in "${SEEDS[@]}"; do
        seed_dir="$EXP_DIR/seed_$seed"
        # Find the latest time.XXXXXXX file in this seed dir
        latest=$(ls "$seed_dir"/time.[0-9]* 2>/dev/null | sort | tail -1)
        if [[ -z "$latest" ]]; then
            echo "ERROR: No checkpoint files in $seed_dir"
            exit 1
        fi
        t=$(basename "$latest" | sed 's/time\.//')
        t=$((10#$t))  # strip leading zeros → decimal
        if [[ -z "$COMMON_TIME" || $t -lt $COMMON_TIME ]]; then
            COMMON_TIME=$t
        fi
    done
    RESTART_TIME=$COMMON_TIME
fi

echo "=== Complexity restart: $EXPERIMENT / $RAD_TYPE ==="
echo "  Restart time : ${RESTART_TIME}s"
echo "  Wall time    : $WALL_TIME"

# Validate all restart files exist
for seed in "${SEEDS[@]}"; do
    rf="$EXP_DIR/seed_$seed/time.$(printf '%07d' "$RESTART_TIME")"
    if [[ ! -f "$rf" ]]; then
        echo "ERROR: Missing restart file: $rf"
        exit 1
    fi
done

# --- Patch starttime in each seed's ini ---
for seed in "${SEEDS[@]}"; do
    ini="$EXP_DIR/seed_$seed/cabauw.ini"
    sed -i "s/^starttime = .*/starttime = ${RESTART_TIME}/" "$ini"
    echo "  Patched starttime=$RESTART_TIME in seed_$seed/cabauw.ini"
done

# --- Build SIM_DIRS string ---
SIM_DIRS_LIST=()
for seed in "${SEEDS[@]}"; do
    SIM_DIRS_LIST+=("$EXP_DIR/seed_$seed")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

# --- Short name for job ---
declare -A SHORT_NAME=(
    [no_aerosols]="noaer"
    [no_gases]="nogas"
    [prescribed_fluxes]="pflux"
)
JOB_NAME="CPLX_${SHORT_NAME[$EXPERIMENT]}_${RAD_TYPE:0:3}_RST"

# --- Submit ---
sbatch \
    --qos=regular \
    --constraint="gpu&hbm80g" \
    --nodes=1 \
    --ntasks-per-node=3 \
    --gres=gpu:3 \
    --cpus-per-task=16 \
    --time="$WALL_TIME" \
    --job-name="$JOB_NAME" \
    --output="$OUTPUT_BASE/mhh-restart-%j.out" \
    --error="$OUTPUT_BASE/mhh-restart-%j.err" \
    --account=m1266 \
    --mail-user=mp4257@columbia.edu \
    --mail-type=ALL \
    --export=SIM_DIRS="$SIM_DIRS",RESTART_TIME="$RESTART_TIME" \
    "$SCRIPT_DIR/sbatch_restart_complexity.sh"

echo "Submitted restart job: $JOB_NAME"
