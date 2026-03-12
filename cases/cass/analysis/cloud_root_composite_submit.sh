#!/bin/bash
# Submit cloud-root composite prep jobs for all (expt, rt, rep) combinations.
#
# Usage:
#   bash cloud_root_composite_submit.sh [--dry-run] [--expts base,no_aerosols,...]
#
# By default submits all standard experiments. Use --expts to restrict.
# --dry-run prints the sbatch commands without submitting.
#
# Prerequisites per rep directory:
#   3d_to_nc.py must have already been run (see submit_3d_to_nc.sh).

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
COMPOSITE_ROOT="$LES_ROOT/analysis/cloud_root_composite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/cloud_root_composite_prep.sbatch"

DRY_RUN=false
EXPTS_ARG=""

for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        --expts=*)  EXPTS_ARG="${arg#*=}" ;;
        *)          echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$LES_ROOT/logs"
mkdir -p "$COMPOSITE_ROOT"

# ── Helper ────────────────────────────────────────────────────────────────────
submit_one() {
    local run_dir="$1"
    local rel_path="$2"   # e.g. base/2stream/rep_01

    if [[ ! -d "$run_dir" ]]; then
        echo "  SKIP (dir not found): $run_dir"
        return
    fi
    # Check that 3d_to_nc output exists
    if [[ ! -f "$run_dir/thl.nc" ]]; then
        echo "  SKIP (no thl.nc — run 3d_to_nc.py first): $run_dir"
        return
    fi

    local out_dir="$COMPOSITE_ROOT/$rel_path"
    mkdir -p "$out_dir"

    if $DRY_RUN; then
        echo "  [dry] sbatch RUN_DIR=$run_dir OUTPUT_DIR=$out_dir"
    else
        sbatch \
            --export=ALL,RUN_DIR="$run_dir",OUTPUT_DIR="$out_dir" \
            "$SBATCH_SCRIPT"
        echo "  Submitted: $rel_path"
    fi
}

# ── Simple 2-RT experiments (4 reps × 2 RT) ──────────────────────────────────
simple_expts=(base no_aerosols no_aerosols_zero_wind)

for expt in "${simple_expts[@]}"; do
    if [[ -n "$EXPTS_ARG" ]] && [[ "$EXPTS_ARG" != *"$expt"* ]]; then
        continue
    fi
    echo "--- $expt ---"
    for rt in 2stream raytracer; do
        for rep in 01 02 03 04; do
            run_dir="$LES_ROOT/$expt/$rt/rep_$rep"
            # base lives under top-level; experiments under experiments/
            if [[ ! -d "$run_dir" ]]; then
                run_dir="$LES_ROOT/experiments/$expt/$rt/rep_$rep"
            fi
            submit_one "$run_dir" "$expt/$rt/rep_$rep"
        done
    done
done

# ── cs_veg sweep ──────────────────────────────────────────────────────────────
cs_veg_vals=(0 42000 420000 4200000 42000000)
expt="cs_veg"
if [[ -z "$EXPTS_ARG" ]] || [[ "$EXPTS_ARG" == *"$expt"* ]]; then
    echo "--- $expt ---"
    for val in "${cs_veg_vals[@]}"; do
        for rt in 2stream raytracer; do
            for rep in 01 02 03 04; do
                run_dir="$LES_ROOT/experiments/$expt/cs_veg_$val/$rt/rep_$rep"
                submit_one "$run_dir" "$expt/cs_veg_$val/$rt/rep_$rep"
            done
        done
    done
fi

# ── soil_moisture sweep ───────────────────────────────────────────────────────
sm_vals=(0.1 0.2 0.3 0.4)
expt="soil_moisture"
if [[ -z "$EXPTS_ARG" ]] || [[ "$EXPTS_ARG" == *"$expt"* ]]; then
    echo "--- $expt ---"
    for val in "${sm_vals[@]}"; do
        # stored as theta_0p1, theta_0p2 etc.
        dir_val="theta_${val/./p}"
        for rt in 2stream raytracer; do
            for rep in 01 02 03 04; do
                run_dir="$LES_ROOT/experiments/$expt/$dir_val/$rt/rep_$rep"
                submit_one "$run_dir" "$expt/$dir_val/$rt/rep_$rep"
            done
        done
    done
fi

# ── mean_state_nudge (raytracer only) ─────────────────────────────────────────
expt="mean_state_nudge"
if [[ -z "$EXPTS_ARG" ]] || [[ "$EXPTS_ARG" == *"$expt"* ]]; then
    echo "--- $expt ---"
    # Iterate over any nudge_*s subdirs that exist
    for nudge_dir in "$LES_ROOT/experiments/$expt"/nudge_*s; do
        [[ -d "$nudge_dir" ]] || continue
        nudge_label="$(basename "$nudge_dir")"
        for rep in 01 02 03 04; do
            run_dir="$nudge_dir/raytracer/rep_$rep"
            submit_one "$run_dir" "$expt/$nudge_label/raytracer/rep_$rep"
        done
    done
fi

echo ""
echo "Done. Check $LES_ROOT/logs/ for output."
