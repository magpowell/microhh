#!/bin/bash
# Submit all complexity experiment runs for 20140325_t03.
#
# 3 experiments × 2 radiation types × 3 seeds = 18 simulations.
# Seeds are batched by radiation type (3 tasks per node, 3 GPUs used).
#
# Usage:
#   ./submit_complexity.sh              # submit all experiments
#   ./submit_complexity.sh no_aerosols  # submit one experiment only

set -euo pipefail

BASE=/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325
SBATCH_SCRIPT=$BASE/sbatch_complexity.sh

TARGET_EXP="${1:-}"  # optional: restrict to one experiment

# Wall times match the sensitivity experiment convention
RT_TIME="20:00:00"
STD_TIME="10:00:00"

submit_batch() {
    local constraint=$1
    local time=$2
    local job_name=$3
    shift 3
    local dirs=("$@")
    local n=${#dirs[@]}

    local sim_dirs
    sim_dirs=$(IFS=':'; echo "${dirs[*]}")

    echo "  Submitting: $job_name ($n tasks, time=$time)"
    sbatch --constraint="$constraint" --time="$time" \
        --ntasks-per-node="$n" --gres=gpu:"$n" \
        --export=ALL,SIM_DIRS="$sim_dirs" \
        --job-name="$job_name" \
        "$SBATCH_SCRIPT"
}

# Short name map for job labels (Slurm job names capped at 32 chars)
declare -A SHORT_NAME=(
    [no_aerosols]="noaer"
    [no_gases]="nogas"
    [dark_ocean]="dark"
)

echo "=== Complexity experiments: 20140325_t03 ==="

for exp in no_aerosols no_gases dark_ocean; do
    [[ -n "$TARGET_EXP" && "$exp" != "$TARGET_EXP" ]] && continue

    short="${SHORT_NAME[$exp]}"
    echo ""
    echo "--- $exp ---"

    # RT batch (3 seeds, H100 GPU nodes for ray-tracing)
    submit_batch "gpu&hbm80g" "$RT_TIME" "CPLX_${short}_rt" \
        "$BASE/$exp/rt/seed_1" \
        "$BASE/$exp/rt/seed_2" \
        "$BASE/$exp/rt/seed_3"

    # Standard batch (3 seeds, regular GPU nodes)
    submit_batch "gpu" "$STD_TIME" "CPLX_${short}_std" \
        "$BASE/$exp/standard/seed_1" \
        "$BASE/$exp/standard/seed_2" \
        "$BASE/$exp/standard/seed_3"
done

echo ""
echo "Done. $([ -n "$TARGET_EXP" ] && echo "1 experiment" || echo "All 3 experiments") submitted (6 jobs × 3 GPUs each = 18 simulations)."
