#!/bin/bash
# Submit all sensitivity experiment jobs, 4 simulations per node.
# Total: 12 jobs (42 simulations packed 4-per-node).

BASE=/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS
SBATCH_SCRIPT=$BASE/sbatch_sensitivity.sh

submit_batch() {
    local constraint=$1
    local time=$2
    local job_name=$3
    shift 3
    local dirs=("$@")
    local n=${#dirs[@]}

    # Join dirs with colons (colons avoid conflict with sbatch --export comma syntax)
    local sim_dirs
    sim_dirs=$(IFS=':'; echo "${dirs[*]}")

    echo "  Submitting: $job_name ($n tasks, time=$time)"
    sbatch --constraint="$constraint" --time="$time" \
        --ntasks-per-node="$n" --gres=gpu:"$n" \
        --export=ALL,SIM_DIRS="$sim_dirs" \
        --job-name="$job_name" \
        "$SBATCH_SCRIPT"
}

batch_and_submit() {
    local constraint=$1
    local time=$2
    local name_prefix=$3
    shift 3
    local all_dirs=("$@")

    local batch=()
    local batch_num=1

    for dir in "${all_dirs[@]}"; do
        batch+=("$dir")
        if [[ ${#batch[@]} -eq 4 ]]; then
            submit_batch "$constraint" "$time" "${name_prefix}_b${batch_num}" "${batch[@]}"
            batch=()
            ((batch_num++))
        fi
    done

    # Submit remaining partial batch
    if [[ ${#batch[@]} -gt 0 ]]; then
        submit_batch "$constraint" "$time" "${name_prefix}_b${batch_num}" "${batch[@]}"
    fi
}

echo "=== Seed experiment (full domain 512x512, 8 seeds) ==="

dirs=()
for seed in $(seq 1 8); do
    dirs+=("$BASE/seed_experiment/rt/seed_$seed")
done
batch_and_submit "gpu&hbm80g" "20:00:00" "SENS_seed_rt" "${dirs[@]}"

dirs=()
for seed in $(seq 1 8); do
    dirs+=("$BASE/seed_experiment/standard/seed_$seed")
done
batch_and_submit "gpu" "10:00:00" "SENS_seed_std" "${dirs[@]}"

echo ""
echo "=== Domain experiment (1/4 domain 128x128, 16 reps) ==="

dirs=()
for rep in $(seq -w 1 16); do
    dirs+=("$BASE/domain_experiment/rt/rep_$rep")
done
batch_and_submit "gpu&hbm80g" "05:00:00" "SENS_dom_rt" "${dirs[@]}"

dirs=()
for rep in $(seq -w 1 16); do
    dirs+=("$BASE/domain_experiment/standard/rep_$rep")
done
batch_and_submit "gpu" "03:00:00" "SENS_dom_std" "${dirs[@]}"

echo ""
echo "All jobs submitted! (12 node-jobs x 4 tasks each = 48 simulations)"
