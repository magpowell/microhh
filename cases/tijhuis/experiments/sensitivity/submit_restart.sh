#!/bin/bash
# Submit restart jobs for simulations that timed out.
#
# Usage:
#   ./submit_restart.sh                          # auto-detect latest checkpoint
#   ./submit_restart.sh 50400                    # restart from specific time
#   ./submit_restart.sh 50400 seed_experiment/rt # restart only one experiment group

set -euo pipefail

BASE=/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS
SBATCH_SCRIPT=$BASE/sbatch_restart.sh

RESTART_TIME="${1:-auto}"
TARGET_GROUP="${2:-}"  # e.g. "seed_experiment/rt" to restart only that group

# --- Helper functions (same batching logic as submit_all.sh) ---

submit_batch() {
    local constraint=$1
    local time=$2
    local job_name=$3
    local restart_time=$4
    shift 4
    local dirs=("$@")
    local n=${#dirs[@]}

    local sim_dirs
    sim_dirs=$(IFS=':'; echo "${dirs[*]}")

    echo "  Submitting: $job_name ($n tasks, restart_time=$restart_time, wall=$time)"
    sbatch --constraint="$constraint" --time="$time" \
        --ntasks-per-node="$n" --gres=gpu:"$n" \
        --export=ALL,SIM_DIRS="$sim_dirs",RESTART_TIME="$restart_time" \
        --job-name="$job_name" \
        "$SBATCH_SCRIPT"
}

batch_and_submit() {
    local constraint=$1
    local time=$2
    local name_prefix=$3
    local restart_time=$4
    shift 4
    local all_dirs=("$@")

    local batch=()
    local batch_num=1

    for dir in "${all_dirs[@]}"; do
        batch+=("$dir")
        if [[ ${#batch[@]} -eq 4 ]]; then
            submit_batch "$constraint" "$time" "${name_prefix}_b${batch_num}" "$restart_time" "${batch[@]}"
            batch=()
            ((batch_num++))
        fi
    done

    if [[ ${#batch[@]} -gt 0 ]]; then
        submit_batch "$constraint" "$time" "${name_prefix}_b${batch_num}" "$restart_time" "${batch[@]}"
    fi
}

# Auto-detect the latest checkpoint time from a directory
detect_restart_time() {
    local dir=$1
    local latest
    latest=$(ls "$dir"/time.* 2>/dev/null | sort -t. -k2 -n | tail -1)
    if [[ -z "$latest" ]]; then
        echo "ERROR: No checkpoint files found in $dir" >&2
        exit 1
    fi
    # Extract time from filename: time.0050400 -> 50400
    basename "$latest" | sed 's/time\.0*//'
}

# Compute remaining wall time: (endtime - restart_time) / restart_time * original_wall * safety_factor
estimate_wall_time() {
    local restart_time=$1
    local endtime=$2
    local original_wall_hours=$3
    local original_simtime=$4

    local remaining=$((endtime - restart_time))
    # Rate: original_wall covers original_simtime of simulation
    # Add 50% safety margin
    local wall_seconds=$(( (remaining * original_wall_hours * 3600 * 3 / 2) / original_simtime ))
    local hours=$((wall_seconds / 3600))
    local mins=$(( (wall_seconds % 3600) / 60 ))
    printf "%02d:%02d:00" "$hours" "$mins"
}

# --- Determine which groups to restart ---

should_restart() {
    local group=$1
    [[ -z "$TARGET_GROUP" ]] || [[ "$TARGET_GROUP" == "$group" ]]
}

# --- Seed experiment / rt ---
if should_restart "seed_experiment/rt"; then
    sample_dir="$BASE/seed_experiment/rt/seed_1"
    if [[ -d "$sample_dir" ]]; then
        rt=${RESTART_TIME}
        if [[ "$rt" == "auto" ]]; then
            rt=$(detect_restart_time "$sample_dir")
        fi

        # endtime=64800, original wall=15h, original simtime=64800
        wall=$(estimate_wall_time "$rt" 64800 15 64800)

        echo "=== Seed experiment / rt (restart from t=$rt, wall=$wall) ==="
        dirs=()
        for seed in $(seq 1 8); do
            dirs+=("$BASE/seed_experiment/rt/seed_$seed")
        done
        batch_and_submit "gpu&hbm80g" "$wall" "RST_seed_rt" "$rt" "${dirs[@]}"
    fi
fi

# --- Seed experiment / standard ---
if should_restart "seed_experiment/standard"; then
    sample_dir="$BASE/seed_experiment/standard/seed_1"
    if [[ -d "$sample_dir" ]]; then
        rt=${RESTART_TIME}
        if [[ "$rt" == "auto" ]]; then
            rt=$(detect_restart_time "$sample_dir")
        fi

        wall=$(estimate_wall_time "$rt" 64800 10 64800)

        echo ""
        echo "=== Seed experiment / standard (restart from t=$rt, wall=$wall) ==="
        dirs=()
        for seed in $(seq 1 8); do
            dirs+=("$BASE/seed_experiment/standard/seed_$seed")
        done
        batch_and_submit "gpu" "$wall" "RST_seed_std" "$rt" "${dirs[@]}"
    fi
fi

# --- Domain experiment / rt ---
if should_restart "domain_experiment/rt"; then
    sample_dir="$BASE/domain_experiment/rt/rep_01"
    if [[ -d "$sample_dir" ]]; then
        rt=${RESTART_TIME}
        if [[ "$rt" == "auto" ]]; then
            rt=$(detect_restart_time "$sample_dir")
        fi

        wall=$(estimate_wall_time "$rt" 64800 5 64800)

        echo ""
        echo "=== Domain experiment / rt (restart from t=$rt, wall=$wall) ==="
        dirs=()
        for rep in $(seq -w 1 16); do
            dirs+=("$BASE/domain_experiment/rt/rep_$rep")
        done
        batch_and_submit "gpu&hbm80g" "$wall" "RST_dom_rt" "$rt" "${dirs[@]}"
    fi
fi

# --- Domain experiment / standard ---
if should_restart "domain_experiment/standard"; then
    sample_dir="$BASE/domain_experiment/standard/rep_01"
    if [[ -d "$sample_dir" ]]; then
        rt=${RESTART_TIME}
        if [[ "$rt" == "auto" ]]; then
            rt=$(detect_restart_time "$sample_dir")
        fi

        wall=$(estimate_wall_time "$rt" 64800 3 64800)

        echo ""
        echo "=== Domain experiment / standard (restart from t=$rt, wall=$wall) ==="
        dirs=()
        for rep in $(seq -w 1 16); do
            dirs+=("$BASE/domain_experiment/standard/rep_$rep")
        done
        batch_and_submit "gpu" "$wall" "RST_dom_std" "$rt" "${dirs[@]}"
    fi
fi

echo ""
echo "Restart jobs submitted."
