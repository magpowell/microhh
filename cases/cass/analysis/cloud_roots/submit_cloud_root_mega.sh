#!/bin/bash
# submit_cloud_root_mega.sh
#
# This script: 2 jobs total.
#
#   Job 1  3d_to_nc   1 CPU node, all reps in parallel (≤MAX_PAR_3D at once)
#   Job 2  composite  1 CPU node, depends on Job 1, all reps (≤MAX_PAR_CR at once)
#
# Both jobs skip reps whose output already exists (thl.nc / events_xz.nc).
# u_10p0/raytracer is included only if rep_01 stats are present at submit time.
#
# Usage:
#   bash submit_cloud_root_mega.sh [--dry-run]

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
COMPOSITE_ROOT="$LES_ROOT/analysis/cloud_root_composite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"

# Parallelism limits (per-node)
# 3d_to_nc: ~2-4 GB RAM each → 32 safe on 512 GB node
# composite: ~32 GB RAM each  → 8-16 safe; use 12 (leaves headroom)
MAX_PAR_3D=32
MAX_PAR_CR=12

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$COMPOSITE_ROOT"

# ── Build run lists ───────────────────────────────────────────────────────────
# Each group: (label, run_root) → reps found under run_root/rep_{01..04}/
# RUN_DIRS / OUT_DIRS: flat lists of per-rep paths

RUN_DIRS=()
OUT_DIRS=()

add_group() {
    local run_root="$1"
    local out_root="$2"
    for rep in 01 02 03 04; do
        local rd="$run_root/rep_$rep"
        if [[ -d "$rd" ]]; then
            RUN_DIRS+=("$rd")
            OUT_DIRS+=("$out_root/rep_$rep")
        fi
    done
}

# ── base ──────────────────────────────────────────────────────────────────────
add_group "$LES_ROOT/base/2stream"   "$COMPOSITE_ROOT/base/2stream"
add_group "$LES_ROOT/base/raytracer" "$COMPOSITE_ROOT/base/raytracer"

# ── no_aerosols ───────────────────────────────────────────────────────────────
add_group "$LES_ROOT/experiments/no_aerosols/2stream"   "$COMPOSITE_ROOT/no_aerosols/2stream"
add_group "$LES_ROOT/experiments/no_aerosols/raytracer" "$COMPOSITE_ROOT/no_aerosols/raytracer"

# ── no_aerosols_zero_wind ─────────────────────────────────────────────────────
add_group "$LES_ROOT/experiments/no_aerosols_zero_wind/2stream"   "$COMPOSITE_ROOT/no_aerosols_zero_wind/2stream"
add_group "$LES_ROOT/experiments/no_aerosols_zero_wind/raytracer" "$COMPOSITE_ROOT/no_aerosols_zero_wind/raytracer"

# ── cs_veg (3 lowest values; two highest suppress clouds) ────────────────────
for val in cs_veg_0 cs_veg_41840 cs_veg_418400; do
    add_group "$LES_ROOT/experiments/cs_veg/$val/2stream"   "$COMPOSITE_ROOT/cs_veg/$val/2stream"
    add_group "$LES_ROOT/experiments/cs_veg/$val/raytracer" "$COMPOSITE_ROOT/cs_veg/$val/raytracer"
done

# ── soil_moisture ─────────────────────────────────────────────────────────────
# Enumerate whatever theta_* dirs exist — handles 0p1, 0p155, 0p17, 0p185, etc.
for val_path in "$LES_ROOT/experiments/soil_moisture/theta_"*/; do
    [[ -d "$val_path" ]] || continue
    val="$(basename "$val_path")"
    add_group "$LES_ROOT/experiments/soil_moisture/$val/2stream"   "$COMPOSITE_ROOT/soil_moisture/$val/2stream"
    add_group "$LES_ROOT/experiments/soil_moisture/$val/raytracer" "$COMPOSITE_ROOT/soil_moisture/$val/raytracer"
done

# ── mean_state_nudge (raytracer only) ─────────────────────────────────────────
add_group "$LES_ROOT/experiments/mean_state_nudge/nudge_3600s/raytracer" \
          "$COMPOSITE_ROOT/mean_state_nudge/nudge_3600s/raytracer"

# ── wind_u ────────────────────────────────────────────────────────────────────
for val in u_0p0 u_2p5 u_5p0 u_7p5; do
    add_group "$LES_ROOT/experiments/wind_u/$val/2stream"   "$COMPOSITE_ROOT/wind_u/$val/2stream"
    add_group "$LES_ROOT/experiments/wind_u/$val/raytracer" "$COMPOSITE_ROOT/wind_u/$val/raytracer"
done
# u_10p0 2stream always; raytracer conditional on run completion
add_group "$LES_ROOT/experiments/wind_u/u_10p0/2stream" "$COMPOSITE_ROOT/wind_u/u_10p0/2stream"
if [[ -f "$LES_ROOT/experiments/wind_u/u_10p0/raytracer/rep_01/cass.default.0000000.nc" ]]; then
    add_group "$LES_ROOT/experiments/wind_u/u_10p0/raytracer" "$COMPOSITE_ROOT/wind_u/u_10p0/raytracer"
else
    echo "NOTE: wind_u/u_10p0/raytracer not yet complete — skipped."
fi

N_REPS=${#RUN_DIRS[@]}
echo "Total reps to process: $N_REPS"
for i in "${!RUN_DIRS[@]}"; do
    printf "  %3d  %s\n" "$((i+1))" "${RUN_DIRS[$i]}"
done

if $DRY_RUN; then
    echo ""
    echo "[dry-run] Would submit 2 jobs for $N_REPS reps."
    echo "  Job 1: 3d_to_nc   (≤${MAX_PAR_3D} parallel)"
    echo "  Job 2: composite  (≤${MAX_PAR_CR} parallel, depends on Job 1)"
    exit 0
fi

# ── Serialise run/out arrays for embedding in batch scripts ──────────────────
_run_array_lines=""
_out_array_lines=""
for rd in "${RUN_DIRS[@]}"; do _run_array_lines+="    '${rd}'"$'\n'; done
for od in "${OUT_DIRS[@]}"; do _out_array_lines+="    '${od}'"$'\n'; done

# ── Job 1: 3d_to_nc ───────────────────────────────────────────────────────────
tmp_3d=$(mktemp /tmp/mega_3d_to_nc_XXXXXX.sh)
cat > "$tmp_3d" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --job-name=3d_to_nc_all
#SBATCH --output=${LOG_DIR}/3d_to_nc_mega-%j.out
#SBATCH --error=${LOG_DIR}/3d_to_nc_mega-%j.err
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR_3D}

declare -a RUN_DIRS=(
${_run_array_lines})

HEADER

cat >> "$tmp_3d" <<'LOGIC'
echo "=== 3d_to_nc (all reps) ==="
echo "Start: $(date)"
echo "Total reps: ${#RUN_DIRS[@]}"

PIDS=()
RUNNING=0
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for rd in "${RUN_DIRS[@]}"; do
    if [[ -f "$rd/b.nc" && -f "$rd/u.nc" && -f "$rd/v.nc" ]]; then
        echo "SKIP (b.nc u.nc v.nc exist): $rd"
        continue
    fi
    echo "START: $rd"
    throttle
    (cd "$rd" && python 3d_to_nc.py -v thl qt ql w b u v \
        >> "${rd}/3d_to_nc.log" 2>&1 \
        && echo "DONE: $rd" \
        || echo "FAIL: $rd") &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} background job(s)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "One or more 3d_to_nc calls failed — check logs in run dirs."; exit 1; }
LOGIC

# ── Job 2: composite ──────────────────────────────────────────────────────────
tmp_cr=$(mktemp /tmp/mega_composite_XXXXXX.sh)
cat > "$tmp_cr" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=460G
#SBATCH --time=06:00:00
#SBATCH --job-name=cass_cr_all
#SBATCH --output=${LOG_DIR}/cr_composite_mega-%j.out
#SBATCH --error=${LOG_DIR}/cr_composite_mega-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env
export OMP_NUM_THREADS=4

MAX_PAR=${MAX_PAR_CR}
COMPOSITE_PREP='${SCRIPT_DIR}/cloud_root_composite_prep.py'

declare -a RUN_DIRS=(
${_run_array_lines})
declare -a OUT_DIRS=(
${_out_array_lines})

HEADER

cat >> "$tmp_cr" <<'LOGIC'
echo "=== cloud_root_composite (all reps) ==="
echo "Start: $(date)"
echo "Total reps: ${#RUN_DIRS[@]}"

PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 5; done
}

for i in "${!RUN_DIRS[@]}"; do
    rd="${RUN_DIRS[$i]}"
    od="${OUT_DIRS[$i]}"
    # Skip only if events_xz.nc already has the circulation fields (b_prime, horiz_wind_prime).
    # Old events without these must be regenerated.
    if [[ -f "$od/events_xz.nc" ]] && \
       ncdump -h "$od/events_xz.nc" 2>/dev/null | grep -q 'b_prime'; then
        echo "SKIP (events_xz.nc has circ fields): $od"
        continue
    fi
    echo "START: $rd"
    mkdir -p "$od"
    throttle
    (python "$COMPOSITE_PREP" --run-dir "$rd" --output-dir "$od" \
        >> "${od}/composite.log" 2>&1 \
        && echo "DONE: $rd" \
        || echo "FAIL: $rd") &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} composite job(s)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "One or more composites failed — check logs in output dirs."; exit 1; }
LOGIC

# ── Job 3: sw_surface_composite ───────────────────────────────────────────────
# Depends on Job 2 (events files must exist). Skips reps whose sw_composite.nc
# already exists so it is safe to re-run without triggering a full recompute.
MAX_PAR_SW=32

tmp_sw=$(mktemp /tmp/mega_sw_composite_XXXXXX.sh)
cat > "$tmp_sw" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=sw_composite_all
#SBATCH --output=${LOG_DIR}/sw_composite_mega-%j.out
#SBATCH --error=${LOG_DIR}/sw_composite_mega-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR_SW}
SW_SCRIPT='${SCRIPT_DIR}/sw_surface_composite.py'

declare -a RUN_DIRS=(
${_run_array_lines})
declare -a OUT_DIRS=(
${_out_array_lines})

HEADER

cat >> "$tmp_sw" <<'LOGIC'
echo "=== sw_surface_composite (all reps) ==="
echo "Start: $(date)"
echo "Total reps: ${#RUN_DIRS[@]}"

PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for i in "${!RUN_DIRS[@]}"; do
    rd="${RUN_DIRS[$i]}"
    od="${OUT_DIRS[$i]}"

    # Skip if output already exists
    if [[ -f "$od/sw_composite.nc" ]]; then
        echo "SKIP (sw_composite.nc exists): $od"
        continue
    fi

    # Skip if no events files (e.g. run produced no clouds)
    if [[ ! -f "$od/events_xz.nc" ]] && [[ ! -f "$od/events_yz.nc" ]]; then
        echo "SKIP (no events): $od"
        continue
    fi

    # Derive rt from path: parent of rep_XX dir is the rt type
    rt=$(basename "$(dirname "$rd")")   # 2stream or raytracer

    echo "START: $rt  $rd"
    throttle
    (python "$SW_SCRIPT" \
        --run-dir  "$rd" \
        --rt       "$rt" \
        --comp-dir "$od" \
        >> "${od}/sw_composite.log" 2>&1 \
        && echo "DONE: $rd" \
        || echo "FAIL: $rd  (see ${od}/sw_composite.log)") &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} background job(s)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "One or more sw_composite calls failed — check per-rep logs."; exit 1; }
LOGIC

# ── Submit ────────────────────────────────────────────────────────────────────
jid_3d=$(sbatch "$tmp_3d" | awk '{print $NF}')
echo "Submitted 3d_to_nc:   $jid_3d"

jid_cr=$(sbatch --dependency=afterok:"$jid_3d" "$tmp_cr" | awk '{print $NF}')
echo "Submitted composite:  $jid_cr  (after $jid_3d)"

jid_sw=$(sbatch --dependency=afterok:"$jid_cr" "$tmp_sw" | awk '{print $NF}')
echo "Submitted sw_composite: $jid_sw  (after $jid_cr)"

rm -f "$tmp_3d" "$tmp_cr" "$tmp_sw"

echo ""
echo "3 jobs submitted for $N_REPS reps. Logs: $LOG_DIR"
