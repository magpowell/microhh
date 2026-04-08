#!/bin/bash
# submit_nazw_only.sh — targeted composite + sw_composite for nazw reps only.
# Skips 3d_to_nc (b.nc already exists). Two sequential jobs: composite then sw_composite.

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
COMPOSITE_ROOT="$LES_ROOT/analysis/cloud_root_composite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

# ── Build nazw-only run/out lists ─────────────────────────────────────────────
RUN_DIRS=()
OUT_DIRS=()
for rt in 2stream raytracer; do
    for rep in 01 02 03 04; do
        rd="$LES_ROOT/experiments/no_aerosols_zero_wind/$rt/rep_$rep"
        od="$COMPOSITE_ROOT/no_aerosols_zero_wind/$rt/rep_$rep"
        if [[ -d "$rd" ]]; then
            RUN_DIRS+=("$rd")
            OUT_DIRS+=("$od")
        fi
    done
done

echo "nazw reps: ${#RUN_DIRS[@]}"
for i in "${!RUN_DIRS[@]}"; do
    echo "  $((i+1))  ${RUN_DIRS[$i]}"
done

if $DRY_RUN; then
    echo "[dry-run] Would submit 2 jobs for ${#RUN_DIRS[@]} reps."
    exit 0
fi

# ── Build array lines for heredocs ────────────────────────────────────────────
_run_lines=""
_out_lines=""
for i in "${!RUN_DIRS[@]}"; do
    _run_lines+="  '${RUN_DIRS[$i]}'"$'\n'
    _out_lines+="  '${OUT_DIRS[$i]}'"$'\n'
done

# ── Job 1: composite ─────────────────────────────────────────────────────────
tmp_cr=$(mktemp /tmp/nazw_composite_XXXXXX.sh)
cat > "$tmp_cr" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --job-name=nazw_cr
#SBATCH --output=${LOG_DIR}/nazw_composite-%j.out
#SBATCH --error=${LOG_DIR}/nazw_composite-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env
export OMP_NUM_THREADS=4

COMPOSITE_PREP='${SCRIPT_DIR}/cloud_root_composite_prep.py'

declare -a RUN_DIRS=(
${_run_lines})
declare -a OUT_DIRS=(
${_out_lines})
HEADER

cat >> "$tmp_cr" <<'LOGIC'
echo "=== nazw composite ==="
echo "Start: $(date)"
echo "Total reps: ${#RUN_DIRS[@]}"

PIDS=()
FAIL=0

for i in "${!RUN_DIRS[@]}"; do
    rd="${RUN_DIRS[$i]}"
    od="${OUT_DIRS[$i]}"
    if [[ -f "$od/events_xz.nc" ]] && \
       ncdump -h "$od/events_xz.nc" 2>/dev/null | grep -q 'b_prime'; then
        echo "SKIP: $od"
        continue
    fi
    echo "START: $rd"
    mkdir -p "$od"
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
[[ $FAIL -eq 0 ]] || { echo "FAILED"; exit 1; }
LOGIC

# ── Job 2: sw_composite ──────────────────────────────────────────────────────
tmp_sw=$(mktemp /tmp/nazw_sw_composite_XXXXXX.sh)
cat > "$tmp_sw" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=nazw_sw
#SBATCH --output=${LOG_DIR}/nazw_sw_composite-%j.out
#SBATCH --error=${LOG_DIR}/nazw_sw_composite-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

SW_SCRIPT='${SCRIPT_DIR}/sw_surface_composite.py'

declare -a RUN_DIRS=(
${_run_lines})
declare -a OUT_DIRS=(
${_out_lines})
HEADER

cat >> "$tmp_sw" <<'LOGIC'
echo "=== nazw sw_composite ==="
echo "Start: $(date)"

PIDS=()
FAIL=0

for i in "${!RUN_DIRS[@]}"; do
    rd="${RUN_DIRS[$i]}"
    od="${OUT_DIRS[$i]}"

    if [[ ! -f "$od/events_xz.nc" ]] && [[ ! -f "$od/events_yz.nc" ]]; then
        echo "SKIP (no events): $od"
        continue
    fi

    rt=$(basename "$(dirname "$rd")")
    echo "START: $rt  $rd"
    (python "$SW_SCRIPT" \
        --run-dir  "$rd" \
        --rt       "$rt" \
        --comp-dir "$od" \
        >> "${od}/sw_composite.log" 2>&1 \
        && echo "DONE: $rd" \
        || echo "FAIL: $rd") &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} background job(s)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "FAILED"; exit 1; }
LOGIC

# ── Submit ────────────────────────────────────────────────────────────────────
jid_cr=$(sbatch --parsable "$tmp_cr")
echo "Submitted composite:    $jid_cr"

jid_sw=$(sbatch --parsable --dependency=afterok:"$jid_cr" "$tmp_sw")
echo "Submitted sw_composite: $jid_sw  (after $jid_cr)"

echo "Logs: $LOG_DIR"
